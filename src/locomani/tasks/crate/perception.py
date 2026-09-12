"""Estimate a painted crate's pose from RGB pixels and robot encoder FK only."""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    rotation: np.ndarray  # OpenCV optical axes, expressed in the pelvis frame
    intrinsic: np.ndarray

    def project(self, points):
        local = (np.asarray(points) - self.position) @ self.rotation
        pixel = local @ self.intrinsic.T
        return pixel[:, :2] / np.maximum(local[:, 2:], 1e-5), local[:, 2]


@dataclass(frozen=True)
class CrateEstimate:
    position: np.ndarray
    rotation: np.ndarray
    timestamp: float
    reprojection_px: float
    detections: dict

    def transform(self, points):
        return np.asarray(points) @ self.rotation.T + self.position


@dataclass(frozen=True)
class ApproachTarget:
    position: np.ndarray
    timestamp: float
    source: str


def detect_markers(image):
    """Centroids of four physical colour patches; accepts only the actual RGB image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    masks = {
        name: (hue >= lo) & (hue <= hi) & (saturation > 140) & (value > 90)
        for name, (lo, hi) in {"cyan": (82, 98), "green": (48, 75), "magenta": (138, 162), "yellow": (25, 37)}.items()
    }
    detections = {}
    for name, mask in masks.items():
        count, _, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        candidates = [i for i in range(1, count) if 4 <= stats[i, cv2.CC_STAT_AREA] <= 3000]
        if candidates:
            index = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
            detections[name] = centres[index]
    return detections


class CratePerception:
    streams = {"ego_view": "head_camera", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}

    def __init__(self, ik: ArmIK, geometry: dict, *, max_reprojection_px=4.0):
        self.ik = ik
        self.markers = {name: np.array(marker["position"]) for name, marker in geometry["markers"].items()}
        self.max_reprojection_px = max_reprojection_px
        self.last = None

    def cameras(self, joints, images):
        # FK uses robot joints and fixed sensor extrinsics. Object/world state is absent.
        self.ik.fk(joints, "right")
        m, d = self.ik.model, self.ik.data
        cameras = {}
        for stream, name in self.streams.items():
            if stream not in images:
                continue
            camera = m.camera(name)
            body = int(camera.bodyid[0])
            body_rot = d.xmat[body].reshape(3, 3)
            quat = camera.quat.reshape(4)
            rotation = body_rot @ Rotation.from_quat(np.r_[quat[1:], quat[0]]).as_matrix() @ np.diag([1.0, -1.0, -1.0])
            position = d.xpos[body] + body_rot @ camera.pos.reshape(3)
            height, width = images[stream].shape[:2]
            focal = height / (2 * np.tan(np.radians(float(camera.fovy[0])) / 2))
            cameras[stream] = Camera(
                position.copy(), rotation, np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
            )
        return cameras

    def update(self, images, joints, timestamp):
        cameras = self.cameras(joints, images)
        detections = {stream: detect_markers(images[stream]) for stream in cameras}
        pairs = [
            (stream, name, uv)
            for stream, points in detections.items()
            for name, uv in points.items()
            if name in self.markers
        ]
        if len({name for _, name, _ in pairs}) < 3:
            return None
        candidates = []
        for stream, points in detections.items():
            if len(points) != 4:
                continue
            names = list(self.markers)
            obj = np.array([self.markers[name] for name in names], np.float64)
            pixels = np.array([points[name] for name in names], np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, pixels, cameras[stream].intrinsic, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                rotation = cameras[stream].rotation @ cv2.Rodrigues(rvec)[0]
                position = cameras[stream].position + cameras[stream].rotation @ tvec.ravel()
                candidates.append(np.r_[position, Rotation.from_matrix(rotation).as_rotvec()])
        if self.last is not None:
            candidates.append(np.r_[self.last.position, Rotation.from_matrix(self.last.rotation).as_rotvec()])
        if not candidates:
            return None

        def residual(values, observations):
            rotation = Rotation.from_rotvec(values[3:]).as_matrix()
            return np.concatenate(
                [
                    cameras[stream].project([rotation @ self.markers[name] + values[:3]])[0][0] - uv
                    for stream, name, uv in observations
                ]
            )

        # Reject coloured background fragments using agreement with a valid
        # camera/history hypothesis before combining observations across views.
        support = [
            np.linalg.norm(residual(candidate, pairs).reshape(-1, 2), axis=1) < self.max_reprojection_px * 3
            for candidate in candidates
        ]
        best = int(np.argmax([np.count_nonzero(mask) for mask in support]))
        inliers = [pair for pair, keep in zip(pairs, support[best], strict=True) if keep]
        if len({name for _, name, _ in inliers}) < 3:
            return None
        fit = least_squares(
            lambda values: residual(values, inliers), candidates[best], max_nfev=12, loss="soft_l1", f_scale=2.0
        )
        error = float(np.sqrt(np.mean(residual(fit.x, inliers) ** 2)))
        rotation = Rotation.from_rotvec(fit.x[3:]).as_matrix()
        depths = [
            cameras[stream].project([rotation @ self.markers[name] + fit.x[:3]])[1][0] for stream, name, _ in inliers
        ]
        if error > self.max_reprojection_px or min(depths) < 0.03 or not (0.1 < fit.x[0] < 1.5):
            return None
        self.last = CrateEstimate(
            fit.x[:3],
            rotation,
            timestamp,
            error,
            {key: {name: uv.tolist() for stream, name, uv in inliers if stream == key} for key in cameras},
        )
        return self.last

    def approach_target(self, images, joints, timestamp):
        """Coarse range/bearing from the visible top pair; never a grasp pose.

        The fixed upright basket and roughly frontal approach are shared scene
        priors. Occluded lower markers are not filled in from simulator state.
        """
        if "ego_view" not in images:
            return None
        points = detect_markers(images["ego_view"])
        if not {"cyan", "magenta"} <= points.keys():
            return None
        camera = self.cameras(joints, {"ego_view": images["ego_view"]})["ego_view"]
        rays = [camera.rotation @ np.linalg.solve(camera.intrinsic, [*points[name], 1]) for name in ("cyan", "magenta")]
        span = np.linalg.norm(self.markers["cyan"] - self.markers["magenta"])
        matrix = np.column_stack((rays[0], -rays[1]))
        scales = np.linalg.lstsq(matrix, [0, span, 0], rcond=None)[0]
        if min(scales) <= 0 or np.linalg.norm(matrix @ scales - [0, span, 0]) > 0.025:
            return None
        midpoint = camera.position + (rays[0] * scales[0] + rays[1] * scales[1]) / 2
        local_midpoint = (self.markers["cyan"] + self.markers["magenta"]) / 2
        position = midpoint - Rotation.from_euler("z", 90, degrees=True).apply(local_midpoint)
        if not 0.45 < position[0] < 3.0 or abs(position[1]) > 0.6:
            return None
        return ApproachTarget(position, timestamp, "visible_top_pair")
