"""RGB cap localization for the calibrated, upright bottled_water_1 asset.

The green cap is a known cylinder, not a privileged object-pose measurement.
Fit its silhouette in the actual ego/wrist images using robot FK and calibration.
This deliberately narrow detector is not a general bottle recognizer.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from locomani.tasks.bottle.image_servo_config import BottleCalibration

DEFAULT_CALIBRATION = BottleCalibration()


@dataclass(frozen=True)
class CameraGeometry:
    position: np.ndarray
    rotation: np.ndarray  # MuJoCo camera: x right, y up, -z forward.
    fovy: float
    height: int = 480
    width: int = 640

    @property
    def focal(self):
        return self.height / (2 * np.tan(np.deg2rad(self.fovy) / 2))

    def project(self, points):
        local = (np.asarray(points) - self.position) @ self.rotation
        depth = -local[:, 2]
        xy = local[:, :2] / np.maximum(depth[:, None], 1e-6)
        return np.column_stack((self.width / 2 + self.focal * xy[:, 0], self.height / 2 - self.focal * xy[:, 1]))


def detect_cap(rgb, image_up, calibration=DEFAULT_CALIBRATION):
    """Highest isolated green component; reject cropped/too-small silhouettes."""
    hsv = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([32, 55, 35]), np.array([91, 255, 255]))
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    # JPEG ringing on metal produces small dark green components. Require
    # support from the cap's bright green interior, retaining its darker rim
    # in the fitted silhouette instead of shrinking the depth cue.
    core = cv2.inRange(hsv, np.array([32, 55, 100]), np.array([68, 255, 255])) > 0
    candidates = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if np.count_nonzero(core & (labels == i)) < 0.25 * area:
            continue
        if (
            np.count_nonzero((hsv[:, :, 2] >= calibration.bright_value) & (labels == i))
            < calibration.bright_fraction * area
        ):
            continue
        candidates.append((float(centres[i] @ image_up), np.array([x, y, x + w, y + h], dtype=float), int(area)))
    if not candidates:
        return None
    _, box, area = max(candidates, key=lambda row: row[0])
    x, y, right, bottom = box
    if area < calibration.min_cap_pixels or min(right - x, bottom - y) < 2:
        return None
    margin = max(2.0, right - x)
    if x <= margin or y <= 1 or right >= rgb.shape[1] - margin or bottom >= rgb.shape[0] - 1:
        return None
    return dict(box=box, pixels=area)


def cap_points(up, calibration=DEFAULT_CALIBRATION):
    up = np.asarray(up, dtype=float)
    side = np.cross(up, [1.0, 0.0, 0.0])
    if np.linalg.norm(side) < 0.1:
        side = np.cross(up, [0.0, 1.0, 0.0])
    side /= np.linalg.norm(side)
    other = np.cross(up, side)
    angle = np.linspace(0.0, 2 * np.pi, 48, endpoint=False)
    ring = calibration.cap_radius * (np.cos(angle)[:, None] * side + np.sin(angle)[:, None] * other)
    return np.concatenate([ring + up * z for z in (calibration.cap_bottom, calibration.cap_top)])


def fit_bottle(images, cameras, up, calibration=DEFAULT_CALIBRATION):
    """Return current RGB-derived root-frame centre, or None; no temporal prior."""
    observations = []
    offsets = cap_points(up, calibration)
    mid = (calibration.cap_top + calibration.cap_bottom) / 2
    for key, camera in cameras.items():
        if key not in images or images[key] is None:
            continue
        local_up = camera.rotation.T @ up
        image_up = np.array([local_up[0], -local_up[1]])
        found = detect_cap(images[key], image_up, calibration)
        if found is None:
            continue
        box = found["box"]
        width = box[2] - box[0]
        depth = camera.focal * (2 * calibration.cap_radius) / width
        pixel = (box[:2] + box[2:]) / 2
        ray = np.array(
            [(pixel[0] - camera.width / 2) / camera.focal, -(pixel[1] - camera.height / 2) / camera.focal, -1.0]
        )
        initial = camera.position + camera.rotation @ (depth * ray) - mid * up

        def residual(centre, cam=camera, bounds=box):
            uv = cam.project(centre + offsets)
            return np.r_[uv.min(0), uv.max(0)] - bounds

        fitted = least_squares(residual, initial, max_nfev=16, diff_step=1e-4)
        error = float(np.sqrt(np.mean(fitted.fun**2)))
        if error > calibration.max_fit_error_px or not 0.05 < np.linalg.norm(fitted.x - camera.position) < 3:
            continue
        observations.append(
            dict(camera=key, centre=fitted.x, box=box, pixels=found["pixels"], error_px=error, jac=fitted.jac)
        )
    if not observations:
        return None
    # Multi-camera consistency rejects a label mistakenly identified as the cap.
    best = max(observations, key=lambda row: row["pixels"])
    compatible = observations
    if len(compatible) > 1:

        def combined(centre):
            parts = []
            for row in compatible:
                uv = cameras[row["camera"]].project(centre + offsets)
                parts.extend(np.r_[uv.min(0), uv.max(0)] - row["box"])
            return parts

        fitted = least_squares(combined, best["centre"], max_nfev=24)
        if np.sqrt(np.mean(fitted.fun**2)) > calibration.max_fit_error_px:
            # Moving wrist proprioception can lag the RGB capture. Keep a
            # coarse single view available, but never authorize CLOSE from
            # an inconsistent camera pair.
            #
            # Prefer the HEAD view here, not the best-conditioned one. Selecting
            # by conditioning picks whichever silhouette is larger, which near
            # the table is the wrist -- the camera this disagreement is blamed
            # on. Measured over attempts 08/09 (622 logged fits, audited against
            # object truth): the fused pair is calibrated (uncertainty 0.017 m,
            # true error 0.018 m), while wrist-only fits reported 0.076 m and
            # were off by 0.545 m, of which 98% was pure depth, placing the
            # bottle 0.53 m NEARER than it is. A single silhouette fixes bearing
            # well and depth badly, and the head camera is rigid to the pelvis
            # while the wrist camera is the one that moves and lags.
            head = [row for row in observations if row["camera"] == "ego_view"]
            best = head[0] if head else observations[0]
            compatible = [best]
            centre, jac = best["centre"], best["jac"]
        else:
            centre = fitted.x
            jac = fitted.jac
    else:
        centre = best["centre"]
        jac = best["jac"]
    # Single-view depth from a small silhouette is poorly conditioned. It may
    # guide a view/approach, but must not authorize precision closure.
    enough_pixels = len(compatible) > 1 or (
        best["camera"] == "right_wrist" and best["pixels"] >= calibration.precise_wrist_pixels
    )
    singular = np.linalg.svd(jac, compute_uv=False)
    uncertainty = float(calibration.edge_noise_px / max(singular[-1], 1e-9))
    # NOTE (2026-09-11): on a disagreeing pair this number is not a safety net.
    # Audited over attempts 08/09 it read 0.076 m on wrist-only fits whose true
    # error was 0.545 m -- a confidently wrong monocular depth is self-consistent,
    # so its own conditioning looks fine. A width-sensitivity term was tried and
    # still let 98% of those through, because the fit believes the bottle is near.
    # What actually separates them is cross-view agreement, which is why the
    # disagreement branch above now keeps the head view rather than the wrist.
    precise = enough_pixels and uncertainty < calibration.close_uncertainty_m and len(compatible) == len(observations)
    return dict(
        centre=centre,
        views=observations,
        selected=[row["camera"] for row in compatible],
        precise=precise,
        uncertainty_m=uncertainty,
    )


class BottleImageObserver:
    """Camera transport plus FK; object truth never enters fit_bottle."""

    def __init__(self, ik, calibration=DEFAULT_CALIBRATION):
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        from locomani.sim import ports

        self.client = ComposedCameraClientSensor(port=ports.camera())
        self.ik = ik
        self.calibration = calibration
        self.latest = None

    def observe(self, state):
        import time

        from locomani.backends.sonic.primitives import Primitives

        message = self.client.read()
        if message is None:
            return None
        images = {
            key: rgb
            for key, rgb in message["images"].items()
            if key in ("ego_view", "right_wrist")
            and 0 <= time.time() - message["timestamps"].get(key, 0) < self.calibration.max_image_age_s
        }
        self.ik.fk(np.asarray(state["joints"]), "right")
        mj, model, data = self.ik._mj, self.ik.model, self.ik.data
        mj.mj_camlight(model, data)
        cameras = {}
        for key, name in (("ego_view", "head_camera"), ("right_wrist", "right_wrist")):
            if key not in images:
                continue
            cid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, name)
            cameras[key] = CameraGeometry(
                data.cam_xpos[cid].copy(),
                data.cam_xmat[cid].reshape(3, 3).copy(),
                float(model.cam_fovy[cid]),
                *images[key].shape[:2],
            )
        up = Primitives._rot(state["pelvis"]["quat"]).T @ np.array([0.0, 0.0, 1.0])
        self.latest = fit_bottle(images, cameras, up, self.calibration)
        return self.latest

    def close(self):
        self.client.close()
