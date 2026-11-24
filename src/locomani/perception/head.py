"""The head camera as the robot has it, and what it can read from a picture (moved from simple_eval_20261112/see.py, TODO 49.2).

The head's colour and depth images, the camera's calibration, where it sits on the pelvis from the joint readings
(HeadKin), SAM 3 on the colour image (find), the table's level and edge from the points around the target (support_of,
near_edge) and the largest thing on a surface (largest_thing). Nothing here reads the simulator's state of the scene. The
privileged segmentation helpers of the old file (head_points, head_cloud: the target's pixels from the simulator) are not
perception and were left out; SIMPLE's head camera constants live here because the scene builder and the view both need them.
"""

import mujoco
import numpy as np

from locomani.perception.services import Client, unpack_mask

HEAD_CAM = "head"
HEAD_CAM_POS = [0.05366004 + 0.0039635, 0.01752999, 0.4738702 - 0.044]  # in torso_link
HEAD_CAM_QUAT = [0.91496, 0.0, 0.40355, 0.0]  # the camera's x forward, z up frame in torso_link
HEAD_CAM_WH = (640, 360)
HEAD_CAM_FOVY = float(np.degrees(2 * np.arctan(np.tan(np.radians(110) / 2) * 360 / 640)))  # 110 deg across the width


# ---- what the robot itself has: the head camera's colour and depth, its joint readings, a segmenter (no simulator state)


def head_frame(sim):
    """The head camera's colour image and depth image (m along the optical axis)."""
    if hasattr(sim, "eyes"):  # SIMPLE's environment: the Isaac Sim image, depth from the stereo pair
        return sim.eyes()
    W, H = HEAD_CAM_WH
    g = sim.m.vis.global_
    g.offwidth, g.offheight = max(W, g.offwidth), max(H, g.offheight)
    r = mujoco.Renderer(sim.m, H, W)
    r.update_scene(sim.d, camera=HEAD_CAM)
    rgb = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(sim.d, camera=HEAD_CAM)
    depth = r.render().copy()
    r.close()
    return rgb, depth


def intrinsics() -> np.ndarray:
    """The head camera's pinhole intrinsics (pixels): the calibration cam_points uses, as the K a Frame carries."""
    W, H = HEAD_CAM_WH
    f = 0.5 * H / np.tan(np.radians(HEAD_CAM_FOVY) / 2)
    return np.array([[f, 0.0, W / 2 - 0.5], [0.0, f, H / 2 - 0.5], [0.0, 0.0, 1.0]])


def cam_points(depth: np.ndarray) -> np.ndarray:
    """Each pixel's point in the camera's frame (x right, y up, -z forward), H x W x 3, by the camera's calibration. A pixel with
    no depth (NaN or infinite: a RealSense-like hole, the renderer's sky) is depth 0, like the stereo model's invalid pixels:
    its point is the camera's origin and callers mask it by `depth > 0.2`."""
    depth = np.where(np.isfinite(depth), depth, 0.0)
    H, W = depth.shape
    f = 0.5 * H / np.tan(np.radians(HEAD_CAM_FOVY) / 2)
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([(u - W / 2 + 0.5) * depth / f, -(v - H / 2 + 0.5) * depth / f, -depth], axis=2)


class HeadKin:
    """Where the head camera sits on the pelvis, from the joint readings and the robot's own model."""

    def __init__(self, model, qadr):
        self.m, self.d, self.qadr, self.cam = model, mujoco.MjData(model), qadr, model.camera(HEAD_CAM).id

    def __call__(self, joints):
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[:3], d.qpos[3:7], d.qpos[self.qadr] = 0.0, (1.0, 0.0, 0.0, 0.0), joints
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        return d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3).copy()


_SAM: dict = {}  # address -> the connection to that SAM 3 service (one per process; the bed is single-threaded)


def sam_results(sam3, rgb: np.ndarray, phrases) -> dict:
    """Every instance SAM 3 gives for each phrase: {phrase: [(score, packed mask)]}. `sam3` is the service's (host, port);
    the connection opens on the first call."""
    key = tuple(sam3)
    if key not in _SAM:
        _SAM[key] = Client(key)
    return _SAM[key].call(dict(image=rgb, prompts=list(phrases)))["results"]


def find(sam3, rgb: np.ndarray, phrases, floor: float = 0.3):
    """SAM 3's best-scored instance over the phrases: (mask, score, phrase), or (None, 0, None)."""
    found = sam_results(sam3, rgb, phrases)
    best = (None, 0.0, None)
    for p in phrases:
        for score, packed in found.get(p, []):
            if score > max(best[1], floor):
                best = (unpack_mask(packed).astype(bool), float(score), p)
    return best


def support_of(pts, ok, mask) -> float:
    """The height of what the masked thing stands on: the level of the points around its foot."""
    obj = pts[mask & ok]
    lo, c = np.percentile(obj[:, 2], 2), np.median(obj[:, :2], axis=0)
    near = ok & ~mask & (np.linalg.norm(pts[..., :2] - c, axis=2) < 0.35) & (np.abs(pts[..., 2] - lo) < 0.04)
    z = pts[near][:, 2]
    if len(z) < 50:
        return float(lo)
    h, e = np.histogram(z, bins=np.arange(lo - 0.04, lo + 0.045, 0.005))
    top = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    return float(np.median(z[np.abs(z - top) < 0.006]))


def near_edge(pts, ok, top: float, ahead, lateral: float, after: float, half: float = 0.3):
    """Where the surface at height `top` begins along the unit vector `ahead` (the coordinate p . ahead of its near
    edge), in a band of `half` to either side of `lateral` (p . left); only points beyond `after` count (nearer ones
    are the robot's own body)."""
    u, w = pts[..., :2] @ ahead[:2], pts[..., :2] @ np.array([-ahead[1], ahead[0]])
    sel = ok & (np.abs(pts[..., 2] - top) < 0.012) & (np.abs(w - lateral) < half) & (u > after)
    return float(np.percentile(u[sel], 0.5)) if sel.sum() >= 50 else None


def largest_thing(pts, ok, ahead, after: float):
    """The mask of the largest thing standing on the surface that fills most of the view beyond `after` along `ahead`
    (for an instruction that names no object), or None."""
    from scipy import ndimage

    far = ok & (pts[..., :2] @ ahead[:2] > after)
    z = pts[far][:, 2]
    if len(z) < 500:
        return None
    lo = np.percentile(z, 1) + 0.2  # above the floor
    h, e = np.histogram(z[z > lo], bins=np.arange(lo, z.max() + 0.02, 0.01))
    top = 0.5 * (e[h.argmax()] + e[h.argmax() + 1])
    top = float(np.median(z[np.abs(z - top) < 0.008]))
    lab, n = ndimage.label(far & (pts[..., 2] > top + 0.012) & (pts[..., 2] < top + 0.4))
    best = None
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() >= 150 and pts[m][:, 2].min() < top + 0.03 and (best is None or m.sum() > best.sum()):
            best = m
    return best
