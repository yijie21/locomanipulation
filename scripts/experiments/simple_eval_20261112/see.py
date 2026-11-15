"""One look from SIMPLE's head camera at the start of each episode: the target's points from the depth image.

The robot stands at the episode's start for a second and the head camera's depth is rendered. The target's pixels come
from the simulator's segmentation (privileged, used by the baselines pick.py and place.py; blind.py uses the functions
after `head_cloud`, which read no simulator state; SIMPLE's GraspNet objects have no textures in MuJoCo for a segmenter to
read), the points from the depth and the camera's true pose. The target's pose is saved only to carry grasps to pick.py
in the object's frame.

    BFM_ROOT=true MUJOCO_GL=egl PYTHONPATH=src python see.py XMovePickTeleop
-> <out>/cloud/<task>_L<level>_e<index>.npz (pts, p_o, R_o, table_top, depth, mask)
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import episodes as EP  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402
from locomani.backends.scalebfm.motor import Motor, MotorConfig  # noqa: E402


def head_points(sim, body: int):
    """The body's points in the head camera's frame (MuJoCo's: x right, y up, looking along -z), from its depth image;
    also the depth image and the body's mask (from the simulator's segmentation)."""
    W, H = EP.HEAD_CAM_WH
    sim.m.vis.quality.offsamples = 0  # multisampling blends segment ids at edges into ids of nothing
    r = mujoco.Renderer(sim.m, H, W)
    r.update_scene(sim.d, camera=EP.HEAD_CAM)
    r.enable_depth_rendering()
    depth = r.render().copy()
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(sim.d, camera=EP.HEAD_CAM)  # the scene's segment ids are set when it is updated in this mode
    geom = r.render()[..., 0]
    r.close()
    mask = np.isin(geom, np.flatnonzero(sim.m.geom_bodyid == body)) & (depth > 0.2) & (depth < 5)
    f = 0.5 * H / np.tan(np.radians(sim.m.cam_fovy[sim.m.camera(EP.HEAD_CAM).id]) / 2)
    v, u = np.nonzero(mask)
    z = depth[v, u]
    return np.stack([(u - W / 2 + 0.5) * z / f, -(v - H / 2 + 0.5) * z / f, -z], axis=1), depth, mask


def head_cloud(sim, body: int):
    """The body's points in the world from the head camera's depth; also the depth image and the body's mask."""
    c, depth, mask = head_points(sim, body)
    cam = sim.m.camera(EP.HEAD_CAM).id
    return sim.d.cam_xpos[cam] + c @ sim.d.cam_xmat[cam].reshape(3, 3).T, depth, mask


# ---- what the robot itself has: the head camera's colour and depth, its joint readings, a segmenter (no simulator state)

SAM3 = ("127.0.0.1", int(__import__("os").environ.get("SAM3_PORT", 18832)))  # scripts/rsi/sam3_server.py


def head_frame(sim):
    """The head camera's colour image and depth image (m along the optical axis)."""
    if hasattr(sim, "eyes"):  # SIMPLE's environment: the Isaac Sim image, depth from the stereo pair
        return sim.eyes()
    W, H = EP.HEAD_CAM_WH
    g = sim.m.vis.global_
    g.offwidth, g.offheight = max(W, g.offwidth), max(H, g.offheight)
    r = mujoco.Renderer(sim.m, H, W)
    r.update_scene(sim.d, camera=EP.HEAD_CAM)
    rgb = r.render().copy()
    r.enable_depth_rendering()
    r.update_scene(sim.d, camera=EP.HEAD_CAM)
    depth = r.render().copy()
    r.close()
    return rgb, depth


def cam_points(depth: np.ndarray) -> np.ndarray:
    """Each pixel's point in the camera's frame (x right, y up, -z forward), H x W x 3, by the camera's calibration."""
    H, W = depth.shape
    f = 0.5 * H / np.tan(np.radians(EP.HEAD_CAM_FOVY) / 2)
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([(u - W / 2 + 0.5) * depth / f, -(v - H / 2 + 0.5) * depth / f, -depth], axis=2)


class HeadKin:
    """Where the head camera sits on the pelvis, from the joint readings and the robot's own model."""

    def __init__(self, model, qadr):
        self.m, self.d, self.qadr, self.cam = model, mujoco.MjData(model), qadr, model.camera(EP.HEAD_CAM).id

    def __call__(self, joints):
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[:3], d.qpos[3:7], d.qpos[self.qadr] = 0.0, (1.0, 0.0, 0.0, 0.0), joints
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_camlight(self.m, d)
        return d.cam_xpos[self.cam].copy(), d.cam_xmat[self.cam].reshape(3, 3).copy()


_sam = None


def find(rgb: np.ndarray, phrases, floor: float = 0.3):
    """SAM 3's best-scored instance over the phrases: (mask, score, phrase), or (None, 0, None)."""
    global _sam
    from locomani.tasks.scene.services import Client, unpack_mask

    _sam = _sam or Client(SAM3)
    found = _sam.call(dict(image=rgb, prompts=list(phrases)))["results"]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    out = EP.OUT / "cloud"
    out.mkdir(parents=True, exist_ok=True)
    pol = R.Policy("cuda", ScaleBfmBackendConfig())
    for ep in EP.episodes(a.task)[: a.n]:
        info = EP.build(ep)
        sim = R.Sim(pol.meta, backend=ScaleBfmBackendConfig(scene_dir=info["dir"], object_name=""))
        mo = Motor(sim, pol, MotorConfig(), model_xml=info["dir"] + "/scene_locomani.xml")
        mo.reset(info["start"]["xy"], info["start"]["yaw"])
        EP.pull_objects(sim.m, sim.d, info)
        mo.hold(1.0)
        b = sim.m.body(info["target"]).id
        pts, depth, mask = head_cloud(sim, b)
        np.savez_compressed(out / f"{a.task}_L{ep['level']}_e{ep['index']}.npz", pts=pts.astype(np.float32),
                            p_o=sim.d.xpos[b].copy(), R_o=sim.d.xmat[b].reshape(3, 3).copy(), depth=depth, mask=mask,
                            table_top=max(t["top"] for t in info["tables"].values()))  # fmt: skip
        err = np.abs(np.linalg.norm(pts - sim.d.xpos[b], axis=1)).max() if len(pts) else float("nan")
        print(
            f"L{ep['level']} e{ep['index']}: {len(pts)} points, farthest from the body origin {100 * err:.1f} cm",
            flush=True,
        )


if __name__ == "__main__":
    main()
