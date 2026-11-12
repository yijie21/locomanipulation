"""One look from SIMPLE's head camera at the start of each episode: the target's points from the depth image.

The robot stands at the episode's start for a second and the head camera's depth is rendered. The target's pixels come
from the simulator's segmentation (privileged; SIMPLE's GraspNet objects have no textures in MuJoCo for a segmenter to
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


def head_cloud(sim, body: int):
    """The body's points in the world from the head camera's depth; also the depth image and the body's mask."""
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
    cam = sim.m.camera(EP.HEAD_CAM).id
    f = 0.5 * H / np.tan(np.radians(sim.m.cam_fovy[cam]) / 2)
    v, u = np.nonzero(mask)
    z = depth[v, u]
    c = np.stack([(u - W / 2 + 0.5) * z / f, -(v - H / 2 + 0.5) * z / f, -z], axis=1)  # MuJoCo's camera looks along -z
    return sim.d.cam_xpos[cam] + c @ sim.d.cam_xmat[cam].reshape(3, 3).T, depth, mask


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
