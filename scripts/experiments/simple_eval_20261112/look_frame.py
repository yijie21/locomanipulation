"""The RGB-D look's dump for the path planner (selfwm TODO 38.11, 2026-11-24): each episode's first look through the
replica's MuJoCo head camera, the whole frame -- colour, each pixel's point in the robot's frame, which pixels have a
depth -- with the pelvis belief and the camera's pose, in the fields blind.py look writes for the stereo look
(look_dump_eyes) and curobo_avoid_20261120/obstacles_box.py reads (ok, pts, pb, yaw, cam_p).

blind.py look writes its dump only with --eyes, so the RGB-D look (cloud_blind, 2026-11-19) has none. This stands the
robot as blind.py look does (blind.setup: the replica, ScaleBFM, the 1 s hold) and takes the same frame
(place.Belief.scene), without SAM 3: the target's points stay cloud_blind's. Each episode prints how far cloud_blind's
points lie from this frame's points (median and 95th percentile, cm); near 0 says both share one frame.

    BFM_ROOT=odo OMP_NUM_THREADS=1 MUJOCO_GL=egl PYTHONPATH=src $PY look_frame.py <task> [--only <i> ...]
        -> outputs/simple_eval_20261112/look_dump_blind/<task>_L<level>_e<index>.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blind as B  # noqa: E402
import episodes as EP  # noqa: E402
import pick as P  # noqa: E402

from locomani.backends.scalebfm import runtime as R  # noqa: E402
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig  # noqa: E402

DUMP = "look_dump_blind"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--only", type=int, nargs="*", help="positions in the task's list of episodes")
    ap.add_argument("--cloud", default="cloud_blind", help="the look whose target points are checked against the frame")
    a = ap.parse_args()
    out = EP.OUT / DUMP
    out.mkdir(parents=True, exist_ok=True)
    R.OFF[:] = P.PALM  # as blind.main
    pol = R.Policy("cuda", ScaleBfmBackendConfig())
    eps = EP.episodes(a.task)
    for ep in [eps[i] for i in a.only] if a.only is not None else eps:
        info, sim, mo, bel = B.setup(pol, ep)
        name = f"{a.task}_L{ep['level']}_e{ep['index']}"
        rgb, pts, ok = bel.scene()
        pb, yaw = bel.pelvis()
        cam = sim.m.camera(EP.HEAD_CAM).id
        np.savez_compressed(
            out / f"{name}.npz",
            rgb=rgb,
            pts=pts.astype(np.float32),
            ok=ok,
            pb=pb,
            yaw=yaw,
            cam_p=sim.d.cam_xpos[cam].copy(),
            cam_R=sim.d.cam_xmat[cam].reshape(3, 3).copy(),
            fovy=sim.m.cam_fovy[cam],
            instruction=ep["instruction"],
        )
        row = dict(name=name, ok_px=int(ok.sum()), pb=np.round(pb, 4).tolist(), yaw=round(float(yaw), 5))
        cloud = EP.OUT / a.cloud / f"{name}.npz"
        if cloud.exists():
            d = cKDTree(pts[ok]).query(np.load(cloud)["pts"])[0]
            row.update(
                cloud_cm_median=round(100 * float(np.median(d)), 3),
                cloud_cm_p95=round(100 * float(np.percentile(d, 95)), 3),
            )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
