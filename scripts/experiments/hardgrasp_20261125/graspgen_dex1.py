"""selfmodel_hardgrasp_20261125 (BiGym): GraspGen-X grasps for the Dex1 on the look's object clouds, as poses of BiGym's
right_end_effector (the Dex1 pinch-centre site) in the look's pelvis frame F0.

Runs in GraspGen-X's own venv, one GPU (1-3; GPU 0 is not ours):

    G=/wuji-vefps-D/wuji-il/yijie/tools/perception/GraspGenX
    CUDA_VISIBLE_DEVICES=1 $G/.venv/bin/python graspgen_dex1.py <look_dir> [--only <prefix>] [--num-grasps 300] [--watch]

Input: every <look_dir>/<name>_obj.npz with key `pts` ((N, 3) object points, F0, metres; other keys ignored).
Output: <look_dir>/<name>_grasps.npz, skipped when it exists and is newer than its cloud:
    T      (n, 4, 4) float64  right_end_effector's pose in F0 (column vectors): R[:, 0] the feed (the way the fingers
                              point and go in), R[:, 1] the fingers' sliding (closing) axis, R[:, 2] = R[:, 0] x R[:, 1]
    score  (n,) float32       GraspGen-X's discriminator confidence, best first (T in the same order)
    n_pts  int                points given to GraspGen-X (after dropping stray pixels > 0.3 m from the median)
--watch: keep scanning for new clouds every 2 s (the look stage writes them while this runs).

The fixed transform from GraspGen-X's Dex1 grasp frame G to right_end_effector S, T_GS = BASE_T @ T_BS:
  BASE_T  x_grippers/dex1/config.json "base_rotation": the pose of the description's root link (gripper.urdf's root is
          right_dex1_base_link: +X approach, +-Y closing axis) in G (GraspGen-X's scripts/vis_gripper_desc.py renders
          every link at base_T @ T_link); G: +z approach (fingertip at z = 0.1433), x the fingers' opening axis.
  T_BS    right_end_effector in right_dex1_base_link, measured in BiGym's model (bigym.loco.make("dishwasher_unload_cups"),
          inner_env.model, reset 732000, mj_kinematics): rotation identity, translation (0.1065, 0, 0) (the base link
          sits 0.0415 m beyond right_wrist_yaw_link, the site 0.148 m: BiGym's G1Dex1._EE_SITE_POS).
  -> T_GS: S's x = G's +z (approach), S's y = G's +x (opening axis), S at z = 0.1065 in G (the fingertip 0.1433 less
     3.7 cm: grasp_tol.py's PAD_X = (-0.029, 0.037)).
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg

HAND = "dex1"
ASSETS = "/wuji-vefps-D/wuji-il/yijie/tools/perception/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets"
T_BS = np.array([[1.0, 0, 0, 0.1065], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])


def base_T(assets_dir: str) -> np.ndarray:
    cfg = json.load(open(os.path.join(assets_dir, "x_grippers", HAND, "config.json")))
    return np.array(cfg["base_rotation"], float)


def self_check(T_GS: np.ndarray) -> None:
    R, p = T_GS[:3, :3], T_GS[:3, 3]
    print("T_GS (right_end_effector in GraspGen-X's Dex1 grasp frame):\n", np.round(T_GS, 4))
    assert np.allclose(R[:, 0], [0, 0, 1]), "site x must be G's +z (approach)"
    assert np.allclose(np.abs(R[:, 1]), [1, 0, 0]), "site y must be G's +-x (opening axis)"
    assert 0.10 <= p[2] <= 0.11 and np.allclose(p[:2], 0), "site must sit 0.10-0.11 m along G's approach"
    print("self-check ok: site x = G +z, site y = G +x, site at z =", round(float(p[2]), 4), flush=True)


def one(f: str, out: str, sampler, T_GS, num_grasps: int) -> None:
    z = np.load(f)
    pts = np.asarray(z["pts"], float)
    pts = pts[np.linalg.norm(pts - np.median(pts, axis=0), axis=1) < 0.3]  # stray pixels of the mask
    t0 = time.time()
    if len(pts) < 10:
        np.savez_compressed(out, T=np.zeros((0, 4, 4)), score=np.zeros(0, np.float32), n_pts=len(pts))
        print(os.path.basename(f), len(pts), "points: too few, no grasps", flush=True)
        return
    np.random.seed(0)
    torch.manual_seed(0)
    G, conf, _, _ = run_planner_on_object(pts.astype(np.float32), sampler, planner="graspmoe", grasp_threshold=-1.0,
                                          num_grasps=num_grasps, topk_num_grasps=-1)  # fmt: skip
    conf = np.asarray(conf, np.float32)
    o = np.argsort(-conf)
    T = np.asarray(G, float)[o] @ T_GS if len(o) else np.zeros((0, 4, 4))
    tmp = out[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, T=T, score=conf[o], n_pts=len(pts))
    os.replace(tmp, out)  # atomic: a reader never sees half a file
    print(os.path.basename(f), len(pts), "points,", len(o), "grasps, top", conf[o][:3].round(3),
          f"{time.time() - t0:.1f}s", flush=True)  # fmt: skip


def scan(a, sampler, T_GS) -> int:
    n = 0
    for f in sorted(glob.glob(os.path.join(a.dir, a.only + "*_obj.npz"))):
        out = f[: -len("_obj.npz")] + "_grasps.npz"
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(f):
            continue
        try:
            one(f, out, sampler, T_GS, a.num_grasps)
        except Exception as e:  # noqa: BLE001  (a bad cloud must not stop the others)
            print(os.path.basename(f), "FAILED:", type(e).__name__, e, flush=True)
            np.savez_compressed(out, T=np.zeros((0, 4, 4)), score=np.zeros(0, np.float32), n_pts=-1)
        n += 1
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--assets-dir", default=ASSETS)
    ap.add_argument("--num-grasps", type=int, default=300)
    ap.add_argument("--only", default="", help="clouds whose name starts with this")
    ap.add_argument("--watch", action="store_true", help="keep scanning every 2 s")
    a = ap.parse_args()
    T_GS = base_T(a.assets_dir) @ T_BS
    self_check(T_GS)
    ckpt = str(get_checkpoints_version_dir())
    sampler = GraspGenXSampler(
        load_model_cfg(os.path.join(ckpt, "gen"), os.path.join(ckpt, "dis")), HAND, assets_dir=a.assets_dir
    )
    scan(a, sampler, T_GS)
    while a.watch:
        time.sleep(2.0)
        scan(a, sampler, T_GS)
