"""GraspGen-X with its Unitree G1 three-finger hand (the right Dex3) on the head-camera clouds of the look pass, and the
truth grasp library's loader (moved from simple_eval_20261112/grasps_graspgen.py and escalation_20261116/grasp_lib.py, TODO 49.2).

Runs in GraspGen-X's own venv, one GPU. GraspGen-X gives the pose of the hand's grasp frame and no finger angles; its
gripper description carries one open and one closed finger posture, saved here in the Dex3 joint order of the ScaleBFM
runtime (thumb 0-2, index 0-1, middle 0-1). The palm poses are saved in the object's frame, best score first.

The process is started by locomani.bench.simple.session in GraspGen-X's own venv as a plain script (this file imports no
locomani module, only numpy/torch/graspgenx there):

    G=third_party/GraspGenX      # under the main checkout
    CUDA_VISIBLE_DEVICES=1 $G/.venv/bin/python src/locomani/perception/grasps.py <out>/cloud \
        --assets-dir $G/ext/gripper_descriptions/gripper_descriptions/assets
-> <out>/cloud/grasp_<name>.npz (T_op: n x 4 x 4 palm poses, score, f_open, f_close)
"""

import argparse
import glob
import json
import os

import numpy as np

HAND = "unitree_g1"
JOINTS = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
)  # the description's names for thumb 0-2, index 0-1, middle 0-1
# The description's palm link in its grasp frame (gripper.urdf, world_joint: xyz 0.07 0 0.02, rpy 90 deg 0 180 deg), then
# to the palm link of SIMPLE's model: the two hands differ a little, the finger bases line up 7.3 mm further along x.
T_GRASP_PALM = np.array([[-1, 0, 0, 0.07 - 0.0073], [0, 0, 1, 0.0], [0, 1, 0, 0.02], [0, 0, 0, 1.0]])


def load_library(lib_dir, task: str):
    """The truth track's grasp library of a task (T_op, score, f_close, f_open; built from the released episodes' clouds)."""
    return np.load(os.path.join(str(lib_dir), f"{task}.npz"))


def main() -> None:
    import torch
    from graspgenx import get_checkpoints_version_dir
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.samplers import run_planner_on_object
    from graspgenx.utils.checkpoint_io import load_model_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--num-grasps", type=int, default=500)
    ap.add_argument("--planner", default="graspmoe")
    ap.add_argument("--only", default="", help="clouds whose name starts with this")
    a = ap.parse_args()
    ckpt = str(get_checkpoints_version_dir())
    sampler = GraspGenXSampler(
        load_model_cfg(os.path.join(ckpt, "gen"), os.path.join(ckpt, "dis")), HAND, assets_dir=a.assets_dir
    )
    cfg = json.load(open(os.path.join(a.assets_dir, "x_grippers", HAND, "config.json")))
    f_open, f_close = (np.array([cfg[k][f"right_{j}_joint"] for j in JOINTS], float) for k in ("open", "close"))
    for f in sorted(glob.glob(os.path.join(a.dir, a.only + "*_L*_e*.npz"))):
        name = os.path.basename(f)[:-4]
        g = os.path.join(a.dir, f"grasp_{name}.npz")
        if name.startswith("grasp_") or (os.path.exists(g) and os.path.getmtime(g) >= os.path.getmtime(f)):
            continue  # its grasps are newer than the cloud: other runs may be reading them
        if name.startswith("grasp_"):
            continue
        z = np.load(f)
        pts = z["pts"][np.linalg.norm(z["pts"] - np.median(z["pts"], axis=0), axis=1) < 0.3]  # stray pixels of the mask
        np.random.seed(0)
        torch.manual_seed(0)
        G, conf, _, _ = run_planner_on_object(pts.astype(np.float32), sampler, planner=a.planner, grasp_threshold=-1.0,
                                              num_grasps=a.num_grasps, topk_num_grasps=-1)  # fmt: skip
        o = np.argsort(-np.asarray(conf))
        T_wo = np.eye(4)
        T_wo[:3, :3], T_wo[:3, 3] = z["R_o"], z["p_o"]
        T_op = np.linalg.inv(T_wo) @ np.asarray(G, float)[o] @ T_GRASP_PALM
        conf = np.asarray(conf, np.float32)[o]
        np.savez_compressed(
            os.path.join(a.dir, f"grasp_{name}.npz"), T_op=T_op, score=conf, f_open=f_open, f_close=f_close
        )
        print(
            name,
            len(pts),
            "points,",
            len(conf),
            "grasps, top",
            conf[:3].round(2),
            "at least 0.5:",
            int((conf >= 0.5).sum()),
            flush=True,
        )


if __name__ == "__main__":
    main()
