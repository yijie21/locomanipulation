"""GraspGen-X (Dex1 config) on the head-camera point clouds of vision_grasp.py see. Runs in GraspGen-X's own venv, one GPU.

Per placement with at least --min-points target points: grasps in the world, turned into room_grasp's cache fields in the object's
frame (tool = the jaw centre 110 mm along the Dex1 base link's x, R = the base link's attitude: x approach, y closing; the object's
pose only moves them into the frame the cache is stored in, and room_grasp moves them back with the same pose).
-> <set>/vision/grasp_<pose>_L<k>_s<j>.npz (tool, R, score, branch)

    G=third_party/GraspGenX; O=outputs/scene_rsi_20261016/graspgenx_20261026     # both under the main checkout
    CUDA_VISIBLE_DEVICES=0 $G/.venv/bin/python scripts/experiments/reachwm_20261030/vision_grasp_infer.py <set>/vision --assets-dir $O/gripper_assets
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

TOOL_X = 0.1515 - 0.0415  # the jaw centre from the Dex1 base link origin, along its x

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--pose", default="start")
    ap.add_argument("--min-points", type=int, default=100)
    ap.add_argument("--num-grasps", type=int, default=500)
    ap.add_argument("--planner", default="graspmoe")
    a = ap.parse_args()
    ckpt = str(get_checkpoints_version_dir())
    sampler = GraspGenXSampler(
        load_model_cfg(os.path.join(ckpt, "gen"), os.path.join(ckpt, "dis")), "dex1", assets_dir=a.assets_dir
    )
    base_rot = np.array(
        json.load(open(os.path.join(a.assets_dir, "x_grippers", "dex1", "config.json")))["base_rotation"], float
    )
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, f"see_{a.pose}_L*_s*.npz"))):
        z = np.load(f)
        name = os.path.basename(f)[4:-4]
        if len(z["pts"]) < a.min_points:
            rows.append(dict(name=name, n_pts=int(len(z["pts"])), n=0))
            continue
        np.random.seed(0)
        torch.manual_seed(0)
        t = time.perf_counter()
        G, conf, tags, _ = run_planner_on_object(z["pts"].astype(np.float32), sampler, planner=a.planner, grasp_threshold=-1.0,
                                                 num_grasps=a.num_grasps, topk_num_grasps=-1)  # fmt: skip
        if len(conf) == 0:  # the planner kept nothing (few points left after its outlier removal)
            rows.append(dict(name=name, n_pts=int(len(z["pts"])), n=0))
            continue
        o = np.argsort(-np.asarray(conf))
        G = np.asarray(G, float)[o] @ base_rot
        conf = np.asarray(conf, np.float32)[o]
        Rw, pw = G[:, :3, :3], G[:, :3, 3] + G[:, :3, 0] * TOOL_X
        R_o, p_o = z["R_o"], z["p_o"]
        np.savez_compressed(os.path.join(a.dir, f"grasp_{name}.npz"), tool=(pw - p_o) @ R_o, R=np.einsum("ji,njk->nik", R_o, Rw),
                            score=conf, branch=np.array([tags[i] != "diff" for i in o]))  # fmt: skip
        rows.append(dict(name=name, n_pts=int(len(z["pts"])), n=int(len(conf)), top=float(conf[0]), n05=int((conf >= 0.5).sum()),
                         s=round(time.perf_counter() - t, 2)))  # fmt: skip
        print(rows[-1], flush=True)
    json.dump(rows, open(os.path.join(a.dir, f"infer_{a.pose}.json"), "w"))
