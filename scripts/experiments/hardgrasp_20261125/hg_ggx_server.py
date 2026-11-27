"""GraspGen-X service for selfmodel_hardgrasp_20261125 (LocoManiLab part): the Unitree G1 three-finger hand (`unitree_g1`,
the right Dex3) on one object cloud per request. Runs in GraspGen-X's own venv, one GPU, the same connection protocol as
locomani.tasks.scene.services (multiprocessing Listener, AUTHKEY), inlined so this venv needs nothing from our src.

Request  dict(points=(N,3) float32 in any frame, num_grasps=int, seed=int)
Reply    dict(G=(n,4,4) grasp-frame poses in the cloud's frame, best score first, conf=(n,), n_points=int, s=float)

The grasp frame is the description's (gripper.urdf world link: approach along +z, the palm link at xyz 0.07 0 0.02,
rpy 90 deg 0 180 deg); hg_look.py turns it into the host's wrist_yaw_link.

    P=/wuji-vefps-D/wuji-il/yijie/tools/perception
    CUDA_VISIBLE_DEVICES=1 $P/GraspGenX/.venv/bin/python hg_ggx_server.py --port 18853 \
        --assets-dir $P/GraspGenX/ext/gripper_descriptions/gripper_descriptions/assets
"""

import argparse
import os
import threading
import time
import traceback
from multiprocessing.connection import Listener

import numpy as np
import torch
from graspgenx import get_checkpoints_version_dir
from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_object
from graspgenx.utils.checkpoint_io import load_model_cfg

AUTHKEY = b"locomani-vision-rsi"
HAND = "unitree_g1"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18853)
    ap.add_argument("--assets-dir", required=True)
    ap.add_argument("--planner", default="graspmoe")
    a = ap.parse_args()
    ckpt = str(get_checkpoints_version_dir())
    sampler = GraspGenXSampler(
        load_model_cfg(os.path.join(ckpt, "gen"), os.path.join(ckpt, "dis")), HAND, assets_dir=a.assets_dir
    )
    lock = threading.Lock()
    # 2026-11-25 design: the default 0.014 m kNN outlier threshold dropped all points of clouds seen from ~2.8 m
    # (depth pixels ~1.5 cm apart): looser for this head camera's sparse far clouds
    OUTLIER = 0.05

    def handle(req: dict) -> dict:
        t0 = time.time()
        pts = np.asarray(req["points"], np.float32)
        c = np.median(pts, axis=0)
        pts = pts[np.linalg.norm(pts - c, axis=1) < 0.3]  # stray pixels of the mask
        np.random.seed(int(req.get("seed", 0)))
        torch.manual_seed(int(req.get("seed", 0)))
        G, conf, _, _ = run_planner_on_object(pts, sampler, planner=a.planner, grasp_threshold=-1.0,
                                              num_grasps=int(req.get("num_grasps", 400)), topk_num_grasps=-1,
                                              moe_outlier_threshold=float(req.get("outlier", OUTLIER)))  # fmt: skip
        G, conf = np.asarray(G, np.float64), np.asarray(conf, np.float64)
        o = np.argsort(-conf)
        return dict(G=G[o], conf=conf[o], n_points=len(pts), s=time.time() - t0)

    def session(conn) -> None:
        with conn:
            while True:
                try:
                    req = conn.recv()
                except (EOFError, OSError):
                    return
                try:
                    with lock:
                        rep = handle(req)
                except Exception as e:
                    rep = dict(error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-2000:])
                conn.send(rep)

    with Listener(("127.0.0.1", a.port), authkey=AUTHKEY) as listener:
        print(f"serving on 127.0.0.1:{a.port}", flush=True)
        while True:
            try:
                conn = listener.accept()
            except Exception as e:
                print(f"dropped a connection: {type(e).__name__}: {e}", flush=True)
                continue
            threading.Thread(target=session, args=(conn,), daemon=True).start()
