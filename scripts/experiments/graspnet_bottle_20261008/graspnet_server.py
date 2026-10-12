"""TODO 21.3.4: graspnet-baseline as a grasp module, served over ZMQ to the episode process.

Runs in graspnet-baseline's own venv (torch 2.7 cu128, pointnet2 / knn rebuilt for sm_120; third_party/graspnet-baseline,
licence: non-commercial research, weights not redistributed). One REQ/REP round trip per grasp query:

    request:  npz bytes with `cloud` (N x 3 float32, camera frame, metres: x right, y down, z forward, the
              OpenCV convention the network was trained in) and optional `collision_thresh` (default 0.01)
    reply:    npz bytes with `grasps` (M x 17 float32, graspnetAPI GraspGroup rows: score, width, height, depth,
              rotation 3x3 row-major, translation 3, object id), sorted by score after NMS and collision filtering
    ping:     a request npz with a `ping` entry is answered with a `pong` entry at once (grasp_planned's readiness check)

Installed by install_graspnet.sh, started by servers.sh (`servers.sh start graspnet`); `--selftest` loads the network,
runs graspnet-baseline's example scene, prints the selftest line and exits. `--graspnet-dir` points at another
checkout (install_graspnet.sh's own check of a fresh install).

    third_party/graspnet-baseline/.venv/bin/python scripts/experiments/graspnet_bottle_20261008/graspnet_server.py \
        --port 5790 --gpu 1 [--selftest]
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
GN = REPO / "third_party" / "graspnet-baseline"
CHECKPOINT = GN / "logs" / "checkpoint-rs.tar"  # trained on RealSense captures; the authors' pick for transfer
NUM_POINT = 20000


def load_net(checkpoint: Path):
    import torch
    from graspnet import GraspNet

    net = GraspNet(
        input_feature_dim=0,
        num_view=300,
        num_angle=12,
        num_depth=4,
        cylinder_radius=0.05,
        hmin=-0.02,
        hmax_list=[0.01, 0.02, 0.03, 0.04],
        is_training=False,
    )
    net.to("cuda:0")
    net.load_state_dict(torch.load(checkpoint, weights_only=False)["model_state_dict"])
    net.eval()
    return net


def predict(net, cloud: np.ndarray, collision_thresh: float = 0.01, seed: int = 0) -> np.ndarray:
    import torch
    from collision_detector import ModelFreeCollisionDetector
    from graspnet import pred_decode
    from graspnetAPI import GraspGroup

    rng = np.random.default_rng(seed)
    n = len(cloud)
    idx = rng.choice(n, NUM_POINT, replace=n < NUM_POINT)
    pts = torch.from_numpy(cloud[idx][None].astype(np.float32)).to("cuda:0")
    with torch.no_grad():
        gg = GraspGroup(pred_decode(net({"point_clouds": pts, "cloud_colors": None}))[0].detach().cpu().numpy())
    if collision_thresh > 0:
        detector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
        gg = gg[~detector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)]
    gg.nms()
    gg.sort_by_score()
    return gg.grasp_group_array.astype(np.float32)


def selftest(net, gn: Path) -> None:
    """The repo's example scene: the network must find grasps there before it is asked about ours."""
    import scipy.io as scio
    from data_utils import CameraInfo, create_point_cloud_from_depth_image
    from PIL import Image

    d = gn / "doc" / "example_data"
    depth = np.array(Image.open(d / "depth.png"))
    mask = np.array(Image.open(d / "workspace_mask.png"))
    meta = scio.loadmat(d / "meta.mat")
    K = meta["intrinsic_matrix"]
    cam = CameraInfo(1280.0, 720.0, K[0][0], K[1][1], K[0][2], K[1][2], meta["factor_depth"])
    cloud = create_point_cloud_from_depth_image(depth, cam, organized=True)[(mask > 0) & (depth > 0)]
    t0 = time.time()
    g = predict(net, cloud)
    print(f"[graspnet] selftest: {len(g)} grasps, best score {g[0, 0]:.3f}, {time.time() - t0:.2f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=5790)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--selftest", action="store_true", help="run the example scene and exit")
    ap.add_argument("--graspnet-dir", type=Path, default=GN, help="the graspnet-baseline checkout (weights in logs/)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gn = args.graspnet_dir.resolve()
    sys.path[:0] = [str(gn / "models"), str(gn / "dataset"), str(gn / "utils")]
    net = load_net(gn / CHECKPOINT.relative_to(GN))
    selftest(net, gn)
    if args.selftest:
        return 0
    import zmq

    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[graspnet] listening on :{args.port}", flush=True)
    while True:
        req = np.load(io.BytesIO(sock.recv()))
        if "ping" in req.files:
            buf = io.BytesIO()
            np.savez(buf, pong=np.array(1))
            sock.send(buf.getvalue())
            continue
        t0 = time.time()
        try:
            grasps = predict(
                net,
                np.asarray(req["cloud"], np.float32),
                float(req["collision_thresh"]) if "collision_thresh" in req else 0.01,
            )
            err = ""
        except Exception as exc:  # noqa: BLE001 -- the episode decides what a failed query means
            grasps, err = np.zeros((0, 17), np.float32), repr(exc)
        buf = io.BytesIO()
        np.savez(buf, grasps=grasps, error=np.array(err), seconds=np.array(time.time() - t0))
        sock.send(buf.getvalue())
        print(
            f"[graspnet] {len(req['cloud'])} points -> {len(grasps)} grasps {time.time() - t0:.2f} s {err}", flush=True
        )


if __name__ == "__main__":
    raise SystemExit(main())
