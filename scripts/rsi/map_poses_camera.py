"""Vision RSI D2 step 2b without LiDAR (docs/handoff/vision-rsi-design-20261023.md sections 6 and 17.7): the head
camera's pose along the mapping tour from MapAnything (facebook/map-anything, CC-BY-NC weights from the local Hugging
Face cache) on its own frames, offline on one recording (tour_sensors.py), against the truth.

The tour's head-camera frames, in time order, go to MapAnything in chunks of `--chunk` frames overlapping by `--overlap`,
each frame with its intrinsics and its depth (the stereo error model, map_common.stereo_noise: the D455's depth) so the
scale is metric. MapAnything gives every frame of a chunk a pose in the chunk's own frame; each chunk is placed on the
ones before through the overlap frames (the rigid motion that best maps the chunk's poses of those frames onto their
poses so far, rotation by SVD), and the first chunk on the true start pose (as map_poses.py anchors the LiDAR).

Reported as map_poses.py does, for the camera: the position error over the tour, at each viewpoint's head scan, at the
end, the heading error, the seconds per frame.

    HF_HOME=/workspace/huggingface_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \\
        /workspace/envs/mapanything/bin/python scripts/rsi/map_poses_camera.py outputs/vision_rsi_20261023/tour/sensors \\
        outputs/vision_rsi_20261023/tour/poses
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from map_common import MAX_DEPTH_M, kmat, se3, stereo_noise  # noqa: E402

FLIP = np.diag([1.0, -1.0, -1.0])  # MuJoCo camera axes (x right, y up, z back) <-> OpenCV's (x right, y down, z ahead)


def rigid(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """The rigid transform T (4 x 4) minimising sum |T @ A_k - B_k| over poses (n x 4 x 4): rotation by SVD of the
    summed rotation products, translation from the centres."""
    M = sum(b[:3, :3] @ a[:3, :3].T for a, b in zip(A, B, strict=True))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    t = np.mean([b[:3, 3] - R @ a[:3, 3] for a, b in zip(A, B, strict=True)], axis=0)
    return se3(R, t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sensors", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--no-depth", action="store_true", help="images and intrinsics only (no metric depth input)")
    a = ap.parse_args(argv)
    from mapanything.models import MapAnything
    from mapanything.utils.image import preprocess_inputs

    cz = dict(np.load(a.sensors / "camera.npz"))
    K = kmat(cz["K"])
    n = len(cz["t"])
    true = np.array([se3(R @ FLIP, p) for R, p in zip(cz["R"], cz["pos"], strict=True)])  # OpenCV camera to world
    info = json.loads((a.sensors / "sensors.json").read_text())
    holds = json.loads((Path(info["record"]) / "tour.json").read_text())["holds"]
    model = MapAnything.from_pretrained("facebook/map-anything").to("cuda").eval()
    rng = np.random.default_rng(0)
    est: dict[int, np.ndarray] = {}
    step = a.chunk - a.overlap
    t0 = time.time()
    for s in range(0, n, step):
        idx = list(range(s, min(n, s + a.chunk)))
        if len(idx) < 2:
            break
        views = []
        for i in idx:
            img = cv2.cvtColor(cv2.imread(str(a.sensors / "rgb" / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
            v = dict(img=torch.from_numpy(img), intrinsics=torch.from_numpy(K.astype(np.float32)))
            if not a.no_depth:
                z = cv2.imread(str(a.sensors / "depth" / f"{i:05d}.png"), cv2.IMREAD_UNCHANGED) / 1000.0
                z = stereo_noise(z, K[1, 1], rng)
                z[(z < 0.1) | (z > MAX_DEPTH_M)] = 0.0
                v["depth_z"] = torch.from_numpy(z.astype(np.float32))
            views.append(v)
        with torch.no_grad():
            pred = model.infer(preprocess_inputs(views), memory_efficient_inference=True, use_amp=True)
        P = [p["camera_poses"][0].float().cpu().numpy() for p in pred]
        shared = [k for k, i in enumerate(idx) if i in est]
        A = (
            rigid([P[k] for k in shared], [est[idx[k]] for k in shared])
            if shared
            else true[idx[0]] @ np.linalg.inv(P[0])
        )
        for k, i in enumerate(idx):
            if i not in est:
                est[i] = A @ P[k]
        print(f"chunk {s}-{idx[-1]} {time.time() - t0:.0f} s", flush=True)
        if idx[-1] == n - 1:
            break
    E = np.array([est[i] for i in range(n)])
    e = np.linalg.norm(E[:, :3, 3] - true[:, :3, 3], axis=1)
    eh = np.linalg.norm(E[:, :2, 3] - true[:, :2, 3], axis=1)
    fwd = lambda T: T[:, :3, 2]  # noqa: E731  OpenCV forward
    yaw = lambda T: np.arctan2(fwd(T)[:, 1], fwd(T)[:, 0])  # noqa: E731
    dyaw = np.degrees(np.abs(np.angle(np.exp(1j * (yaw(E) - yaw(true))))))
    t = cz["t"]
    at = [round(100 * float(eh[(t >= h["t0"]) & (t < h["t1"])].mean()), 2) if ((t >= h["t0"]) & (t < h["t1"])).any()
          else None for h in holds]  # fmt: skip
    name = "mapanything" + ("_nodepth" if a.no_depth else "")
    res = dict(rmse_cm=round(100 * float(np.sqrt((e**2).mean())), 2), p95_cm=round(100 * float(np.percentile(e, 95)), 2),
               max_cm=round(100 * float(e.max()), 2), end_cm=round(100 * float(eh[-1]), 2),
               yaw_max_deg=round(float(dyaw.max()), 2), at_viewpoints_cm=at, frames=n, chunk=a.chunk,
               overlap=a.overlap, s_per_frame=round((time.time() - t0) / n, 3))  # fmt: skip
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / f"{name}_camera.npy", E)
    prev = json.loads((a.out / "poses_camera.json").read_text()) if (a.out / "poses_camera.json").exists() else {}
    (a.out / "poses_camera.json").write_text(json.dumps({**prev, name: res}, indent=1))
    print(name, json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
