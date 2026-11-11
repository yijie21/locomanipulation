"""LingBot-Depth on the head camera's colour + RealSense-like depth (vision_grasp.py see): holes filled, surfaces cleaned.
Runs in the sam3d env with the official code on the path (third_party/lingbot-depth), one GPU; weights from Hugging Face.

-> <set>/vision/lb_<pose>_L<k>_s<j>.npz (depth_lb, m; 0 where the model gives nothing), fix_<pose>.json (error against the clean depth)

    PYTHONPATH=third_party/lingbot-depth HF_HOME=/workspace/huggingface_cache CUDA_VISIBLE_DEVICES=0 \\
        /workspace/miniconda3/envs/sam3d/bin/python -s scripts/experiments/reachwm_20261030/vision_depth_fix.py <set>/vision
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
from mdm.model.v2 import MDMModel

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--pose", default="start")
    ap.add_argument("--model", default="robbyant/lingbot-depth-pretrain-vitl-14-v0.5")
    a = ap.parse_args()
    model = MDMModel.from_pretrained(a.model).cuda().eval()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, f"see_{a.pose}_L*_s*.npz"))):
        z = np.load(f)
        H, W = z["depth"].shape
        fpx = 0.5 * H / math.tan(math.radians(float(z["fovy"])) / 2)
        K = torch.tensor([[fpx / W, 0, 0.5], [0, fpx / H, 0.5], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
        img = torch.tensor(z["rgb"] / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None]
        with torch.no_grad():
            out = model.infer(
                img, depth_in=torch.tensor(z["depth_rs"], dtype=torch.float32, device="cuda")[None], intrinsics=K
            )
        d = out["depth"][0].float().cpu().numpy()
        d = np.where(np.isfinite(d) & (d > 0), d, 0.0).astype(np.float32)
        name = os.path.basename(f)[4:]
        np.savez_compressed(os.path.join(a.dir, "lb_" + name), depth_lb=d)
        m, clean, rs = z["mask"], z["depth"], z["depth_rs"]
        row = dict(name=name[:-4], px=int(m.sum()))
        if m.sum() >= 100:
            row.update(rs_hole=float((rs[m] == 0).mean()), lb_hole=float((d[m] == 0).mean()),
                       rs_err_mm=1000 * float(np.median(np.abs(rs - clean)[m & (rs > 0)])) if (m & (rs > 0)).any() else None,
                       lb_err_mm=1000 * float(np.median(np.abs(d - clean)[m & (d > 0)])) if (m & (d > 0)).any() else None,
                       lb_bias_mm=1000 * float(np.median((d - clean)[m & (d > 0)])) if (m & (d > 0)).any() else None)  # fmt: skip
        rows.append(row)
        print(row, flush=True)
    json.dump(rows, open(os.path.join(a.dir, f"fix_{a.pose}.json"), "w"))
