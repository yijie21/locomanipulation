"""Stereo depth service of the SIMPLE bed (moved from escalation_20261116/depth_server_esc.py, TODO 49.2): a client that dies
mid-request no longer kills the service.

Depth from the head stereo pair: semi-global matching, then LingBot-Depth fills what matching leaves out or gets wrong
(the glossy table top matches its reflections). Runs in the sam3d env with third_party/lingbot-depth on the path:

    cd <main checkout> && PYTHONPATH=third_party/lingbot-depth HF_HOME=/workspace/huggingface_cache HF_HUB_OFFLINE=1 \\
        CUDA_VISIBLE_DEVICES=0 /workspace/miniconda3/envs/sam3d/bin/python -s src/locomani/perception/stereo_server.py <socket path>
"""

import math
import sys
from multiprocessing.connection import Listener

import cv2
import numpy as np
import torch
from mdm.model.v2 import MDMModel

FOV, BASE, UP, AGREE = math.radians(110), 0.05, 2, 0.08


def matched(left, right):
    g = [
        cv2.resize(cv2.cvtColor(i, cv2.COLOR_RGB2GRAY), None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
        for i in (left, right)
    ]
    m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64 * UP, blockSize=5, P1=200, P2=800, uniquenessRatio=10,
                              speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_HH)  # fmt: skip
    d = cv2.resize(m.compute(*g).astype(np.float32) / 16.0 / UP, left.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    fx = left.shape[1] / (2 * math.tan(FOV / 2))
    return np.where(d > 0.5, fx * BASE / np.maximum(d, 1e-3), 0.0).astype(np.float32)


model = MDMModel.from_pretrained("robbyant/lingbot-depth-pretrain-vitl-14-v0.5").cuda().eval()


def depth(left, right):
    sg = matched(left, right)
    H, W = sg.shape
    f = W / (2 * math.tan(FOV / 2))
    K = torch.tensor([[f / W, 0, 0.5], [0, f / H, 0.5], [0, 0, 1]], dtype=torch.float32, device="cuda")[None]
    img = torch.tensor(left / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)[None]
    with torch.no_grad():
        lb = (
            model.infer(img, depth_in=torch.tensor(sg, device="cuda")[None], intrinsics=K)["depth"][0]
            .float()
            .cpu()
            .numpy()
        )
    lb = np.where(np.isfinite(lb) & (lb > 0), lb, 0.0).astype(np.float32)
    keep = (
        (sg > 0) & (lb > 0) & (np.abs(sg - lb) < AGREE * lb)
    )  # matching where the two agree, the filled depth elsewhere
    return np.where(keep, sg, lb), sg, lb


with Listener(sys.argv[1], family="AF_UNIX") as lis:
    print("listening", sys.argv[1], flush=True)
    while True:
        with lis.accept() as c:
            try:
                while True:
                    left, right = c.recv()
                    c.send(depth(left, right))
            except (EOFError, BrokenPipeError, ConnectionResetError):
                pass
