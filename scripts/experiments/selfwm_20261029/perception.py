"""Synthetic head-camera perception with structured errors (the minimal proof's stand-in for SAM 3 + depth).

A point P is read in the TRUE camera frame, then its depth is scaled and jittered: x' = (1 + s) R_c^T (P - c) + n,
n ~ N(0, (sig0 + k d^2)^2) per axis. The robot maps x' into its pelvis frame with the camera pose it BELIEVES:
its own kinematics rotated by an extrinsic error dR (the mount, the head bracket, the waist chain). Box and jaw are
read through the same camera, so what is common to both cancels in their difference and what is not does not.

Works on numpy arrays (the simulator) and torch tensors (the world model's imagination) alike.
"""

from __future__ import annotations

import math

import numpy as np

FOVY_DEG = 65.0  # walk_mars head_camera
W_PX, H_PX = 640, 480
SIG0, SIG_K = 0.002, 0.005  # m, m^-1: jitter 2 mm + 5 mm/m^2 * d^2


def focal() -> float:
    return (H_PX / 2) / math.tan(math.radians(FOVY_DEG) / 2)


def rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Extrinsic error rotation from small angles (rad) about the camera's own x, y, z axes."""
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def visible(x_cam) -> bool:
    """In front of the camera (MuJoCo cameras look along -z) and inside the image."""
    z = -float(x_cam[2])
    if z <= 0.05:
        return False
    f = focal()
    u, v = f * float(x_cam[0]) / z, f * float(x_cam[1]) / z
    return abs(u) < W_PX / 2 and abs(v) < H_PX / 2


def read_np(P_pel: np.ndarray, cam_pel: np.ndarray, R_cam_pel: np.ndarray, err: dict, rng) -> tuple[np.ndarray, bool]:
    """One reading of pelvis-frame point P_pel (true) as the robot reports it in its pelvis frame.
    cam_pel / R_cam_pel: the TRUE camera pose in the pelvis frame. err: scale, dR (3x3), and jitter on/off."""
    x = R_cam_pel.T @ (P_pel - cam_pel)
    ok = visible(x)
    d = -float(x[2])
    x = (1.0 + err.get("scale", 0.0)) * x
    if err.get("jitter", True):
        x = x + rng.normal(0.0, SIG0 + SIG_K * d * d, 3)
    R_believed = R_cam_pel @ err.get("dR", np.eye(3))
    return cam_pel + R_believed @ x, ok


def read_torch(P_pel, cam_pel, R_cam_pel, scale, dR, noise):
    """Batched torch version: P_pel, cam_pel (B, 3); R_cam_pel, dR (B, 3, 3); scale (B, 1); noise (B, 3) standard
    normal draws (scaled here by the depth law). Returns (reading (B, 3), visible (B,) float)."""
    import torch

    x = torch.einsum("bji,bj->bi", R_cam_pel, P_pel - cam_pel)
    d = (-x[:, 2]).clamp(min=1e-3)
    f = focal()
    u, v = f * x[:, 0] / d, f * x[:, 1] / d
    vis = ((-x[:, 2] > 0.05) & (u.abs() < W_PX / 2) & (v.abs() < H_PX / 2)).float()
    x = (1.0 + scale) * x + noise * (SIG0 + SIG_K * d * d)[:, None]
    Rb = torch.einsum("bij,bjk->bik", R_cam_pel, dR)
    return cam_pel + torch.einsum("bij,bj->bi", Rb, x), vis
