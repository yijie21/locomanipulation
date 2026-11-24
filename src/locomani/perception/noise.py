"""Depth noise models for sensor=rgbd (TODO 49.2 step 6): what a real depth camera would have seen instead of the
renderer's exact depth.

`realsense` is the RealSense D435-like model of the reach world-model work (esc
scripts/experiments/reachwm_20261030/vision_depth.py, 2026-11-11), with the defects of active stereo by their causes:

- occlusion shadow: a point the second imager (`baseline` to the right) does not see has no disparity -> hole; needs
  the clean depth of that second imager (`depth_right`), skipped when it is not given
- edges: matching fails across depth steps -> pixels next to a step dropped with P_EDGE, a few flying pixels between
  the two depths
- range: disparity noise of SUBPIXEL px, quantised to 1/32 px -> error growing with the square of the distance;
  nothing under MIN_Z or beyond MAX_Z
- material: glass and glossy bodies (a mask from the caller) lose most of their pixels in blobs
- speckle: small random holes everywhere

Changes from the original: the focal length comes from the intrinsics (K[1, 1]) instead of a vertical field of view
(the same number for a pinhole camera), and holes are NaN, the Frame convention, instead of 0.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

BASELINE, MIN_Z, MAX_Z, SUBPIXEL = 0.050, 0.28, 6.0, 0.08
P_EDGE, EDGE_STEP, P_FLY = 0.7, 0.02, 0.15
P_SPECKLE, P_GLASS = 0.004, 0.75
MODELS = ("none", "realsense")


def blobs(rng: np.random.Generator, shape, p: float, sigma: float) -> np.ndarray:
    """A random mask covering about p of the frame in blobs of about sigma px."""
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return f > np.quantile(f, 1 - p)


def realsense(
    depth: np.ndarray,
    K: np.ndarray,
    rng: np.random.Generator,
    depth_right: np.ndarray | None = None,
    glass: np.ndarray | None = None,
    baseline: float = BASELINE,
) -> np.ndarray:
    """depth, depth_right: clean depth (m, NaN or <= 0 where nothing) of the imager and of one `baseline` to its right,
    same attitude. -> noisy depth (m, float32, NaN holes)."""
    H, W = depth.shape
    f = float(K[1, 1])
    out = np.nan_to_num(depth.astype(np.float64), nan=0.0)
    hole = (out < MIN_Z) | (out > MAX_Z)
    if depth_right is not None:  # shadow: the point's column in the right imager, and whether it sees that depth there
        right = np.nan_to_num(depth_right.astype(np.float64), nan=np.inf)
        v, u = np.mgrid[0:H, 0:W]
        ur = np.rint(u - f * baseline / np.maximum(out, 1e-3)).astype(int)
        seen = (ur >= 0) & (ur < W)
        zr = right[v, np.clip(ur, 0, W - 1)]
        hole |= ~seen | (zr < out - (0.01 + 0.01 * out))
    step = np.zeros_like(hole)
    for ax in (0, 1):
        d = np.abs(np.diff(out, axis=ax)) > EDGE_STEP * np.minimum(out.take(range(out.shape[ax] - 1), ax), 3.0)
        pad = [(0, 0), (0, 0)]
        pad[ax] = (0, 1)
        d = np.pad(d, pad)
        step |= d | np.roll(d, 1, ax)
    step = ndimage.binary_dilation(step, iterations=1)
    r = rng.random(out.shape)
    fly = step & (r < P_FLY)
    out[fly] = ndimage.uniform_filter(out, 5)[fly]  # between the near and the far side
    hole |= step & ~fly & (r < P_FLY + P_EDGE)
    hole |= blobs(rng, out.shape, P_SPECKLE, 1.2)
    if glass is not None and glass.any():
        hole |= glass & blobs(rng, out.shape, P_GLASS, 2.0)
    disp = f * baseline / np.maximum(out, 1e-3) + rng.normal(0.0, SUBPIXEL, out.shape)  # disparity noise, quantised
    out = f * baseline / np.maximum(np.round(disp * 32) / 32, 1e-3)
    out[hole] = np.nan
    return out.astype(np.float32)
