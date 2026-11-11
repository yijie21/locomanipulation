"""A RealSense D435-like depth frame from a clean rendered one (vision_grasp.py see): the defects of active stereo by their causes.

- occlusion shadow: a point the second imager (BASELINE to the right) does not see has no disparity -> hole (0)
- edges: matching fails across depth steps -> pixels next to a step dropped with P_EDGE, a few flying pixels between the two depths
- range: disparity noise of SUBPIXEL px, quantised to 1/32 px -> error growing with the square of the distance; nothing under MIN_Z
- material: glass and glossy bodies (a mask from the caller) lose most of their pixels in blobs
- speckle: small random holes everywhere

Holes are 0, as in the camera's own frames."""

import math

import numpy as np
from scipy import ndimage

BASELINE, MIN_Z, SUBPIXEL = 0.050, 0.28, 0.08
P_EDGE, EDGE_STEP, P_FLY = 0.7, 0.02, 0.15
P_SPECKLE, P_GLASS = 0.004, 0.75


def blobs(rng, shape, p: float, sigma: float) -> np.ndarray:
    """A random mask covering about p of the frame in blobs of about sigma px."""
    f = ndimage.gaussian_filter(rng.standard_normal(shape), sigma)
    return f > np.quantile(f, 1 - p)


def realsense(depth, depth_right, fovy: float, rng, glass=None):
    """depth, depth_right: clean depth (m) of the left imager and of one BASELINE to its right, same attitude."""
    H, W = depth.shape
    f = 0.5 * H / math.tan(math.radians(fovy) / 2)
    out = depth.astype(np.float64).copy()
    hole = (out < MIN_Z) | (out > 6.0)
    # shadow: the point's column in the right imager, and whether the right imager sees that depth there
    v, u = np.mgrid[0:H, 0:W]
    ur = np.rint(u - f * BASELINE / np.maximum(out, 1e-3)).astype(int)
    seen = (ur >= 0) & (ur < W)
    zr = depth_right[v, np.clip(ur, 0, W - 1)]
    hole |= ~seen | (zr < out - (0.01 + 0.01 * out))
    # edges
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
    # disparity noise and quantisation
    disp = f * BASELINE / np.maximum(out, 1e-3) + rng.normal(0.0, SUBPIXEL, out.shape)
    out = f * BASELINE / np.maximum(np.round(disp * 32) / 32, 1e-3)
    out[hole] = 0.0
    return out.astype(np.float32)
