"""Vision RSI D2 (docs/handoff/vision-rsi-design-20261023.md section 17): what the mapping scripts share and the sim
venv can import (no Open3D): the head camera's stereo error model, poses, the camera matrix."""

from __future__ import annotations

import numpy as np

# a D455-like stereo depth error: sigma = z^2 * STEREO_PX / (f * BASELINE_M); depth beyond MAX_DEPTH_M dropped
STEREO_PX, BASELINE_M, MAX_DEPTH_M = 0.08, 0.095, 6.0


def se3(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p
    return T


def kmat(K) -> np.ndarray:
    """The camera matrix, from either a 3 x 3 or tour_sensors' first (fx, fy, cx, cy)."""
    K = np.asarray(K, float)
    return K if K.shape == (3, 3) else np.array([[K[0], 0.0, K[2]], [0.0, K[1], K[3]], [0.0, 0.0, 1.0]])


def stereo_noise(z: np.ndarray, fy: float, rng) -> np.ndarray:
    """Depth with the stereo error added (z in metres)."""
    return z + rng.normal(size=z.shape) * z**2 * STEREO_PX / (fy * BASELINE_M)
