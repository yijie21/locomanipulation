"""RgbdSensor (sensor=rgbd, TODO 49.2 step 6): one camera's image and the renderer's depth, as is or through a noise
model (depth_noise=none|realsense). The benchmark's camera channel gives the exact frame (bench/channels.py
Cameras.rgbd); a benchmark that can also render the depth of a second imager one baseline to the right hands it in as
`depth_right` so the realsense model can cast occlusion shadows."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from locomani.perception import noise as N
from locomani.perception.sensor import Frame


class RgbdSensor:
    def __init__(
        self,
        exact: Callable[[], Frame],
        depth_noise: str = "none",
        seed: int = 0,
        depth_right: Callable[[], np.ndarray] | None = None,
    ):
        if depth_noise not in N.MODELS:
            raise ValueError(f"depth_noise {depth_noise!r} not in {N.MODELS}")
        self.exact, self.noise, self.depth_right = exact, depth_noise, depth_right
        self.rng = np.random.default_rng(seed)
        self.name = "rgbd" if depth_noise == "none" else f"rgbd:{depth_noise}"

    def frame(self) -> Frame:
        f = self.exact()
        if self.noise == "none":
            return replace(f, source=self.name)
        right = self.depth_right() if self.depth_right is not None else None
        return replace(f, depth=N.realsense(f.depth, f.K, self.rng, depth_right=right), source=self.name)
