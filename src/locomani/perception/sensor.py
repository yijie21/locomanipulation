"""Sensor: one colour image with metric depth, whatever produced the depth (TODO 49.1).

    sensor=stereo                        the head's left image; depth from the stereo pair through a stereo model
    sensor=rgbd, depth_noise=none        one camera's image and the simulator's rendered depth, as is
    sensor=rgbd, depth_noise=realsense   the same depth through a RealSense-like noise model

A CameraView reads only Frames, so it does not know which of these it runs on; `Frame.source` records it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Frame:
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32 metres along the optical axis, NaN where unknown
    K: np.ndarray  # (3, 3) intrinsics shared by rgb and depth
    T_body_cam: np.ndarray  # (4, 4) the camera in the pelvis frame, from the robot's own forward kinematics
    source: str  # "stereo:<model>", "rgbd" or "rgbd:<noise model>"
    t: float  # simulation time of the image (s)

    def __post_init__(self):
        h, w = self.depth.shape
        if self.rgb.shape != (h, w, 3):
            raise ValueError(f"Frame: rgb {self.rgb.shape} and depth {self.depth.shape} differ")
        if self.K.shape != (3, 3) or self.T_body_cam.shape != (4, 4):
            raise ValueError("Frame: K must be 3x3 and T_body_cam 4x4")


@runtime_checkable
class Sensor(Protocol):
    name: str  # "stereo", "rgbd" or "rgbd:<noise model>"

    def frame(self) -> Frame:
        """The current image and depth."""
        ...
