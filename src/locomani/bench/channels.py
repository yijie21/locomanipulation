"""What a benchmark session hands out (TODO 49.1): camera streams for the Sensors, and the privileged state.

A benchmark adapter implements the camera streams its capability table supports. `Privileged` is the simulator's
truth: only the truth View, the truth SelfPose, the judge (success and compliance) and the records may hold one; a
skill package never receives it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from locomani.perception.sensor import Frame


@dataclass(frozen=True)
class StereoPair:
    left: np.ndarray  # (H, W, 3) uint8
    right: np.ndarray  # (H, W, 3) uint8
    K: np.ndarray  # (3, 3) intrinsics of the left camera (the pair is rectified)
    baseline: float  # metres between the two optical centres
    T_body_cam: np.ndarray  # (4, 4) the left camera in the pelvis frame
    t: float  # simulation time (s)


@runtime_checkable
class Cameras(Protocol):
    def stereo(self) -> StereoPair:
        """The head's stereo pair (benchmarks whose table supports sensor=stereo)."""
        ...

    def rgbd(self) -> Frame:
        """One camera's image with the rendered depth, exact (benchmarks whose table supports sensor=rgbd)."""
        ...


@runtime_checkable
class Privileged(Protocol):
    def pelvis(self) -> tuple[np.ndarray, float]:
        """The pelvis's world position and yaw."""
        ...

    def body_pose(self, name: str) -> np.ndarray:
        """(4, 4) world pose of a named body (the target, a container, a door leaf)."""
        ...

    def body_vertices(self, name: str) -> np.ndarray:
        """(N, 3) world vertices of a named body's collision geometry."""
        ...

    def episode(self) -> dict:
        """The episode's task facts: instruction, target and container names, tables, start pose."""
        ...
