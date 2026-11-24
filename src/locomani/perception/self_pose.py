"""SelfPose: where the robot believes its pelvis is (option self_pose, TODO 49.1 / 49.3).

The robot's frame is the world as it was at the episode's start. `truth` reads the simulator's pelvis; `odometry`
carries the start along with leg odometry (joint encoders, gyro, gravity and foot loads; leg_odo.LegOdometry on the
esc bed). What ScaleBFM itself is told is a separate option, controller_root.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SelfPose(Protocol):
    name: str  # "truth" or "odometry"

    def start(self) -> None:
        """Make the robot's frame the current pose (episode start)."""
        ...

    def pelvis(self) -> tuple[np.ndarray, float]:
        """(xyz position, yaw) of the pelvis in the robot's frame."""
        ...
