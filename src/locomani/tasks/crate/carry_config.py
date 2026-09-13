"""Carry trajectory and stance settings, resolved by the task entry."""

import math
from dataclasses import dataclass

START_EDGE_RANGE_M = (1.2, 1.5)  # recorded locomotion starts outside the near table edge


@dataclass(frozen=True)
class CrateCarryConfig:
    settle_seconds: float = 0.3  # continuous observed quiet interval, seconds
    joint_speed: float = 0.7  # maximum reference speed, rad/s
    joint_acceleration: float = 2.0  # maximum reference acceleration, rad/s²
    waist_pitch_max: float = 0.4  # forward waist bound, rad; also clipped to the loaded model
    pelvis_height: float | None = None  # measured crouch target, m; None keeps the standing flow
    height_speed: float = 0.04  # maximum planner height-command slew, m/s

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "pelvis_height" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"carry.{name} must be finite and positive")
        if self.waist_pitch_max > 0.52:
            raise ValueError("carry.waist_pitch_max must not exceed 0.52 rad for G1")
        if self.pelvis_height is not None and not 0.50 <= self.pelvis_height <= 0.75:
            raise ValueError("carry.pelvis_height must be between 0.50 and 0.75 m")


DEFAULT_CARRY = CrateCarryConfig()
