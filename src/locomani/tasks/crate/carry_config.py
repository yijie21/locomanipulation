"""Carry trajectory and stance settings, resolved by the task entry."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrateCarryConfig:
    settle_seconds: float = 0.3  # continuous observed quiet interval, seconds
    joint_speed: float = 0.7  # maximum reference speed, rad/s
    joint_acceleration: float = 2.0  # maximum reference acceleration, rad/s²

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"carry.{name} must be finite and positive")


DEFAULT_CARRY = CrateCarryConfig()
