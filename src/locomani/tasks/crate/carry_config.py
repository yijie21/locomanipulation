"""Carry trajectory and stance settings, resolved by the task entry."""

import math
from dataclasses import dataclass

START_EDGE_RANGE_M = (1.2, 1.5)  # recorded locomotion starts outside the near table edge


@dataclass(frozen=True)
class CrateCarryConfig:
    reach_control: str = "planner"  # planner upper-body reference or whole_body pose stream
    waist_dofs: str = "all"  # all physical waist axes, or pitch-only comparison
    settle_seconds: float = 0.3  # continuous observed quiet interval, seconds
    joint_speed: float = 0.7  # maximum reference speed, rad/s
    joint_acceleration: float = 2.0  # maximum reference acceleration, rad/s²
    waist_pitch_max: float = 0.4  # forward waist bound, rad; also clipped to the loaded model
    pelvis_height: float | None = None  # measured crouch target, m; None keeps the standing flow
    height_speed: float = 0.04  # maximum planner height-command slew, m/s
    pelvis_pitch_max: float = 0.70  # whole-body forward pelvis orientation bound, rad
    pelvis_height_min: float = 0.65  # whole-body minimum pelvis height, m; encourages shallow knee flexion

    def __post_init__(self):
        if self.reach_control not in ("planner", "whole_body"):
            raise ValueError("carry.reach_control must be planner or whole_body")
        if self.waist_dofs not in ("pitch", "all"):
            raise ValueError("carry.waist_dofs must be pitch or all")
        for name, value in vars(self).items():
            if name in ("reach_control", "waist_dofs"):
                continue
            if name == "pelvis_height" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"carry.{name} must be finite and positive")
        if self.waist_pitch_max > 0.52:
            raise ValueError("carry.waist_pitch_max must not exceed 0.52 rad for G1")
        if self.pelvis_height is not None and not 0.50 <= self.pelvis_height <= 0.75:
            raise ValueError("carry.pelvis_height must be between 0.50 and 0.75 m")
        if self.reach_control == "whole_body" and self.pelvis_height is not None:
            raise ValueError("whole_body controls pelvis posture; do not also request a planner squat height")
        if self.pelvis_pitch_max > 0.85 or not 0.55 <= self.pelvis_height_min <= 0.78:
            raise ValueError("whole-body pelvis bounds require pitch <= 0.85 rad and height between 0.55 and 0.78 m")


DEFAULT_CARRY = CrateCarryConfig()
