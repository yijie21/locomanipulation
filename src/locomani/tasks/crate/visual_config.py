"""Limits for the crate RGB expert; independent of the legacy wall/carry experts."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrateVisualConfig:
    waist_pitch_max: float = 0.40  # radians; additionally bounded by the robot model
    palm_pitch_deg: float = 60.0  # downward grasp orientation relative to the observed basket
    stand_off: float = 0.36  # initial pelvis distance from the nominal basket centre, metres
    position_jitter: float = 0.02  # visible basket x/y randomisation at reset, metres
    yaw_jitter_deg: float = 5.0  # visible basket yaw randomisation at reset
    pregrasp_above: float = 0.09  # metres above each grasp point
    approach_speed: float = 0.055  # maximum Cartesian reference speed, metres/second
    joint_speed: float = 0.7  # maximum reference joint speed, radians/second
    tracking_gain: float = 0.5  # proportional correction from observed tool-point error
    tracking_cap: float = 0.025  # maximum Cartesian tracking correction, metres
    align_tolerance: float = 0.018  # both measured grasp points must align within this distance
    close_seconds: float = 1.6  # coordinated gripper closing duration
    probe_height: float = 0.025  # initial lift used to observe bilateral attachment
    lift_height: float = 0.10  # total desired rise after the probe
    hold_seconds: float = 2.0  # stable hold before ending the attempt
    timeout_seconds: float = 35.0  # bounded expert attempt
    vision_timeout: float = 0.7  # stop advancing when the last valid RGB estimate is older
    reprojection_px: float = 4.0  # maximum RMS marker reprojection residual
    max_tilt_deg: float = 15.0  # observed basket tilt limit during lifting

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"visual.{name} must be a finite non-negative number")
        if not 0 <= self.waist_pitch_max <= 0.52:
            raise ValueError("visual.waist_pitch_max must be in [0, 0.52]")
        if not 0 < self.probe_height < self.lift_height:
            raise ValueError("visual.probe_height must be smaller than lift_height")
        for name in (
            "stand_off",
            "joint_speed",
            "approach_speed",
            "close_seconds",
            "timeout_seconds",
            "vision_timeout",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"visual.{name} must be positive")
