"""Limits for the crate RGB expert; independent of the legacy wall/carry experts."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CrateVisualConfig:
    pose_source: str = "markers"  # markers: RGB fitting; assisted: visible target with simulation pose/feedback
    waist_pitch_max: float = 0.40  # radians; additionally bounded by the robot model
    palm_pitch_deg: float = 60.0  # downward grasp orientation relative to the observed basket
    start_edge_min: float = 1.2  # initial pelvis distance from the near table edge, metres
    start_edge_max: float = 1.5  # upper end of initial table-edge distance sampling
    walk_speed: float = 0.35  # maximum SONIC approach speed command, metres/second
    walk_brake_margin: float = 0.07  # stop commanding steps before the final basket distance
    walk_timeout: float = 30.0  # maximum walking and stance-settling duration
    settle_seconds: float = 1.0  # continuously quiet visual/IMU interval before reaching
    stand_lateral_tolerance: float = 0.04  # metres; final body IK still owns low-reach admission
    lift_seconds: float = 4.0  # lift ramp duration; assisted standing overlaps after the measured probe
    stand_seconds: float = 5.0  # attached recovery duration after the measured probe, seconds
    upright_tolerance_deg: float = 12.0  # measured torso lean allowed at successful completion
    stand_off: float = 0.36  # final pelvis distance from the visible basket centre, metres
    position_jitter: float = 0.02  # visible basket x/y randomisation at reset, metres
    yaw_jitter_deg: float = 5.0  # visible basket yaw randomisation at reset
    pregrasp_above: float = 0.09  # metres above each grasp point
    approach_speed: float = 0.055  # maximum Cartesian reference speed, metres/second
    joint_speed: float = 0.7  # maximum reference joint speed, radians/second
    joint_acceleration: float = 2.0  # maximum reference velocity change, radians/second squared
    palm_rotation_speed: float = 0.6  # maximum palm reference rotation speed, radians/second
    palm_align_tolerance_deg: float = 10.0  # both measured wrists must align before closing
    tracking_gain: float = 0.5  # proportional correction from observed tool-point error
    tracking_cap: float = 0.025  # maximum Cartesian tracking correction, metres
    align_tolerance: float = 0.018  # both measured grasp points must align within this distance
    close_seconds: float = 1.6  # coordinated gripper closing duration
    probe_height: float = 0.025  # initial lift used to observe bilateral attachment
    lift_height: float = 0.10  # total desired rise after the probe
    hold_seconds: float = 2.0  # stable hold before ending the attempt
    hold_position_tolerance: float = 0.025  # maximum load-position radius over the final hold window, metres
    timeout_seconds: float = 35.0  # bounded expert attempt
    vision_timeout: float = 0.7  # stop advancing when the last valid RGB estimate is older
    reprojection_px: float = 4.0  # maximum RMS marker reprojection residual
    max_tilt_deg: float = 15.0  # observed basket tilt limit during lifting

    def __post_init__(self):
        if self.pose_source not in ("markers", "assisted"):
            raise ValueError("visual.pose_source must be markers or assisted")
        for name, value in vars(self).items():
            if name == "pose_source":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"visual.{name} must be a finite non-negative number")
        if not 0 <= self.waist_pitch_max <= 0.52:
            raise ValueError("visual.waist_pitch_max must be in [0, 0.52]")
        if not 0 < self.upright_tolerance_deg <= 20:
            raise ValueError("visual.upright_tolerance_deg must be in (0, 20]")
        if not 0 < self.probe_height < self.lift_height:
            raise ValueError("visual.probe_height must be smaller than lift_height")
        if not 1.2 <= self.start_edge_min <= self.start_edge_max <= 1.5:
            raise ValueError("visual.start_edge_min/max must define an interval within [1.2, 1.5] metres")
        for name in (
            "stand_off",
            "stand_lateral_tolerance",
            "lift_seconds",
            "stand_seconds",
            "upright_tolerance_deg",
            "walk_speed",
            "walk_timeout",
            "settle_seconds",
            "pregrasp_above",
            "align_tolerance",
            "reprojection_px",
            "hold_seconds",
            "hold_position_tolerance",
            "max_tilt_deg",
            "joint_speed",
            "joint_acceleration",
            "palm_rotation_speed",
            "palm_align_tolerance_deg",
            "approach_speed",
            "close_seconds",
            "timeout_seconds",
            "vision_timeout",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"visual.{name} must be positive")
