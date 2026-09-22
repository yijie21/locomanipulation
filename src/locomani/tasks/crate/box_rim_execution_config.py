"""Dependency-light configuration contract for generic box-rim execution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector(value, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {length} finite values")
    return result.copy()


@dataclass(frozen=True)
class BoxRimExecutionConfig:
    approach_mode: str = "supported_pose"
    # coordinated only: do not stop on planned/measured wall contacts before the grasp -- the
    # controller reaches, closes and lifts; the box's own rise/slip/tilt verdict decides. Falls and
    # the object guards still apply. A "just try it" mode for the grasp itself (2026-09-22).
    direct_grasp: bool = False
    observation_max_age_s: float = 0.25
    watchdog_wall_s: float = 1.0
    settle_s: float = 0.5
    settle_timeout_s: float = 5.0
    planning_timeout_s: float = 12.0
    settle_linear_speed_mps: float = 0.05
    upright_tolerance_deg: float = 15.0
    minimum_pelvis_height_m: float = 0.66
    pregrasp_clearance_m: float = 0.055
    pregrasp_position_tolerance_m: float = 0.035
    align_position_tolerance_m: float = 0.018
    align_orientation_tolerance_rad: float = 0.22
    align_hold_s: float = 0.25
    align_timeout_s: float = 5.0
    maximum_joint_speed_rps: float = 0.35
    maximum_joint_acceleration_rps2: float = 0.70
    swept_samples: int = 12
    initial_escape_max_penetration_m: float = 0.0005
    initial_escape_tracking_growth_m: float = 0.00025
    ready_shoulder_clearance_rad: float = 0.15
    observed_foot_anchor_tolerance_m: float = 0.005
    waypoint_settle_s: float = 0.25
    low_reach_minimum_bend_s: float = 4.0
    low_reach_tracking_tolerance_m: float = 0.065
    initial_escape_samples: int = 41
    initial_escape_attempts: int = 2
    close_s: float = 1.6
    contact_hold_s: float = 0.25
    probe_lift_m: float = 0.025
    probe_timeout_s: float = 3.0
    minimum_object_rise_m: float = 0.018
    maximum_slip_m: float = 0.025
    maximum_object_tilt_rad: float = 0.16
    maximum_box_motion_before_close_m: float = 0.018
    maximum_box_tilt_before_close_rad: float = 0.10
    coordinated_stand_off_m: float = 0.39
    coordinated_stand_lateral_tolerance_m: float = 0.14
    coordinated_walk_speed_mps: float = 0.35
    coordinated_walk_brake_margin_m: float = 0.10
    coordinated_settle_s: float = 0.30
    coordinated_pregrasp_above_m: float = 0.04
    coordinated_approach_speed_mps: float = 0.16
    coordinated_tracking_gain: float = 0.70
    coordinated_tracking_cap_m: float = 0.04
    enable_reposition: bool = True
    maximum_reposition_yaw_change_rad: float = 0.35
    reposition_candidates: tuple[tuple[float, float, float], ...] = (
        (0.10, 0.0, 0.0),
        (0.08, 0.05, 0.0),
        (0.08, -0.05, 0.0),
        (0.06, 0.0, 0.12),
        (0.06, 0.0, -0.12),
    )
    # SONIC's historical 0.10 speed command sits in its gait dead band.  This is
    # still bounded by measured travel, target feedback and an explicit stop.
    reposition_speed_mps: float = 0.25
    reposition_position_tolerance_m: float = 0.035
    reposition_braking_horizon_s: float = 0.40
    reposition_yaw_tolerance_rad: float = 0.12
    reposition_timeout_s: float = 6.0
    reposition_retries: int = 2
    maximum_total_reposition_m: float = 0.22

    def __post_init__(self):
        if self.approach_mode not in ("supported_pose", "planner_arms", "legacy_low_reach", "coordinated"):
            raise ValueError("approach_mode must be supported_pose, planner_arms, legacy_low_reach or coordinated")
        if not isinstance(self.direct_grasp, bool):
            raise ValueError("direct_grasp must be boolean")
        if self.direct_grasp and self.approach_mode != "coordinated":
            raise ValueError("direct_grasp is only defined for approach_mode=coordinated")
        positive = (
            self.observation_max_age_s,
            self.watchdog_wall_s,
            self.settle_s,
            self.settle_timeout_s,
            self.planning_timeout_s,
            self.settle_linear_speed_mps,
            self.upright_tolerance_deg,
            self.minimum_pelvis_height_m,
            self.pregrasp_clearance_m,
            self.pregrasp_position_tolerance_m,
            self.align_position_tolerance_m,
            self.align_orientation_tolerance_rad,
            self.align_hold_s,
            self.align_timeout_s,
            self.maximum_joint_speed_rps,
            self.maximum_joint_acceleration_rps2,
            self.initial_escape_max_penetration_m,
            self.initial_escape_tracking_growth_m,
            self.ready_shoulder_clearance_rad,
            self.observed_foot_anchor_tolerance_m,
            self.waypoint_settle_s,
            self.low_reach_minimum_bend_s,
            self.low_reach_tracking_tolerance_m,
            self.close_s,
            self.contact_hold_s,
            self.probe_lift_m,
            self.probe_timeout_s,
            self.minimum_object_rise_m,
            self.maximum_slip_m,
            self.maximum_object_tilt_rad,
            self.maximum_box_motion_before_close_m,
            self.maximum_box_tilt_before_close_rad,
            self.coordinated_stand_off_m,
            self.coordinated_stand_lateral_tolerance_m,
            self.coordinated_walk_speed_mps,
            self.coordinated_walk_brake_margin_m,
            self.coordinated_settle_s,
            self.coordinated_pregrasp_above_m,
            self.coordinated_approach_speed_mps,
            self.coordinated_tracking_gain,
            self.coordinated_tracking_cap_m,
            self.maximum_reposition_yaw_change_rad,
            self.reposition_speed_mps,
            self.reposition_position_tolerance_m,
            self.reposition_yaw_tolerance_rad,
            self.reposition_timeout_s,
            self.maximum_total_reposition_m,
        )
        if not np.isfinite(positive).all() or any(value <= 0 for value in positive):
            raise ValueError("execution timing, motion and acceptance limits must be finite positive")
        if not isinstance(self.enable_reposition, bool):
            raise ValueError("enable_reposition must be boolean")
        if not np.isfinite(self.reposition_braking_horizon_s) or self.reposition_braking_horizon_s < 0:
            raise ValueError("reposition_braking_horizon_s must be finite nonnegative")
        if (
            self.swept_samples < 2
            or self.initial_escape_samples < 3
            or self.initial_escape_attempts < 1
            or self.reposition_retries < 0
        ):
            raise ValueError(
                "swept_samples must be >=2, initial_escape_samples >=3, "
                "initial_escape_attempts >=1 and reposition_retries nonnegative"
            )
        candidates = []
        for index, candidate in enumerate(self.reposition_candidates):
            values = _vector(candidate, 3, f"reposition_candidates[{index}]")
            if np.linalg.norm(values[:2]) > self.maximum_total_reposition_m + 1e-12:
                raise ValueError("reposition candidate exceeds total travel budget")
            candidates.append(tuple(float(value) for value in values))
        if not candidates or not any(np.linalg.norm(value[:2]) > 0 for value in candidates):
            raise ValueError("reposition_candidates need at least one nonzero translation")
        object.__setattr__(self, "reposition_candidates", tuple(candidates))


def execution_config_from_dict(values: dict | None) -> BoxRimExecutionConfig:
    raw = {} if values is None else dict(values)
    allowed = set(BoxRimExecutionConfig.__dataclass_fields__)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"execution unknown fields: {', '.join(unknown)}")
    if "reposition_candidates" in raw:
        raw["reposition_candidates"] = tuple(tuple(item) for item in raw["reposition_candidates"])
    return BoxRimExecutionConfig(**raw)


__all__ = ["BoxRimExecutionConfig", "execution_config_from_dict"]
