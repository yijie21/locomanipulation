"""Explicit, versioned calibration of the current tabletop reach controller."""
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class ReachConfig:
    post_min: float = .22
    post_max: float = .40
    pre_min: float = .18
    pre_max: float = .27
    backshift: float = .08
    depth_margin: float = .01
    azimuth_deg: float = 36.5
    approach_timeout: float = 10.
    max_drop: float = .13
    transport_margin: float = .004
    sweep_joint_step: float = .04
    transport_guard: bool = True
    coordinated_arrival: bool = False
    ready_grasp_baseline: bool = False
    ready_grasp_recovery: bool = False
    recovery_region_stop: bool = False
    grip_below_center: float = .01
    arrival_advance: float = 0.
    measured_squat_handover: bool = False
    coupled_squat: bool = False
    timely_capture: bool = False
    capture_reference_handover: bool = False
    brake_preplan: bool = False
    continuous_capture: bool = False
    tool_point_grasp: bool = False
    horizontal_entry: bool = False
    grasp_depth_preference: float = 0.
    local_point_lead: float = .035
    coupled_hand_lead: float = 0.  # shorter IK retry only when the full target is infeasible
    local_point_speed: float = .125
    squat_prediction_backshift_ratio: float = 0.
    squat_prediction_drop_shortfall: float = 0.
    recovery_attempts: int = 2
    recovery_timeout: float = 45.
    # Approach geometry (2026-09-07). Defaults reproduce the 07 controller
    # exactly; the closed-loop failure anatomy of ready07 (15 finger sweeps,
    # 15 topples with no finger contact) motivates the non-default values.
    pregrasp_rise: float = .08            # safe_pregrasp height above the grasp centre
    pregrasp_gap: float = .07             # safe_pregrasp set-back along the entry axis (was PRE_GAP)
    pregrasp_lateral_jitter: float = 0.   # +-uniform lateral offset of safe_pregrasp; demonstrates the lateral correction
    local_reach_limit: float = .20        # local alignment aborts beyond this wrist error
    pregrasp_min_forward: float = 0.      # clamp safe_pregrasp to >= this far ahead of the pelvis (0 = off); a large gap at a close stand otherwise puts the palm inside the torso
    arrival_advance_jitter: float = 0.    # per-episode uniform +-jitter on arrival_advance: spreads the expert's stand so the squat label covers the policy's own stop spread (ready07d: P(squat)=0.12 at 0.38-0.43 m)
    # Visible expert (2026-09-09): every action must be a function of what the
    # image policy observes. Removes the per-episode RNG stand cell, the
    # arrival_advance latch (brake at a fixed measured depth instead), and the
    # recovery reposition branch (an infeasible stand discards the episode).
    visible_expert: bool = False
    visible_brake_depth: float = .31      # pelvis-frame object depth at which the walk stops (07d brake median: first_reachable 0.392 - advance 0.08)
    visible_decel_band: float = .12       # speed ramps down over this depth band above the brake depth

    def __post_init__(self):
        values = asdict(self)
        if any(type(v) is not bool for v in (self.transport_guard, self.coordinated_arrival, self.ready_grasp_baseline, self.ready_grasp_recovery, self.measured_squat_handover, self.coupled_squat, self.timely_capture, self.brake_preplan, self.continuous_capture, self.tool_point_grasp, self.horizontal_entry, self.recovery_region_stop, self.capture_reference_handover, self.visible_expert)):
            raise ValueError('strategy and guard switches must be boolean')
        for name, value in values.items():
            if name not in ('transport_guard', 'coordinated_arrival', 'ready_grasp_baseline', 'ready_grasp_recovery', 'measured_squat_handover', 'coupled_squat', 'timely_capture', 'brake_preplan', 'continuous_capture', 'tool_point_grasp', 'horizontal_entry', 'recovery_region_stop', 'capture_reference_handover', 'visible_expert') and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f'{name} must be a finite number')
        if not (0 < self.post_min < self.post_max and 0 < self.pre_min < self.pre_max):
            raise ValueError('Depth bounds must be positive and ordered')
        if not (0 <= self.backshift and 0 <= self.depth_margin and 0 < self.azimuth_deg < 90
                and self.approach_timeout > 0 and 0 <= self.max_drop <= .13
                and 0 <= self.transport_margin <= .03 and 0 < self.sweep_joint_step <= .10):
            raise ValueError('Reach calibration outside supported controller bounds')
        if self.capture_reference_handover and (not (self.coupled_squat and self.timely_capture)
                or self.tool_point_grasp or self.continuous_capture):
            raise ValueError('Capture reference handover requires the timely coupled wrist controller')
        if self.coupled_hand_lead and (not self.coupled_squat or not .005 <= self.coupled_hand_lead <= .05):
            raise ValueError('Coupled hand lead requires coupled squat and a 5–50 mm bound')
        if self.recovery_region_stop and not self.ready_grasp_recovery:
            raise ValueError('Region stopping requires recovery policy')
        if self.coordinated_arrival and self.ready_grasp_baseline:
            raise ValueError("Choose one collection policy")
        if self.continuous_capture and not (self.coupled_squat and self.timely_capture):
            raise ValueError('Continuous capture requires coupled squat and timely capture')
        if self.tool_point_grasp and not (self.coupled_squat and self.timely_capture):
            raise ValueError('Tool-point grasp requires coupled squat and timely capture')
        if self.tool_point_grasp and self.continuous_capture:
            raise ValueError('Tool-point control and legacy continuous wrist feedback are separate experiments')
        if self.horizontal_entry and not self.tool_point_grasp:
            raise ValueError('Horizontal entry requires tool-point control')
        if not (0 <= self.grasp_depth_preference <= .015
                and .005 <= self.local_point_lead <= .035
                and .01 <= self.local_point_speed <= .125):
            raise ValueError('Grasp preference or point tracking outside supported bounds')
        if not (0 <= self.squat_prediction_backshift_ratio <= 1.2
                and 0 <= self.squat_prediction_drop_shortfall <= .02):
            raise ValueError('Squat prediction calibration outside supported bounds')
        if (self.squat_prediction_backshift_ratio or self.squat_prediction_drop_shortfall) and not self.coupled_squat:
            raise ValueError('Squat prediction calibration requires coupled squat')
        if self.grasp_depth_preference and not self.coupled_squat:
            raise ValueError('Grasp depth preference requires coupled capture')
        if (self.timely_capture or self.brake_preplan) and not self.coupled_squat:
            raise ValueError('Timing experiments require coupled squat')
        if self.ready_grasp_recovery and (not self.ready_grasp_baseline or not self.transport_guard):
            raise ValueError("Recovery requires ready baseline and collision guard")
        if type(self.recovery_attempts) is not int or not 0 <= self.recovery_attempts <= 3 or not 5 <= self.recovery_timeout <= 90:
            raise ValueError("Recovery budget outside supported bounds")
        if not 0 <= self.grip_below_center <= .03:
            raise ValueError("grip_below_center must be between 0 and .03 m")
        # 0..5 cm was the calibration range of the 07 measured-prediction runs.
        # Raised to 10 cm on 2026-09-07: the learned policy inherits the
        # expert's stop distribution (first_stop median 0.36 m) and succeeds
        # only in its near half (closest approach <= 0.34 m), so the next
        # collection stands ~5 cm closer. The table-edge gate still bounds it.
        if not 0 <= self.arrival_advance <= .10:
            raise ValueError('arrival_advance must be within 0..10 cm')
        if (self.arrival_advance or self.measured_squat_handover or self.coupled_squat) and not self.ready_grasp_recovery:
            raise ValueError('Arrival experiment requires recovery strategy')
        if not (.05 <= self.pregrasp_rise <= .20 and .03 <= self.pregrasp_gap <= .15
                and 0 <= self.pregrasp_lateral_jitter <= .05 and .15 <= self.local_reach_limit <= .35
                and 0 <= self.pregrasp_min_forward <= .25 and 0 <= self.arrival_advance_jitter <= .05
                and 0 <= self.arrival_advance-self.arrival_advance_jitter and self.arrival_advance+self.arrival_advance_jitter <= .10):
            raise ValueError('approach geometry outside supported bounds')
        if self.visible_expert:
            if not (self.ready_grasp_baseline and self.ready_grasp_recovery):
                raise ValueError('visible_expert requires the ready baseline with the recovery controller')
            if self.arrival_advance_jitter or self.pregrasp_lateral_jitter or self.recovery_region_stop or self.brake_preplan:
                raise ValueError('visible_expert excludes per-episode jitter, region stopping and brake preplanning')
            if not (.20 <= self.visible_brake_depth <= .45 and 0 <= self.visible_decel_band <= .30):
                raise ValueError('visible brake geometry outside supported bounds')
        self.approach_band()

    def approach_band(self):
        lo = max(self.pre_min, self.post_min+self.depth_margin-self.backshift)
        hi = min(self.pre_max, self.post_max-self.depth_margin-self.backshift)
        if lo >= hi: raise ValueError(f'Empty pre-crouch depth band: {lo:.3f}..{hi:.3f}')
        return lo, hi

    def to_dict(self):
        return dict(schema_version=1, **asdict(self))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.pop('schema_version', None) != 1:
            raise ValueError('reach config requires schema_version=1')
        return cls(**data)


DEFAULT_REACH = ReachConfig()
