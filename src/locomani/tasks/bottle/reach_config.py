"""Explicit, versioned calibration of the current tabletop reach controller."""

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from locomani.tasks.bottle.image_servo_config import BottleCalibration, ServoLimits

# Fields deleted from ReachConfig that may still appear in a committed
# outputs/experiments/*/reach_config.json snapshot. `load` drops them with a
# warning instead of failing, so a historical run keeps loading verbatim.
RETIRED_KEYS = frozenset(
    {
        # refactor step 2, family 6 (2026-09-09): coupled_hand_lead/brake_preplan/
        # pregrasp_lateral_jitter/arrival_advance_jitter/pregrasp_min_forward.
        # brake_preplan is false in every live config; the jitter/forward-clamp
        # knobs were nonzero only in single-variable outputs/experiments records
        # (ready07e_spread_20260908, ready07c_geom_20260907) that must keep loading.
        "brake_preplan",
        "pregrasp_lateral_jitter",
        "pregrasp_min_forward",
        "arrival_advance_jitter",
    }
)


BOOL_FIELDS = (
    "transport_guard",
    "ready_grasp_baseline",
    "ready_grasp_recovery",
    "measured_squat_handover",
    "coupled_squat",
    "timely_capture",
    "visible_expert",
    "image_servo",
    "image_servo_truth_control",
)


@dataclass(frozen=True)
class ReachConfig:
    post_min: float = 0.22
    post_max: float = 0.40
    pre_min: float = 0.18
    pre_max: float = 0.27
    backshift: float = 0.08
    depth_margin: float = 0.01
    azimuth_deg: float = 36.5
    approach_timeout: float = 10.0
    max_drop: float = 0.13
    transport_margin: float = 0.004
    sweep_joint_step: float = 0.04
    transport_guard: bool = True
    ready_grasp_baseline: bool = False
    ready_grasp_recovery: bool = False
    grip_below_center: float = 0.01
    arrival_advance: float = 0.0
    measured_squat_handover: bool = False
    coupled_squat: bool = False
    timely_capture: bool = False
    squat_prediction_backshift_ratio: float = 0.0
    squat_prediction_drop_shortfall: float = 0.0
    recovery_attempts: int = 2
    recovery_timeout: float = 45.0
    # Approach geometry (2026-09-07). Defaults reproduce the 07 controller
    # exactly; the closed-loop failure anatomy of ready07 (15 finger sweeps,
    # 15 topples with no finger contact) motivates the non-default values.
    pregrasp_rise: float = 0.08  # safe_pregrasp height above the grasp centre
    pregrasp_gap: float = 0.07  # safe_pregrasp set-back along the entry axis (was PRE_GAP)
    local_reach_limit: float = 0.20  # local alignment aborts beyond this wrist error
    # Visible expert (2026-09-09): every action must be a function of what the
    # image policy observes. Removes the per-episode RNG stand cell, the
    # arrival_advance latch (brake at a fixed measured depth instead), and the
    # recovery reposition branch (an infeasible stand discards the episode).
    visible_expert: bool = False
    visible_brake_depth: float = 0.31  # pelvis-frame object depth at which the walk stops (07d brake median: first_reachable 0.392 - advance 0.08)
    visible_decel_band: float = 0.12  # speed ramps down over this depth band above the brake depth
    image_servo: bool = False  # RGB cap feedback and re-entrant arm control (ledger bottle §19).
    # Privileged-teacher variant (ledger bottle §19.9): the expert CONTROLS from
    # object truth so it is reliable, while the RGB fit is still computed every
    # tick and the SAME decide() is evaluated on it, recording per frame whether
    # an image-only student would have chosen the same intent. Truth never enters
    # the recorded observation; it selects the action, and the audit says whether
    # that action was reproducible from the images.
    image_servo_truth_control: bool = False
    image_calibration: BottleCalibration = field(default_factory=BottleCalibration)  # RGB feature calibration.
    image_control: ServoLimits = field(default_factory=ServoLimits)  # Explicit feedback and execution limits.

    def __post_init__(self):
        values = asdict(self)
        if any(type(v) is not bool for v in (getattr(self, name) for name in BOOL_FIELDS)):
            raise ValueError("strategy and guard switches must be boolean")
        for name, value in values.items():
            if name not in (
                "transport_guard",
                "ready_grasp_baseline",
                "ready_grasp_recovery",
                "measured_squat_handover",
                "coupled_squat",
                "timely_capture",
                "visible_expert",
                "image_servo",
                "image_servo_truth_control",
                "image_calibration",
                "image_control",
            ) and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError(f"{name} must be a finite number")
        if not (0 < self.post_min < self.post_max and 0 < self.pre_min < self.pre_max):
            raise ValueError("Depth bounds must be positive and ordered")
        if not (
            0 <= self.backshift
            and 0 <= self.depth_margin
            and 0 < self.azimuth_deg < 90
            and self.approach_timeout > 0
            and 0 <= self.max_drop <= 0.13
            and 0 <= self.transport_margin <= 0.03
            and 0 < self.sweep_joint_step <= 0.10
        ):
            raise ValueError("Reach calibration outside supported controller bounds")
        if not (
            0 <= self.squat_prediction_backshift_ratio <= 1.2 and 0 <= self.squat_prediction_drop_shortfall <= 0.02
        ):
            raise ValueError("Squat prediction calibration outside supported bounds")
        if (self.squat_prediction_backshift_ratio or self.squat_prediction_drop_shortfall) and not self.coupled_squat:
            raise ValueError("Squat prediction calibration requires coupled squat")
        if self.timely_capture and not self.coupled_squat:
            raise ValueError("Timing experiments require coupled squat")
        if self.ready_grasp_recovery and (not self.ready_grasp_baseline or not self.transport_guard):
            raise ValueError("Recovery requires ready baseline and collision guard")
        if (
            type(self.recovery_attempts) is not int
            or not 0 <= self.recovery_attempts <= 3
            or not 5 <= self.recovery_timeout <= 90
        ):
            raise ValueError("Recovery budget outside supported bounds")
        # Negative = grip ABOVE the box centre: a tall can (pick_place, 2026-09-10) is
        # gripped near its top so the pitched palm clears the body above the grip.
        if not -0.05 <= self.grip_below_center <= 0.03:
            raise ValueError("grip_below_center must be between -0.05 and .03 m")
        # 0..5 cm was the calibration range of the 07 measured-prediction runs.
        # Raised to 10 cm on 2026-09-07: the learned policy inherits the
        # expert's stop distribution (first_stop median 0.36 m) and succeeds
        # only in its near half (closest approach <= 0.34 m), so the next
        # collection stands ~5 cm closer. The table-edge gate still bounds it.
        if not 0 <= self.arrival_advance <= 0.10:
            raise ValueError("arrival_advance must be within 0..10 cm")
        if (
            self.arrival_advance or self.measured_squat_handover or self.coupled_squat
        ) and not self.ready_grasp_recovery:
            raise ValueError("Arrival experiment requires recovery strategy")
        if not (
            0.05 <= self.pregrasp_rise <= 0.20
            and 0.03 <= self.pregrasp_gap <= 0.15
            and 0.15 <= self.local_reach_limit <= 0.35
        ):
            raise ValueError("approach geometry outside supported bounds")
        if self.visible_expert:
            if not (self.ready_grasp_baseline and self.ready_grasp_recovery):
                raise ValueError("visible_expert requires the ready baseline with the recovery controller")
            if not (0.20 <= self.visible_brake_depth <= 0.45 and 0 <= self.visible_decel_band <= 0.30):
                raise ValueError("visible brake geometry outside supported bounds")
        if self.image_servo and not (self.visible_expert and self.coupled_squat and self.timely_capture):
            raise ValueError("image_servo requires visible_expert, coupled_squat and timely_capture")
        if not isinstance(self.image_calibration, BottleCalibration) or not isinstance(self.image_control, ServoLimits):
            raise ValueError("image calibration/control must be typed configurations")
        self.approach_band()

    def approach_band(self):
        lo = max(self.pre_min, self.post_min + self.depth_margin - self.backshift)
        hi = min(self.pre_max, self.post_max - self.depth_margin - self.backshift)
        if lo >= hi:
            raise ValueError(f"Empty pre-crouch depth band: {lo:.3f}..{hi:.3f}")
        return lo, hi

    def to_dict(self):
        return dict(schema_version=1, **asdict(self))

    @classmethod
    def from_dict(cls, data: dict, *, source: str = "<dict>") -> "ReachConfig":
        """Reconstruct from a `to_dict()`-shaped mapping, dropping RETIRED_KEYS.

        Used both by `load` (a JSON file) and by BottleTaskConfig's nested
        `reach` field (an in-memory dict already parsed from the task's JSON).
        """
        data = dict(data)
        if data.pop("schema_version", None) != 1:
            raise ValueError("reach config requires schema_version=1")
        for key in RETIRED_KEYS.intersection(data):
            warnings.warn(f"reach config {source}: ignoring retired key {key!r}", stacklevel=2)
            del data[key]
        for key, factory in (("image_calibration", BottleCalibration), ("image_control", ServoLimits)):
            if key in data and isinstance(data[key], dict):
                data[key] = factory(**data[key])
        return cls(**data)

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()), source=str(path))

    def dump(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1) + "\n")


DEFAULT_REACH = ReachConfig()


def require_runnable(config: "ReachConfig", source: str | None = None) -> None:
    """Refuse a calibration that cannot produce an episode, before any actuation.

    `ready_grasp_recovery=False` is the old non-recovery approach/crouch/servo
    tail, deleted in refactor step 2. `execute_grasp` fails fast on it, but only
    after the round has paid for a reset, a walk and an arrival -- and since the
    approach itself succeeds, the log reads like a broken controller rather than
    a configuration that was never applied (2026-09-10, cost a stack cycle).
    A calibration is not loaded from anywhere by default, so this also catches
    the commonest cause: no calibration was passed at all and the dataclass
    default came through.
    """
    if config.ready_grasp_recovery:
        return
    where = (
        f"--reach-config {source} sets it False"
        if source
        else ("no reach calibration was passed, so this is ReachConfig()'s default")
    )
    raise SystemExit(
        "reach calibration has ready_grasp_recovery=False, which the removal of the "
        "non-recovery path made unrunnable: every round would walk to the object and then "
        f"fail in the grasp stage.\n  {where}\n"
        "  pass the collection profile's calibration -- --task-config "
        "configs/tasks/bottle_visible.json applies its `reach` block, or give --reach-config <file>."
    )


def require_expert_calibration(vla_policy, config: "ReachConfig", source: str | None = None) -> None:
    """`require_runnable`, but only where the expert grasp actually runs.

    A closed-loop evaluation sets grasp_gym.VLA_POLICY, and `_p_handover`
    returns the model's round before `_p_approach` or `_p_grasp` execute, so
    `execute_grasp` -- the only thing `ready_grasp_recovery` gates -- is never
    reached. `eval_closed_loop.py` passes --reach-config only when one was asked
    for, so refusing there would block every eval that did not name one: the
    same shape of false positive as the exporter probe that matched one entry
    point of two (both 2026-09-10).

    Do NOT read this as "the calibration does not matter to a VLA round". It
    still does: `ready_grasp_policy.sample_stand` reads `max_drop`, `backshift`
    and `ready_grasp_recovery`, so the calibration moves the sampled stand and
    therefore the case distribution, and `round_phases` has a
    `ready_grasp_baseline` branch that runs specifically when vla_policy is not
    None. What is safe is not refusing; silently SUBSTITUTING one calibration
    for another in an eval is not (see cli_args._apply_task_config).
    """
    if vla_policy is None:
        require_runnable(config, source)
