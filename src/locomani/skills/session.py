"""The state carried between primitive calls in one episode (design doc §3).

`Session` collects what today's expert code keeps scattered across `RoundContext`,
crate's `run_attempt` locals and each arm class's own fields: who owns the upper-body
channel, the walking heading anchor, the held object, the last approach's stance, and
the run's `ExecutionRecord`. `invariant_violations` is the read side of design doc §3's
five invariants; the executor (`skills/executor.py`) is what enforces them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import numpy as np

from locomani.pipeline.execution_record import ExecutionRecord
from locomani.skills.contract import Violation


class ArmOwner(Enum):
    PLANNER_SWING = "planner_swing"  # the planner swings its own arms (HangingWalk); for walking
    PLANNER_FROZEN = "planner_frozen"  # upper body pinned at q17 (PlannerArm); legs stay the planner's
    POSE_STREAM = "pose_stream"  # full-body 29-DoF POSE stream (StreamedArm); legs are ours too


@dataclass
class HeldObject:
    key: str
    sides: tuple[str, ...]  # ("right",) or ("left", "right")
    grip_in_palm: dict[str, np.ndarray]  # object position in each palm's frame; place's servo reference
    hold_offset: np.ndarray | None = None  # single-arm: torso-frame hold offset; bimanual: None
    start_z: float | None = (
        None  # world z of the object at the moment it was grasped; retreat's load floor is relative to it
    )


@dataclass
class Stance:
    target: str  # which object/surface the approach stopped in front of
    profile: str  # which profile (skills/profiles.py) the approach used
    depth_m: float  # front line to the nearest point, at the stop
    lateral_m: float
    yaw_err_rad: float
    quiet: bool  # did the stationary-admission gate pass


@dataclass
class Session:
    stack: Any  # the Primitives-like commander (backends/sonic/primitives.py); tests inject a fake
    scene: Any  # skills.scene_model.SceneModel
    record: ExecutionRecord = field(default_factory=ExecutionRecord)
    clock: Callable[[], float] = time.monotonic  # sim time; wall-clock vs lockstep differ only here
    arm_owner: ArmOwner = ArmOwner.PLANNER_SWING
    arm: Any = None  # HangingWalk | PlannerArm | StreamedArm
    heading: float | None = None  # world heading anchor; None = not yet re-anchored
    hands: dict = field(default_factory=lambda: {"left": 0.0, "right": 0.0})  # 0 open .. 1 closed
    held: HeldObject | None = None
    stance: Stance | None = None
    trace: list = field(default_factory=list)  # one entry per primitive call (executor.py)
    reset: Any = None  # sim EpisodeControlClient (pin_arms / ghost); None on a stack without one
    camera: Any = None  # dataset camera subscriber (read() -> images + timestamps); the bimanual line needs it
    directory: Any = None  # Path for per-episode artefacts (recordings, ghost trace); None = do not record
    scratch: dict = field(default_factory=dict)  # adapter-private hand-over state, keyed by adapter name


def invariant_violations(session: Session) -> list[str]:
    """The design doc §3 invariants an executor checks around every primitive call.

    Invariants 4 and 5 (who is allowed to write `stance`/`held`) are a property of
    which primitive ran, not of the session's contents, so the executor enforces
    those itself; this only checks the two that are readable off `session` alone.
    """
    violations: list[str] = []

    owner = session.arm_owner
    arm_cls = type(session.arm).__name__ if session.arm is not None else None
    if owner is ArmOwner.POSE_STREAM:
        if arm_cls != "StreamedArm" or getattr(session.arm, "_stream_released", False):
            violations.append(f"arm_owner is pose_stream but arm is {arm_cls!r} (or already released)")
    elif owner is ArmOwner.PLANNER_FROZEN:
        if session.arm is not None and arm_cls != "PlannerArm":
            violations.append(f"arm_owner is planner_frozen but arm is {arm_cls!r}")
    elif owner is ArmOwner.PLANNER_SWING:
        if session.arm is not None and arm_cls != "HangingWalk":
            violations.append(f"arm_owner is planner_swing but arm is {arm_cls!r}")

    if session.held is not None:
        for side in session.held.sides:
            if not (session.hands.get(side, 0.0) > 0):
                violations.append(f"held {session.held.key!r} on {side!r} but hands[{side!r}] is not closed")
    return violations


def needs_heading(session: Session) -> Violation | None:
    """Precondition helper: a primitive that turns/reaches in place needs an anchored heading first."""
    if session.heading is None:
        return Violation("heading", None, note="heading is not anchored; approach or turn_to must run first")
    return None


def stance_violations(stance: Stance | None, band: dict | None) -> list[Violation]:
    """Pure: the stance against a profile's admission band ({"depth": (lo, hi), "lateral": (lo, hi)}; None = open).

    This is the part of a grasp's precondition that `scripts/replay_preconditions.py` replays on recorded
    stances, so it takes the numbers, not the session.
    """
    out: list[Violation] = []
    if stance is None or not band:
        return out
    for name, value in (("depth", stance.depth_m), ("lateral", stance.lateral_m)):
        lo_hi = band.get(name)
        if lo_hi is None or value is None or not np.isfinite(value):
            continue
        lo, hi = lo_hi
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            out.append(Violation(f"stance_{name}", float(value), (lo, hi), "outside the profile's admission band"))
    return out


def snapshot(session: Session) -> dict:
    """JSON-safe view of `session` for the executor's trace and the agent's turn."""
    held = None
    if session.held is not None:
        hold_offset = session.held.hold_offset
        held = {
            "key": session.held.key,
            "sides": list(session.held.sides),
            "hold_offset": None if hold_offset is None else np.asarray(hold_offset).tolist(),
            "start_z": session.held.start_z,
        }
    stance = None
    if session.stance is not None:
        stance = {
            "target": session.stance.target,
            "profile": session.stance.profile,
            "depth_m": session.stance.depth_m,
            "lateral_m": session.stance.lateral_m,
            "yaw_err_rad": session.stance.yaw_err_rad,
            "quiet": session.stance.quiet,
        }
    return {
        "arm_owner": session.arm_owner.value,
        "arm": type(session.arm).__name__ if session.arm is not None else None,
        "heading": session.heading,
        "hands": dict(session.hands),
        "held": held,
        "stance": stance,
        "stage": session.record.stage,
    }
