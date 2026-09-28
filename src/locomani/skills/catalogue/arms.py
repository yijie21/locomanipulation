"""Arm-channel primitives: ready the arms, take the POSE stream over, hand it back, open a hand.

These are the transitions between the three `ArmOwner` states. Each wraps the existing entry
(`tasks/crate/expert.prepare_stationary_ready`, `StreamedArm.start/stop`, `Primitives.release`);
constructors take factories so a test substitutes fakes for the live stack objects.
"""

from __future__ import annotations

import numpy as np

from locomani.skills.contract import Outcome, Violation
from locomani.skills.session import ArmOwner, Session, needs_heading

SIDES = ("left", "right")


def _bimanual_solver(cfg, carry):
    from locomani import paths
    from locomani.primitives.arm_ik import ArmIK
    from locomani.tasks.crate.bimanual_ik import BimanualIK

    return BimanualIK(
        ArmIK(paths.SCENE_XML),
        waist_pitch_max=carry.waist_pitch_max,
        waist_dofs=carry.waist_dofs,
        elbow_min_deg=cfg.elbow_min_deg,
    )


class Ready:
    """Ready the arms in place after a brake (bimanual profiles; the single-arm line readies inside its grasp).

    Always the in-place path (`after_brake=True`, `visual.ready_after_brake=quick`): the `full` variant is the
    episode-start sequence (rest pose, pin release, `start_control`), which is a reset, not a primitive.
    """

    name = "ready"
    requires = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})
    yields = ArmOwner.PLANNER_FROZEN

    def __init__(self, profiles, prepare=None, solver_factory=_bimanual_solver):
        if prepare is None:
            from locomani.tasks.crate.expert import prepare_stationary_ready as prepare
        self._profiles, self._prepare, self._solver = profiles, prepare, solver_factory

    def preconditions(self, session: Session, *, profile: str) -> list[Violation]:
        out = []
        v = needs_heading(session)
        if v is not None:
            out.append(v)
        prof = self._profiles.get(profile)
        if prof is None:
            out.append(Violation("profile", profile, note=f"known: {sorted(self._profiles)}"))
        elif prof.hands != "bimanual":
            out.append(Violation("profile", profile, note="single-arm profiles ready inside grasp_single"))
        if session.held is not None:
            out.append(Violation("held", session.held.key, note="ready opens both hands"))
        return out

    def run(self, session: Session, *, profile: str) -> Outcome:
        t0 = session.clock()
        prof = self._profiles[profile]
        cfg, carry = prof.config.visual, prof.config.carry
        session.record.enter("ready", profile=profile, pose=cfg.ready_pose)
        arm, info = self._prepare(
            session.stack,
            None,
            cfg,
            self._solver(cfg, carry),
            placement=lambda *_a, **_k: None,
            whole_body=True,
            heading=float(session.heading),
            after_brake=True,
            on_phase=lambda name, details: session.record.enter(f"ready.{name}", **details),
        )
        session.arm, session.arm_owner = arm, ArmOwner.PLANNER_FROZEN
        session.hands = {side: 0.0 for side in SIDES}
        ok = bool(info.get("initial_stance_ready", True))
        reason = "" if ok else "[ready] quiet admission failed"
        return Outcome(ok, reason, duration_s=session.clock() - t0, metrics={"initial_stance_ready": ok})


class TakeOver:
    """Enter the POSE stream holding the planner's current pose; nothing moves."""

    name = "take_over"
    requires = frozenset({ArmOwner.PLANNER_FROZEN})
    yields = ArmOwner.POSE_STREAM

    def __init__(self, arm_factory=None):
        self._factory = arm_factory or _streamed_arm

    def preconditions(self, session: Session, *, side: str = "right") -> list[Violation]:
        return [] if side in SIDES else [Violation("side", side, note="left or right")]

    def run(self, session: Session, *, side: str = "right") -> Outcome:
        t0 = session.clock()
        session.record.enter("take_over", side=side)
        arm = self._factory(session.stack, side)
        arm.start(hold_planner_pose=True)
        session.arm, session.arm_owner = arm, ArmOwner.POSE_STREAM
        return Outcome(True, "", duration_s=session.clock() - t0, metrics=dict(getattr(arm, "handoff_diagnostics", {})))


def _streamed_arm(stack, side):
    from locomani import paths
    from locomani.backends.sonic.streamed_arm import StreamedArm
    from locomani.primitives.arm_ik import ArmIK

    return StreamedArm(stack, side, ik=ArmIK(paths.SCENE_XML), left_guard=False)


class HandBack:
    """Leave the POSE stream, pinning the upper body at the streamed reference and keeping the hands."""

    name = "hand_back"
    requires = frozenset({ArmOwner.POSE_STREAM})
    yields = ArmOwner.PLANNER_FROZEN

    def preconditions(self, session: Session) -> list[Violation]:
        return []

    def run(self, session: Session) -> Outcome:
        t0 = session.clock()
        session.record.enter("hand_back")
        arm = session.arm
        session.stack.cmd.set_hands(left=arm.hands["left"], right=arm.hands["right"])  # stop() re-sends one side
        arm.stop(preserve_reference=True)
        session.arm, session.arm_owner = None, ArmOwner.PLANNER_FROZEN
        return Outcome(True, "", duration_s=session.clock() - t0)


class Release:
    """Open one or both hands on whichever channel owns them; clears `held`."""

    name = "release"
    requires = frozenset({ArmOwner.POSE_STREAM, ArmOwner.PLANNER_FROZEN})
    yields = None

    def __init__(self, hand_open=None):
        if hand_open is None:
            from locomani.backends.sonic.agent_server import HAND_OPEN as hand_open
        self._open = np.asarray(hand_open, dtype=np.float32)

    def preconditions(self, session: Session, *, sides=("right",), seconds: float = 1.0) -> list[Violation]:
        bad = [s for s in sides if s not in SIDES]
        return [Violation("sides", str(bad), note="left or right")] if bad else []

    def run(self, session: Session, *, sides=("right",), seconds: float = 1.0) -> Outcome:
        t0 = session.clock()
        session.record.enter("release", sides=list(sides))
        if session.arm_owner is ArmOwner.POSE_STREAM:
            for side in sides:
                session.arm.hands[side] = self._open.copy()
            session.arm.hold(seconds)
        else:
            for side in sides:
                session.stack.release(side, hold=seconds)
        for side in sides:
            session.hands[side] = 0.0
        session.held = None
        return Outcome(True, "", duration_s=session.clock() - t0)
