"""ArmOwner/HeldObject/Stance/Session and the two read-side invariant helpers."""

import numpy as np
import pytest

from locomani.skills.contract import Violation
from locomani.skills.session import (
    ArmOwner,
    HeldObject,
    Session,
    Stance,
    invariant_violations,
    needs_heading,
    snapshot,
)


# named to match the literal class-name checks in session.invariant_violations
class HangingWalk:
    hands = {"left": 0.0, "right": 0.0}


class PlannerArm:
    hands = {"left": 0.0, "right": 1.0}


def _session(**kwargs) -> Session:
    defaults = dict(stack=object(), scene=object())
    defaults.update(kwargs)
    return Session(**defaults)


def test_held_object_start_z_defaults_to_none_and_is_settable():
    held = HeldObject(key="soup_can", sides=("right",), grip_in_palm={"right": np.zeros(3)})
    assert held.start_z is None
    held2 = HeldObject(key="soup_can", sides=("right",), grip_in_palm={"right": np.zeros(3)}, start_z=0.36)
    assert held2.start_z == 0.36


def test_session_defaults():
    session = _session()
    assert session.arm_owner is ArmOwner.PLANNER_SWING
    assert session.heading is None
    assert session.held is None
    assert session.stance is None
    assert session.hands == {"left": 0.0, "right": 0.0}
    assert session.trace == []


def test_invariant_violations_empty_for_a_fresh_session():
    session = _session()
    assert invariant_violations(session) == []


def test_invariant_violations_flags_held_object_with_open_hand():
    session = _session(held=HeldObject(key="soup_can", sides=("right",), grip_in_palm={"right": np.zeros(3)}))
    violations = invariant_violations(session)
    assert any("soup_can" in v and "right" in v for v in violations)


def test_invariant_violations_clean_when_hand_and_arm_agree():
    session = _session(
        arm_owner=ArmOwner.PLANNER_FROZEN,
        arm=PlannerArm(),
        hands={"left": 0.0, "right": 1.0},
        held=HeldObject(key="soup_can", sides=("right",), grip_in_palm={"right": np.zeros(3)}),
    )
    assert invariant_violations(session) == []


def test_invariant_violations_flags_arm_class_mismatch():
    session = _session(arm_owner=ArmOwner.PLANNER_SWING, arm=PlannerArm())
    violations = invariant_violations(session)
    assert any("planner_swing" in v for v in violations)


def test_needs_heading_before_and_after_anchoring():
    session = _session()
    violation = needs_heading(session)
    assert isinstance(violation, Violation)
    assert violation.check == "heading"

    session.heading = 0.0
    assert needs_heading(session) is None


def test_snapshot_includes_start_z_and_stance():
    held = HeldObject(
        key="soup_can",
        sides=("right",),
        grip_in_palm={"right": np.zeros(3)},
        hold_offset=np.array([0.1, 0.0, 0.0]),
        start_z=0.36,
    )
    stance = Stance(target="soup_can", profile="bottle_simple", depth_m=0.4, lateral_m=0.0, yaw_err_rad=0.0, quiet=True)
    session = _session(held=held, stance=stance, heading=1.57)
    snap = snapshot(session)
    assert snap["held"]["start_z"] == 0.36
    assert snap["held"]["hold_offset"] == [0.1, 0.0, 0.0]
    assert snap["stance"]["target"] == "soup_can"
    assert snap["heading"] == pytest.approx(1.57)
    assert snap["arm_owner"] == "planner_swing"
