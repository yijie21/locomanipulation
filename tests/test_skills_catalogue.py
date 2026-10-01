"""Catalogue adapters over fakes: owner transitions, session updates and rejections through the executor."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from locomani.skills import ArmOwner, Executor, HeldObject, Session
from locomani.skills.catalogue import PENDING, RUNNABLE, build_catalogue
from locomani.skills.catalogue.arms import HandBack, Ready, Release, TakeOver
from locomani.skills.catalogue.locomotion import TurnTo, WalkTo
from locomani.skills.profiles import load_profiles


class FakeStack:
    hold_yaw = None

    def __init__(self):
        self.calls = []
        self.cmd = SimpleNamespace(set_hands=lambda **kw: self.calls.append(("set_hands", kw)))
        self.left_hand = np.zeros(7)
        self.right_hand = np.zeros(7)

    def turn_to(self, yaw, tol=0.15, **kw):
        self.calls.append(("turn_to", yaw, tol))
        return True

    def walk_to(self, xy, stop_dist=0.05, timeout=30.0):
        self.calls.append(("walk_to", tuple(xy)))
        return True

    def release(self, side, hold=1.0):
        self.calls.append(("release", side, hold))

    def obs(self):
        return {"fallen": False, "pelvis": {"pos": [0, 0, 0.8], "yaw": 0.0}, "objects": {}}


class StreamedArm:  # the invariant check keys on the class name
    def __init__(self, stack, side):
        self.side, self.calls, self._stream_released = side, [], False
        self.hands = {"left": np.ones(7), "right": np.ones(7)}
        self.handoff_diagnostics = {"q_start_error": 0.0}

    def start(self, **kw):
        self.calls.append(("start", kw))

    def stop(self, *, preserve_reference=False):
        self.calls.append(("stop", preserve_reference))
        self._stream_released = True

    def hold(self, seconds, **kw):
        self.calls.append(("hold", seconds))


class PlannerArm:
    def __init__(self):
        self.hands = {"left": np.zeros(7), "right": np.zeros(7)}


def session(owner=ArmOwner.PLANNER_SWING, heading=0.0, arm=None, held=None):
    s = Session(stack=FakeStack(), scene=None, arm_owner=owner, arm=arm, heading=heading, held=held)
    if held is not None:
        for side in held.sides:
            s.hands[side] = 1.0
    return s


def executor(s, **entries):
    return Executor(s, build_catalogue(profiles={}, **entries))


def test_catalogue_names_cover_the_design_table():
    names = set(build_catalogue(profiles={}))
    assert set(RUNNABLE) <= names and set(PENDING) <= names
    assert names == set(RUNNABLE) | set(PENDING)


def test_turn_to_needs_a_heading_anchor_then_updates_it():
    s = session(heading=None)
    entry = executor(s).step("turn_to", {"yaw": 0.5})
    assert [v["check"] for v in entry["rejected"]] == ["heading"]
    s = session(heading=0.0)
    s.stance = SimpleNamespace(target="x")
    entry = executor(s).step("turn_to", {"yaw": 0.5})
    assert entry["outcome"]["ok"] and s.heading == 0.5 and s.stance is None and s.stack.hold_yaw == 0.5


def test_walk_to_refuses_a_load_and_clears_the_stance():
    held = HeldObject(key="can", sides=("right",), grip_in_palm={}, start_z=0.8)
    entry = executor(session(held=held)).step("walk_to", {"xy": [1.0, 0.0]})
    assert entry["rejected"][0]["check"] == "held"
    s = session()
    entry = executor(s).step("walk_to", {"xy": [1.0, 0.0]})
    assert entry["outcome"]["ok"] and ("walk_to", (1.0, 0.0)) in s.stack.calls


def test_take_over_then_hand_back_round_trip_keeps_the_load():
    held = HeldObject(key="can", sides=("right",), grip_in_palm={}, start_z=0.8)
    s = session(owner=ArmOwner.PLANNER_FROZEN, held=held)
    ex = executor(s, take_over=TakeOver(arm_factory=StreamedArm))
    assert ex.step("take_over", {})["outcome"]["ok"]
    assert s.arm_owner is ArmOwner.POSE_STREAM and s.arm.calls == [("start", {"hold_planner_pose": True})]
    arm = s.arm
    assert ex.step("hand_back", {})["outcome"]["ok"]
    assert s.arm_owner is ArmOwner.PLANNER_FROZEN and s.arm is None and s.held is held
    assert ("stop", True) in arm.calls and s.stack.calls[-1][0] == "set_hands"


def test_release_opens_on_either_channel_and_clears_held():
    held = HeldObject(key="can", sides=("right",), grip_in_palm={}, start_z=0.8)
    s = session(owner=ArmOwner.POSE_STREAM, arm=StreamedArm(None, "right"), held=held)
    ex = executor(s, release=Release(hand_open=np.zeros(7)))
    assert ex.step("release", {"sides": ["right"], "seconds": 0.5})["outcome"]["ok"]
    assert s.held is None and s.hands["right"] == 0.0 and np.all(s.arm.hands["right"] == 0)
    assert ("hold", 0.5) in s.arm.calls
    s = session(owner=ArmOwner.PLANNER_FROZEN, held=held)
    ex = executor(s, release=Release(hand_open=np.zeros(7)))
    assert ex.step("release", {})["outcome"]["ok"] and ("release", "right", 1.0) in s.stack.calls


def test_ready_uses_the_bimanual_profile_and_rejects_the_single_arm_one():
    profiles = load_profiles()
    seen = {}

    def fake_prepare(p, reset, cfg, solver, **kw):
        seen.update(pose=cfg.ready_pose, after_brake=kw["after_brake"], heading=kw["heading"], solver=solver)
        kw["on_phase"]("reanchor_started", {})
        return PlannerArm(), {"initial_stance_ready": True}

    ready = Ready(profiles, prepare=fake_prepare, solver_factory=lambda cfg, carry: ("solver", carry.waist_dofs))
    s = session(heading=0.7)
    entry = executor(s, ready=ready).step("ready", {"profile": "warehouse_box"})
    assert entry["outcome"]["ok"] and s.arm_owner is ArmOwner.PLANNER_FROZEN and s.hands == {"left": 0.0, "right": 0.0}
    assert seen["pose"] == "lifted" and seen["after_brake"] is True and seen["heading"] == 0.7
    assert seen["solver"][1] == profiles["warehouse_box"].grasp["carry"]["waist_dofs"]
    assert any(e["stage"] == "ready.reanchor_started" for e in s.record.events)
    entry = executor(session(heading=0.7), ready=ready).step("ready", {"profile": "bottle_simple"})
    assert entry["rejected"][0]["check"] == "profile"


def test_pending_entries_are_rejected_before_run():
    for name in PENDING:
        s = session(owner=next(iter(PENDING[name][0])))
        entry = executor(s).step(name, {})
        assert entry["rejected"][0]["check"] == "unimplemented", name


class CarryStack(FakeStack):
    """walk_to drives `on_tick` through a scripted state stream; the load slips at the given tick."""

    def __init__(self, slip_at=None, drop_at=None, ticks=6, bimanual=False):
        super().__init__()
        self.slip_at, self.drop_at, self.ticks, self.bimanual = slip_at, drop_at, ticks, bimanual
        self.t, self.pos, self.aborted = 0.0, np.array([0.0, 0.0]), None

    def state(self, k):
        touching = ["left", "right"] if self.bimanual else ["right"]
        if self.drop_at is not None and k >= self.drop_at:
            touching = []
        z = 0.5 if self.bimanual else 0.9
        return {
            "t": self.t,
            "fallen": False,
            "pelvis": {"pos": [*self.pos, 0.8], "yaw": 0.0},
            "objects": {"box": {"pos": [0.3, 0.0, z], "quat": [1, 0, 0, 0], "touching": touching}},
            "_k": k,
        }

    def obs(self):
        return self.state(self.ticks)

    def pelvis_xy(self, s=None):
        return self.pos

    def pelvis_yaw(self, s=None):
        return 0.0

    def turn_to(self, yaw, tol=0.15, mode=None, **kw):
        self.calls.append(("turn_to", round(float(yaw), 3)))
        self.turn_modes = getattr(self, "turn_modes", []) + [getattr(mode, "name", None)]
        return True

    def walk_to(self, xy, stop_dist=0.05, timeout=30.0, face_target=True, face_yaw=None, on_tick=None):
        self.calls.append(
            ("walk_to", tuple(np.round(xy, 3)), face_target, None if face_yaw is None else round(face_yaw, 3))
        )
        for k in range(self.ticks):
            self.t += 0.1
            if on_tick is not None and on_tick(self.state(k), 1.0) is False:
                self.aborted = "walking callback rejected the next command"
                return False
        self.pos = np.asarray(xy, dtype=float)
        return True


def carry_prim(slip_at=None):
    def offset(s, key, side):
        k = s["_k"]
        return np.array([0.0, 0.0, 0.05 if slip_at is not None and k >= slip_at else 0.0])

    return WalkTo(palm_offset=offset)


def test_walk_to_with_a_load_takes_a_relative_target_and_reports_drift():
    held = HeldObject(key="box", sides=("right",), grip_in_palm={}, start_z=0.9)
    s = session(owner=ArmOwner.PLANNER_FROZEN, held=held, heading=0.0)
    s.stack = CarryStack()
    s.hands["right"] = 1.0
    entry = executor(s, walk_to=carry_prim()).step("walk_to", {"relative": [-1.0, 0.0]})
    assert entry["outcome"]["ok"], entry
    # a target behind the robot: turn in place first (180 deg), then walk holding that line
    assert s.stack.calls[-2] == ("turn_to", round(np.pi, 3)) and s.stack.calls[-1] == (
        "walk_to",
        (-1.0, 0.0),
        False,
        round(np.pi, 3),
    )
    assert s.held is held and s.stance is None
    m = entry["outcome"]["metrics"]
    assert (
        m["turned"] is True
        and m["max_drift_m"] == 0.0
        and m["walked_m"] == pytest.approx(1.0)
        and len(m["samples"]) >= 5
    )
    s.stack.calls.clear()
    entry = executor(s, walk_to=carry_prim()).step("walk_to", {"relative": [1.0, 0.0]})
    assert entry["outcome"]["ok"] and s.stack.calls == [("walk_to", (0.0, 0.0), False, 0.0)]  # ahead: no turn


def test_walk_to_with_a_load_aborts_on_palm_slip_and_on_lost_contact():
    held = HeldObject(key="box", sides=("right",), grip_in_palm={}, start_z=0.9)
    s = session(owner=ArmOwner.PLANNER_FROZEN, held=held, heading=0.0)
    s.stack, s.hands["right"] = CarryStack(), 1.0
    entry = executor(s, walk_to=carry_prim(slip_at=3)).step("walk_to", {"xy": [1.0, 0.0]})
    assert entry["outcome"]["ok"] is False and "slipped" in entry["outcome"]["reason"]
    s = session(owner=ArmOwner.PLANNER_FROZEN, held=held, heading=0.0)
    s.stack, s.hands["right"] = CarryStack(drop_at=1, ticks=8), 1.0
    entry = executor(s, walk_to=carry_prim()).step("walk_to", {"xy": [1.0, 0.0], "contact_gap_s": 0.25})
    assert entry["outcome"]["ok"] is False and "dropped" in entry["outcome"]["reason"]


def test_walk_to_with_a_bimanual_load_uses_the_height_floor_and_both_hands():
    held = HeldObject(key="box", sides=("left", "right"), grip_in_palm={}, start_z=0.40)
    s = session(owner=ArmOwner.PLANNER_FROZEN, held=held, heading=0.0)
    s.stack = CarryStack(bimanual=True)
    s.hands = {"left": 1.0, "right": 1.0}
    entry = executor(s, walk_to=carry_prim()).step("walk_to", {"xy": [1.0, 0.0]})
    assert entry["outcome"]["ok"] and entry["outcome"]["metrics"]["min_load_z"] == 0.5
    entry = executor(s, walk_to=carry_prim()).step("walk_to", {"xy": [1.0, 0.0], "lift_floor_m": 0.2})
    assert entry["outcome"]["ok"] is False and "dropped" in entry["outcome"]["reason"]


def test_walk_to_free_handed_takes_a_relative_target_and_refuses_the_load_arguments():
    s = session(owner=ArmOwner.PLANNER_SWING, heading=0.0)
    s.stack = CarryStack()
    entry = executor(s, walk_to=carry_prim()).step("walk_to", {"relative": [-1.0, 0.0]})
    assert entry["outcome"]["ok"] and s.stack.calls == [("walk_to", (-1.0, 0.0), True, None)]  # faces as it walks
    entry = executor(s, walk_to=carry_prim()).step("walk_to", {"xy": [1.0, 0.0], "turn_first": False})
    assert entry["rejected"][0]["check"] == "load_args" and entry["rejected"][0]["measured"] == "turn_first"


def test_walk_to_rejections():
    held = HeldObject(key="box", sides=("right",), grip_in_palm={}, start_z=0.9)
    s = session(owner=ArmOwner.PLANNER_FROZEN, held=held, heading=0.0)
    s.hands["right"] = 1.0
    entry = executor(s, walk_to=carry_prim()).step("walk_to", {"xy": [1.0, 0.0], "relative": [1.0, 0.0]})
    assert entry["rejected"][0]["check"] == "target"
