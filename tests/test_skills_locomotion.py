"""walk_to with a load (delegates to carry_to), navigate_to over a fake distance field, LoadWatch rules."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from locomani.skills import ArmOwner, Executor, HeldObject, Session
from locomani.skills.catalogue import build_catalogue
from locomani.skills.catalogue.locomotion import CarryTo, LoadWatch, NavigateTo, WalkTo


class FakeNav:
    """A straight-line stand-in for `Navigator`: the field is the Euclidean distance to the goal."""

    def __init__(self):
        self.goal, self.snap, self.calls = None, 0.0, []

    def set_goal_point(self, xy, here_xy=None):
        self.goal = np.asarray(xy, dtype=float)
        self.calls.append(("point", tuple(self.goal)))
        return True

    def set_goal_ring(self, centre, r_lo, r_hi, here_xy=None):
        centre = np.asarray(centre, dtype=float)
        d = centre - np.asarray(here_xy, dtype=float)
        self.goal = centre - d / max(np.linalg.norm(d), 1e-6) * 0.5 * (r_lo + r_hi)
        self.calls.append(("ring", tuple(centre), r_lo, r_hi))
        return True

    def replan(self, here):
        self.calls.append(("replan",))
        return True

    def reachable(self, here):
        return True

    def distance_at(self, here):
        return float(np.linalg.norm(self.goal - np.asarray(here, dtype=float)))

    def waypoint(self, here, step):
        here = np.asarray(here, dtype=float)
        d = self.goal - here
        n = float(np.linalg.norm(d))
        return self.goal.copy() if n <= step else here + d / n * step


class WalkStack:
    """walk_to teleports to the target in `ticks` steps, feeding `on_tick` a state each step."""

    hold_yaw = None

    def __init__(self, held_key="water_bottle", slip_at=None, planner=True, ticks=4):
        self.pos, self.yaw, self.t = np.array([0.0, 0.0]), 0.0, 0.0
        self.key, self.slip_at, self.ticks, self.calls = held_key, slip_at, ticks, []
        self.planner = FakeNav() if planner else None
        self.drift, self.aborted = 0.0, None
        self.cmd = SimpleNamespace(frame=lambda **kw: None)

    def state(self):
        return {
            "t": self.t,
            "fallen": False,
            "pelvis": {"pos": [*self.pos, 0.79], "yaw": self.yaw},
            "objects": {self.key: {"pos": [0.5, 0.0, 0.8], "touching": ["right"], "quat": [1, 0, 0, 0]}},
        }

    def obs(self):
        return self.state()

    def pelvis_xy(self, s=None):
        return self.pos.copy()

    def pelvis_yaw(self, s=None):
        return self.yaw

    def turn_to(self, yaw, tol=0.15, timeout=15.0, mode=None):
        self.calls.append(("turn_to", round(float(yaw), 3), getattr(mode, "name", mode)))
        self.yaw = float(yaw)
        return True

    def walk_to(self, xy, stop_dist=0.05, timeout=30.0, mode=None, face_target=True, face_yaw=None, on_tick=None):
        self.calls.append(("walk_to", tuple(np.round(xy, 3)), face_target))
        target = np.asarray(xy, dtype=float)
        start = self.pos.copy()
        for k in range(1, self.ticks + 1):
            self.t += 0.1
            self.pos = start + (target - start) * k / self.ticks
            if self.slip_at is not None and len(self.calls) >= self.slip_at:
                self.drift = 0.05
            if on_tick is not None and on_tick(self.state(), float(np.linalg.norm(target - self.pos))) is False:
                self.aborted = "walking callback rejected the next command"
                return False
        return True

    def log(self, text):
        pass


def palm_offset_of(stack):
    return lambda state, key, side: np.array([0.18 + stack.drift, 0.0, 0.0])


def session(stack, owner=ArmOwner.PLANNER_FROZEN, held=True):
    s = Session(stack=stack, scene=None, arm_owner=owner, heading=0.0)
    if held:
        s.held = HeldObject("water_bottle", ("right",), {"right": np.zeros(3)}, start_z=0.8)
        s.hands["right"] = 1.0
    return s


def catalogue(stack):
    palm = palm_offset_of(stack)
    return build_catalogue(
        profiles={},
        walk_to=WalkTo(carry=CarryTo(palm_offset=palm)),
        carry_to=CarryTo(palm_offset=palm),
        navigate_to=NavigateTo(palm_offset=palm),
    )


def test_walk_to_with_a_load_is_the_carried_walk():
    stack = WalkStack()
    s = session(stack)
    entry = Executor(s, catalogue(stack)).step("walk_to", {"xy": [1.0, 0.0]})
    assert entry["outcome"]["ok"], entry
    assert [e["stage"] for e in s.record.events] == ["carry_to"]
    assert entry["outcome"]["metrics"]["max_drift_m"] == 0.0 and s.stance is None
    assert s.arm_owner is ArmOwner.PLANNER_FROZEN


def test_walk_to_with_a_load_needs_the_frozen_channel():
    stack = WalkStack()
    entry = Executor(session(stack, owner=ArmOwner.PLANNER_SWING), catalogue(stack)).step("walk_to", {"xy": [1.0, 0.0]})
    assert entry["rejected"][0]["check"] == "held"


def test_walk_to_free_handed_is_unchanged():
    stack = WalkStack()
    s = session(stack, owner=ArmOwner.PLANNER_SWING, held=False)
    entry = Executor(s, catalogue(stack)).step("walk_to", {"xy": [1.0, 0.0]})
    assert entry["outcome"]["ok"] and [e["stage"] for e in s.record.events] == ["walk_to"]


def test_navigate_to_free_handed_walks_the_field_in_segments():
    stack = WalkStack()
    s = session(stack, owner=ArmOwner.PLANNER_SWING, held=False)
    entry = Executor(s, catalogue(stack)).step("navigate_to", {"xy": [1.2, 0.0], "stop_dist": 0.1})
    assert entry["outcome"]["ok"], entry
    walks = [c for c in stack.calls if c[0] == "walk_to"]
    assert len(walks) == 3 and all(c[2] is True for c in walks)  # 0.45 + 0.45 + the rest, facing the waypoint
    assert entry["outcome"]["metrics"]["field_left_m"] < 0.1 and s.stance is None
    assert stack.planner.calls[0][0] == "point"


def test_navigate_to_with_a_load_turns_first_then_holds_the_line():
    stack = WalkStack()
    s = session(stack)
    entry = Executor(s, catalogue(stack)).step("navigate_to", {"xy": [0.0, 1.0]})  # 90 deg off the heading
    assert entry["outcome"]["ok"], entry
    turns = [c for c in stack.calls if c[0] == "turn_to"]
    walks = [c for c in stack.calls if c[0] == "walk_to"]
    assert turns and turns[0][1] == round(np.pi / 2, 3) and turns[0][2] == "SLOW_WALK"
    assert walks and all(c[2] is False for c in walks)
    assert entry["outcome"]["metrics"]["max_drift_m"] == 0.0 and s.held is not None


def test_navigate_to_with_a_load_reports_the_slip():
    stack = WalkStack(slip_at=2)
    s = session(stack)
    entry = Executor(s, catalogue(stack)).step("navigate_to", {"xy": [1.5, 0.0]})
    assert not entry["outcome"]["ok"] and entry["outcome"]["reason"].startswith("[carry] load slipped")


def test_navigate_to_target_ring_and_missing_map():
    stack = WalkStack()
    scene = SimpleNamespace(objects={"water_bottle": object()})
    s = session(stack, owner=ArmOwner.PLANNER_SWING, held=False)
    s.scene = scene
    entry = Executor(s, catalogue(stack)).step("navigate_to", {"target": "water_bottle", "stand_off": 0.4})
    assert entry["outcome"]["ok"] and stack.planner.calls[0][0] == "ring"
    bare = WalkStack(planner=False)
    entry = Executor(session(bare, held=False), catalogue(bare)).step("navigate_to", {"xy": [1.0, 0.0]})
    assert entry["rejected"][0]["check"] == "map"
    entry = Executor(session(stack, held=False), catalogue(stack)).step("navigate_to", {})
    assert entry["rejected"][0]["check"] == "target"


def test_load_watch_rules():
    stack = WalkStack()
    held = HeldObject("water_bottle", ("right",), {}, start_z=0.8)
    watch = LoadWatch(stack, held, palm_offset_of(stack), drift_max=0.03, contact_gap_s=0.3)
    assert watch(stack.state(), 1.0) is True
    stack.drift = 0.04
    assert watch(stack.state(), 0.9) is False and "slipped" in watch.why
    both = HeldObject("box", ("left", "right"), {}, start_z=0.385)
    stack.key = "box"
    stack.drift = 0.0
    bim = LoadWatch(stack, both, palm_offset_of(stack), lift_floor_m=0.06)
    state = stack.state()
    state["objects"]["box"] = {"pos": [0.5, 0.0, 0.5], "touching": ["left", "right"]}
    assert bim(state, 1.0) is True
    state["objects"]["box"]["touching"] = ["left"]
    state["t"] = 0.1
    assert bim(state, 0.9) is True  # inside the contact gap
    state["t"] = 0.5
    assert bim(state, 0.8) is False and bim.why == "[carry] load dropped"
    assert bim.metrics()["min_load_z"] == pytest.approx(0.5)
