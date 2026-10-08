"""The 2026-09-29 recovery-path fixes (TODO 18.3.1 / 8.3.2): turn_to edge clearance, navigate_to edge margin,
nudge keeping the stance, the loaded approach strafing on the same edge, the free approach fixing a lateral offset,
and the runner's until-injected loop."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from locomani.skills import ArmOwner, Executor, HeldObject, SceneModel, Session, Stance
from locomani.skills.catalogue import build_catalogue
from locomani.skills.catalogue.bottle import BottleApproach
from locomani.skills.catalogue.locomotion import (
    EDGE_MARGIN,
    NavigateTo,
    clear_of_surfaces,
    remeasured_stance,
    surface_clearance,
)
from locomani.skills.contract import Outcome
from locomani.skills.planner import SYSTEM
from locomani.skills.profiles import load_profiles

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

META = {
    "robot_start": {"xy": [-1.5, 0.0], "yaw_deg": 0.0},
    "surfaces": {"A": {"top": 0.62, "xy": [0.0, 0.0], "half_extent": [0.5, 0.5], "body": "table"}},
    "objects": {
        "soup_can": {
            "box": {"half_extent": [0.03, 0.03, 0.07]},
            "body": "soup_can",
            "bimanual": False,
            "container": False,
            "surface": "A",
            "height": 0.14,
            "init_pos": [-0.4, -0.28, 0.692],
        },
        "basket": {
            "box": {"half_extent": [0.15, 0.10, 0.076], "wall": 0.01},
            "body": "basket",
            "bimanual": False,
            "container": True,
            "surface": "A",
            "height": 0.152,
            "init_pos": [-0.34, 0.0, 0.698],
        },
    },
}


class Stack:
    hold_yaw = None
    planner = None

    def __init__(self, pos=(-0.9, -0.3), yaw=0.0):
        self.pos, self.yaw, self.calls, self.t = np.array(pos, dtype=float), yaw, [], 0.0
        self.cmd = SimpleNamespace(set_hands=lambda **kw: None, frame=lambda **kw: None)
        self.can = np.array([-0.4, -0.28, 0.692])

    def obs(self):
        self.t += 0.02
        return {
            "t": self.t,
            "fallen": False,
            "pelvis": {"pos": [*self.pos, 0.79], "yaw": self.yaw, "vel": [0.0, 0.0, 0.0]},
            "objects": {
                "soup_can": {"pos": self.can.tolist(), "touching": ["right", "table"], "quat": [1, 0, 0, 0]},
                "basket": {"pos": [-0.34, 0.0, 0.698], "touching": [], "quat": [1, 0, 0, 0]},
            },
        }

    def pelvis_xy(self, s=None):
        return self.pos.copy()

    def pelvis_yaw(self, s=None):
        return self.yaw

    def object_pos(self, key, s=None):
        return np.array(self.obs()["objects"][key]["pos"])

    def world_to_root(self, pos, s=None):
        d = np.asarray(pos[:2]) - self.pos
        c, si = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([c * d[0] + si * d[1], -si * d[0] + c * d[1], pos[2] - 0.79])

    def turn_to(self, yaw, tol=0.15, timeout=15.0, mode=None):
        self.calls.append(("turn_to", round(float(yaw), 3)))
        self.yaw = float(yaw)
        return True

    def walk_to(self, xy, stop_dist=0.05, timeout=30.0, mode=None, face_target=True, face_yaw=None, on_tick=None):
        self.calls.append(("walk_to", tuple(np.round(xy, 3)), face_target, face_yaw))
        self.pos = np.asarray(xy, dtype=float)
        return True

    def log(self, text):
        pass


def scene():
    return SceneModel(META)


# ---- geometry helpers -------------------------------------------------------------------------------------------


def test_surface_clearance_and_clear_of_surfaces():
    surfaces = scene().surfaces
    assert surface_clearance([-0.64, 0.24], surfaces)[0] == pytest.approx(0.14)
    assert surface_clearance([0.0, 0.0], surfaces)[0] == pytest.approx(-0.5)  # inside: negative
    assert surface_clearance([-0.9, 0.9], surfaces)[0] == pytest.approx(np.hypot(0.4, 0.4))
    moved = clear_of_surfaces([-0.64, 0.24], surfaces, 0.30)
    assert moved.tolist() == pytest.approx([-0.80, 0.24])  # pushed out along the -x edge normal only
    assert clear_of_surfaces([-0.95, 0.24], surfaces, 0.30).tolist() == [-0.95, 0.24]  # already clear: untouched
    assert clear_of_surfaces([0.1, 0.45], surfaces, 0.30).tolist() == pytest.approx([0.1, 0.80])  # inside: out via +y


# ---- turn_to ----------------------------------------------------------------------------------------------------


def test_turn_to_is_refused_beside_the_table_and_allowed_at_the_default_margin():
    stack = Stack(pos=(-0.64, 0.24))
    s = Session(stack=stack, scene=scene(), heading=0.0)
    entry = Executor(s, build_catalogue(profiles={})).step("turn_to", {"yaw": 0.0})
    assert entry["rejected"][0]["check"] == "edge_clearance" and entry["rejected"][0]["measured"] == pytest.approx(0.14)
    stack.pos = np.array([-0.80, 0.24])
    entry = Executor(s, build_catalogue(profiles={})).step("turn_to", {"yaw": 0.0})
    assert entry["outcome"]["ok"] and stack.calls == [("turn_to", 0.0)]
    stack.pos = np.array([-0.64, 0.24])  # the check is a parameter, not a wall
    entry = Executor(s, build_catalogue(profiles={})).step("turn_to", {"yaw": 0.0, "edge_clear": 0.1})
    assert entry["outcome"]["ok"]


# ---- navigate_to ------------------------------------------------------------------------------------------------


class Nav:
    def __init__(self):
        self.goal, self.snap = None, 0.0

    def set_goal_point(self, xy, here_xy=None):
        self.goal = np.asarray(xy, dtype=float)
        return True

    def replan(self, here):
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


def test_navigate_to_moves_a_point_goal_off_the_table_edge():
    stack = Stack(pos=(-1.2, 0.0))
    stack.planner = Nav()
    s = Session(stack=stack, scene=scene(), heading=0.0)
    entry = Executor(s, {"navigate_to": NavigateTo(palm_offset=lambda *a: np.zeros(3))}).step(
        "navigate_to", {"xy": [-0.64, 0.24]}
    )
    assert entry["outcome"]["ok"], entry
    m = entry["outcome"]["metrics"]
    assert m["goal"] == pytest.approx([-0.80, 0.24]) and m["goal_moved_m"] == pytest.approx(0.16)
    assert np.linalg.norm(stack.pos - [-0.80, 0.24]) < 0.11 and EDGE_MARGIN == 0.30  # within stop_dist of the goal


# ---- nudge keeps the stance --------------------------------------------------------------------------------------


def test_remeasured_stance_follows_the_pelvis_and_survives_a_nudge():
    stack = Stack(pos=(-0.9, -0.28))
    s = Session(stack=stack, scene=scene(), heading=0.0)
    s.stance = Stance("soup_can", "bottle_simple", 0.50, 0.0, 0.0, True)
    s.scratch["bottle"] = SimpleNamespace(obj_key="soup_can", coordinated=object())
    stack.pos = np.array([-0.82, -0.20])  # a nudge moved the pelvis 8 cm forward, 8 cm left
    st = remeasured_stance(s, stack.obs())
    assert st.depth_m == pytest.approx(0.42) and st.lateral_m == pytest.approx(-0.08) and st.quiet is True
    assert s.scratch["bottle"].coordinated is not None
    s.stance = None
    assert remeasured_stance(s, stack.obs()) is None  # nothing to keep


# ---- loaded approach strafes on the same edge ---------------------------------------------------------------------


class Carry:
    _palm_offset = staticmethod(lambda state, key, side: np.array([0.18, 0.0, 0.0]))

    def __init__(self, stack):
        self.stack, self.asked = stack, []

    def run(self, session, *, xy=None, relative=None, stop_dist=0.08, timeout=30.0, turn_first=True, drift_max=0.03):
        self.asked.append((None if xy is None else tuple(np.round(xy, 3)), relative, turn_first))
        if xy is not None:
            self.stack.pos = np.asarray(xy, dtype=float)
        return Outcome(True, "", metrics={"walked_m": 0.3})


def stand_same_edge(meta, target, obj_xy, reach, rm):
    return np.array([0.35, -0.11]), 0.0, np.asarray(obj_xy) - np.array([0.35, -0.11]), 0.0  # face +x


def stand_other_edge(meta, target, obj_xy, reach, rm):
    return np.array([0.35, -0.11]), np.pi / 2, np.asarray(obj_xy) - np.array([0.11, 0.35]), 0.0  # face +y


def loaded_session(stack):
    s = Session(stack=stack, scene=scene(), arm_owner=ArmOwner.PLANNER_FROZEN, heading=0.0)
    s.scene.reach = SimpleNamespace(plan_core_stand=lambda z: None, graspable=lambda rel, arm: True)
    s.held = HeldObject("soup_can", ("right",), {"right": np.array([0.18, 0.0, 0.0])}, start_z=0.69)
    s.hands["right"] = 1.0
    return s


@pytest.mark.parametrize(
    "stand, expect_turn_first", [(stand_same_edge, False), (stand_other_edge, True)], ids=["same-edge", "other-edge"]
)
def test_loaded_approach_strafes_when_already_facing_the_edge(stand, expect_turn_first):
    stack = Stack(pos=(-0.85, 0.32), yaw=0.3)  # re-grasped at the moved can (body wrapped 0.3 rad), basket to the right
    s = loaded_session(stack)
    carry = Carry(stack)
    profiles = load_profiles()
    approach = BottleApproach(profiles, carry=carry, stand=stand)
    entry = Executor(s, build_catalogue(profiles=profiles, approach_single=approach)).step(
        "approach", {"target": "basket", "profile": "bottle_simple"}
    )
    assert carry.asked[0][2] is expect_turn_first  # the decision is made before the carry
    if not expect_turn_first:
        assert entry["outcome"]["ok"], entry
        assert entry["outcome"]["metrics"]["strafe"] is True
        assert stack.calls == [("turn_to", 0.0)]  # only the small facing correction after the strafe, never before


# ---- free approach walks to the approach line first --------------------------------------------------------------


def fake_context(session, profile, target):
    return SimpleNamespace(
        p=session.stack,
        obj_key=target,
        psi=0.0,
        rel=np.array([0.35, -0.11]),
        spawn_xy=np.array([-0.75, -0.17]),  # the stand for the can at (-0.4, -0.28), facing +x
        surf_a=META["surfaces"]["A"],
        squat_hint=None,
        arm_pin=False,
        reach_config=profile.config.reach,
        coordinated=SimpleNamespace(stats={"stand": {"settled": True}}, close=lambda: None),
        task_grasp=SimpleNamespace(grip=lambda s: np.array([-0.4, -0.28, 0.66]), replan=None, transport_guard=None),
    )


def test_free_approach_prewalks_a_lateral_offset_holding_the_facing():
    profiles = load_profiles()
    prim = BottleApproach(
        profiles, phases=(lambda ctx: None,), prepare_ready=lambda p, arms: True, context=fake_context
    )
    stack = Stack(pos=(-0.85, 0.43), yaw=0.0)  # 0.6 m to the left of the approach line
    s = Session(stack=stack, scene=scene(), heading=0.0)
    entry = Executor(s, {"approach": prim}).step("approach", {"target": "soup_can", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"], entry
    pre = entry["outcome"]["metrics"]["prewalk"]
    assert pre["ok"] and pre["lateral_m"] == pytest.approx(0.60) and pre["lead"] == pytest.approx([-1.35, -0.17])
    assert stack.calls == [("walk_to", (-1.35, -0.17), False, 0.0)]  # face_yaw held at the table normal
    stack = Stack(pos=(-1.4, -0.20), yaw=0.0)  # on the line: no pre-walk
    entry = Executor(Session(stack=stack, scene=scene(), heading=0.0), {"approach": prim}).step(
        "approach", {"target": "soup_can", "profile": "bottle_simple"}
    )
    assert entry["outcome"]["ok"] and "prewalk" not in entry["outcome"]["metrics"] and stack.calls == []


# ---- runner loop + prompt ---------------------------------------------------------------------------------------


def test_more_rounds_until_injected_and_prompt_rule():
    from plan_episode import more_rounds

    assert [more_rounds(i, 0, 3, None) for i in range(4)] == [True, True, True, False]
    assert more_rounds(5, 2, 30, 3) and not more_rounds(5, 3, 30, 3) and not more_rounds(30, 1, 30, 3)
    assert "trust the scene" in SYSTEM


# ---- place refuses a container outside the placing band ----------------------------------------------------------


def test_place_refuses_a_container_outside_the_placing_band():
    from locomani.skills.catalogue.bottle import Place, placing_band_violation

    profiles = load_profiles()
    prof = profiles["bottle_simple"]
    stack = Stack(pos=(-0.75, 0.43), yaw=0.0)  # at the moved can's stand: the basket is 0.41 m ahead, 0.43 m right
    s = Session(stack=stack, scene=scene(), arm_owner=ArmOwner.PLANNER_FROZEN, heading=0.0)
    s.held = HeldObject("soup_can", ("right",), {"right": np.zeros(3)}, start_z=0.69)
    s.hands["right"] = 1.0
    state = stack.obs()
    state["objects"]["basket"]["cam"] = [0.0, 0.0]
    stack.obs = lambda: state
    v = placing_band_violation(s, prof, "basket", state, 0.10)
    assert v is not None and v.check == "placing_band" and v.measured == pytest.approx([0.41, -0.43], abs=0.01)
    entry = Executor(s, {"place": Place(profiles)}).step("place", {"container": "basket", "profile": "bottle_simple"})
    assert [x["check"] for x in entry["rejected"]] == ["placing_band"]
    stack.pos = np.array([-0.69, 0.11])  # inside band_x [0.26, 0.44] x band_y [-0.16, -0.06]
    assert placing_band_violation(s, prof, "basket", state, 0.10) is None
    coupled = profiles["pick_place_coupled"]
    stack.pos = np.array([-0.75, 0.43])
    assert placing_band_violation(s, coupled, "basket", state, 0.10) is None  # coupled places from its grasp stand
