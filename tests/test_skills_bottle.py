"""Bottle black-box adapters over fakes: context hand-over, owner transitions, held bookkeeping, rejections."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from locomani.skills import ArmOwner, Executor, HeldObject, SceneModel, Session, Stance
from locomani.skills.catalogue import Approach, build_catalogue
from locomani.skills.catalogue.bottle import BottleApproach, GraspSingle, Place, stand_plan, surfaces_edge_distance
from locomani.skills.catalogue.pending import Unimplemented
from locomani.skills.profiles import load_profiles

META = {
    "object_phrase": {"soup_can": "the soup can"},
    "robot_start": {"xy": [0.0, -1.2], "yaw_deg": 90.0},
    "surfaces": {"A": {"top": 0.75, "xy": [0.0, 0.0], "half_extent": [0.6, 0.4], "body": "table"}},
    "objects": {
        "soup_can": {
            "box": {"half_extent": [0.03, 0.03, 0.07], "centre": [0.0, 0.0, 0.0]},
            "body": "soup_can",
            "bimanual": False,
            "container": False,
            "surface": "A",
            "height": 0.14,
            "init_pos": [0.2, -0.35, 0.82],
        },
        "basket": {
            "box": {"half_extent": [0.15, 0.10, 0.076], "centre": [0.0, 0.0, 0.0], "wall": 0.01},
            "body": "basket",
            "bimanual": False,
            "container": True,
            "surface": "A",
            "height": 0.152,
            "init_pos": [-0.2, -0.35, 0.826],
        },
    },
}


class FakeStack:
    hold_yaw = None

    def __init__(self, yaw=0.0):
        self.yaw, self.calls = yaw, []
        self.cmd = SimpleNamespace(set_hands=lambda **kw: None)

    def obs(self):
        return {
            "fallen": False,
            "seq": 1,
            "t": 1.0,
            "joints": np.zeros(29),
            "pelvis": {"pos": [0.0, -1.0, 0.79], "yaw": self.yaw, "quat": [1, 0, 0, 0], "vel": [0, 0, 0]},
            "palms": {"right": {"pos": [0.1, -0.9, 0.9], "quat": [1, 0, 0, 0]}},
            "objects": {
                "soup_can": {"pos": [0.2, -0.35, 0.82], "quat": [1, 0, 0, 0], "cam": [0.1, 0.1]},
                "basket": {"pos": [-0.2, -0.35, 0.826], "quat": [1, 0, 0, 0], "cam": [0.2, 0.1]},
            },
        }

    def pelvis_xy(self, s=None):
        return np.array([0.0, -1.0])

    def pelvis_yaw(self, s=None):
        return self.yaw

    def object_pos(self, key, s=None):
        return np.array(self.obs()["objects"][key]["pos"])

    def turn_to(self, yaw, **kw):
        self.calls.append(("turn_to", yaw))
        self.yaw = yaw
        return True

    def log(self, text):
        pass


class Policy:
    def __init__(self):
        self.stats, self.closed = {"stand": {"settled": True}}, False

    def close(self):
        self.closed = True


def fake_context(session, profile, target):
    ctx = SimpleNamespace(
        p=session.stack,
        obj_key=target,
        psi=np.pi / 2,
        rel=np.array([0.24, -0.05]),
        surf_a=META["surfaces"]["A"],
        squat_hint=None,
        arm_pin=False,
        reach_config=profile.config.reach,
        coordinated=None,
        task_grasp=SimpleNamespace(grip=lambda s: np.array([0.2, -0.35, 0.80]), replan=None, transport_guard=None),
    )
    return ctx


def navigate(ctx):
    ctx.coordinated = Policy()


def approach_ok(ctx):
    return None


def session(**kw):
    kw.setdefault("stack", FakeStack(yaw=np.pi / 2))
    return Session(scene=SceneModel(META), **kw)


@pytest.fixture(scope="module")
def profiles():
    return load_profiles()


def bottle_approach(profiles, phases=(navigate, approach_ok)):
    return BottleApproach(profiles, phases=phases, prepare_ready=lambda p, arms: True, context=fake_context)


def test_stand_plan_is_the_round_samplers_geometry_without_jitter(profiles):
    edge = surfaces_edge_distance(META)
    assert edge([0.0, -1.0]) == pytest.approx(0.6) and edge([0.0, 0.0]) < 0
    rm = SimpleNamespace(plan_core_stand=lambda z: None, graspable=lambda rel, arm: True)
    rel, psi, stand, _ = stand_plan(META, "soup_can", [0.2, -0.35], profiles["bottle_simple"].config.reach, rm)
    assert psi == pytest.approx(np.pi / 2)  # the can hugs the -y edge: the robot stands at -y and faces +y
    c, s = np.cos(psi), np.sin(psi)
    back = np.array([0.2, -0.35]) - stand
    assert np.allclose([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]], back)


def test_approach_dispatches_on_profile_and_rejects_bimanual_until_wrapped(profiles):
    s = session()
    stub = Unimplemented("approach_bimanual", BottleApproach.requires, ArmOwner.PLANNER_SWING, "wraps_pending", "")
    cat = {"approach": Approach(profiles, bottle_approach(profiles), stub)}
    entry = Executor(s, cat).step("approach", {"target": "soup_can", "profile": "warehouse_box"})
    assert entry["rejected"][0]["check"] == "unimplemented"  # the dispatcher picked the bimanual half
    entry = Executor(s, cat).step("approach", {"target": "basket", "profile": "bottle_simple"})
    assert entry["rejected"][0]["check"] == "target"


def test_approach_faces_the_table_runs_the_phases_and_hands_the_policy_over(profiles):
    s = session(stack=FakeStack(yaw=0.0), arm_owner=ArmOwner.PLANNER_FROZEN)
    entry = Executor(s, {"approach": bottle_approach(profiles)}).step(
        "approach", {"target": "soup_can", "profile": "bottle_simple"}
    )
    assert entry["outcome"]["ok"], entry
    assert s.stack.calls == [("turn_to", np.pi / 2)] and s.heading == pytest.approx(np.pi / 2)
    assert (
        s.arm_owner is ArmOwner.PLANNER_SWING and s.stance.target == "soup_can" and s.stance.profile == "bottle_simple"
    )
    assert isinstance(s.scratch["bottle"].coordinated, Policy) and entry["outcome"]["produced"]["quiet"] is True


def test_approach_failure_closes_the_policy_and_reports_the_phase_tag(profiles):
    s = session()
    prim = bottle_approach(profiles, phases=(navigate, lambda ctx: (False, "[stand] fell during the approach")))
    entry = Executor(s, {"approach": prim}).step("approach", {"target": "soup_can", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"] is False and entry["outcome"]["tag"] == "stand"
    assert "bottle" not in s.scratch and s.stance is None


def test_grasp_single_needs_the_approach_hand_over_and_ends_frozen_holding(profiles):
    s = session(arm_owner=ArmOwner.PLANNER_SWING, heading=np.pi / 2)
    seen = {}

    def execute(p, target, grip, plan, **kw):
        seen.update(target=target, coordinated=kw["coordinated"], side=kw["side"])
        return SimpleNamespace(ok=True, why="", visibility=0.9)

    prim = GraspSingle(profiles, execute_grasp=execute, palm_offset=lambda s, k, side: np.array([0.0, 0.0, 0.1]))
    ex = Executor(s, {"grasp_single": prim})
    entry = ex.step("grasp_single", {"target": "soup_can", "profile": "bottle_simple"})
    assert entry["rejected"][0]["check"] == "stance"
    s.stance = Stance("soup_can", "bottle_simple", 0.24, -0.05, 0.0, True)
    ctx = fake_context(s, profiles["bottle_simple"], "soup_can")
    ctx.coordinated = Policy()
    s.scratch["bottle"] = ctx
    entry = ex.step("grasp_single", {"target": "soup_can", "profile": "warehouse_box"})
    assert entry["rejected"][0]["check"] == "profile"
    entry = ex.step("grasp_single", {"target": "soup_can", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"] and seen["coordinated"] is ctx.coordinated and seen["side"] == "right"
    assert s.arm_owner is ArmOwner.PLANNER_FROZEN and s.held.key == "soup_can" and s.held.start_z == 0.82
    assert s.hands["right"] == 1.0 and ctx.coordinated.closed and "bottle" not in s.scratch


def test_grasp_single_failure_keeps_nothing_and_tags_grasp(profiles):
    s = session(arm_owner=ArmOwner.PLANNER_SWING, heading=np.pi / 2)
    s.stance = Stance("soup_can", "bottle_simple", 0.24, -0.05, 0.0, True)
    ctx = fake_context(s, profiles["bottle_simple"], "soup_can")
    ctx.coordinated = Policy()
    s.scratch["bottle"] = ctx
    prim = GraspSingle(
        profiles,
        execute_grasp=lambda *a, **k: SimpleNamespace(ok=False, why="no bilateral contact", visibility=None),
        palm_offset=lambda *a: np.zeros(3),
    )
    entry = Executor(s, {"grasp_single": prim}).step("grasp_single", {"target": "soup_can", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"] is False and entry["outcome"]["tag"] == "grasp"
    assert s.held is None and s.arm_owner is ArmOwner.PLANNER_FROZEN and ctx.coordinated.closed


class PlannerArm:
    def __init__(self):
        self.calls = []
        self.hands = {"left": np.zeros(7), "right": np.ones(7)}

    def start(self, **kw):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")


def test_place_builds_a_planner_arm_and_clears_the_load(profiles):
    held = HeldObject("soup_can", ("right",), {"right": np.zeros(3)}, start_z=0.82)
    s = session(arm_owner=ArmOwner.PLANNER_FROZEN, heading=np.pi / 2, held=held)
    s.hands["right"] = 1.0
    arm = PlannerArm()
    seen = {}

    def place(ctx):
        seen.update(goal=ctx.goal_key, arm=ctx.coordinated.arm, cfg=ctx.place_config, surf=ctx.surf_a["top"])
        ctx.execution.metrics["place"] = {"xy_error": 0.01}
        return True, ""

    prim = Place(profiles, place=place, arm_factory=lambda p, side, cfg: arm)
    ex = Executor(s, {"place": prim})
    assert (
        ex.step("place", {"container": "soup_can", "profile": "bottle_simple"})["rejected"][0]["check"] == "container"
    )
    entry = ex.step("place", {"container": "basket", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"] and entry["outcome"]["metrics"] == {"xy_error": 0.01}
    assert seen["goal"] == "basket" and seen["arm"] is arm and seen["surf"] == 0.75
    assert seen["cfg"] is profiles["bottle_simple"].config.place
    assert arm.calls == ["start", "stop"] and s.held is None and s.hands["right"] == 0.0
    assert s.arm is None and s.arm_owner is ArmOwner.PLANNER_FROZEN and s.stance is None


def test_place_failure_still_releases_bookkeeping(profiles):
    held = HeldObject("soup_can", ("right",), {"right": np.zeros(3)}, start_z=0.82)
    s = session(arm_owner=ArmOwner.PLANNER_FROZEN, heading=np.pi / 2, held=held)
    s.hands["right"] = 1.0
    prim = Place(
        profiles, place=lambda ctx: (False, "[place] missed: outside"), arm_factory=lambda p, side, cfg: PlannerArm()
    )
    entry = Executor(s, {"place": prim}).step("place", {"container": "basket", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"] is False and entry["outcome"]["tag"] == "place" and s.held is None


def test_build_catalogue_wires_the_bottle_adapters(profiles):
    cat = build_catalogue(profiles)
    assert isinstance(cat["grasp_single"], GraspSingle) and isinstance(cat["place"], Place)
    assert isinstance(cat["approach"], Approach) and "approach_bimanual" not in cat
