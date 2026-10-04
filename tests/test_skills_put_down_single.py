"""put_down_single and the loaded approach to a surface, over fakes: spot choice, stance and spot hand-over,
refusals, bookkeeping, and the plan / criterion plan_episode builds for an on(X, surface) goal."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from locomani.skills import ArmOwner, Executor, HeldObject, SceneModel, Session
from locomani.skills.catalogue import build_catalogue
from locomani.skills.catalogue.bottle import BottleApproach
from locomani.skills.catalogue.put_down_single import PutDownSingle
from locomani.skills.catalogue.surface_spot import reachable_spot, surface_occupants, surface_spot
from locomani.skills.contract import Outcome
from locomani.skills.profiles import load_profiles

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import plan_episode  # noqa: E402

CAN = {"half_extent": [0.041, 0.0615, 0.098], "centre": [0.0, 0.0, 0.0]}
META = {
    "robot_start": {"xy": [0.0, -1.2], "yaw_deg": 90.0},
    "surfaces": {
        "A": {"top": 0.62, "xy": [0.0, 0.0], "half_extent": [0.6, 0.4], "body": "table"},
        # an ADT-style round table: square box, outline fills pi/4 of it, turned, one open edge (+x = world -y)
        "round": {
            "top": 0.52,
            "xy": [0.0, 1.0],
            "half_extent": [0.25, 0.25],
            "yaw": -np.pi / 2,
            "fill": 0.789,
            "approach_edges": ["+x"],
            "body": "adt_env",
        },
    },
    "objects": {
        "can": {"box": CAN, "body": "can", "surface": "A", "height": 0.196, "init_pos": [0.2, -0.3, 0.718]},
        "cup": {
            "box": {"half_extent": [0.03, 0.03, 0.05], "centre": [0.0, 0.0, 0.0]},
            "body": "cup",
            "surface": "A",
            "height": 0.10,
            "init_pos": [0.0, -0.28, 0.67],
        },
    },
}


class Stack:
    hold_yaw = None

    def __init__(self, pos=(0.4, -1.0), yaw=0.0, touching=("right",), cup=(0.0, -0.28, 0.67)):
        self.pos, self.yaw, self.calls, self.t = np.array(pos, float), yaw, [], 0.0
        self.touching, self.cup = list(touching), list(cup)

    def obs(self):
        self.t += 0.02
        return {
            "t": self.t,
            "fallen": False,
            "pelvis": {"pos": [*self.pos, 0.79], "yaw": self.yaw},
            "objects": {
                "can": {"pos": [0.2, -0.3, 0.80], "touching": list(self.touching)},
                "cup": {"pos": list(self.cup), "touching": []},
            },
        }

    def pelvis_xy(self, s=None):
        return self.pos.copy()

    def pelvis_yaw(self, s=None):
        return self.yaw

    def turn_to(self, yaw, tol=0.15, timeout=15.0, mode=None):
        self.calls.append(("turn_to", round(float(yaw), 3)))
        self.yaw = float(yaw)
        return True

    def world_to_root(self, pos, s=None):
        d = np.asarray(pos[:2], float) - self.pos
        c, si = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([c * d[0] + si * d[1], -si * d[0] + c * d[1], float(pos[2]) - 0.79])

    def log(self, text):
        pass


class Carry:
    """A loaded walk_to stand-in that teleports the stack to the stand."""

    _palm_offset = staticmethod(lambda state, key, side: np.array([0.18, 0.0, 0.0]))

    def __init__(self, stack):
        self.stack, self.asked = stack, None

    def run(self, session, *, xy=None, relative=None, stop_dist=0.08, timeout=30.0, turn_first=True, drift_max=0.03):
        self.asked = np.asarray(xy, dtype=float)
        self.stack.pos = self.asked.copy()
        return Outcome(True, "", metrics={"walked_m": 0.6})


@pytest.fixture(scope="module")
def profiles():
    return load_profiles()


def session(stack, held=True, **kw):
    s = Session(stack=stack, scene=SceneModel(META), arm_owner=ArmOwner.PLANNER_FROZEN, heading=0.0, **kw)
    if held:
        s.held = HeldObject("can", ("right",), {"right": np.array([0.18, 0.0, 0.0])}, start_z=0.718)
        s.hands["right"] = 1.0
    return s


def occupants(stack):
    return surface_occupants(META, "A", stack.obs())


def test_spot_on_a_round_table_is_on_its_open_edge_inside_the_disc():
    spot = surface_spot(META, "round", [0.4, 0.0], "can", {})
    # +x of a table turned -90 deg is world -y: stand south of it, face north; 0.12 in, clipped into the disc
    assert spot["edge"] == [0, 1.0] and abs(spot["psi"] - np.pi / 2) < 1e-9 and spot["inboard"] == 0.12
    local = np.array([1.0 - spot["xy"][1], spot["xy"][0]])  # (x, y) in the table's own frame
    assert abs(local[0] - 0.13) < 1e-9 and np.hypot(*local) <= 0.25 - (0.0615 + 0.03) + 1e-9
    assert spot["xy"][0] > 0  # as near the robot's side as the disc allows


def test_spot_steps_along_the_edge_past_another_object():
    stack = Stack()
    spot = surface_spot(META, "A", [0.0, -1.0], "can", occupants(stack))
    assert np.allclose(spot["xy"], [0.15, -0.28])  # 0, +-0.05, +-0.10 all touch the cup's footprint
    assert surface_spot(META, "A", [0.0, -1.0], "cup", {}) is not None
    small = {**META, "surfaces": {**META["surfaces"], "A": {**META["surfaces"]["A"], "half_extent": [0.6, 0.08]}}}
    assert surface_spot(small, "A", [0.0, -1.0], "can", {}) is None  # shallower than the can's clear footprint


def test_reachable_spot_is_the_free_point_nearest_the_aim():
    # aim far off the round table's centre: the answer is on the shrunk disc (radius 0.25 - 0.0915), toward the aim
    xy = reachable_spot(META, "round", "can", {}, [0.0, 0.3])
    assert np.isclose(np.hypot(xy[0], xy[1] - 1.0), 0.1585, atol=0.02) and xy[1] < 1.0
    # an aim on the table is kept (to the grid step); a cup standing there pushes the spot off by the clear radii
    assert np.allclose(reachable_spot(META, "A", "can", {}, [0.1, -0.2]), [0.1, -0.2], atol=0.011)
    cup = {"cup": (np.array([0.1, -0.2]), 0.03)}
    moved = reachable_spot(META, "A", "can", cup, [0.1, -0.2])
    assert np.linalg.norm(moved - [0.1, -0.2]) >= 0.0915 + 0.03 - 1e-9
    tiny = {**META, "surfaces": {**META["surfaces"], "A": {**META["surfaces"]["A"], "half_extent": [0.6, 0.05]}}}
    assert reachable_spot(tiny, "A", "can", {}, [0.0, 0.0]) is None


def test_occupants_skip_the_far_away_and_take_live_positions():
    stack = Stack(cup=(0.3, -0.2, 0.67))
    occ = surface_occupants(META, "A", stack.obs())
    assert set(occ) == {"can", "cup"} and np.allclose(occ["cup"][0], [0.3, -0.2])
    stack.cup = [3.0, 3.0, 0.67]  # knocked off and away
    assert "cup" not in surface_occupants(META, "A", stack.obs())


def test_loaded_approach_to_a_surface_aims_the_band_at_the_spot(profiles):
    stack = Stack(pos=(0.4, -1.0), yaw=0.0, cup=(2.0, 2.0, 0.0))
    s = session(stack)
    carry = Carry(stack)
    ex = Executor(s, build_catalogue(profiles=profiles, approach_single=BottleApproach(profiles, carry=carry)))
    entry = ex.step("approach", {"target": "A", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"], entry
    band = profiles["bottle_simple"].config.place
    want = np.array([np.mean(band.band_x), np.mean(band.band_y)])
    assert np.allclose(s.scratch["spot"]["xy"], [0.4, -0.28]) and abs(s.scratch["spot"]["psi"] - np.pi / 2) < 1e-9
    assert np.allclose(carry.asked, np.array([0.4, -0.28]) - np.array([-want[1], want[0]]))
    assert s.stance.target == "A" and np.allclose([s.stance.depth_m, s.stance.lateral_m], want)
    assert entry["outcome"]["metrics"]["spot"]["surface"] == "A"
    assert "approach.spot" in [e["stage"] for e in s.record.events]


def test_empty_handed_approach_to_a_surface_is_refused(profiles):
    stack = Stack()
    s = session(stack, held=False)
    s.arm_owner = ArmOwner.PLANNER_SWING
    entry = Executor(s, build_catalogue(profiles=profiles)).step(
        "approach", {"target": "A", "profile": "bottle_simple"}
    )
    assert entry["rejected"][0]["check"] == "target" and "put_down_single" in entry["rejected"][0]["note"]


def put_down_session(stack, spot_xy=(0.4, -0.28), surface="A"):
    s = session(stack)
    s.scratch["spot"] = {"surface": surface, "xy": list(spot_xy), "psi": np.pi / 2, "inboard": 0.12}
    return s


def test_put_down_single_refusals(profiles):
    band = profiles["bottle_simple"].config.place
    want = np.array([np.mean(band.band_x), np.mean(band.band_y)])
    stand = np.array([0.4, -0.28]) - np.array([-want[1], want[0]])
    prim = PutDownSingle(profiles, place=lambda ctx: (True, ""), arm_factory=lambda p, side, cfg: None)

    def checks(s, **args):
        args = {"surface": "A", "profile": "bottle_simple", **args}
        return [v.check for v in prim.preconditions(s, **args)]

    far = Stack(pos=stand, yaw=np.pi / 2, cup=(2.0, 2.0, 0.0))
    assert checks(put_down_session(far)) == []
    s = put_down_session(far)
    s.scratch.pop("spot")
    assert checks(s) == ["spot"]
    assert checks(put_down_session(far, surface="round")) == ["spot"]  # the spot is another surface's
    assert checks(put_down_session(far), surface="nowhere") == ["surface", "spot"]
    assert checks(put_down_session(far), drop_m=0.2) == ["args"]
    s = put_down_session(far)
    s.held = None
    assert checks(s) == ["held"]
    assert checks(put_down_session(Stack(pos=stand + [0.0, -0.4], yaw=np.pi / 2, cup=(2.0, 2.0, 0.0)))) == [
        "placing_band"
    ]
    assert checks(put_down_session(Stack(pos=stand, yaw=np.pi / 2, cup=(0.42, -0.28, 0.67)))) == ["spot_clear"]


class Arm:
    def __init__(self):
        self.calls, self.start_kw = [], None

    def start(self, **kw):
        self.calls.append("start")
        self.start_kw = kw

    def stop(self):
        self.calls.append("stop")


def test_put_down_single_runs_the_place_phase_on_the_surface_and_clears_the_load(profiles):
    stack = Stack(pos=(0.29, -0.63), yaw=np.pi / 2, cup=(2.0, 2.0, 0.0))
    s = put_down_session(stack)
    arm, seen = Arm(), {}

    def place(ctx):
        seen.update(goal=ctx.goal_key, surface=ctx.surface, spot=ctx.spot, drop=ctx.drop_m, arm=ctx.coordinated.arm)
        seen["respot"] = ctx.respot
        ctx.execution.metrics["place"] = {"xy_error": 0.004, "squat_m": 0.12}
        return True, ""

    entry = Executor(
        s, {"put_down_single": PutDownSingle(profiles, place=place, arm_factory=lambda p, d, c: arm)}
    ).step("put_down_single", {"surface": "A", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"], entry
    assert seen["goal"] is None and seen["surface"]["key"] == "A" and seen["surface"]["top"] == 0.62
    assert seen["surface"]["half_extent"] == [0.6, 0.4] and seen["spot"] == [0.4, -0.28] and seen["drop"] == 0.01
    assert arm.calls == ["start", "stop"] and s.held is None and s.hands["right"] == 0.0 and "spot" not in s.scratch
    assert entry["outcome"]["metrics"]["squat_m"] == 0.12 and entry["outcome"]["metrics"]["spot"]["surface"] == "A"
    assert set(entry["outcome"]["metrics"]["poses"]) == {"entry", "after_start"}
    assert stack.hold_yaw == np.pi / 2 and s.heading == np.pi / 2  # the yaw it stands at, not a new facing
    assert arm.start_kw == {} and entry["outcome"]["metrics"]["takeover"] == "measured_joints"  # nothing commanded yet
    assert np.allclose(seen["respot"](stack.obs(), [0.4, -0.28]), [0.4, -0.28], atol=0.011)  # a free point on A


def test_put_down_single_takes_over_at_the_commanded_reference(profiles):
    stack = Stack(pos=(0.29, -0.63), yaw=np.pi / 2, cup=(2.0, 2.0, 0.0))
    stack.cmd = SimpleNamespace(upper_body=np.arange(17, dtype=np.float32) * 0.01)
    s = put_down_session(stack)
    arm = Arm()
    prim = PutDownSingle(profiles, place=lambda ctx: (True, ""), arm_factory=lambda *a: arm)
    entry = Executor(s, {"put_down_single": prim}).step("put_down_single", {"surface": "A", "profile": "bottle_simple"})
    assert entry["outcome"]["metrics"]["takeover"] == "owned_reference" and arm.start_kw["settle_seconds"] == 0.0
    assert np.allclose(arm.start_kw["upper_reference"], np.arange(17) * 0.01)


def test_put_down_single_failure_before_the_release_keeps_the_load(profiles):
    stack = Stack(pos=(0.29, -0.63), yaw=np.pi / 2, cup=(2.0, 2.0, 0.0))
    s = put_down_session(stack)
    prim = PutDownSingle(profiles, place=lambda ctx: (False, "[turn] base did not stop"), arm_factory=lambda *a: Arm())
    entry = Executor(s, {"put_down_single": prim}).step("put_down_single", {"surface": "A", "profile": "bottle_simple"})
    assert entry["outcome"]["tag"] == "turn" and s.held is not None and s.arm_owner is ArmOwner.PLANNER_FROZEN
    stack.touching = []  # a slip that let go: the load is gone
    s = put_down_session(stack)
    entry = Executor(s, {"put_down_single": prim}).step("put_down_single", {"surface": "A", "profile": "bottle_simple"})
    assert s.held is None


def test_plan_episode_plans_and_judges_a_surface_goal():
    scene = SceneModel(META)
    steps = plan_episode.default_plan(scene, {"target": "can", "goal": "round"})
    assert [(st["primitive"], st["args"].get("target") or st["args"].get("surface")) for st in steps] == [
        ("approach", "can"),
        ("grasp_single", "can"),
        ("approach", "round"),
        ("put_down_single", "round"),
    ]
    text, check = plan_episode.criterion_for(scene, {"target": "can", "goal": "round"}, steps)
    assert text.startswith("put ") and " on round" in text
    resting = {"objects": {"can": {"pos": [0.0, 0.85, 0.618], "touching": []}}}
    assert check(resting, SimpleNamespace(held=None)) == (True, "")
    assert not check(resting, SimpleNamespace(held=object()))[0]  # still holding it over the table
    held_over = {"objects": {"can": {"pos": [0.0, 0.85, 0.70], "touching": ["right"]}}}
    assert check(held_over, SimpleNamespace(held=None))[1].startswith("[put_on] still in the hand")
