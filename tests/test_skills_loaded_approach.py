"""approach with something held: carry to the target's stand, face the table normal, write the stance."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from locomani.skills import ArmOwner, Executor, HeldObject, SceneModel, Session
from locomani.skills.catalogue import build_catalogue
from locomani.skills.catalogue.bottle import BottleApproach
from locomani.skills.contract import Outcome
from locomani.skills.profiles import load_profiles

META = {
    "robot_start": {"xy": [0.0, -1.2], "yaw_deg": 90.0},
    "surfaces": {"A": {"top": 0.75, "xy": [0.0, 0.0], "half_extent": [0.6, 0.4], "body": "table"}},
    "objects": {
        "soup_can": {
            "box": {"half_extent": [0.03, 0.03, 0.07]},
            "body": "soup_can",
            "bimanual": False,
            "container": False,
            "surface": "A",
            "height": 0.14,
            "init_pos": [0.2, -0.35, 0.82],
        },
        "basket": {
            "box": {"half_extent": [0.15, 0.10, 0.076], "wall": 0.01},
            "body": "basket",
            "bimanual": False,
            "container": True,
            "surface": "A",
            "height": 0.152,
            "init_pos": [-0.2, -0.35, 0.826],
        },
    },
}


class Stack:
    hold_yaw = None

    def __init__(self, yaw=0.0):
        self.pos, self.yaw, self.calls, self.t = np.array([0.4, -1.0]), yaw, [], 0.0

    def obs(self):
        self.t += 0.02
        return {
            "t": self.t,
            "fallen": False,
            "pelvis": {"pos": [*self.pos, 0.79], "yaw": self.yaw},
            "objects": {
                "soup_can": {"pos": [0.2, -0.35, 0.82], "touching": ["right"]},
                "basket": {"pos": [-0.2, -0.35, 0.826], "touching": []},
            },
        }

    def pelvis_xy(self, s=None):
        return self.pos.copy()

    def pelvis_yaw(self, s=None):
        return self.yaw

    def turn_to(self, yaw, tol=0.15, timeout=15.0, mode=None):
        self.calls.append(("turn_to", round(float(yaw), 3), getattr(mode, "name", mode)))
        self.yaw = float(yaw)
        return True

    def world_to_root(self, pos, s=None):
        d = np.asarray(pos[:2]) - self.pos
        c, si = np.cos(self.yaw), np.sin(self.yaw)
        return np.array([c * d[0] + si * d[1], -si * d[0] + c * d[1], pos[2] - 0.79])

    def log(self, text):
        pass


class Carry:
    """A carry_to stand-in that teleports the stack to the stand (plus a drift) and records what it was asked."""

    _palm_offset = staticmethod(lambda state, key, side: np.array([0.18, 0.0, 0.0]))

    def __init__(self, stack, ok=True, drift=(0.0, 0.0)):
        self.stack, self.ok, self.asked, self.drift, self.relatives = stack, ok, None, np.asarray(drift, float), []

    def run(self, session, *, xy=None, relative=None, stop_dist=0.08, timeout=30.0, turn_first=True):
        if relative is not None:  # a correction: move in the current heading frame, exactly
            self.relatives.append((list(relative), turn_first))
            c, sn = np.cos(self.stack.yaw), np.sin(self.stack.yaw)
            self.stack.pos = self.stack.pos + np.array(
                [c * relative[0] - sn * relative[1], sn * relative[0] + c * relative[1]]
            )
            session.record.enter("carry_to", target=self.stack.pos.tolist())
            return Outcome(True, "", metrics={"walked_m": float(np.hypot(*relative))})
        self.asked = np.asarray(xy, dtype=float)
        session.record.enter("carry_to", target=self.asked.tolist())
        if not self.ok:
            return Outcome(False, "[carry] load dropped", metrics={"walked_m": 0.1})
        self.stack.pos = self.asked + self.drift  # the standing turn drifts the pelvis
        session.stance = None
        return Outcome(True, "", metrics={"walked_m": 0.6, "max_drift_m": 0.004})


def stand_for_basket(meta, target, obj_xy, reach, rm):
    psi = np.pi / 2  # the basket hugs the table's near (-y) edge: face +y
    return np.array([0.35, -0.11]), psi, np.asarray(obj_xy) - np.array([0.11, 0.35]), 0.0


def session(stack):
    s = Session(stack=stack, scene=SceneModel(META), arm_owner=ArmOwner.PLANNER_FROZEN, heading=0.0)
    s.scene.reach = SimpleNamespace(plan_core_stand=lambda z: None, graspable=lambda rel, arm: True)
    s.held = HeldObject("soup_can", ("right",), {"right": np.array([0.18, 0.0, 0.0])}, start_z=0.82)
    s.hands["right"] = 1.0
    return s


def executor(s, carry):
    profiles = load_profiles()
    approach = BottleApproach(profiles, carry=carry, stand=stand_for_basket)
    return Executor(s, build_catalogue(profiles=profiles, approach_single=approach))


def test_loaded_approach_carries_faces_and_writes_the_stance():
    stack = Stack(yaw=0.0)
    s = session(stack)
    carry = Carry(stack)
    entry = executor(s, carry).step("approach", {"target": "basket", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"], entry
    assert np.allclose(carry.asked, [-0.31, -0.70])
    assert stack.calls == [("turn_to", round(np.pi / 2, 3), "SLOW_WALK")]
    st = s.stance
    assert st.target == "basket" and st.profile == "bottle_simple" and s.heading == np.pi / 2
    assert abs(st.depth_m - 0.35) < 1e-6 and abs(st.lateral_m + 0.11) < 1e-6  # container 0.35 ahead, 0.11 right
    assert s.arm_owner is ArmOwner.PLANNER_FROZEN and s.held is not None
    assert [e["stage"] for e in s.record.events] == ["approach.carry", "carry_to", "approach.face"]
    assert entry["outcome"]["metrics"]["max_drift_m"] == 0.004 and entry["outcome"]["metrics"]["corrections"] == []


def test_loaded_approach_corrects_the_turn_drift():
    stack = Stack(yaw=0.0)
    s = session(stack)
    carry = Carry(stack, drift=(0.18, 0.0))  # after the turn to +y the container reads 0.18 m to the right
    entry = executor(s, carry).step("approach", {"target": "basket", "profile": "bottle_simple"})
    assert entry["outcome"]["ok"], entry
    assert len(carry.relatives) == 1 and carry.relatives[0][1] is False
    assert np.allclose(
        carry.relatives[0][0], [0.0, 0.18], atol=1e-6
    )  # the container read 0.18 m left of the aim: step left
    assert abs(s.stance.lateral_m + 0.11) < 1e-6 and entry["outcome"]["metrics"]["corrections"][0][1] is True
    assert "approach.correct" in [e["stage"] for e in s.record.events]


def test_loaded_approach_reports_the_carry_failure_as_approach():
    stack = Stack()
    s = session(stack)
    entry = executor(s, Carry(stack, ok=False)).step("approach", {"target": "basket", "profile": "bottle_simple"})
    assert not entry["outcome"]["ok"] and entry["outcome"]["reason"] == "[approach] load dropped"
    assert s.stance is None


def test_loaded_approach_refusals():
    stack = Stack()
    s = session(stack)
    entry = executor(s, Carry(stack)).step("approach", {"target": "soup_can", "profile": "bottle_simple"})
    assert entry["rejected"][0]["check"] == "target"  # cannot approach what is being carried
    s.arm_owner = ArmOwner.PLANNER_SWING
    entry = executor(s, Carry(stack)).step("approach", {"target": "basket", "profile": "bottle_simple"})
    assert entry["rejected"][0]["check"] == "held"
    s.arm_owner, s.heading = ArmOwner.PLANNER_FROZEN, None
    entry = executor(s, Carry(stack)).step("approach", {"target": "basket", "profile": "bottle_simple"})
    assert [v["check"] for v in entry["rejected"]] == ["heading"]


def test_approach_refuses_a_profile_whose_strategy_is_not_wrapped():
    from types import SimpleNamespace as NS

    from locomani.skills.catalogue import build_catalogue

    profiles = load_profiles()
    coupled = NS(hands="single", config=NS(reach=NS(strategy="coupled")))
    stack = Stack()
    s = Session(stack=stack, scene=SceneModel(META), arm_owner=ArmOwner.PLANNER_SWING, heading=0.0)
    approach = BottleApproach({**profiles, "coupled": coupled}, carry=Carry(stack), stand=stand_for_basket)
    ex = Executor(s, build_catalogue(profiles={**profiles, "coupled": coupled}, approach_single=approach))
    entry = ex.step("approach", {"target": "soup_can", "profile": "coupled"})
    assert entry["rejected"][0]["check"] == "profile" and "only simple" in entry["rejected"][0]["note"]
