"""put_down_bimanual over a fake solver, arm and stack: lowering to rest, release, raise, verdict, refusals."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from locomani.skills import ArmOwner, Executor, HeldObject, SceneModel, Session
from locomani.skills.catalogue import build_catalogue
from locomani.skills.catalogue.crate_put_down import PutDownBimanual, resting_height, tilt_deg
from locomani.skills.profiles import load_profiles

META = {
    "robot_start": {"xy": [-1.8, 0.0], "yaw_deg": 0.0},
    "surfaces": {"A": {"top": 0.30, "xy": [0.0, 0.0], "half_extent": [0.5, 0.5]}},
    "objects": {
        "blue_crate": {
            "box": {"half_extent": [0.20, 0.13, 0.085], "centre": [0.0, 0.0, 0.0]},
            "bimanual": True,
            "container": False,
            "surface": "A",
            "height": 0.17,
            "init_pos": [-0.3, 0.0, 0.385],
        }
    },
}
REST_Z = 0.385


class FakeSolver:
    """tool z rides in joints[0]; `solve` sets it to the commanded z so the stack can read it back."""

    def tool_poses(self, q):
        z = float(q[0])
        return np.array([[0.3, 0.1, z], [0.3, -0.1, z]]), np.array([np.eye(3), np.eye(3)])

    def solve(self, q, positions, rotations, **kw):
        out = np.asarray(q, dtype=float).copy()
        out[0] = float(np.mean(np.asarray(positions)[:, 2]))
        out[1] = float(np.asarray(positions)[0, 0])
        return SimpleNamespace(joints=out, errors=np.zeros(2))


class StreamedArm:  # the class name is what the session invariant and the precondition key on
    def __init__(self, z0):
        self.q = np.zeros(29)
        self.q[0] = z0
        self.hands = {"left": np.full(7, 0.5, dtype=np.float32), "right": np.full(7, 0.5, dtype=np.float32)}
        self._stream_released, self.pushes = False, []

    def _push(self, q, **kw):
        self.pushes.append((float(q[0]), float(self.hands["left"][0])))


class Stack:
    """The object follows the commanded tool z down to the rest height; contact holds while the hands are closed."""

    def __init__(self, arm, drop_fails=False, tilt=0.0):
        self.arm, self.drop_fails, self.tilt, self.t = arm, drop_fails, tilt, 0.0
        self.z_offset = 0.10  # object origin above the reference tool z at grasp
        self.quat = [np.cos(np.radians(tilt) / 2), np.sin(np.radians(tilt) / 2), 0.0, 0.0]

    def obs(self):
        self.t += 0.02
        open_hands = float(self.arm.hands["left"][0]) < 0.05
        z = REST_Z if open_hands else max(REST_Z, float(self.arm.q[0]) + self.z_offset)  # released: stays put
        if self.drop_fails and len(self.arm.pushes) > 5:
            return dict(
                t=self.t, fallen=False, objects={"blue_crate": dict(pos=[0, 0, z + 0.05], quat=self.quat, touching=[])}
            )
        touching = [] if open_hands else ["left", "right"]
        return dict(
            t=self.t, fallen=False, objects={"blue_crate": dict(pos=[0, 0, z], quat=self.quat, touching=touching)}
        )


def session(stack, arm):
    s = Session(stack=stack, scene=SceneModel(META), arm_owner=ArmOwner.POSE_STREAM, arm=arm, heading=0.0)
    s.held = HeldObject("blue_crate", ("left", "right"), {}, start_z=REST_Z)
    s.hands = {"left": 1.0, "right": 1.0}
    return s


def executor(s, **kw):
    prim = PutDownBimanual(
        load_profiles(), solver_factory=lambda cfg, carry: FakeSolver(), hand_open=np.zeros(7, dtype=np.float32), **kw
    )
    return Executor(s, build_catalogue(profiles=load_profiles(), put_down_bimanual=prim))


def test_resting_height_and_tilt():
    assert resting_height(SceneModel(META), "blue_crate", "A") == REST_Z
    assert tilt_deg([1, 0, 0, 0]) == 0.0 and abs(tilt_deg([np.cos(0.2), np.sin(0.2), 0, 0]) - np.degrees(0.4)) < 1e-6


def test_put_down_lowers_opens_raises_and_clears_the_load():
    arm = StreamedArm(z0=REST_Z)  # reference tool z; the object sits 0.10 above it (lifted)
    stack = Stack(arm)
    s = session(stack, arm)
    entry = executor(s).step("put_down_bimanual", {"profile": "warehouse_box", "release_seconds": 0.2, "raise_m": 0.08})
    assert entry["outcome"]["ok"], entry
    m = entry["outcome"]["metrics"]
    assert abs(m["drop_m"] - 0.10) < 1e-9 and abs(m["final_z_error_m"]) < 0.01 and m["touching"] == []
    zs = [z for z, _ in arm.pushes]
    assert min(zs) < REST_Z - 0.05 and zs[-1] > min(zs) + 0.07  # went down, then rose raise_m
    opened = [h for _, h in arm.pushes]
    assert opened[0] == 0.5 and opened[-1] == 0.0
    assert s.held is None and s.hands == {"left": 0.0, "right": 0.0} and s.arm_owner is ArmOwner.POSE_STREAM
    assert [e["stage"] for e in s.record.events] == ["put_down.lower", "put_down.release", "put_down.raise"]


def test_put_down_reports_a_dropped_load_while_lowering():
    arm = StreamedArm(z0=REST_Z)
    stack = Stack(arm, drop_fails=True)
    s = session(stack, arm)
    entry = executor(s).step("put_down_bimanual", {"profile": "warehouse_box"})
    assert not entry["outcome"]["ok"] and entry["outcome"]["reason"] == "[put_down] load dropped while lowering"
    assert s.held is None  # the sim says nothing touches it any more


def test_put_down_refusals():
    arm = StreamedArm(z0=REST_Z)
    s = session(Stack(arm), arm)
    entry = executor(s).step("put_down_bimanual", {"profile": "bottle_simple"})
    assert entry["rejected"][0]["check"] == "profile"
    entry = executor(s).step("put_down_bimanual", {"profile": "warehouse_box", "surface": "B"})
    assert entry["rejected"][0]["check"] == "surface"
    single = session(Stack(arm), arm)
    single.held = HeldObject("blue_crate", ("right",), {}, start_z=REST_Z)
    single.hands = {"left": 0.0, "right": 1.0}
    entry = executor(single).step("put_down_bimanual", {"profile": "warehouse_box"})
    assert entry["rejected"][0]["check"] == "held"
    frozen = session(Stack(arm), None)
    frozen.arm_owner = ArmOwner.PLANNER_FROZEN
    entry = executor(frozen).step("put_down_bimanual", {"profile": "warehouse_box"})
    assert entry["rejected"][0]["check"] == "arm_owner"
