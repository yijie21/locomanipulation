"""Crate black-box adapters over the same fakes the offline expert test uses (monkeypatched on `expert`)."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from geometry_fixture import scene_ik

from locomani.skills import ArmOwner, Executor, SceneModel, Session
from locomani.skills.catalogue import build_catalogue
from locomani.skills.profiles import load_profiles
from locomani.tasks.crate import expert
from locomani.tasks.crate.perception import CrateEstimate
from locomani.tasks.crate.visual_approach import ApproachCommand
from locomani.tasks.crate.visual_low_reach import LowReachCommand

META = {
    "robot_start": {"xy": [-1.8, 0.0], "yaw_deg": 0.0},
    "surfaces": {"A": {"top": 0.30, "xy": [0.0, 0.0], "half_extent": [0.5, 0.5], "flat_half_extent": [0.5, 0.5]}},
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


class Stack:
    hold_yaw = None

    def __init__(self):
        self.seq = 0
        self.keys = []
        self.cmd = SimpleNamespace(
            reference_observer=None,
            key=self.keys.append,
            hold=lambda *a, **kw: None,
            frame=lambda **kw: None,
            set_hands=lambda **kw: None,
            set_upper_body_vel=lambda v: None,
            stream_mode=None,
            upper_body=None,
        )

    def obs(self):
        self.seq += 1
        return dict(
            t=self.seq * 0.02,
            seq=self.seq,
            joints=np.zeros(29),
            fallen=False,
            pelvis=dict(pos=[-1.8, 0, 0.78], quat=[1, 0, 0, 0], vel=[0, 0, 0]),
            palms={side: dict(hand_q=[-0.015] * 7) for side in ("left", "right")},
            objects={"blue_crate": dict(pos=[-0.3, 0, 0.385], quat=[1, 0, 0, 0], touching=["left", "right"])},
        )

    def pelvis_yaw(self, raw=None):
        return 0.0

    def _facing(self, s=None):
        return (1.0, 0.0, 0.0)


class PlannerArm:  # class name matters: the session invariant keys on it
    def __init__(self, p=None, *a, **k):
        self.p = p
        self.q, self.root_quat, self.calls = np.zeros(29), np.array([1, 0, 0, 0]), []
        self.hands = {"left": np.zeros(7), "right": np.zeros(7)}

    def _push(self, q, **kw):
        self.calls.append("push")
        observer = getattr(getattr(self.p, "cmd", None), "reference_observer", None)
        if observer is not None:  # what the real arms do: every reference send is journaled
            observer(
                "test_reference",
                dict(joints=np.asarray(q), **{k: v for k, v in kw.items() if k == "reference_velocity"}),
            )

    def stop(self):
        self.calls.append("stop")


class StreamedArm(PlannerArm):
    _stream_released = False

    def start(self, **kw):
        self.calls.append(("start", kw))

    def stop(self, **kw):
        self.calls.append("stop")
        self._stream_released = True


class Body:
    def __init__(self, *a, accepted=True, **k):
        self.count, self.velocity, self.diagnostics, self.retries = 0, np.zeros(29), {}, {"descend": 0}

    accepted = True

    def step(self, wall, estimate, raw, openings, **kw):
        self.count += 1
        phase = "hold" if self.count >= 3 else "reach"
        finished = self.count >= 5
        reason = ("completed" if Body.accepted else "unstable_hold") if finished else ""
        return LowReachCommand(np.full(29, 0.1), self.velocity, np.array([1, 0, 0, 0]), 1.0, phase, reason, finished)


class Noop:
    def __init__(self, *a, **k):
        pass

    def start(self, label):
        pass

    def stop(self):
        pass


@pytest.fixture
def fakes(monkeypatch):
    images = {n: np.ones((16, 16, 3), dtype=np.uint8) for n in ("ego_view", "left_wrist", "right_wrist")}
    # A live camera stamps every frame; the guard walk's quiet admission alone outlasts vision_timeout (0.7 s).
    camera = SimpleNamespace(read=lambda: dict(images=images, timestamps={n: time.time() for n in images}))
    guidance = SimpleNamespace(
        diagnostics={},
        update=lambda images, raw, stamp: CrateEstimate(np.array([0.4, 0, -0.4]), np.eye(3), stamp, None, {}),
    )
    monkeypatch.setattr(expert, "ArmIK", lambda path: scene_ik())
    monkeypatch.setattr(expert, "StreamedArm", StreamedArm)
    monkeypatch.setattr(expert, "CoordinatedReach", Body)
    monkeypatch.setattr(expert, "AssistedGuidance", lambda ik, **_kw: guidance)
    monkeypatch.setattr(expert, "GhostTargetTrace", Noop)
    monkeypatch.setattr(
        expert, "ApproachPreparation", lambda *a: SimpleNamespace(step=lambda *args: (np.zeros(29), np.zeros(29)))
    )
    monkeypatch.setattr(
        expert,
        "VisualApproach",
        lambda cfg: SimpleNamespace(
            relative_speed=0,
            angular_speed=0,
            stand_attempts=[],
            step=lambda *a, **kw: ApproachCommand("settle", np.zeros(3), ready=True),
        ),
    )
    monkeypatch.setattr(
        expert, "prepare_stationary_ready", lambda *a, **kw: (PlannerArm(), {"initial_stance_ready": True})
    )
    monkeypatch.setattr(expert, "evaluate", lambda *a, **kw: dict(ok=a[2] == "completed", why=a[2]))
    monkeypatch.setattr(
        expert,
        "FootWatch",
        lambda path: SimpleNamespace(
            update=lambda *a: (True, -1.5, 0.14), last_strides=[0.14], frontmost_low=lambda *a, **k: -1.4
        ),
    )
    monkeypatch.setattr(expert, "GuardBrake", lambda **kw: SimpleNamespace(should_brake=lambda *a: (True, 0.14)))
    Body.accepted = True
    return camera


def session(tmp_path, camera, owner=ArmOwner.PLANNER_SWING):
    return Session(stack=Stack(), scene=SceneModel(META), camera=camera, directory=tmp_path, arm_owner=owner)


@pytest.fixture(scope="module")
def profiles():
    return load_profiles()


def test_depth_approach_then_grasp_holds_the_crate_on_the_stream(tmp_path, fakes, profiles):
    s = session(tmp_path, fakes)
    ex = Executor(s, build_catalogue(profiles))
    entry = ex.step("approach", {"target": "blue_crate", "profile": "crate_low_table"})
    assert entry["outcome"]["ok"], entry
    assert s.arm_owner is ArmOwner.PLANNER_FROZEN and type(s.arm).__name__ == "PlannerArm"
    assert s.stance.target == "blue_crate" and s.stance.depth_m == pytest.approx(0.4) and s.heading == 0.0
    assert "crate" in s.scratch and s.stack.cmd.reference_observer is not None
    entry = ex.step("grasp_bimanual", {"target": "blue_crate", "profile": "crate_low_table"})
    assert entry["outcome"]["ok"], entry
    assert s.arm_owner is ArmOwner.POSE_STREAM and type(s.arm).__name__ == "StreamedArm"
    assert s.held.key == "blue_crate" and s.held.sides == ("left", "right") and s.held.start_z == pytest.approx(0.385)
    assert s.hands == {"left": 1.0, "right": 1.0} and "crate" not in s.scratch
    assert s.stack.cmd.reference_observer is None and entry["outcome"]["metrics"]["reason"] == "completed"
    rows = [json.loads(line) for line in (tmp_path / "blue_crate_000/commands.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows if r["phase"] in ("reach", "hold")] == ["reach", "reach", "hold", "hold", "hold"]
    stages = [e["stage"] for e in s.record.events]
    assert "approach.settle" in stages and "grasp_bimanual.hold" in stages


def test_guard_approach_hands_the_arms_to_the_planner_and_grasp_needs_ready(tmp_path, fakes, profiles):
    s = session(tmp_path, fakes, owner=ArmOwner.PLANNER_FROZEN)
    ex = Executor(s, build_catalogue(profiles))
    entry = ex.step("approach", {"target": "blue_crate", "profile": "warehouse_box"})
    assert entry["outcome"]["ok"], entry
    assert s.arm_owner is ArmOwner.PLANNER_SWING and s.arm is None and s.stack.hold_yaw == 0.0
    assert entry["outcome"]["metrics"]["brake"]["stride"] == 0.14
    entry = ex.step("grasp_bimanual", {"target": "blue_crate", "profile": "warehouse_box"})
    assert entry["rejected"][0]["check"] == "arm_owner"
    s.arm, s.arm_owner = PlannerArm(), ArmOwner.PLANNER_FROZEN  # what `ready` leaves behind
    entry = ex.step("grasp_bimanual", {"target": "blue_crate", "profile": "crate_low_table"})
    assert entry["rejected"][0]["check"] == "profile"
    entry = ex.step("grasp_bimanual", {"target": "blue_crate", "profile": "warehouse_box"})
    assert entry["outcome"]["ok"] and s.held is not None


def test_failed_reach_hands_the_stream_back(tmp_path, fakes, profiles):
    Body.accepted = False
    s = session(tmp_path, fakes)
    ex = Executor(s, build_catalogue(profiles))
    assert ex.step("approach", {"target": "blue_crate", "profile": "crate_low_table"})["outcome"]["ok"]
    entry = ex.step("grasp_bimanual", {"target": "blue_crate", "profile": "crate_low_table"})
    assert entry["outcome"]["ok"] is False and entry["outcome"]["tag"] == "reach"
    assert s.arm_owner is ArmOwner.PLANNER_FROZEN and s.arm is None and s.held is None


def test_approach_rejections(tmp_path, fakes, profiles):
    ex = Executor(session(tmp_path, None), build_catalogue(profiles))
    entry = ex.step("approach", {"target": "blue_crate", "profile": "warehouse_box"})
    assert [v["check"] for v in entry["rejected"]] == ["camera"]
    ex = Executor(session(tmp_path, fakes), build_catalogue(profiles))
    entry = ex.step("approach", {"target": "blue_crate", "profile": "bottle_simple"})
    assert entry["rejected"][0]["check"] == "target"  # dispatched to the single-arm half: not single-hand graspable
