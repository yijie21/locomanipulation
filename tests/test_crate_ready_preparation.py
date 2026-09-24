"""Regression checks for the shared visual-expert ready preparation."""

from types import SimpleNamespace

import numpy as np

from locomani.backends.sonic.agent_server import HAND_OPEN
from locomani.tasks.crate import box_rim_execution, expert
from locomani.tasks.crate import ghost_reach_execution_cli as ghost_cli


def test_whole_body_preparation_keeps_original_sequence_and_returns_started_arm(monkeypatch):
    events = []
    phases = []
    state = {
        "seq": 0,
        "t": 0.0,
        "joints": np.zeros(29),
        "pelvis": {
            "pos": [0.0, 0.0, 0.76],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
            "yaw": 0.0,
        },
    }

    class Command:
        def reset_pose(self):
            events.append("reset_pose")

        def set_hands(self, **_kwargs):
            events.append("open_hands")

        def exit_streamed(self):
            events.append("exit_streamed")

        def hold(self, seconds, **_kwargs):
            events.append(("idle", seconds))

        def start_control(self):
            events.append("start_control")

    class Primitives:
        cmd = Command()
        hold_yaw = None

        def obs(self):
            state["t"] += 0.1
            state["seq"] += 5
            return state

        def _facing(self, _state):
            return (1.0, 0.0, 0.0)

    class Reset:
        def pin_arms(self):
            events.append("release_pins")

    class PlannerArm:
        def __init__(self, *_args, **_kwargs):
            self.q = np.full(29, 0.25)
            self.starts = []
            self.pushes = 0
            events.append("construct_arm")

        def start(self, **kwargs):
            self.starts.append(kwargs)
            events.append(("arm_start", kwargs.copy()))

        def _push(self, _q):
            self.pushes += 1
            events.append("push")

        def hold(self, seconds):
            events.append(("arm_hold", seconds))

    ready = SimpleNamespace(joints=np.full(29, 9.0))
    solver = SimpleNamespace(solve=lambda *_args, **_kwargs: ready)
    cfg = SimpleNamespace(
        walk_timeout=3.0,
        vision_timeout=0.7,
        upright_tolerance_deg=12.0,
        settle_seconds=0.3,
        ready_pose="forward",  # the original sequence; `lifted` solves its own pose (test_crate_lifted_ready.py)
    )

    monkeypatch.setattr(expert, "PlannerArm", PlannerArm)
    monkeypatch.setattr(expert, "reanchor", lambda _p: events.append("reanchor"))
    monkeypatch.setattr(
        expert.upper_ref,
        "shared",
        lambda: SimpleNamespace(wait=lambda seconds: events.append(("upper_ref", seconds))),
    )
    monkeypatch.setattr(expert.time, "sleep", lambda seconds: events.append(("sleep", seconds)))

    def placement(reset_id):
        events.append(("placement", reset_id))

    arm, report = expert.prepare_stationary_ready(
        Primitives(),
        Reset(),
        cfg,
        solver,
        placement=placement,
        whole_body=True,
        on_phase=lambda name, details: phases.append((name, details)),
    )

    assert arm.starts[0] == {}
    assert arm.starts[1]["settle_seconds"] == 0.0
    assert arm.pushes >= 125
    assert events[: events.index(("arm_hold", 1.2))].count("push") == 125
    assert [(name, details.get("reset_id")) for name, details in phases if name == "reset_commanded"] == [
        ("reset_commanded", "initial_placement"),
        ("reset_commanded", "prepared_placement"),
    ]
    assert events.index(("placement", "initial_placement")) < events.index("construct_arm")
    assert events.index(("arm_hold", 1.2)) < events.index(("placement", "prepared_placement"))
    assert ("idle", 1.5) in events
    assert "reanchor" in events
    assert ("idle", 3.0) in events
    assert report["initial_stance_ready"] is True
    np.testing.assert_array_equal(arm.q[12:15], np.zeros(3))
    np.testing.assert_array_equal(arm.q[15:22], [-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0])


def test_runtime_reuses_prepared_planner_arm_without_start_or_push(monkeypatch):
    prepared = SimpleNamespace(q=np.zeros(29), hands={"left": HAND_OPEN.copy(), "right": HAND_OPEN.copy()})
    monkeypatch.setattr(
        box_rim_execution,
        "PlannerArm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("constructed a second PlannerArm")),
    )

    runtime = box_rim_execution.SonicBoxRimRuntime(
        SimpleNamespace(),
        ik=object(),
        approach_mode="coordinated",
        initial_hands=HAND_OPEN,
        planner_arm=prepared,
    )

    assert runtime.planner_arm is prepared
    assert runtime.pose_arm is None


def test_session_prime_recording_contains_reset_and_full_prime(tmp_path, monkeypatch):
    events = []
    state = {"seq": 10, "t": 0.2}

    class Command:
        def reset_pose(self):
            events.append("reset_pose")

        def start_control(self):
            events.append("start_control")

        def hold(self, seconds, **_kwargs):
            events.append(("idle", seconds))

    class Primitives:
        cmd = Command()

        def obs(self):
            state["seq"] += 1
            state["t"] += 0.02
            return state

    class Reset:
        def start_video(self, label):
            events.append(("video_start", label))

        def pin_arms(self):
            events.append("release_pins")

        def reset(self, *, label):
            events.append(("reset", label))

        def end_video(self):
            events.append("video_end")

    monkeypatch.setattr(ghost_cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(expert.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr("locomani.tasks.crate.round_runner.time.sleep", lambda _seconds: None)

    report = ghost_cli._prime_session(Primitives(), Reset(), tmp_path / "result.json", "probe_session_prime")

    assert report["status"] == "ready"
    assert events[0] == ("video_start", "probe_session_prime")
    assert events[-1] == "video_end"
    assert events.index(("reset", "probe_session_prime_reset")) < events.index("start_control")
    assert events.index("start_control") < events.index(("idle", 2.5))
    assert report["resets"][0]["reset_id"] == "session_prime"
    assert report["resets"][0]["label"] == "probe_session_prime_reset"
