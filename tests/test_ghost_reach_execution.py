from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.tasks.crate import ghost_reach_execution as execution
from locomani.tasks.crate import ghost_reach_execution_cli as execution_cli
from locomani.tasks.crate.coordinated_reach import CoordinatedTarget
from scripts.probe_ghost_reach import stack_environment


def _case() -> execution.GhostReachCase:
    return execution.GhostReachCase(
        index=1,
        label_zh="nominal",
        midpoint_robot_m=(0.4, 0.0, 0.7),
        separation_m=0.4,
        line_yaw_deg=0.0,
        source_centres_robot_m=((0.4, 0.2, 0.7), (0.4, -0.2, 0.7)),
    )


def _state(t, *, seq=None, pelvis_xy=(0.0, 0.0), feet=None, tools=None, rotations=None, fallen=False):
    return {
        "seq": int(round(t * 10)) if seq is None else seq,
        "t": float(t),
        "wall": 1000.0 + float(t),
        "pelvis": {
            "pos": [pelvis_xy[0], pelvis_xy[1], 0.76],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "yaw": 0.0,
            "vel": [0.0, 0.0, 0.0],
        },
        "joints": np.zeros(29).tolist(),
        "objects": {},
        "palms": {"left": {"hand_q": [0.0, 0.0]}, "right": {"hand_q": [0.0, 0.0]}},
        "feet": np.asarray(feet if feet is not None else [[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
        "tools": np.asarray(tools if tools is not None else [[0.0, 0.2, 0.7], [0.0, -0.2, 0.7]]),
        "rotations": np.asarray(rotations if rotations is not None else [np.eye(3), np.eye(3)]),
        "fallen": fallen,
    }


class _Runtime:
    def __init__(self, states, *, inject_walk=False):
        self.states = list(states)
        self.index = 0
        self.walk_command_count = 0
        self.inject_walk = inject_walk
        self.commands = []
        self.stopped = False

    def observe(self):
        state = self.states[min(self.index, len(self.states) - 1)]
        self.index += 1
        return state

    def enter_pose(self, acceleration, joint_limits):
        self.entered = (acceleration, joint_limits)

    def pose_step(self, joints, root_quaternion_wxyz, hands, *, reference_velocity=None):
        self.commands.append((np.asarray(joints), np.asarray(root_quaternion_wxyz), np.asarray(hands)))
        if self.inject_walk:
            self.walk_command_count += 1

    def walk_step(self, *_args):
        raise AssertionError("executor must not call walking")

    def safe_stop(self):
        self.stopped = True


class _Controller:
    def __init__(self, _solver, _config, state, **_kwargs):
        self.x = np.r_[state["pelvis"]["pos"], 0.0, state["joints"]]
        self.steps = 0
        self.diagnostics = {"admission_hand_error": [0.0, 0.0]}

    def _command(self, reason="", finished=False):
        return SimpleNamespace(
            joints=np.full(29, 0.5),
            velocity=np.zeros(29),
            root_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            phase="reach",
            reason=reason,
            finished=finished,
        )

    def step(self, *_args, **_kwargs):
        self.steps += 1
        return self._command()

    def result(self, reason="", finished=False):
        return self._command(reason, finished)


def _executor(monkeypatch, runtime, *, timeout=1.0):
    monkeypatch.setattr(execution, "foot_positions", lambda _ik, state: np.asarray(state["feet"], dtype=float))
    monkeypatch.setattr(
        execution,
        "_measured_tools",
        lambda _solver, state: (np.asarray(state["tools"], dtype=float), np.asarray(state["rotations"], dtype=float)),
    )
    solver = SimpleNamespace(ik=object(), lo=np.full(29, -2.0), hi=np.full(29, 2.0))
    criteria = execution.GhostReachCriteria(
        target_timeout_sim_s=timeout,
        success_hold_sim_s=0.5,
        position_tolerance_m=0.02,
        orientation_tolerance_deg=15.0,
        foot_horizontal_drift_m=0.03,
        foot_vertical_lift_m=0.03,
        wall_watchdog_s=5.0,
    )
    return execution.GhostReachExecutor(
        solver,
        runtime,
        criteria=criteria,
        controller_factory=_Controller,
        wall_clock=lambda: 0.0,
        event_clock=lambda: 1000.0,
    )


def test_world_target_is_fixed_while_pelvis_moves():
    case = _case()
    initial = _state(0.0, pelvis_xy=(0.1, -0.2))
    target = execution.world_target(case, initial)
    moved = _state(0.2, pelvis_xy=(0.14, -0.23))
    moved["pelvis"]["quat"] = Rotation.from_euler("z", 0.2).as_quat()[[3, 0, 1, 2]].tolist()
    estimate = execution.synthetic_estimate(target, moved, 1000.0)
    local = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    base = Rotation.from_quat(np.asarray(moved["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
    reconstructed = estimate.transform(local) @ base.T + np.asarray(moved["pelvis"]["pos"])
    np.testing.assert_allclose(reconstructed, target.centres_m, atol=1e-12)
    assert CoordinatedTarget().reach_only is False


def test_commanded_reference_cannot_count_as_measured_success(monkeypatch):
    initial = _state(0.0)
    states = [_state(value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)]
    runtime = _Runtime(states)
    result = _executor(monkeypatch, runtime).run_case(_case(), initial)
    assert result["status"] == "failed"
    assert result["reason"] == "target_timeout"
    assert runtime.commands
    assert runtime.walk_command_count == 0


def test_measured_success_requires_full_stable_window(monkeypatch):
    initial = _state(0.0)
    target = execution.world_target(_case(), initial)
    states = [
        _state(value, seq=index, tools=target.centres_m, rotations=[target.permitted_tool_rotation] * 2)
        for index, value in enumerate((0.0, 0.25, 0.49, 0.50))
    ]
    runtime = _Runtime(states)
    result = _executor(monkeypatch, runtime, timeout=2.0).run_case(_case(), initial)
    assert result["status"] == "success"
    assert result["reason"] == "measured_reach_held"
    assert result["traces"][-1]["sim_time_s"] == 0.5
    assert result["walk_command_count"] == 0


def test_drift_failure_records_trigger_frame(monkeypatch):
    initial = _state(0.0)
    drifted_feet = [[0.031, 0.1, 0.0], [0.0, -0.1, 0.0]]
    runtime = _Runtime([_state(0.0), _state(0.1, feet=drifted_feet)])
    result = _executor(monkeypatch, runtime).run_case(_case(), initial)
    assert result["reason"] == "stance_drift_horizontal"
    assert result["traces"][-1]["seq"] == 1
    assert result["traces"][-1]["terminal_reason"] == "stance_drift_horizontal"
    assert max(result["traces"][-1]["foot_horizontal_drift_m"]) > 0.03


def test_any_walking_instrumentation_fails_trial(monkeypatch):
    runtime = _Runtime([_state(0.0), _state(0.1)], inject_walk=True)
    result = _executor(monkeypatch, runtime).run_case(_case(), _state(0.0))
    assert result["reason"] == "walking_command_forbidden"
    assert result["walk_command_count"] == 1


def test_source_file_is_exact_fourteen_case_identity(tmp_path):
    sample = {
        "label_zh": "fixture",
        "parameters": {
            "pair_midpoint_x_m": 0.4,
            "pair_midpoint_y_m": 0.0,
            "ground_height_z_m": 0.7,
            "pair_separation_m": 0.4,
            "pair_line_yaw_deg": 0.0,
        },
        "capture_centres_xyz": {"left": [0.4, 0.2, 0.7], "right": [0.4, -0.2, 0.7]},
        "orientation_endpoints": {},
    }
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps({"schema_version": 1, "samples": [{**sample, "index": index} for index in range(1, 15)]})
    )
    cases, identity = execution.load_source_cases(path)
    assert [case.index for case in cases] == list(range(1, 15))
    assert identity["sample_count"] == 14
    assert len(identity["sha256"]) == 64


def test_launcher_delivers_absolute_native_state_trace_directory(tmp_path):
    backend = SimpleNamespace(to_dict=lambda: {"schema_version": 1, "no_encoder": False})
    output = tmp_path / "run" / "result.json"
    env = stack_environment(backend, output)
    assert env["SIM_STATE_TRACE_DIR"] == str(output.resolve().parent / "videos" / "trace")
    assert json.loads(env["LOCOMANI_BACKEND_JSON"])["no_encoder"] is False


def test_single_case_recording_separates_initialization_from_fresh_task_boundary(tmp_path, monkeypatch):
    scene_xml = tmp_path / "scene_locomani.xml"
    scene_meta = tmp_path / "scene_meta.json"
    request_path = tmp_path / "request.json"
    for path in (scene_xml, scene_meta, request_path):
        path.write_text("{}")

    events = []
    sequence = {"value": 0}
    video_dir = tmp_path / "videos"
    video_dir.mkdir()

    class Primitives:
        def __init__(self, **_kwargs):
            pass

        def obs(self):
            sequence["value"] += 1
            return _state(sequence["value"] * 0.02, seq=sequence["value"])

        def close(self):
            events.append("primitives_closed")

    class Reset:
        def start_video(self, label):
            events.append(("start", label))
            (video_dir / f"{label}.mp4").write_bytes(b"video-open")

        def end_video(self):
            events.append("end")

        def pin_arms(self):
            events.append("release_pins")

        def close(self):
            events.append("reset_closed")

    class Runtime:
        def safe_stop(self):
            events.append("safe_stop")

    class Executor:
        def __init__(self, *_args, **_kwargs):
            pass

        def run_case(self, case, initial_state):
            events.append(("execute", initial_state["seq"]))
            return {
                "index": case.index,
                "label_zh": case.label_zh,
                "status": "success",
                "reason": "measured_reach_held",
                "initialization": {},
                "traces": [{"seq": initial_state["seq"], "sim_time_s": initial_state["t"]}],
            }

    def initialize(primitives, _reset, _ik, _criteria, _label, initialization):
        state = primitives.obs()
        initialization.update(status="ready", reason=None, initial_stance_ready=True)
        return Runtime(), state, initialization

    monkeypatch.setattr(execution_cli, "require_pose_ownership", lambda: None)
    monkeypatch.setattr(execution_cli, "ArmIK", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(execution_cli, "BimanualIK", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(execution_cli, "MeasuredCollisionChecker", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(execution_cli, "Primitives", Primitives)
    monkeypatch.setattr(execution_cli, "EpisodeControlClient", Reset)
    monkeypatch.setattr(execution_cli, "GhostReachExecutor", Executor)
    monkeypatch.setattr(execution_cli, "_initialize_trial", initialize)
    monkeypatch.setattr(execution_cli, "_prime_session", lambda *_args: {"status": "ready"})

    request = {
        "path": request_path,
        "sha256": "request-sha",
        "request_identity": "full-fourteen",
        "source_identity": {"sample_count": 14},
        "seed": 7000,
        "cases": [replace(_case(), index=index) for index in range(1, 15)],
        "criteria": execution.GhostReachCriteria(),
        "runtime": {
            "scene_xml": scene_xml,
            "scene_meta_path": scene_meta,
            "scene_meta": {},
            "scene_dir": tmp_path,
            "instance": 2,
            "gpu": "1",
            "video_label_prefix": "ghost",
        },
    }

    result = execution_cli.run_live(request, tmp_path / "result.json", case_index=1)

    assert result["request"]["source_case_count"] == 14
    assert result["request"]["case_count"] == 1
    assert result["request"]["selected_case_index"] == 1
    case = result["cases"][0]
    assert case["initialization_artifacts"]["label"] == "ghost_01_initialization"
    assert case["artifacts"]["label"] == "ghost_01"
    assert case["reach_boundary"]["start_seq"] > case["task_recording_boundary"]["initialization_admission_seq"]
    assert events.index("end") < events.index(("start", "ghost_01"))
    assert ("start", "ghost_01_initialization") in events
