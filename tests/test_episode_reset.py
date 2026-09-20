"""Episode reset joint-vector contract and failure isolation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES
from locomani.sim.privileged_pub import EpisodeControlServer
from locomani.tasks.crate import ghost_reach_execution_cli
from locomani.tasks.crate.ghost_reach_execution import GhostReachCriteria


class _Messages:
    def __init__(self, *messages: dict):
        self.messages = [json.dumps(message) for message in messages]

    def poll(self, _timeout: int) -> bool:
        return bool(self.messages)

    def recv_string(self) -> str:
        return self.messages.pop(0)


def _server():
    qpos0 = np.zeros(7 + len(MUJOCO_JOINT_NAMES), dtype=float)
    qpos0[2] = 0.76
    default_joints = np.linspace(-0.29, 0.29, len(MUJOCO_JOINT_NAMES))
    data = SimpleNamespace(qpos=np.full_like(qpos0, 9.0), qvel=np.ones(len(qpos0) - 1))
    calls = {"appearance": 0, "reset": 0, "forward": 0}

    def reset_data(_model, target):
        calls["reset"] += 1
        target.qpos[:] = -3.0
        target.qvel[:] = -4.0

    server = EpisodeControlServer.__new__(EpisodeControlServer)
    server._mj = SimpleNamespace(
        mj_resetData=reset_data,
        mj_forward=lambda _model, _data: calls.__setitem__("forward", calls["forward"] + 1),
    )
    server.model = object()
    server.data = data
    server.qpos0 = qpos0
    server.default_dof_angles = default_joints
    server.obj_qadr = {}
    server._dof_qadr = list(range(7, 7 + len(MUJOCO_JOINT_NAMES)))
    server._appearance = lambda _seed: calls.__setitem__("appearance", calls["appearance"] + 1)
    server._seat_on_floor = lambda: None
    server._apply_obj_pin = lambda: None
    server._place_ghost = lambda _pos, _quat: None
    return server, default_joints, calls


def _ready_reference() -> np.ndarray:
    ready = np.zeros(len(MUJOCO_JOINT_NAMES), dtype=float)
    ready[15:22] = [-0.35, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
    ready[22:29] = [-0.35, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
    return ready


def test_reset_applies_full_joint_vector_from_requested_index():
    server, default_joints, calls = _server()
    ready = _ready_reference()

    server._apply({"robot": {"dxy": [0.0, 0.0], "yaw": 0.0, "dof": ready.tolist(), "dof_from": 12}})

    joints = server.data.qpos[7:]
    np.testing.assert_array_equal(joints[:12], default_joints[:12])
    np.testing.assert_array_equal(joints[12:15], np.zeros(3))
    assert joints[MUJOCO_JOINT_NAMES.index("left_elbow_joint")] == pytest.approx(0.8)
    assert joints[MUJOCO_JOINT_NAMES.index("right_elbow_joint")] == pytest.approx(0.8)
    np.testing.assert_array_equal(server.data.qvel, np.zeros_like(server.data.qvel))
    assert calls == {"appearance": 1, "reset": 1, "forward": 1}


def test_ghost_initialization_uses_scene_only_resets_and_reuses_prepared_arm(monkeypatch):
    server, default_joints, _calls = _server()
    events = []
    prepared_arm = SimpleNamespace(
        q=np.zeros(29),
        hands={
            "left": np.asarray(ghost_reach_execution_cli.HAND_OPEN),
            "right": np.asarray(ghost_reach_execution_cli.HAND_OPEN),
        },
    )
    state = {
        "seq": 40,
        "t": 0.8,
        "fallen": False,
        "joints": default_joints.copy(),
        "pelvis": {
            "pos": [0.0, 0.0, 0.76],
            "quat": [1.0, 0.0, 0.0, 0.0],
            "vel": [0.0, 0.0, 0.0],
        },
    }

    class Reset:
        def start_video(self, label):
            events.append(("video", label))

        def reset(self, *, robot, label, **_kwargs):
            events.append(("reset", label, robot.copy()))
            server._apply({"robot": robot})

    class Runtime:
        def __init__(self, _primitives, _ik, *, planner_arm):
            assert planner_arm is prepared_arm
            events.append(("runtime", planner_arm))

    def shared(_primitives, _reset, cfg, _solver, *, placement, whole_body, on_phase):
        assert cfg == ghost_reach_execution_cli.original_reach_control_config().visual
        assert whole_body is True
        for reset_id in ("initial_placement", "prepared_placement"):
            on_phase("reset_commanded", {"reset_id": reset_id})
            placement(reset_id)
        return prepared_arm, {
            "initialization_resets": 2,
            "initial_stance_ready": True,
            "ready_reference": _ready_reference(),
            "admission_state": state,
        }

    primitives = SimpleNamespace(obs=lambda: state)
    monkeypatch.setattr(ghost_reach_execution_cli, "prepare_stationary_ready", shared)
    monkeypatch.setattr(ghost_reach_execution_cli, "SonicGhostReachRuntime", Runtime)
    monkeypatch.setattr(ghost_reach_execution_cli, "BimanualIK", lambda *_args, **_kwargs: object())

    runtime, observed, initialization = ghost_reach_execution_cli._initialize_trial(
        primitives,
        Reset(),
        object(),
        GhostReachCriteria(),
        "case",
    )

    assert runtime is not None
    assert observed is state
    assert [event[0] for event in events] == ["reset", "reset", "runtime"]
    assert all(set(event[2]) == {"dxy", "yaw"} for event in events if event[0] == "reset")
    assert [entry["reset_id"] for entry in initialization["resets"]] == [
        "initial_placement",
        "prepared_placement",
    ]
    np.testing.assert_array_equal(server.data.qpos[7:], default_joints)
    assert initialization["status"] == "ready"


@pytest.mark.parametrize(
    "dof",
    [np.zeros(17), np.r_[np.zeros(len(MUJOCO_JOINT_NAMES) - 1), np.nan]],
    ids=["sliced-upper-body", "non-finite"],
)
def test_reset_rejects_malformed_joint_vector_before_state_changes(dof):
    server, _default_joints, calls = _server()
    qpos_before = server.data.qpos.copy()
    qvel_before = server.data.qvel.copy()

    with pytest.raises(ValueError, match="robot.dof"):
        server._apply({"robot": {"dxy": [0.0, 0.0], "yaw": 0.0, "dof": dof.tolist(), "dof_from": 12}})

    np.testing.assert_array_equal(server.data.qpos, qpos_before)
    np.testing.assert_array_equal(server.data.qvel, qvel_before)
    assert calls == {"appearance": 0, "reset": 0, "forward": 0}


@pytest.mark.parametrize("dof_from", [12.5, float("inf"), -1, len(MUJOCO_JOINT_NAMES) + 1])
def test_reset_rejects_invalid_joint_start_before_state_changes(dof_from):
    server, _default_joints, calls = _server()
    qpos_before = server.data.qpos.copy()
    qvel_before = server.data.qvel.copy()

    with pytest.raises(ValueError, match="robot.dof_from"):
        server._apply(
            {
                "robot": {
                    "dxy": [0.0, 0.0],
                    "yaw": 0.0,
                    "dof": _ready_reference().tolist(),
                    "dof_from": dof_from,
                }
            }
        )

    np.testing.assert_array_equal(server.data.qpos, qpos_before)
    np.testing.assert_array_equal(server.data.qvel, qvel_before)
    assert calls == {"appearance": 0, "reset": 0, "forward": 0}


def test_poll_isolates_invalid_reset_request():
    server, _default_joints, calls = _server()
    qpos_before = server.data.qpos.copy()
    server.sock = _Messages(
        {
            "cmd": "reset",
            "robot": {"dxy": [0.0, 0.0], "yaw": 0.0, "dof": np.zeros(17).tolist(), "dof_from": 12},
        }
    )

    server.poll()

    np.testing.assert_array_equal(server.data.qpos, qpos_before)
    assert calls == {"appearance": 0, "reset": 0, "forward": 0}
