import io
import json
import threading
import time
from unittest.mock import Mock

import numpy as np

from locomani.backends.sonic import agent_server
from locomani.backends.sonic.upper_ref import RefTap
from locomani.sim.privileged_pub import PoseSidecar


def test_ref_tap_snapshot_preserves_identity_and_disclaims_exact_consumption():
    tap = RefTap(port=1)
    tap._q = np.arange(29, dtype=np.float32)
    tap._root = np.array([1, 0, 0, 0], dtype=np.float32)
    tap._t = time.time()
    tap._sample_id = 17

    snapshot = tap.snapshot()

    assert snapshot["sample_id"] == 17
    assert snapshot["source"] == "deploy_current_motion_output"
    assert snapshot["timestamp_semantics"] == "agent_receive_wall_no_controller_ack"
    snapshot["body_q_target"][0] = -99
    assert tap._q[0] == 0


def test_pose_diagnostics_do_not_change_wire_payload(monkeypatch):
    packed = []
    monkeypatch.setattr(agent_server, "pack_pose_message", lambda data, **kwargs: packed.append(data) or b"pose")
    commander = object.__new__(agent_server.AgentCommander)
    commander.sonic = Mock()
    commander.reference_observer = Mock()
    joints = np.arange(3 * 29, dtype=float).reshape(3, 29)

    commander.pose_chunk(joints, [1, 0, 0, 0], 10, diagnostic_current_offset=1)

    assert "diagnostic_current" not in packed[0]
    observed = commander.reference_observer.call_args.args[1]
    assert observed["packet_sample_count"] == 3
    assert observed["diagnostic_current"]["frame_index"] == 11
    assert agent_server.LAST_CMD["pose_send"]["timestamp_semantics"] == "agent_post_send_wall_no_deploy_ack"


def _run_one_sidecar_row(reference_tap):
    sidecar = object.__new__(PoseSidecar)
    sidecar._alive = True
    sidecar._lock = threading.Lock()
    sidecar._fh = io.StringIO()
    sidecar._reference_tap = reference_tap

    message = json.dumps(
        {
            "seq": 4,
            "t": 0.08,
            "wall": 100.02,
            "pelvis": {"pos": [0, 0, 0.8], "quat": [1, 0, 0, 0]},
            "objects": {},
            "joints": np.arange(29).tolist(),
        }
    )

    class Socket:
        def __init__(self):
            self.first = True

        def recv_string(self):
            if self.first:
                self.first = False
                return message
            sidecar._alive = False
            raise TimeoutError

    sidecar.sock = Socket()
    sidecar._loop()
    return json.loads(sidecar._fh.getvalue())


def test_sidecar_adds_reference_and_measured_joints_without_claiming_acknowledgement():
    tap = Mock()
    tap.snapshot.return_value = {
        "schema_version": 1,
        "source": "deploy_current_motion_output",
        "timestamp_semantics": "agent_receive_wall_no_controller_ack",
        "sample_id": 8,
        "received_wall": 100.01,
        "body_q_target": np.zeros(29),
        "base_quat_target": np.array([1, 0, 0, 0]),
    }
    agent_server.LAST_CMD.update(
        control_owner_requested="streamed",
        stream_mode_requested=1,
        pose_send={"sent_wall": 100.0},
    )

    row = _run_one_sidecar_row(tap)

    assert row["joints"] == np.arange(29).tolist()
    assert row["deploy_reference"]["sample_id"] == 8
    assert row["deploy_reference"]["age_at_state_wall_s"] > 0
    assert row["reference_command_state"]["stream_mode_requested"] == 1


def test_sidecar_still_writes_state_when_optional_reference_tap_fails():
    tap = Mock()
    tap.snapshot.side_effect = RuntimeError("diagnostic unavailable")

    row = _run_one_sidecar_row(tap)

    assert row["seq"] == 4
    assert "deploy_reference" not in row
