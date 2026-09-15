from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.backends.sonic import agent_server
from locomani.backends.sonic.pose_preview import PosePreview
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.primitives.g1_joints import mujoco_to_isaac


def test_physical_time_frames_supply_current_pose_and_consistent_braking_future(monkeypatch):
    packets = []
    monkeypatch.setattr(agent_server, "pack_pose_message", lambda data, **kw: packets.append(data) or b"pose")
    cmd = object.__new__(agent_server.AgentCommander)
    cmd.sonic = Mock()
    cmd.left_hand = cmd.right_hand = np.zeros(7)
    preview = PosePreview(acceleration=2.0, joint_limits=(np.full(29, -2), np.full(29, 2)))
    arm = StreamedArm(SimpleNamespace(cmd=cmd), "right", ik=scene_ik(), left_guard=False, preview=preview)
    arm.next_t = 0
    velocity = np.linspace(-0.6, 0.6, 29)
    for t in (0, 0.04, 0.10, 0.14, 0.22):
        arm.root_quat = Rotation.from_euler("y", 0.2 * t).as_quat()[[3, 0, 1, 2]]
        arm._push(velocity * t, reference_velocity=velocity, reference_time=10 + t)
        packet = packets[-1]
        current = list(packet["frame_index"]).index(round(t / 0.02))
        # The real deploy stops playback before its 46-frame observation window.
        # A current pose must be playable immediately, without waiting for 46 sends.
        assert current < len(packet["frame_index"]) - 46
        np.testing.assert_allclose(packet["joint_pos"][current], mujoco_to_isaac(velocity * t), atol=1e-7)
        np.testing.assert_allclose(packet["body_quat"][current, 0], arm.root_quat, atol=1e-7)
        q, v = packet["joint_pos"][current:], packet["joint_vel"][current:]
        np.testing.assert_allclose(np.diff(q, axis=0) / 0.02, v[1:], atol=2e-6)
        assert np.max(np.abs(np.diff(v, axis=0))) <= 2 * 0.02 + 1e-6
        np.testing.assert_allclose(v[-1], 0, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(packet["body_quat"][:, 0], axis=1), 1, atol=1e-7)
    with pytest.raises(ValueError, match="backwards"):
        preview.sample(9, np.zeros(29), velocity, [1, 0, 0, 0])


def test_future_brakes_within_joint_bounds_without_moving_a_stationary_reference():
    preview = PosePreview(acceleration=2, joint_limits=(np.full(29, -1), np.full(29, 1)))
    q = np.zeros(29)
    q[0] = 0.99
    v = np.zeros(29)
    v[0] = 0.1
    positions, velocity, _, _ = preview.sample(0, q, v, [1, 0, 0, 0])
    assert np.max(positions) <= 1
    np.testing.assert_allclose(positions[:, 1:], 0)
    np.testing.assert_allclose(velocity[-1], 0)


def test_continuous_preview_handles_sparse_duplicate_samples_and_out_of_limit_takeover():
    preview = PosePreview(
        acceleration=2.0,
        joint_limits=(np.full(29, -1.0), np.full(29, 1.0)),
        continuous_handoff=True,
    )
    q0, v0 = np.zeros(29), np.full(29, 0.6)
    preview.sample(0.0, q0, v0, [1, 0, 0, 0])
    # Duplicate physical ticks replace the current reference instead of
    # manufacturing time or integrating velocity.
    duplicate = q0 + 0.01
    p_duplicate, _, _, _ = preview.sample(0.0, duplicate, v0, [1, 0, 0, 0])
    np.testing.assert_allclose(p_duplicate[0], duplicate)

    # Incompatible sparse endpoint velocities cannot make Hermite history
    # overshoot either observed endpoint.
    q1 = np.full(29, 0.04)
    positions, velocities, _, _ = preview.sample(0.10, q1, -v0, [1, 0, 0, 0])
    history = positions[: preview.diagnostics["current_offset"] + 1]
    assert np.min(history) >= 0.01 - 1e-12
    assert np.max(history) <= 0.04 + 1e-12
    np.testing.assert_allclose(np.diff(positions, axis=0) / preview.dt, velocities[1:], atol=1e-8)

    # A planner pose already just outside a nominal bound is the first emitted
    # pose and the preview never snaps it back inside in one frame.
    outside = np.zeros(29)
    outside[0] = 1.03
    second = PosePreview(
        acceleration=2.0,
        joint_limits=(np.full(29, -1.0), np.full(29, 1.0)),
        continuous_handoff=True,
    )
    positions, _, _, _ = second.sample(0.0, outside, np.zeros(29), [1, 0, 0, 0])
    assert positions[0, 0] == pytest.approx(1.03)
    np.testing.assert_allclose(positions[:, 0], 1.03)
