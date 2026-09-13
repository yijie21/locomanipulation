from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from geometry_fixture import scene_ik

from locomani.backends.sonic import agent_server
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.primitives.g1_joints import mujoco_to_isaac


def test_each_buffered_pose_keeps_its_root_rotation_and_joint_velocity(monkeypatch):
    captured = []
    monkeypatch.setattr(agent_server, "pack_pose_message", lambda data, **kw: captured.append(data) or b"pose")
    commander = object.__new__(agent_server.AgentCommander)
    commander.sonic = Mock()
    commander.left_hand = commander.right_hand = np.zeros(7)
    p = SimpleNamespace(cmd=commander)
    arm = StreamedArm(p, "right", ik=scene_ik(), left_guard=False)
    arm.next_t = 0
    expected_roots = []
    for i in range(3):
        q = np.arange(29) * 0.01 + i * 0.001
        velocity = np.arange(29) * 0.001 + i * 0.01
        arm.root_quat = np.array([np.cos(i * 0.01), 0, np.sin(i * 0.01), 0])
        expected_roots.append(arm.root_quat.copy())
        arm._push(q, reference_velocity=velocity)
    packet = captured[-1]
    np.testing.assert_allclose(packet["body_quat"][:, 0], expected_roots, atol=1e-7)
    for i in range(3):
        np.testing.assert_allclose(packet["joint_vel"][i], mujoco_to_isaac(np.arange(29) * 0.001 + i * 0.01), atol=1e-7)
    np.testing.assert_array_equal(packet["frame_index"], [0, 1, 2])
