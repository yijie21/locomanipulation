from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from geometry_fixture import scene_ik

from locomani.backends.sonic import streamed_arm, upper_ref


def test_streamed_handoff_never_emits_an_unconstrained_planner_frame(monkeypatch):
    """Exercise the actual handoff with a clock and capture its planner packets."""
    clock = [0.0]
    monkeypatch.setattr(
        streamed_arm,
        "time",
        SimpleNamespace(time=lambda: clock[0], sleep=lambda dt: clock.__setitem__(0, clock[0] + dt)),
    )
    monkeypatch.setattr(
        upper_ref, "shared", lambda: SimpleNamespace(pose=lambda: (np.zeros(29), np.array([1, 0, 0, 0])))
    )
    commander = Mock(left_hand=np.zeros(7), right_hand=np.zeros(7), upper_body=np.ones(17), vr_targets=False)
    unprotected = []
    commander.frame.side_effect = lambda **kw: unprotected.append(
        commander.upper_body is None and not commander.vr_targets
    )
    state = dict(joints=np.zeros(29), pelvis=dict(quat=[1, 0, 0, 0]))
    primitive = SimpleNamespace(cmd=commander, obs=lambda: state, _facing=lambda _: (1, 0, 0))
    arm = streamed_arm.StreamedArm(primitive, "right", ik=scene_ik(), left_guard=False)
    arm.start(hold_planner_pose=True)
    commander.set_wrists.assert_not_called()
    commander.set_neck.assert_not_called()
    commander.frame.assert_not_called()
    commander.enter_streamed.assert_called_once()
    assert not any(unprotected), f"{sum(unprotected)} planner frames released both arm references before POSE takeover"


def test_pose_takeover_preserves_the_incoming_reference_instead_of_its_tracking_error(monkeypatch):
    from locomani.primitives.g1_joints import mujoco_to_isaac

    clock = [0.0]
    monkeypatch.setattr(
        streamed_arm,
        "time",
        SimpleNamespace(time=lambda: clock[0], sleep=lambda dt: clock.__setitem__(0, clock[0] + dt)),
    )
    planner = np.linspace(-0.3, 0.3, 29)
    held_upper = np.linspace(-0.1, 0.1, 17)
    expected = planner.copy()
    expected[12:15], expected[15:22], expected[22:29] = held_upper[:3], held_upper[3::2], held_upper[4::2]
    root = np.array([np.cos(0.025), 0, np.sin(0.025), 0])
    commander = Mock(left_hand=np.zeros(7), right_hand=np.zeros(7), upper_body=held_upper, vr_targets=False)
    state = dict(joints=np.zeros(29), pelvis=dict(quat=[1, 0, 0, 0]))
    primitive = SimpleNamespace(cmd=commander, obs=lambda: state, _facing=lambda _: (1, 0, 0))
    arm = streamed_arm.StreamedArm(primitive, "right", ik=scene_ik(), left_guard=False)
    monkeypatch.setattr(upper_ref, "shared", lambda: SimpleNamespace(pose=lambda: (planner, root)))
    arm.start(hold_planner_pose=True)
    for call in commander.pose_chunk.call_args_list:
        np.testing.assert_allclose(call.args[0], np.tile(mujoco_to_isaac(expected), (len(call.args[0]), 1)), atol=1e-7)
        np.testing.assert_allclose(call.args[1], np.tile(root, (len(call.args[1]), 1)), atol=1e-7)


def test_reference_tap_keeps_joint_and_root_from_one_fresh_packet(monkeypatch):
    now = [10.0]
    tap = upper_ref.RefTap(port=5599)
    tap._run = True
    tap._sub = SimpleNamespace(
        get_msg=lambda **kw: dict(body_q_target=np.arange(29) * 0.01, base_quat_target=[1, 0, 0, 0])
    )
    monkeypatch.setattr(
        upper_ref, "time", SimpleNamespace(time=lambda: now[0], sleep=lambda _: setattr(tap, "_run", False))
    )
    tap._loop()
    q, root = tap.pose()
    np.testing.assert_allclose(q, np.arange(29) * 0.01, atol=1e-7)
    np.testing.assert_array_equal(root, [1, 0, 0, 0])
    q[:] = 99
    assert np.max(tap.pose()[0]) < 1
    now[0] += 0.3
    assert tap.pose() is None
