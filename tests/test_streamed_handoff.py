from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
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


def test_opt_in_takeover_inherits_dq_and_emits_consistent_overlapping_windows(monkeypatch):
    from locomani.backends.sonic.pose_preview import PosePreview

    clock = [0.0]
    monkeypatch.setattr(
        streamed_arm,
        "time",
        SimpleNamespace(time=lambda: clock[0], sleep=lambda dt: clock.__setitem__(0, clock[0] + dt)),
    )
    planner = np.linspace(-0.2, 0.2, 29)
    velocity = np.linspace(-0.8, 0.8, 29)
    root = np.array([1.0, 0.0, 0.0, 0.0])
    angular = np.array([0.0, 0.3, 0.0])
    tap = SimpleNamespace(
        pose_state=lambda: dict(
            joints=planner,
            velocity=velocity,
            root=root,
            root_angular_velocity=angular,
            timestamp=0.0,
        ),
        pose=lambda: (_ for _ in ()).throw(AssertionError("pose_state should own opt-in takeover")),
    )
    monkeypatch.setattr(upper_ref, "shared", lambda: tap)
    commander = Mock(
        left_hand=np.zeros(7),
        right_hand=np.zeros(7),
        upper_body=None,
        upper_body_vel=None,
        vr_targets=False,
    )
    state = dict(t=clock[0], joints=np.zeros(29), pelvis=dict(quat=[1, 0, 0, 0]))
    primitive = SimpleNamespace(cmd=commander, obs=lambda: dict(state, t=clock[0]), _facing=lambda _: (1, 0, 0))
    arm = streamed_arm.StreamedArm(primitive, "right", ik=scene_ik(), left_guard=False)
    arm.preview = PosePreview(
        acceleration=2.0,
        joint_limits=(np.full(29, -2.0), np.full(29, 2.0)),
        continuous_handoff=True,
    )

    arm.start(hold_planner_pose=True, inherit_planner_velocity=True)

    assert arm.handoff_diagnostics["q_start_error"] < 1e-12
    assert arm.handoff_diagnostics["dq_start_error"] < 1e-12
    assert arm.handoff_diagnostics["incoming_speed"] == np.max(np.abs(velocity))
    calls = commander.pose_chunk.call_args_list
    assert len(calls) >= 4
    for call in calls:
        positions, _, _ = call.args[:3]
        speeds = call.kwargs["joint_vel_isaac"]
        np.testing.assert_allclose(np.diff(positions, axis=0) / 0.02, speeds[1:], atol=3e-6)

    # Successive packets revise overlap only where a newer current state makes
    # it physically necessary. Here the hold follows the same braking law, so
    # shared future frame predictions agree exactly.
    for left, right in zip(calls, calls[1:]):
        left_pos, left_root, left_first = left.args[:3]
        right_pos, right_root, right_first = right.args[:3]
        left_frames = {left_first + i: (left_pos[i], left_root[i]) for i in range(len(left_pos))}
        right_frames = {right_first + i: (right_pos[i], right_root[i]) for i in range(len(right_pos))}
        overlap = sorted(set(left_frames) & set(right_frames))
        assert len(overlap) >= 45
        for frame in overlap:
            np.testing.assert_allclose(left_frames[frame][0], right_frames[frame][0], atol=3e-6)
            np.testing.assert_allclose(left_frames[frame][1], right_frames[frame][1], atol=3e-6)

    # Later settling/handback uses an ordinary hold after the explicit
    # physical-time takeover. Its absent timestamp must not be subtracted from
    # the saved explicit-reference time.
    arm.hold(0.04)


def test_loaded_reference_brakes_to_admitted_preview_endpoint_before_handback(monkeypatch):
    from locomani.backends.sonic.pose_preview import PosePreview
    from locomani.backends.sonic.primitives import HAND_CLOSED_RIGHT

    clock = [0.0]
    actual = np.zeros(29)
    arm_box = {}

    def sleep(dt):
        clock[0] += dt
        if "arm" in arm_box:
            actual[:] = arm_box["arm"].q

    monkeypatch.setattr(
        streamed_arm,
        "time",
        SimpleNamespace(time=lambda: clock[0], sleep=sleep),
    )
    commander = Mock(left_hand=np.zeros(7), right_hand=np.zeros(7))
    state = lambda: dict(t=clock[0], joints=actual.copy())
    primitive = SimpleNamespace(cmd=commander, obs=state)
    arm = streamed_arm.StreamedArm(primitive, "right", ik=scene_ik(), left_guard=False)
    arm_box["arm"] = arm
    arm.preview = PosePreview(
        acceleration=2.0,
        joint_limits=(np.full(29, -2.0), np.full(29, 2.0)),
        continuous_handoff=True,
    )
    arm.q = np.linspace(-0.2, 0.2, 29)
    arm.root_quat = np.array([1.0, 0, 0, 0])
    arm.reference_velocity = np.linspace(-0.7, 0.7, 29)
    arm.root_angular_velocity = np.array([0.0, 0.3, 0.0])
    arm.reference_time = 0.0
    arm.next_t = 0.0
    arm.hands["right"] = HAND_CLOSED_RIGHT.copy()
    arm.preview.sample(
        0.0,
        arm.q,
        arm.reference_velocity,
        arm.root_quat,
        root_angular_velocity=arm.root_angular_velocity,
    )

    prediction = arm.stopping_reference()
    diagnostics = arm.brake_to_stop(prediction)

    np.testing.assert_allclose(arm.q, prediction["joints"], atol=1e-10)
    np.testing.assert_allclose(arm.root_quat, prediction["root"], atol=1e-10)
    np.testing.assert_allclose(arm.reference_velocity, 0, atol=1e-12)
    np.testing.assert_allclose(arm.root_angular_velocity, 0, atol=1e-12)
    assert diagnostics["packet"]["max_joint_acceleration"] <= 2.0 + 1e-10
    assert diagnostics["packet"]["max_root_angular_acceleration"] <= 2.0 + 1e-10
    assert diagnostics["observed"]["max_joint_acceleration"] is not None

    calls = commander.pose_chunk.call_args_list
    assert len(calls) == prediction["frames"]
    for call in calls:
        positions = call.args[0]
        speeds = call.kwargs["joint_vel_isaac"]
        np.testing.assert_allclose(np.diff(positions, axis=0) / 0.02, speeds[1:], atol=3e-6)
        np.testing.assert_array_equal(call.kwargs["right_hand"], HAND_CLOSED_RIGHT)


def test_braking_uses_the_last_successfully_sent_reference_state(monkeypatch):
    from scipy.spatial.transform import Rotation

    from locomani.backends.sonic.pose_preview import PosePreview

    clock = [0.0]
    actual = np.zeros(29)
    arm_box = {}

    def sleep(dt):
        clock[0] += dt
        if "arm" in arm_box:
            actual[:] = arm_box["arm"].q

    monkeypatch.setattr(
        streamed_arm,
        "time",
        SimpleNamespace(time=lambda: clock[0], sleep=sleep),
    )
    commander = Mock(left_hand=np.zeros(7), right_hand=np.zeros(7))
    primitive = SimpleNamespace(cmd=commander, obs=lambda: dict(t=clock[0], joints=actual.copy()))
    arm = streamed_arm.StreamedArm(primitive, "right", ik=scene_ik(), left_guard=False)
    arm_box["arm"] = arm
    arm.preview = PosePreview(
        acceleration=2.0,
        joint_limits=(np.full(29, -2.0), np.full(29, 2.0)),
        continuous_handoff=True,
    )
    arm.next_t = 0.0

    for stamp, value, speed, pitch in (
        (0.00, 0.00, 0.10, 0.00),
        (0.02, 0.01, 0.35, 0.01),
        (0.04, 0.02, 0.60, 0.018),
    ):
        arm.q = np.full(29, value)
        arm.root_quat = Rotation.from_euler("y", pitch).as_quat()[[3, 0, 1, 2]]
        arm._push(arm.q, reference_velocity=np.full(29, speed), reference_time=stamp)

    assert arm.reference_time == 0.04
    np.testing.assert_allclose(arm.reference_velocity, 0.60)
    assert np.linalg.norm(arm.root_angular_velocity) == pytest.approx(0.4)
    last_q = arm.q.copy()
    prediction = arm.stopping_reference()
    assert np.max(np.abs(prediction["joints"] - last_q)) > 0.05

    diagnostics = arm.brake_to_stop(prediction)

    assert diagnostics["packet"]["initial_joint_speed"] == pytest.approx(0.60)
    assert diagnostics["packet"]["initial_root_angular_speed"] == pytest.approx(0.4)
    np.testing.assert_allclose(arm.q, prediction["joints"], atol=1e-10)
