"""CPU checks for the simple single-arm expert: configuration, dispatch and the phase geometry."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from locomani import profiles
from locomani.pipeline.execution_record import ExecutionRecord
from locomani.tasks.bottle.policy import make_grasp_policy, prepares_during_navigation
from locomani.tasks.bottle.reach_config import ReachConfig
from locomani.tasks.bottle.simple_config import SimpleGraspConfig
from locomani.tasks.bottle.simple_policy import (
    SimpleGraspPolicy,
    entry_aligned,
    entry_captured,
    entry_command,
    finger_touching,
    grasp_rotation,
    hanging_arms,
    lateral_correction,
    lean_angle,
    moving_start,
    native,
    needs_retract,
    object_in_palm,
    polyline_length,
    polyline_point,
    pregrasp_point,
    raise_height,
    raise_point,
    retry_target,
    wrap_direction,
)
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig


def test_simple_task_config_loads_dispatches_and_roundtrips():
    task = profiles.load("task", "bottle_simple")
    assert task.reach.strategy == "simple"
    assert task.reach.ready_grasp_baseline and task.reach.ready_grasp_recovery
    assert task.reach.simple == SimpleGraspConfig()
    assert prepares_during_navigation(task.reach)
    changed = task.with_overrides(["reach.simple.dash_speed_m_s=0.05", "reach.simple.lift_height_m=0.10"])
    assert changed.reach.simple.dash_speed_m_s == 0.05
    assert changed.reach.simple.lift_height_m == 0.10
    assert ReachConfig.from_dict(changed.reach.to_dict()) == changed.reach
    args = [object(), object(), changed.reach, object(), ExecutionRecord()]
    with patch.object(SimpleGraspPolicy, "__init__", return_value=None) as initialize:
        assert isinstance(make_grasp_policy(*args), SimpleGraspPolicy)
    initialize.assert_called_once_with(*args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("pregrasp_back_m", 0.4),
        ("dash_speed_m_s", 0.5),
        ("entry_capture_depth_m", 0.05),
        ("close_seconds", 0.1),
        ("min_rise_m", 0.2),
        ("lean_max_rad", 2.0),
        ("lift_seconds", float("nan")),
        ("entry_attempts", 0),
        ("retreat_m", 0.5),
        ("raise_seconds", 0.1),
        ("raise_above_grip_m", 0.25),
        ("wrap_yaw_max_rad", 2.0),
        ("stand_settle_timeout_s", 0.1),
        ("lateral_correction_max_m", 0.2),
    ],
)
def test_simple_config_rejects_out_of_range_values(field, value):
    with pytest.raises(ValueError):
        replace(SimpleGraspConfig(), **{field: value})


def test_wrap_direction_turns_the_fingers_across_the_body_with_the_bottle_offset():
    forward = np.array([1.0, 0.0, 0.0])
    # Straight ahead: fingers forward. 0.25 m to the right: 1.0 rad to the left of the heading.
    np.testing.assert_allclose(wrap_direction([0, 0], [0.3, 0.0, 0.7], forward, 4.0, 1.2), [1, 0, 0], atol=1e-12)
    right = wrap_direction([0, 0], [0.3, -0.25, 0.7], forward, 4.0, 1.2)
    assert np.arctan2(right[1], right[0]) == pytest.approx(1.0)
    # Capped, and a bottle to the left never turns the fingers right.
    far = wrap_direction([0, 0], [0.3, -0.5, 0.7], forward, 4.0, 1.2)
    assert np.arctan2(far[1], far[0]) == pytest.approx(1.2)
    np.testing.assert_allclose(wrap_direction([0, 0], [0.3, 0.2, 0.7], forward, 4.0, 1.2), [1, 0, 0], atol=1e-12)
    # Heading-relative: the same geometry rotated with the robot.
    heading = np.array([0.0, -1.0, 0.0])
    turned = wrap_direction([0, 0], [-0.25, -0.3, 0.7], heading, 4.0, 1.2)  # bottle to the robot's right
    assert np.arctan2(turned[1], turned[0]) - np.arctan2(heading[1], heading[0]) == pytest.approx(1.0)
    rotation = grasp_rotation(turned)
    np.testing.assert_allclose(rotation[:, 0], turned, atol=1e-12)
    np.testing.assert_allclose(rotation[:, 2], [0, 0, 1], atol=1e-12)
    point = pregrasp_point([0.3, 0.4, 0.7], [0.6, 0.8, 0.0], 0.10, 0.02)
    np.testing.assert_allclose(point, [0.24, 0.32, 0.72])


def test_hanging_arms_mirror_the_left_arm_and_keep_the_waist():
    q = hanging_arms(np.full(17, 0.3), SimpleGraspConfig())
    np.testing.assert_allclose(q[:3], 0.3)
    np.testing.assert_allclose(q[4::2], [0.0, -0.45, 0.0, 1.2, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(q[3::2], [0.0, 0.45, 0.0, 1.2, 0.0, 0.0, 0.0])


def test_moving_start_moves_on_the_first_tick_and_ends_at_rest():
    assert moving_start(0.0) == 0.0 and moving_start(1.0) == 1.0
    assert moving_start(0.05) == pytest.approx(0.05)  # linear head: not the 0.001 of a minimum-jerk start
    assert moving_start(0.5) > 0.4
    assert moving_start(0.99) == pytest.approx(1.0, abs=1e-3)


def test_raise_height_retract_decision_and_polyline_path():
    # Above the grip, and above the table by the hand's extent plus sag, whichever is higher.
    assert raise_height(0.73, 0.62, 0.08, 0.12) == pytest.approx(0.81)
    assert raise_height(0.66, 0.62, 0.08, 0.12) == pytest.approx(0.74)
    raised = raise_point([0.05, -0.20, 0.57], [1.0, 0.0, 0.0], 0.81, 0.10)
    np.testing.assert_allclose(raised, [0.15, -0.20, 0.81])
    # Outboard is to the robot's right for the right hand (heading +x: right is -y).
    np.testing.assert_allclose(raise_point([0.05, -0.20, 0.57], [1.0, 0.0, 0.0], 0.81, 0.05, 0.06), [0.10, -0.26, 0.81])
    # Hand in the table-top band and within 6 cm of the edge: pull back first; farther or lower table: no.
    assert needs_retract(0.57, 0.02, 0.62, 0.12, 0.06)
    assert not needs_retract(0.57, 0.10, 0.62, 0.12, 0.06)
    assert not needs_retract(0.57, 0.02, 0.40, 0.12, 0.06)
    path = [[0, 0, 0], [0, 0, 1], [2, 0, 1]]
    assert polyline_length(path) == pytest.approx(3.0)
    np.testing.assert_allclose(polyline_point(path, 0.5), [0, 0, 0.5])
    np.testing.assert_allclose(polyline_point(path, 1.5), [0.5, 0, 1])
    np.testing.assert_allclose(polyline_point(path, 9.0), [2, 0, 1])
    np.testing.assert_allclose(polyline_point(path, -1.0), [0, 0, 0])


def test_entry_command_advances_at_speed_and_stops_past_the_grip():
    grip = np.array([0.5, 0.0, 0.7])
    direction = np.array([1.0, 0.0, 0.0])
    command = np.array([0.38, 0.0, 0.7])
    stepped = entry_command(command, grip, direction, 0.06, 0.1, 0.02)
    np.testing.assert_allclose(stepped, [0.386, 0.0, 0.7])
    for _ in range(200):
        command = entry_command(command, grip, direction, 0.06, 0.1, 0.02)
    np.testing.assert_allclose(command, [0.52, 0.0, 0.7])
    # Capture is a centimetre-tolerant depth along the line, not a millimetre window.
    assert not entry_captured([0.47, 0.0, 0.7], grip, direction, 0.01)
    assert entry_captured([0.492, 0.0, 0.7], grip, direction, 0.01)
    assert entry_captured([0.51, 0.03, 0.72], grip, direction, 0.01)


def test_lateral_correction_follows_the_jaw_axis_offset_within_rate_and_range():
    palm = np.eye(3)
    # The measured offset is what is left after the shift already applied (an integrating follow).
    shift = lateral_correction(np.zeros(3), palm, [0.0, 0.03, 0.0], 0.03, 0.1, 0.04)
    np.testing.assert_allclose(shift, [0, 0.003, 0])  # 3 cm/s for 0.1 s
    for _ in range(50):
        shift = lateral_correction(shift, palm, [0.0, 0.03 - shift[1], 0.0], 0.03, 0.1, 0.04)
    np.testing.assert_allclose(shift, [0, 0.03, 0], atol=1e-9)  # converged on the offset
    for _ in range(50):
        shift = lateral_correction(shift, palm, [0.0, 0.10 - shift[1], 0.0], 0.03, 0.1, 0.04)
    assert np.linalg.norm(shift) == pytest.approx(0.04)  # range-limited


def test_finger_touch_and_lean_helpers():
    assert finger_touching({"touch_links": ["R.dex1_finger_2"]}, "right")
    assert not finger_touching({"touch_links": ["L.dex1_finger_2", "R.wrist"]}, "right")
    assert not finger_touching({}, "right")
    # Distance only: no lean inside 0.30 m whatever the grip height, the cap at 0.60 m.
    assert lean_angle([0, 0, 0.78], [0.25, 0, 0.70], 0.55) == 0.0
    assert lean_angle([0, 0, 0.78], [0.45, 0, 0.70], 0.55) == pytest.approx(0.275)
    assert lean_angle([0, 0, 0.78], [0.70, 0, 0.70], 0.55) == pytest.approx(0.55)
    # With the heading given, only the forward depth counts: a bottle 0.30 m ahead and
    # 0.40 m to the right (the stand's geometry) needs no lean at all.
    assert lean_angle([0, 0, 0.78], [0.30, -0.40, 0.70], 0.55, forward=[1.0, 0.0, 0.0]) == 0.0
    assert lean_angle([0, 0, 0.78], [0.45, -0.40, 0.70], 0.55, forward=[1.0, 0.0, 0.0]) == pytest.approx(0.275)
    assert lean_angle([0, 0, 0.78], [-0.45, 0.0, 0.70], 0.55, forward=[-1.0, 0.0, 0.0]) == pytest.approx(0.275)


def _simple_policy():
    """A policy object with the simulator, task and IK replaced by mocks."""
    policy = SimpleGraspPolicy.__new__(SimpleGraspPolicy)
    policy.p = SimpleNamespace(
        obs=Mock(),
        cmd=SimpleNamespace(set_upper_body=Mock(), set_upper_body_vel=Mock()),
        pelvis_xy=Mock(return_value=np.array([0.1, 0.2])),
    )
    policy.execution = ExecutionRecord()
    policy.config = ReachConfig(strategy="simple")
    policy.cfg, policy.simple = WholeBodyConfig(), SimpleGraspConfig()
    policy.stats = dict(samples=[], walking=[], phases=[])
    policy.error = None
    policy.completed = False
    policy.arm = None
    policy._stream_released = True
    policy._walk_logged = -np.inf
    policy._walk_q, policy._walk_stamp = np.zeros(17), None
    policy._edge_anchor = None
    policy.guard = SimpleNamespace(check=Mock(return_value=SimpleNamespace(clear=True)))
    policy.target = SimpleNamespace(
        observe=Mock(return_value=None), diagnostics={}, age=Mock(return_value=float("inf"))
    )
    return policy


def test_walk_tick_holds_the_hanging_reference_and_reports_a_guard_hit():
    policy = _simple_policy()
    policy.target = SimpleNamespace(
        observe=Mock(return_value=np.array([0.5, 0.0, 0.7])), diagnostics={"ego_height_px": 40.0}
    )
    hanging = np.full(17, 0.3)
    walking = SimpleNamespace(last=None, q=hanging.copy(), step=Mock(), chosen_stop_depth=0.44, chosen_stop_edge=0.40)
    policy.walking = walking
    policy._walk_q, policy._walk_stamp = hanging.copy(), None
    state = {"t": 1.0, "joints": np.zeros(29)}
    assert policy.walk_tick(state, edge_distance=0.5)
    # First tick: no time has passed, the reference is still the ramp's pose; the walker never plans an arm.
    np.testing.assert_allclose(policy.p.cmd.set_upper_body.call_args.args[0], hanging)
    np.testing.assert_allclose(policy.p.cmd.set_upper_body_vel.call_args.args[0], 0.0)
    walking.step.assert_not_called()
    assert policy.stats["walking"][-1]["chosen_stop_depth"] == 0.44
    assert policy.stats["walking"][-1]["ego_px"] == 40.0
    assert policy._edge_anchor[0] == 0.5 and np.allclose(policy._edge_anchor[1], [0.1, 0.2])
    # The sweep runs on the measured pose (nothing is commanded to move).
    np.testing.assert_allclose(policy.guard.check.call_args_list[0].args[1], np.zeros(29))
    assert policy.walk_tick({"t": 1.0, "joints": np.zeros(29)})  # same stamp: no second push
    assert policy.p.cmd.set_upper_body.call_count == 1
    assert policy.walk_tick(
        {"t": 1.5, "joints": np.zeros(29)}
    )  # later ticks: the same reference, nothing moves during the walk
    np.testing.assert_allclose(policy.p.cmd.set_upper_body.call_args.args[0], hanging)
    policy.guard.check = Mock(
        return_value=SimpleNamespace(clear=False, link="right_dex1_finger_1_link", obstacle="table")
    )
    assert not policy.walk_tick({"t": 2.0, "joints": np.zeros(29)}, edge_distance=0.4)
    assert "simple clearance" in policy.error


def test_prepare_walk_opens_the_jaws_and_ramps_the_arms_standing_until_measured():
    from locomani.backends.sonic.primitives import HAND_OPEN

    policy = _simple_policy()
    policy.p.cmd.set_hands = Mock()
    policy.p.cmd.frame = Mock()
    policy.p._facing = Mock(return_value=np.array([1.0, 0.0]))
    policy.simple = replace(SimpleGraspConfig(), hang_ramp_rad_s=2.0, hang_reached_rad=0.05)
    policy._walk_q = np.zeros(17)
    goal = hanging_arms(np.zeros(17), policy.simple)
    joints = np.zeros(29)
    joints[15:22], joints[22:29] = goal[3::2], goal[4::2]  # the measured arms are already there
    states = [{"t": k * 0.02, "joints": joints.copy()} for k in range(200)]
    policy.p.obs = Mock(side_effect=states)
    policy.prepare_walk()
    np.testing.assert_allclose(policy.p.cmd.set_hands.call_args.kwargs["left"], HAND_OPEN)
    np.testing.assert_allclose(policy._walk_q, goal)  # the reference reached the hang
    assert policy.stats["prepare_walk"]["reached"] and policy.stats["prepare_walk"]["seconds"] < 1.5
    assert policy.p.cmd.frame.call_count >= 1  # standing frames were sent while ramping


def test_settle_for_admission_waits_for_legs_and_base_at_rest_then_records_the_stance():
    policy = _simple_policy()
    policy.walking = SimpleNamespace(q=np.zeros(17))
    policy.ik = SimpleNamespace(
        body_pose=Mock(side_effect=[(np.array([0.05, 0.1, 0.0]), None), (np.array([-0.03, -0.1, 0.0]), None)])
    )
    # Legs move for the first 0.3 s, then everything is still: the settle takes 0.3 s plus the dwell.
    policy.simple = replace(SimpleGraspConfig(), stand_settle_s=0.5, stand_leg_speed_rad_s=0.10)
    states = []
    t = 0.0
    for k in range(80):
        t = round(k * 0.02, 4)
        moving = t < 0.3
        joints = np.zeros(29)
        joints[3] = 0.05 * k if moving else 0.05 * 15
        states.append({"t": t, "joints": joints, "pelvis": {"pos": np.array([0.0, 0.0, 0.78])}})
    policy.p.obs = Mock(side_effect=states)
    assert policy.settle_for_admission()
    stand = policy.stats["stand"]
    assert stand["settled"] and 0.75 <= stand["seconds"] <= 0.85
    assert stand["feet"]["forward_m"] == pytest.approx(0.08) and stand["feet"]["lateral_m"] == pytest.approx(0.2)
    assert policy.p.cmd.set_upper_body.call_count == len(policy.p.obs.call_args_list)
    # A stance that never settles is recorded at the timeout, not failed.
    policy = _simple_policy()
    policy.walking = SimpleNamespace(q=np.zeros(17))
    policy.ik = SimpleNamespace(body_pose=Mock(side_effect=KeyError("no ankle")))
    policy.simple = replace(SimpleGraspConfig(), stand_settle_timeout_s=0.5, stand_leg_speed_rad_s=0.10)
    never = [{"t": k * 0.02, "joints": np.full(29, 0.1 * k), "pelvis": {"pos": np.zeros(3)}} for k in range(40)]
    policy.p.obs = Mock(side_effect=never)
    assert policy.settle_for_admission()
    assert not policy.stats["stand"]["settled"] and policy.stats["stand"]["feet"] is None


def test_release_hands_back_once_and_holds_a_failed_reference_first():
    policy = _simple_policy()
    arm = Mock(q=np.zeros(29))
    policy.arm, policy._stream_released = arm, False
    policy._release(False)
    arm._push.assert_called_once()
    arm.stop.assert_called_once_with(preserve_reference=True)
    policy._release(False)
    arm.stop.assert_called_once()
    success = _simple_policy()
    success.arm, success._stream_released = Mock(q=np.zeros(29)), False
    success._release(True)
    success.arm._push.assert_not_called()
    success.arm.stop.assert_called_once_with(preserve_reference=True)


def test_execute_reports_an_invisible_target_and_still_releases_nothing():
    policy = _simple_policy()
    policy.target = SimpleNamespace(
        observe=Mock(return_value=None), diagnostics={"ego": "none"}, age=Mock(return_value=9.0)
    )
    policy.task = SimpleNamespace(target=SimpleNamespace(key="water_bottle"))
    policy.p.obs.return_value = {"t": 0.0}
    policy.walking = SimpleNamespace(q=np.zeros(17))
    policy.simple = replace(SimpleGraspConfig(), target_grace_s=0.05)
    ok, why = policy.execute(grip_world=None, squat_hint=None, ghost=None)
    assert not ok and "not visible" in why
    assert policy.stats["failure"]["camera"] == {"ego": "none"}
    assert policy.p.cmd.set_upper_body.call_count >= 1  # the hanging reference is held while waiting
    assert policy.stats["outcome"]["ok"] is False
    assert policy.arm is None


def test_stats_are_json_native_after_execute():
    import json

    policy = _simple_policy()
    policy.target = SimpleNamespace(observe=Mock(return_value=None), diagnostics={}, age=Mock(return_value=9.0))
    policy.task = SimpleNamespace(target=SimpleNamespace(key="water_bottle"))
    policy.p.obs.return_value = {"t": 0.0}
    policy.walking = SimpleNamespace(q=np.zeros(17))
    policy.simple = replace(SimpleGraspConfig(), target_grace_s=0.0)
    policy.stats["takeover"] = dict(braking=dict(predicted=dict(joints=np.zeros(29), root=np.array([1.0, 0, 0, 0]))))
    policy.stats["plan"] = dict(direction=np.array([1.0, 0.0, 0.0]), lean=np.float64(0.2), ok=np.bool_(True))
    policy.execute(grip_world=None, squat_hint=None, ghost=None)
    json.dumps(policy.stats)
    assert native({"a": (np.float32(1.5), [np.int64(2)])}) == {"a": [1.5, [2]]}


def test_entry_alignment_and_retry_shift_use_the_measured_palm_offset():
    palm = Rotation.from_euler("z", 0.5).as_matrix()
    tool = np.array([0.4, 0.1, 0.7])
    bottle = tool + palm @ np.array([0.02, 0.03, -0.01])
    offset = object_in_palm(palm, tool, bottle)
    np.testing.assert_allclose(offset, [0.02, 0.03, -0.01], atol=1e-12)
    assert not entry_aligned(offset, 0.015, 0.03)
    assert entry_aligned([0.0, 0.014, 0.029], 0.015, 0.03)
    assert not entry_aligned([0.0, 0.0, 0.031], 0.015, 0.03)
    direction = palm[:, 0]
    target = tool.copy()
    moved = retry_target(target, direction, palm, offset, 0.08)
    # 8 cm back along the line, shifted by the jaw-axis and height offsets.
    np.testing.assert_allclose(moved, tool - 0.08 * direction + palm[:, 1] * 0.03 + [0, 0, -0.01], atol=1e-12)
    np.testing.assert_allclose(object_in_palm(palm, moved + 0.08 * direction, bottle)[1:], [0.0, 0.0], atol=1e-12)


def test_grasp_rotation_is_a_pure_yaw():
    for yaw in (-2.0, 0.3, 2.9):
        direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        rotation = grasp_rotation(direction)
        assert Rotation.from_matrix(rotation).as_euler("xyz")[2] == pytest.approx(yaw)
        assert abs(Rotation.from_matrix(rotation).as_euler("xyz")[:2]).max() < 1e-12
