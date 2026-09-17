import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from geometry_fixture import scene_ik

from locomani import profiles
from locomani.backends.sonic.primitives import upper_body_from_mj
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.pipeline.execution_record import ExecutionRecord
from locomani.primitives.supported_motion import SupportedMotion
from locomani.tasks.bottle.policy import make_grasp_policy
from locomani.tasks.bottle.reach_config import ReachConfig
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig


def test_whole_body_configuration_reaches_actual_factory_and_roundtrips():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    task = profiles.load("task", "bottle_whole_body")
    changed = task.with_overrides(["reach.whole_body.hand_speed=0.12", "reach.whole_body.walk_plan_hz=4"])
    assert changed.reach.whole_body.hand_speed == 0.12
    assert changed.reach.whole_body.walk_plan_hz == 4
    assert changed.reach.whole_body.capture_depth_m == 0.0012
    assert changed.reach.whole_body.capture_depth_tolerance_m == 0.005
    assert changed.reach.whole_body.capture_lateral_tolerance_m == 0.012
    assert changed.reach.whole_body.capture_height_tolerance_m == 0.015
    assert changed.reach.whole_body.contact_recovery_timeout == 0.8
    assert changed.reach.whole_body.capture_position_diameter_max == 0.004
    assert changed.reach.whole_body.capture_drift_speed_max == 0.03
    assert changed.reach.whole_body.capture_rotation_diameter_max == 0.035
    assert changed.reach.whole_body.capture_angular_drift_max == 0.20
    assert changed.reach.whole_body.capture_min_samples == 6
    assert changed.reach.whole_body.capture_brake_acceleration == 0.20
    assert changed.reach.whole_body.capture_brake_lookahead_seconds == 0.10
    assert changed.reach.whole_body.capture_terminal_window_seconds == 0.20
    assert changed.reach.whole_body.capture_terminal_max_gap_seconds == 0.05
    assert changed.reach.whole_body.tracking_correction_dynamics is True
    assert changed.reach.whole_body.ik_velocity_continuity_weight == 0.0
    assert changed.reach.whole_body.comfort_posture is True
    assert changed.reach.whole_body.stance_forward_margin_m == 0.18
    assert changed.reach.whole_body.stance_yaw_margin_deg == 16.0
    assert changed.reach.whole_body.stance_solver_iterations == 40
    assert ReachConfig.from_dict(changed.reach.to_dict()) == changed.reach
    args = [object(), object(), changed.reach, object(), ExecutionRecord()]
    with patch.object(WholeBodyGraspPolicy, "__init__", return_value=None) as initialize:
        assert isinstance(make_grasp_policy(*args), WholeBodyGraspPolicy)
    initialize.assert_called_once_with(*args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("joint_speed", 2),
        ("waist_pitch_max", 1),
        ("vision_timeout", float("nan")),
        ("close_seconds", 0.1),
        ("capture_depth_m", 0.02),
        ("capture_depth_tolerance_m", 0.001),
        ("capture_lateral_tolerance_m", 0.026),
        ("capture_height_tolerance_m", 0.020),
        ("contact_confirm_seconds", 0.01),
        ("contact_recovery_timeout", 0.1),
        ("contact_speed_max", 0.5),
        ("capture_min_samples", 1),
        ("capture_position_diameter_max", 0.5),
        ("capture_drift_speed_max", 0.5),
        ("capture_rotation_diameter_max", 0.001),
        ("capture_angular_drift_max", 0.01),
        ("capture_brake_acceleration", 0.01),
        ("capture_brake_lookahead_seconds", 1.0),
        ("capture_terminal_window_seconds", 0.01),
        ("capture_terminal_max_gap_seconds", 0.2),
        ("tracking_translation_speed", 0.001),
        ("tracking_rotation_acceleration", 3.0),
        ("ik_velocity_continuity_weight", 0.3),
        ("walk_plan_hz", 30),
        ("walk_filter_seconds", 0.01),
        ("stance_forward_margin_m", 0.5),
        ("stance_yaw_margin_deg", 40),
        ("stance_solver_iterations", 20.0),
        ("stance_plan_tolerance_m", 0.1),
        ("stance_settled_timeout", 0.1),
        ("lift_seconds", 0.5),
        ("lift_height", 0.4),
        ("loaded_tracking_translation_full_m", 0.06),
        ("loaded_object_tilt_fail_rad", 0.20),
        ("loaded_grip_swing_full_rad", -0.01),
        ("loaded_orientation_recovery_seconds", 0.05),
    ],
)
def test_invalid_physical_limits_fail_before_execution(field, value):
    config = WholeBodyConfig(natural_walk_prep=field.startswith("stance_"))
    with pytest.raises(ValueError):
        replace(config, **{field: value})


def test_supported_motion_tracks_right_hand_with_planted_feet_and_bounded_rates():
    ik = scene_ik()
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.1, 0.2, -0.1
    q[15:22], q[22:29] = [-0.35, 0.2, 0, 0.8, 0, 0, 0], [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    state = dict(joints=q, pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    cfg = WholeBodyConfig()
    motion = SupportedMotion(ik, state, cfg, q, [1, 0, 0, 0], tool_offset=[0.1515, 0, 0], hand_weights=(0.35, 1))
    initial = motion.body.tools(motion.x).copy()
    target = initial.copy()
    target[1] += [0.05, -0.02, -0.05]
    rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    previous = motion.x.copy()
    for _ in range(8):
        pose = motion.step(target, rotations, motion.body.tools(motion.x).copy(), 0.05)
        np.testing.assert_allclose(motion.v, (motion.x - previous) / 0.05)
        assert max(abs(motion.v[4:])) <= cfg.joint_speed + 1e-8
        assert max(pose.foot_errors) < 0.005
        previous = motion.x.copy()
    assert np.linalg.norm(motion.body.tools(motion.x)[1] - target[1]) < np.linalg.norm(initial[1] - target[1])
    for dt in (0, -0.02, 0.6):
        with pytest.raises(ValueError):
            motion.step(target, rotations, initial, dt)


def test_return_to_planner_preserves_owned_reference_instead_of_tracking_error():
    q = np.linspace(-0.15, 0.15, 29)
    cmd = Mock(left_hand=np.zeros(7), right_hand=np.zeros(7))
    state = dict(joints=np.zeros(29), palms=dict(right=dict(pos=[0.3, -0.2, 0.8])), pelvis=dict(yaw=0.1))
    p = SimpleNamespace(cmd=cmd, obs=lambda: state, world_to_root_full=lambda point, _: point)
    arm = StreamedArm(p, "right", ik=scene_ik())
    arm.q = q
    arm.stop(preserve_reference=True)
    arm.stop(preserve_reference=True)
    np.testing.assert_array_equal(cmd.set_upper_body.call_args.args[0], upper_body_from_mj(q))
    names = [call[0] for call in cmd.method_calls]
    assert names.index("frame") < names.index("exit_streamed")
    assert cmd.exit_streamed.call_count == 1
    np.testing.assert_array_equal(cmd.set_upper_body_vel.call_args.args[0], np.zeros(17))


def test_unseen_target_cache_never_updates_from_new_hidden_truth():
    from locomani.tasks.bottle.assisted_target import AssistedBottleTarget

    # Exercise repeated-image and expiry behavior without rendering or a GPU.
    target = AssistedBottleTarget.__new__(AssistedBottleTarget)
    target.client = SimpleNamespace(read=lambda: None)
    target.seen, target.timeout = 10.0, 0.7
    target.point, target.diagnostics = np.array([0.3, -0.2, 0.7]), {}
    state = {"objects": {"water_bottle": {"pos": [99, 99, 99]}}}
    np.testing.assert_array_equal(target.observe(state, now=10.2), [0.3, -0.2, 0.7])
    assert target.observe(state, now=10.71) is None
    assert target.observe(state, now=9.9) is None


def test_whole_body_trace_can_be_serialized_on_cleanup():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.closed = False
    policy.target = Mock()
    policy.p = SimpleNamespace(cmd=SimpleNamespace(reference_observer=None))
    policy.previous_observer = None
    policy.stats = {"samples": [{"joints": np.arange(29), "value": np.float64(0.1)}]}
    policy.close()
    policy.close()
    json.dumps(policy.stats, allow_nan=False)
    policy.target.close.assert_called_once()


def test_reference_trace_retains_observer_target_and_actual_capture_geometry():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.previous_observer = None
    policy.stats = {"references": []}
    policy.context = dict(
        seq=12,
        observer_target=[0.4, -0.2, 0.7],
        capture_geometry={"depth_m": 0.001, "ready": True},
    )

    policy.observe_reference("upper", {"joint_pos": [0.0] * 17})

    recorded = policy.stats["references"][0]
    assert recorded["observer_target"] == [0.4, -0.2, 0.7]
    assert recorded["capture_geometry"] == {"depth_m": 0.001, "ready": True}
    assert recorded["payload"]["joint_pos"] == [0.0] * 17


def test_supported_reference_bias_does_not_hide_measured_foot_motion():
    ik = scene_ik()
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.1, 0.2, -0.1
    state = dict(joints=q.copy(), pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    reference = q.copy()
    reference[[5, 11]] = [-0.15, 0.15]
    motion = SupportedMotion(ik, state, WholeBodyConfig(), reference, [1, 0, 0, 0], tool_offset=[0.1515, 0, 0])
    np.testing.assert_allclose(motion.measured_foot_drift(state), 0, atol=1e-10)
    moved = dict(state, pelvis=dict(state["pelvis"], pos=[0.06, 0, 0.78]))
    np.testing.assert_allclose(motion.measured_foot_drift(moved), 0.06, atol=1e-10)
    np.testing.assert_array_equal(motion.joints, reference)


def test_successful_grasp_does_not_serialize_logs_before_loaded_transport():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.execution, policy.arm, policy.close = Mock(), Mock(), Mock()
    policy.p = SimpleNamespace(obs=lambda: {}, cmd=SimpleNamespace(reference_observer=None))
    policy.stats, policy.previous_observer = {}, object()
    policy._brake, policy._reach = lambda: True, lambda: (True, "lifted")
    policy._brake_handoff = lambda: (True, "lifted and stopped")
    assert policy.execute(grip_world=None, squat_hint=None, ghost=None) == (True, "lifted and stopped")
    policy.arm.stop.assert_called_once_with(preserve_reference=True)
    policy.close.assert_not_called()
    assert policy.p.cmd.reference_observer is policy.previous_observer


def test_supported_takeover_preserves_existing_decoder_tracking_bias():
    ik = scene_ik()
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.1, 0.2, -0.1
    q[15:22], q[22:29] = [-0.35, 0.2, 0, 0.8, 0, 0, 0], [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    state = dict(joints=q, pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    motion = SupportedMotion(ik, state, WholeBodyConfig(), q, [1, 0, 0, 0], tool_offset=[0.1515, 0, 0])
    reference_tools = motion.body.tools(motion.x).copy()
    rotations = ik.data.xmat[motion.body.hands].reshape(2, 3, 3).copy()
    measured = reference_tools + [[0.02, 0.04, -0.09], [-0.04, 0.01, 0.07]]
    pose = motion.step(measured, rotations, measured, 0.02, measured_rotations=rotations)
    np.testing.assert_allclose(motion.feedback, reference_tools - measured, atol=1e-10)
    np.testing.assert_allclose(pose.joints, q, atol=1e-5)


def test_continuous_entry_clears_cap_before_inserting_and_has_no_raised_waypoint():
    from locomani.tasks.bottle.whole_body_policy import entry_step

    target, axis = np.array([0.4, -0.2, 0.7]), np.array([1.0, 0, 0])
    point = target + [-0.02, 0, 0.15]
    for _ in range(600):
        following, gap = entry_step(point, point, target, axis, 0.02, 0.16)
        assert following[2] <= point[2] + 1e-9
        assert np.linalg.norm(following - point) <= 0.16 * 0.02 + 1e-9
        if point[2] - target[2] >= 0.06:
            assert gap >= 0.115
        point = following
    assert abs(point[2] - target[2]) < 0.001
    assert abs((target - point) @ axis - 0.018) < 0.001
    # At capture height with a clear path, the hand advances directly and
    # never moves upward or takes an intermediate backward waypoint.
    point = target - 0.15 * axis
    for _ in range(200):
        following, _ = entry_step(point, point, target, axis, 0.02, 0.16)
        assert following[0] >= point[0]
        np.testing.assert_allclose(following[1:], point[1:])
        point = following


def test_slow_camera_does_not_block_control_or_freshen_a_late_result():
    import time
    from threading import Event

    from locomani.tasks.bottle.assisted_target import BottleTargetObserver

    entered, release, finished = Event(), Event(), Event()

    class SlowCamera:
        seen = 10.0
        diagnostics = {"visible": {"right_wrist": 10.0}}

        def observe(self, state):
            entered.set()
            release.wait(2)
            return np.array([0.3, -0.2, 0.7])

        def close(self):
            finished.set()

    worker = BottleTargetObserver(None, None, 0.7, observer_factory=SlowCamera)
    try:
        assert worker.observe({"seq": 1}, now=10.1) is None
        assert entered.wait(1)
        # Image work is still blocked, but the actual controller-facing API returns.
        assert worker.observe({"seq": 2}, now=10.2) is None
        release.set()
        deadline = time.monotonic() + 1
        point = None
        while point is None and time.monotonic() < deadline:
            point = worker.observe({"seq": 3}, now=10.3)
            time.sleep(0.001)
        np.testing.assert_array_equal(point, [0.3, -0.2, 0.7])
        assert worker.observe({"seq": 4}, now=10.8) is None
    finally:
        release.set()
        worker.close()
    assert finished.is_set()


def test_lift_target_rides_the_reference_torso_instead_of_a_world_point():
    from locomani.tasks.bottle.whole_body_policy import lift_target, torso_offset

    ik = scene_ik()
    q = np.zeros(29)
    q[15:22], q[22:29] = [-0.35, 0.2, 0, 0.8, 0, 0, 0], [0.3, -0.2, 0, 0.9, 0, 0, 0]
    state = dict(joints=q, pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    motion = SupportedMotion(ik, state, WholeBodyConfig(), q, [1, 0, 0, 0], tool_offset=[0.1515, 0, 0])
    hand = motion.body.tools(motion.x)[1].copy()
    offset = torso_offset(ik, motion, hand)
    np.testing.assert_allclose(lift_target(ik, motion, offset, 0.0), hand, atol=1e-9)
    np.testing.assert_allclose(lift_target(ik, motion, offset, 0.1) - hand, [0, 0, 0.1], atol=1e-9)
    # Stand the reference up (waist pitch 0.4 -> 0): the world hand point would now
    # sit behind the shoulder, but the target keeps the same torso-frame offset.
    bent = motion.x.copy()
    bent[4 + 13] = 0.4  # waist pitch joint (MuJoCo order index 13)
    motion.x = bent
    bent_target = lift_target(ik, motion, offset, 0.0)
    np.testing.assert_allclose(torso_offset(ik, motion, bent_target), offset, atol=1e-9)
    assert np.linalg.norm(bent_target - hand) > 0.05


def test_failed_grasp_pins_a_stopped_reference_and_still_returns_control_to_the_planner():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.execution, policy.arm, policy.close = Mock(), Mock(), Mock()
    policy.arm.q = np.ones(29)
    policy.p = SimpleNamespace(obs=lambda: {}, cmd=SimpleNamespace(reference_observer=None))
    policy.stats, policy.previous_observer = {}, object()
    policy._brake, policy._reach = lambda: True, lambda: (False, "whole-body disturbed target or base")
    assert policy.execute(grip_world=None, squat_hint=None, ghost=None)[0] is False
    policy.arm._push.assert_called_once()
    np.testing.assert_array_equal(policy.arm._push.call_args.kwargs["reference_velocity"], np.zeros(29))
    policy.arm.stop.assert_called_once_with(preserve_reference=True)
    # the stopped reference is pinned before the mode switch, not after
    assert policy.arm.method_calls.index(policy.arm._push.mock_calls and policy.arm.method_calls[0]) == 0
    assert [c[0] for c in policy.arm.method_calls] == ["_push", "stop"]


def test_guard_calls_allow_finger_contact_at_handoff_and_ignore_the_held_bottle_in_lift():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    calls = []

    def check(state, candidate, side, **kw):
        calls.append((side, kw["allow_target_contact"], kw["ignore_target"]))
        return SimpleNamespace(clear=True)

    policy.guard = SimpleNamespace(check=check)
    assert policy._clear({}, np.zeros(29), True) is None
    assert calls == [("left", False, False), ("right", True, False), ("body", False, False)]
    calls.clear()
    policy._clear({}, np.zeros(29), True, attached=True)
    assert calls == [("left", False, False), ("right", True, True), ("body", False, False)]


def test_reference_filter_removes_tick_noise_and_passes_a_ramp_with_bounded_lag():
    from locomani.tasks.bottle.whole_body_policy import ReferenceFilter

    rng = np.random.default_rng(0)
    f = ReferenceFilter(4.0)
    dt, out = 0.02, []
    for k in range(200):
        ramp = np.full(29, 0.5 * k * dt)  # 0.5 rad/s
        q, v = f.step(ramp + 0.02 * rng.standard_normal(29), dt)
        out.append((q.copy(), v.copy()))
    q, v = zip(*out)
    raw = np.array([0.5 * k * dt for k in range(200)])[100:, None] + 0.02 * np.random.default_rng(0).standard_normal(
        (100, 29)
    )
    reversals = lambda x: np.sum(np.diff(x, axis=0) < 0)
    assert reversals(np.array(q)[100:]) < 0.1 * reversals(raw)  # tick noise that reverses the raw stream is gone
    assert abs(np.mean(np.array(v)[100:]) - 0.5) < 0.05  # the filtered velocity tracks the ramp
    assert 0 < (0.5 * 199 * dt) - q[-1][0] < 0.5 * 0.1  # lag under 100 ms at 4 Hz
    first_q, first_v = ReferenceFilter(4.0).step(np.ones(29), dt)
    np.testing.assert_array_equal(first_q, np.ones(29)) and np.testing.assert_array_equal(first_v, np.zeros(29))


def test_minimum_jerk_reach_path_slides_in_before_it_descends_and_rests_at_both_ends():
    from locomani.tasks.bottle.whole_body_policy import reach_path

    start, goal = np.array([0.0, 0.0, 0.05]), np.array([0.15, 0.02, 0.0])
    taus = np.linspace(0, 1, 301)
    path = np.array([reach_path(start, goal, t) for t in taus])
    np.testing.assert_allclose(path[0], start) and np.testing.assert_allclose(path[-1], goal)
    assert np.all(np.diff(path[:, 0]) >= 0)  # monotone slide-in
    assert np.allclose(path[taus <= 0.2, 2], 0.05)  # no descent in the first fifth
    assert np.allclose(path[taus >= 0.8, 2], 0.0)  # at grip height before the last fifth
    v = np.diff(path, axis=0) / (taus[1] - taus[0])
    assert np.linalg.norm(v[0]) < 0.01 and np.linalg.norm(v[-1]) < 0.01  # rests at both ends


def test_optional_smoothness_fields_accept_zero_and_reject_negatives():
    from locomani.tasks.bottle.whole_body_config import WholeBodyConfig

    WholeBodyConfig(pose_lowpass_hz=0.0, reach_seconds=0.0)
    WholeBodyConfig(pose_lowpass_hz=4.0, reach_seconds=3.0)
    with pytest.raises(ValueError):
        WholeBodyConfig(pose_lowpass_hz=-1.0)
    with pytest.raises(ValueError):
        WholeBodyConfig(reach_seconds=11.0)


def test_lean_target_follows_height_or_distance_whichever_asks_more():
    from locomani.tasks.bottle.whole_body_policy import lean_angle

    pelvis = [0.0, 0.0, 0.79]
    near = lean_angle(pelvis, [0.32, -0.2, 0.70], 1.0)
    assert abs(near - (0.79 + 0.12 - 0.70) / 0.6) < 1e-9  # the height term, as before
    far = lean_angle(pelvis, [0.60, -0.2, 0.70], 1.0)
    assert far == 1.0  # the distance term reaches the cap at 0.60 m
    assert lean_angle(pelvis, [0.60, -0.2, 0.70], 0.55) == 0.55  # the default cap reproduces the old behaviour
    assert lean_angle(pelvis, [0.20, -0.2, 0.95], 1.0) == 0.0  # hand above the grip band and close: no lean


def test_visible_expert_stops_at_the_table_edge_or_the_object_depth_whichever_first():
    from types import SimpleNamespace

    from locomani.tasks.bottle.round_phases_approach import visible_speed_scale, visible_stop

    cfg = SimpleNamespace(visible_brake_depth=0.31, visible_brake_edge=0.27, visible_decel_band=0.24)
    assert visible_stop(0.40, 0.27, cfg) and visible_stop(0.31, 0.40, cfg) and not visible_stop(0.40, 0.30, cfg)
    assert visible_speed_scale(0.80, 0.80, cfg) == 1.0
    assert visible_speed_scale(0.80, 0.27, cfg) == 0.0  # the edge rule alone can brake
    assert abs(visible_speed_scale(0.80, 0.39, cfg) - 0.5) < 1e-9
    off = SimpleNamespace(visible_brake_depth=0.31, visible_brake_edge=0.0, visible_decel_band=0.24)
    assert not visible_stop(0.40, 0.10, off) and visible_speed_scale(0.80, 0.10, off) == 1.0
    from locomani.tasks.bottle.whole_body_config import WholeBodyConfig

    WholeBodyConfig(raise_trigger_px=0.0, lean_max=1.0)
    with pytest.raises(ValueError):
        WholeBodyConfig(lean_max=1.5)


def test_admitted_wrist_projection_reports_timestamped_normalized_roi_and_camera_basis():
    from pathlib import Path

    import mujoco

    from locomani.tasks.bottle.assisted_target import AssistedBottleTarget

    ik = scene_ik()
    state = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
    m, d = ik.model, ik.data
    d.qpos[:] = m.qpos0
    d.qpos[ik.qadr] = state["joints"]
    d.qpos[ik.base_qadr : ik.base_qadr + 7] = [*state["pelvis"]["pos"], *state["pelvis"]["quat"]]
    mujoco.mj_fwdPosition(m, d)
    camera = m.camera("right_wrist").id
    camera_rotation = d.cam_xmat[camera].reshape(3, 3)
    target_pos = d.cam_xpos[camera] - camera_rotation[:, 2] * 0.32
    state["objects"]["water_bottle"]["pos"] = target_pos.tolist()
    state["objects"]["water_bottle"]["quat"] = [1, 0, 0, 0]
    state["seq"] = 5
    image = np.repeat(np.indices((480, 640))[0].astype(np.uint8)[:, :, None], 3, axis=2)
    client = SimpleNamespace(
        read=lambda: dict(
            images={"ego_view": image, "right_wrist": image},
            timestamps={"ego_view": 10.0, "right_wrist": 10.0},
        ),
        close=lambda: None,
    )
    task = SimpleNamespace(
        target=SimpleNamespace(
            body="water_bottle_object",
            key="water_bottle",
            box={"half_extent": [0.03, 0.03, 0.10]},
        ),
        grip=lambda sample: np.asarray(sample["objects"]["water_bottle"]["pos"]),
    )
    observer = AssistedBottleTarget(ik, task, 0.7, client=client)

    np.testing.assert_allclose(observer.observe(state, now=10.1), target_pos)
    roi = observer.diagnostics["wrist_roi"]
    assert roi["timestamp"] == 10.0
    assert roi["quality"] > 0 and roi["occupancy"] > 0
    assert np.all((np.asarray(roi["centroid"]) > 0) & (np.asarray(roi["centroid"]) < 1))
    assert np.all((np.asarray(roi["extent"]) > 0) & (np.asarray(roi["extent"]) < 1))
    basis = np.asarray(roi["correction_basis_pelvis"])
    assert basis.shape == (3, 2)
    np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-7)


def test_supported_motion_inherits_joint_root_and_heading_projected_pitch_velocity():
    from pathlib import Path

    from scipy.spatial.transform import Rotation

    state = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
    ik = scene_ik()
    dq = np.linspace(-0.4, 0.4, 29)
    linear = np.array([0.12, -0.03, 0.0])
    root_matrix = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
    omega = root_matrix[:, 1] * 0.35
    motion = SupportedMotion(
        ik,
        state,
        WholeBodyConfig(),
        state["joints"],
        state["pelvis"]["quat"],
        tool_offset=[0.1515, 0, 0],
        reference_velocity=dq,
        root_velocity=linear,
        root_angular_velocity=omega,
    )

    np.testing.assert_allclose(motion.v[4:], dq)
    np.testing.assert_allclose(motion.v[:3], linear)
    assert motion.v[3] == pytest.approx(0.35)
    assert motion.diagnostics["inherited_joint_speed"] == pytest.approx(0.4)


def test_close_transition_preserves_commanded_tools_and_rotations():
    from scipy.spatial.transform import Rotation

    from locomani.tasks.bottle.whole_body_policy import (
        close_transition,
        freeze_rotation_reference,
    )

    arm_q = np.linspace(-0.2, 0.2, 29)
    supported_q = arm_q.copy()
    supported_q[18] += 0.003
    targets = np.array([[0.2, 0.2, 0.8], [0.4, -0.2, 0.7]])
    rotations = Rotation.from_euler("xyz", [[0, 0, 0], [0.1, -0.2, 0.3]]).as_matrix()
    actual = targets + [[0, 0, 0], [0.019, 0, 0]]
    measured_rotations = rotations.copy()
    measured_rotations[1] = Rotation.from_rotvec([0, np.radians(4), 0]).as_matrix() @ rotations[1]

    held_targets, held_rotations, diagnostics = close_transition(
        arm_q,
        supported_q,
        targets,
        rotations,
        actual,
        measured_rotations,
    )

    np.testing.assert_array_equal(held_targets, targets)
    np.testing.assert_array_equal(held_rotations, rotations)
    assert diagnostics["sent_q_continuity_error_rad"] == pytest.approx(0.003)
    np.testing.assert_allclose(diagnostics["actual_tool_error"], [0, 0.019])
    np.testing.assert_allclose(diagnostics["actual_rotation_error_deg"], [0, 4], atol=1e-10)

    reference_rotations = Rotation.from_rotvec([[0, 0, 0], [0, 0, 0.08]]).as_matrix() @ rotations
    rotation_feedback = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.05]])
    frozen = freeze_rotation_reference(reference_rotations, rotation_feedback, tracking_gain=1.0)
    corrected = Rotation.from_rotvec(rotation_feedback).as_matrix() @ frozen
    np.testing.assert_allclose(corrected, reference_rotations, atol=1e-12)


def test_loaded_handoff_does_not_emit_an_unadmitted_stopping_reference():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    q = np.zeros(29)
    state = dict(joints=q.copy(), pelvis=dict(pos=[0, 0, 0.8], quat=[1, 0, 0, 0]))
    arm = Mock(
        q=q.copy(),
        root_quat=np.array([1.0, 0, 0, 0]),
        reference_velocity=np.full(29, 0.4),
        root_angular_velocity=np.zeros(3),
        hands={"right": np.zeros(7)},
    )
    arm.stopping_reference.return_value = dict(
        joints=q + 0.04,
        root=np.array([1, 0, 0, 0]),
        duration=0.2,
        frames=10,
        preview={},
    )
    policy = WholeBodyGraspPolicy.__new__(WholeBodyGraspPolicy)
    policy.arm, policy.stats = arm, {}
    policy.p = SimpleNamespace(obs=lambda: state)
    policy._clear = Mock(return_value="whole-body clearance: right:wrist:table")

    ok, reason = policy._brake_handoff()

    assert not ok and "clearance" in reason
    arm.brake_to_stop.assert_not_called()
    assert policy.stats["handback"]["admitted"] is False


def test_capture_admission_consumer_uses_current_geometry_once():
    from locomani.tasks.bottle.whole_body_policy import CaptureAdmissionConsumer

    consumer = CaptureAdmissionConsumer()
    motion = SimpleNamespace(freeze_posture=Mock())
    execution = SimpleNamespace(enter=Mock())
    captured = {}

    def enter_close():
        motion.freeze_posture()
        execution.enter("grasp.close")
        captured["transition"] = transition

    capture_geometry = {"ready": False, "depth_m": 0.02}
    phase = consumer.consume("reach", capture_geometry, enter_close)

    assert phase == "reach"
    motion.freeze_posture.assert_not_called()
    execution.enter.assert_not_called()
    assert captured == {}

    transition = {"rotation_freeze": "held"}
    capture_geometry = {"ready": True, "depth_m": 0.0012}
    phase = consumer.consume(phase, capture_geometry, enter_close)

    assert phase == "close"
    assert captured["transition"] is transition

    phase = consumer.consume(phase, capture_geometry, enter_close)

    assert phase == "close"
    motion.freeze_posture.assert_called_once_with()
    execution.enter.assert_called_once_with("grasp.close")


def test_whole_body_capture_uses_installed_pad_midpoint_and_preserves_the_legacy_window():
    from locomani.tasks.bottle.grasp_capture import OFFSET, closure_geometry

    state = {"palms": {"right": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]}}}

    def measured(depth):
        return closure_geometry(
            state,
            "right",
            OFFSET + [depth, 0, 0],
            preferred_depth_m=0.0012,
            depth_tolerance_m=0.005,
        )

    assert measured(0.0)["ready"]
    assert measured(0.0012)["ready"]
    assert not measured(0.007)["ready"]  # bottle would be caught toward the pad tips
    assert "depth" in measured(0.007)["rejection_reasons"]
    assert closure_geometry(state, "right", OFFSET + [0.015, 0, 0])["ready"]  # legacy users keep ±20 mm

    r3_like = OFFSET + [0.0012, -0.02239, -0.01683]
    legacy = closure_geometry(
        state,
        "right",
        r3_like,
        preferred_depth_m=0.0012,
        depth_tolerance_m=0.005,
    )
    whole_body = closure_geometry(
        state,
        "right",
        r3_like,
        preferred_depth_m=0.0012,
        depth_tolerance_m=0.005,
        lateral_tolerance_m=0.012,
        height_tolerance_m=0.015,
    )
    assert legacy["ready"]
    assert set(whole_body["rejection_reasons"]) == {"lateral", "height"}


def test_central_capture_window_fits_the_installed_bottle_and_rejects_a_larger_radius():
    from locomani.tasks.bottle.grasp_capture import (
        DEX1_PAD_X_MAX,
        DEX1_PAD_X_MIN,
        DEX1_PALM_X_MAX,
        capture_axial_clearance,
    )

    installed = capture_axial_clearance(0.0012, 0.005, 0.0221589)
    assert installed["ready"]
    assert installed["pad_proximal_m"] == pytest.approx(0.0049531)
    assert installed["pad_distal_m"] == pytest.approx(0.0049001)
    assert installed["palm_m"] == pytest.approx(0.0147411)
    assert installed["bottle_axial_bounds_m"][0] >= DEX1_PAD_X_MIN
    assert installed["bottle_axial_bounds_m"][1] <= DEX1_PAD_X_MAX
    assert installed["bottle_axial_bounds_m"][0] > DEX1_PALM_X_MAX
    assert not capture_axial_clearance(-0.002, 0.005, 0.03)["ready"]


def test_alignment_guarded_depth_retreats_continuously_until_both_axes_are_aligned():
    from locomani.tasks.bottle.grasp_capture import DEX1_PAD_X_MAX, OFFSET
    from locomani.tasks.bottle.whole_body_policy import entry_step, guarded_capture_depth

    capture_depth = 0.0012
    retreat_depth = DEX1_PAD_X_MAX - OFFSET[0] + 0.0226391638 + 0.004
    assert retreat_depth == pytest.approx(0.0598981638)

    target, axis = np.zeros(3), np.array([1.0, 0.0, 0.0])
    command = target - 0.10 * axis
    for _ in range(100):
        command, _ = entry_step(command, command, target, axis, 0.02, 0.16, retreat_depth)
    assert float((target - command) @ axis) == pytest.approx(retreat_depth, abs=1e-6)

    lateral_errors = np.linspace(0.026, 0.012, 101)
    lateral_depths = np.array(
        [guarded_capture_depth(error, 0.0, capture_depth, retreat_depth, 0.012, 0.015) for error in lateral_errors]
    )
    height_errors = np.linspace(0.020, 0.015, 101)
    height_depths = np.array(
        [guarded_capture_depth(0.0, error, capture_depth, retreat_depth, 0.012, 0.015) for error in height_errors]
    )

    assert lateral_depths[0] == pytest.approx(retreat_depth)
    assert height_depths[0] == pytest.approx(retreat_depth)
    assert lateral_depths[-1] == pytest.approx(capture_depth)
    assert height_depths[-1] == pytest.approx(capture_depth)
    assert np.all(np.diff(lateral_depths) <= 0)
    assert np.all(np.diff(height_depths) <= 0)
    assert np.max(np.abs(np.diff(lateral_depths))) < 0.001
    assert np.max(np.abs(np.diff(height_depths))) < 0.001
    assert guarded_capture_depth(0.011, 0.014, capture_depth, retreat_depth, 0.012, 0.015) == pytest.approx(
        capture_depth
    )


def test_terminal_braking_uses_the_current_cap_clearing_goal(monkeypatch):
    from locomani.tasks.bottle import whole_body_policy

    remaining = []

    def record_remaining(value, *_args):
        remaining.append(value)
        return 0.0

    monkeypatch.setattr(whole_body_policy, "terminal_entry_speed", record_remaining)
    target = np.zeros(3)
    command = np.array([-0.121, 0.0, 0.058])
    next_command, gap = whole_body_policy.entry_step(
        command,
        command,
        target,
        np.array([1.0, 0.0, 0.0]),
        0.02,
        0.16,
        0.020,
        brake_acceleration=0.20,
    )

    assert gap == pytest.approx(0.120)
    assert remaining == pytest.approx([0.001])
    assert next_command[0] == pytest.approx(command[0])


def test_central_entry_goal_advances_smoothly_through_the_old_tip_capture_depth():
    from locomani.tasks.bottle.whole_body_policy import entry_step

    target, axis = np.array([0.4, -0.2, 0.7]), np.array([1.0, 0, 0])
    point = target - 0.02 * axis
    positions = [point.copy()]
    for _ in range(200):
        point, _ = entry_step(point, point, target, axis, 0.02, 0.16, -0.002)
        positions.append(point.copy())
    positions = np.asarray(positions)
    assert np.all(np.diff(positions[:, 0]) >= -1e-12)
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)) <= 0.16 * 0.02 + 1e-9
    assert float((target - positions[-1]) @ axis) == pytest.approx(-0.002, abs=1e-6)


def test_terminal_observer_prediction_drives_capture_consumer():
    from locomani.tasks.bottle.whole_body_policy import (
        CaptureAdmissionConsumer,
        TerminalMotionObserver,
        momentum_guarded_depth,
        predictive_capture_geometry,
        signed_stopping_displacement,
    )

    assert signed_stopping_displacement(0.080, 0.20, 0.10) == pytest.approx(0.024)
    assert signed_stopping_displacement(-0.080, 0.20, 0.10) == pytest.approx(-0.024)
    assert momentum_guarded_depth(0.0012, 0.024) == pytest.approx((0.0252, 0.024, False))
    assert momentum_guarded_depth(0.095, 0.024) == pytest.approx((0.10, 0.024, True))

    observer = TerminalMotionObserver(0.20, 0.10)
    status = None
    for stamp in np.linspace(0.0, 0.20, 11):
        status = observer.step(stamp, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0012)
    geometry = dict(
        ready=True,
        rejection_reasons=[],
        depth_m=0.0012,
        preferred_depth_m=0.0012,
        depth_tolerance_m=0.005,
    )
    admitted = predictive_capture_geometry(geometry, status)
    assert admitted["ready"]
    assert admitted["stopping"]["predicted_depth_interval_m"] == pytest.approx([0.0012, 0.0012])

    entered = []
    consumer = CaptureAdmissionConsumer()
    assert consumer.consume("reach", admitted, lambda: entered.append(True)) == "close"
    assert entered == [True]

    moving = TerminalMotionObserver(0.20, 0.10)
    for stamp in np.linspace(0.0, 0.20, 11):
        x = 0.080 * stamp
        status = moving.step(stamp, [x, 0.0, 0.0], [1.0, 0.0, 0.0], 0.0057)
    rejected = predictive_capture_geometry({**geometry, "depth_m": 0.0057}, status)
    assert status["trend_velocity_m_s"] == pytest.approx(0.080)
    assert status["stopping_displacement_m"] == pytest.approx(0.024)
    assert status["predicted_depth_interval_m"] == pytest.approx([-0.0183, -0.0183])
    assert not rejected["ready"]
    assert "predicted_depth_interval" in rejected["rejection_reasons"]


def test_dynamic_momentum_depth_drives_timed_path_and_retracts_ahead_per_tick_command():
    from locomani.tasks.bottle.grasp_capture import capture_tool_goal
    from locomani.tasks.bottle.whole_body_policy import (
        advance_entry_command,
        momentum_guarded_depth,
        reach_path,
    )

    target = np.array([0.4, -0.2, 0.7])
    axis = np.array([1.0, 0.0, 0.0])
    initial = target - 0.10 * axis
    dynamic_depth, _, _ = momentum_guarded_depth(0.0012, 0.015)
    timed, _ = advance_entry_command(
        initial,
        initial,
        initial,
        target,
        axis,
        0.5,
        0.02,
        0.16,
        dynamic_depth,
        brake_acceleration=0.20,
    )
    np.testing.assert_allclose(
        timed,
        reach_path(initial, capture_tool_goal(target, axis, dynamic_depth), 0.5),
    )

    ahead = target - 0.001 * axis
    following, _ = advance_entry_command(
        initial,
        ahead,
        ahead,
        target,
        axis,
        1.0,
        0.02,
        0.16,
        dynamic_depth,
        brake_acceleration=0.20,
    )
    assert float((following - ahead) @ axis) < 0
    assert np.linalg.norm(following - ahead) <= 0.16 * 0.02 + 1e-9


def test_contact_loading_uses_physical_time_and_rejects_intermittent_contact():
    from locomani.tasks.bottle.whole_body_policy import ContactLoadController

    stable = ContactLoadController(0.10, 0.12, 0.80, 0.08)
    relative = [0.15, 0, 0]
    samples = [stable.step(t, True, relative, required=True) for t in (0.0, 0.03, 0.09, 0.101)]
    assert not any(sample["admitted"] for sample in samples[:-1])
    assert samples[-1]["admitted"]

    intermittent = ContactLoadController(0.10, 0.12, 0.80, 0.08)
    sample = None
    for index, stamp in enumerate(np.arange(0.0, 0.91, 0.05)):
        sample = intermittent.step(stamp, index % 2 == 0, relative, required=True)
        assert not sample["admitted"]
    assert sample["timed_out"]

    moving = ContactLoadController(0.10, 0.12, 0.80, 0.08)
    for stamp, x in ((0.0, 0.15), (0.05, 0.17), (0.10, 0.19), (0.15, 0.21)):
        sample = moving.step(stamp, True, [x, 0, 0], required=True)
    assert not sample["quiet"] and not sample["admitted"]


def test_contact_loss_decelerates_monotone_progress_then_resumes_after_stable_reacquisition():
    from locomani.tasks.bottle.whole_body_policy import ContactLoadController, advance_load_progress

    controller = ContactLoadController(0.10, 0.12, 0.80, 0.08)
    progress = 0.0

    def tick(stamp, bilateral):
        nonlocal progress
        status = controller.step(stamp, bilateral, [0.15, 0, 0], required=True)
        before = progress
        progress = advance_load_progress(progress, 0.05, 5.0, status["progress_scale"])
        assert before <= progress <= before + 0.01 + 1e-12
        return status, before, progress

    for stamp in (0.0, 0.05, 0.101, 0.151):
        status, _, _ = tick(stamp, True)
    assert status["admitted"] and status["progress_scale"] > 0

    loss_scales = []
    for stamp in (0.201, 0.251, 0.301, 0.351):
        status, _, _ = tick(stamp, False)
        loss_scales.append(status["progress_scale"])
    assert np.all(np.diff(loss_scales) <= 0) and loss_scales[-1] == 0
    paused = progress

    for stamp in (0.401, 0.451):
        status, _, _ = tick(stamp, True)
        assert not status["admitted"]
    assert progress == paused
    status, _, resumed = tick(0.502, True)
    assert status["admitted"] and resumed > paused
    assert not status["timed_out"]

    controller.step(0.552, False, [0.15, 0, 0], required=True)
    timed_out = controller.step(1.353, False, [0.15, 0, 0], required=True)
    assert timed_out["timed_out"]


def test_rotating_palm_frame_does_not_report_slip_for_a_rigidly_held_bottle():
    from scipy.spatial.transform import Rotation

    from locomani.tasks.bottle.whole_body_policy import object_in_palm_frame

    relative = np.array([0.15, 0.01, 0.0])
    palm0 = np.array([0.2, -0.1, 0.7])
    rotation0 = np.eye(3)
    rotation1 = Rotation.from_euler("z", np.radians(70)).as_matrix()
    palm1 = palm0 + [0.03, 0.02, 0.05]

    def state(position, rotation):
        quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
        return {"palms": {"right": {"pos": position, "quat": quat}}}

    object0 = palm0 + rotation0 @ relative
    object1 = palm1 + rotation1 @ relative
    assert np.linalg.norm((object1 - palm1) - (object0 - palm0)) > 0.1
    np.testing.assert_allclose(object_in_palm_frame(state(palm0, rotation0), "right", object0), relative)
    np.testing.assert_allclose(object_in_palm_frame(state(palm1, rotation1), "right", object1), relative)
