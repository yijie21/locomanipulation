import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.coordinated_reach import CoordinatedReach
from locomani.tasks.crate.perception import CrateEstimate


def setup():
    config = CrateTaskConfig.load("configs/tasks/crate_visual_low_reach.json").with_overrides(
        ["visual.approach_speed=0.1"]
    )
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.1, 0.2, -0.1
    q[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
    q[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    state = dict(joints=q, pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    solver = BimanualIK(scene_ik(), waist_pitch_max=0.52, waist_dofs="all")
    policy = CoordinatedReach(solver, config, state, reference_joints=q, reference_quaternion=state["pelvis"]["quat"])
    return policy, state


def estimate(now, state):
    base = Rotation.from_quat(np.array(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
    return CrateEstimate(
        base.T @ (np.array([0.36, 0.02, 0.385]) - state["pelvis"]["pos"]),
        base.T @ Rotation.from_euler("z", 90, degrees=True).as_matrix(),
        now,
        None,
        {},
    )


def test_continuous_body_feedback_moves_hips_with_hands_without_breaking_support():
    policy, state = setup()
    first = policy.step(0, estimate(0, state), state, [0.08] * 2)
    assert not first.finished, first.reason
    np.testing.assert_array_equal(first.joints, state["joints"])
    previous = first.joints
    velocities, feet = [], []
    for now in np.arange(0.05, 7, 0.05):
        command = policy.step(now, estimate(now, state), state, [0.015] * 2)
        assert not command.finished, command.reason
        velocities.append((command.joints - previous) / 0.05)
        previous = command.joints.copy()
        feet.append(policy.diagnostics.get("foot_error", [0, 0]))
        state = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3].copy(), quat=command.root_quaternion))
    assert policy.phase in ("reach", "close", "lift")
    assert abs(command.joints[0] - first.joints[0]) > 0.1
    assert np.linalg.norm(command.joints[12:14]) > 0.001
    assert np.max(np.abs(velocities)) <= policy.config.visual.joint_speed + 1e-6
    assert np.max(np.abs(np.diff(velocities, axis=0))) / 0.05 <= policy.config.visual.joint_acceleration + 1e-5
    assert np.max(feet) < 0.005
    before = policy.x.copy()
    # A tracking lag requires a further whole-body correction, including the hips.
    state["pelvis"]["pos"] = state["pelvis"]["pos"] + [0, 0, 0.035]
    policy.step(7.05, estimate(7.05, state), state, [0.015] * 2)
    assert np.linalg.norm(policy.x[4:16] - before[4:16]) > 1e-4


def test_unseen_or_stale_target_holds_body_and_cannot_close():
    policy, state = setup()
    policy.step(0, estimate(0, state), state, [0.08] * 2)
    moved = policy.step(0.1, estimate(0.1, state), state, [0.08] * 2)
    paused = policy.step(0.2, None, state, [0.08] * 2)
    resumed = policy.step(0.3, estimate(0.3, state), state, [0.08] * 2)
    np.testing.assert_array_equal(paused.joints, moved.joints)
    np.testing.assert_array_equal(resumed.joints, moved.joints)
    assert paused.closure == 0
    assert policy.step(1.2, estimate(0.1, state), state, [0.08] * 2).reason == "vision_lost"


def test_irregular_observation_spacing_preserves_reference_acceleration():
    policy, state = setup()
    previous = policy.step(0, estimate(0, state), state, [0.08] * 2).joints
    intervals = np.array([0.024, 0.064, 0.032, 0.045] * 24)
    velocities = []
    for now, dt in zip(np.cumsum(intervals), intervals):
        command = policy.step(now, estimate(now, state), state, [0.08] * 2)
        velocities.append((command.joints - previous) / dt)
        previous = command.joints.copy()
        state = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3].copy(), quat=command.root_quaternion))
    acceleration = np.diff(velocities, axis=0) / ((intervals[1:] + intervals[:-1]) / 2)[:, None]
    assert np.max(np.abs(acceleration)) <= policy.config.visual.joint_acceleration + 1e-5


def test_closure_to_lift_does_not_apply_the_tracking_offset_twice():
    policy, state = setup()
    policy.step(0, estimate(0, state), state, [0.08] * 2)
    policy.x = policy.body.seed.copy()  # geometrically admitted contact posture
    policy.q = policy.x[4:].copy()
    policy.root = Rotation.from_matrix(policy.body.put(policy.x)).as_quat()[[3, 0, 1, 2]]
    state = dict(joints=policy.q, pelvis=dict(pos=policy.x[:3] + [0, 0, 0.03], quat=policy.root))
    policy.feedback[:] = [0, 0, -0.03]  # constant observed tracking lag, already in the held reference
    policy.closed_targets = policy.body.tools(policy.x)
    policy.phase, policy.elapsed = "close", 1.01
    policy.step(0.05, estimate(0.05, state), state, [0.015] * 2)
    assert policy.phase == "lift"
    start = policy.body.tools(policy.x).copy()
    for now in np.arange(0.10, 0.31, 0.05):
        command = policy.step(now, estimate(now, state), state, [0.015] * 2)
        state = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3] + [0, 0, 0.03], quat=command.root_quaternion))
    assert np.min(policy.body.tools(policy.x)[:, 2] - start[:, 2]) > -0.002


def test_dropped_basket_aborts_instead_of_waiting_motionless_until_global_timeout():
    policy, state = setup()
    policy.step(0, estimate(0, state), state, [0.08] * 2)
    tools = policy.solver.tool_poses(state["joints"])[0] + state["pelvis"]["pos"]
    policy.closed_object = np.array([0.36, 0.02, 0.385])
    policy.closed_relative = (tools - policy.closed_object) @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
    policy.phase = "hold"
    before = policy.q.copy()
    command = policy.step(0.05, estimate(0.05, state), state, [0.0001] * 2)
    assert command.finished and command.reason == "lost_attachment"
    np.testing.assert_array_equal(command.joints, before)


def test_delayed_camera_frames_do_not_shrink_the_velocity_measurement_interval():
    policy, state = setup()
    previous = policy.step(0, estimate(0, state), state, [0.08] * 2).joints
    intervals = np.array([0.04, 0.11, 0.06, 0.16, 0.04] * 8)
    velocities = []
    for now, dt in zip(np.cumsum(intervals), intervals):
        command = policy.step(now, estimate(now, state), state, [0.08] * 2)
        velocities.append((command.joints - previous) / dt)
        previous = command.joints.copy()
        state = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3].copy(), quat=command.root_quaternion))
    acceleration = np.diff(velocities, axis=0) / ((intervals[1:] + intervals[:-1]) / 2)[:, None]
    assert np.max(np.abs(acceleration)) <= policy.config.visual.joint_acceleration + 1e-5


def test_attached_lift_continues_to_actual_upright_stable_hold():
    policy, state = setup()
    obj = np.array([0.36, 0.02, 0.385])
    orientation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    phases = []
    for now in np.arange(0, 30, 0.05):
        base = Rotation.from_quat(np.array(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
        if policy.phase in ("lift", "stand", "hold"):
            tools = policy.solver.tool_poses(state["joints"])[0] @ base.T + state["pelvis"]["pos"]
            # Ideal attached load follows measured hands, not target commands.
            obj = policy.closed_object + np.mean(tools - policy.closed_tools, axis=0)
        seen = CrateEstimate(base.T @ (obj - state["pelvis"]["pos"]), base.T @ orientation, now, None, {})
        command = policy.step(now, seen, state, [0.015] * 2)
        phases.append(command.phase)
        if command.finished:
            break
        state = dict(joints=command.joints, pelvis=dict(pos=policy.x[:3].copy(), quat=command.root_quaternion))
    assert command.reason == "completed"
    assert "stand" in phases and phases.index("stand") > phases.index("lift")
    assert policy.diagnostics["observed_torso_lean_deg"] <= policy.config.visual.upright_tolerance_deg
    assert policy.diagnostics["observed_basket_speed"] < 0.025
    assert policy.ready_since is not None and now - policy.ready_since >= policy.config.visual.hold_seconds


def test_bent_actual_posture_cannot_complete_a_hold():
    policy, state = setup()
    policy.step(0, estimate(0, state), state, [0.015] * 2)
    policy.x = policy.body.seed.copy()
    policy.q = policy.x[4:].copy()
    policy.root = Rotation.from_matrix(policy.body.put(policy.x)).as_quat()[[3, 0, 1, 2]]
    state = dict(joints=policy.q, pelvis=dict(pos=policy.x[:3].copy(), quat=policy.root))
    tools = policy.body.tools(policy.x)
    obj = np.array([0.36, 0.02, 0.385])
    policy.closed_object = obj - [0, 0, 0.1]
    policy.closed_relative = (tools - obj) @ Rotation.from_euler("z", 90, degrees=True).as_matrix()
    policy.stand_targets = tools.copy()
    policy.enter("hold")
    for now in np.arange(0.1, 5, 0.1):
        command = policy.step(now, estimate(now, state), state, [0.015] * 2)
        assert command.reason != "completed"
        if command.finished:
            break
    assert command.reason == "unstable_hold"


@pytest.mark.parametrize("motion, expected", [("small_sway", "completed"), ("drift", "unstable_hold")])
def test_final_hold_accepts_bounded_sway_but_rejects_a_moving_load(motion, expected):
    policy, initial = setup()
    policy.step(0, estimate(0, initial), initial, [0.015] * 2)
    tools = policy.body.tools(policy.x)
    obj = tools.mean(axis=0) - [0, 0, 0.07]
    orientation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    policy.closed_object = obj - [0, 0, 0.1]
    policy.closed_relative = (tools - obj) @ orientation
    policy.stand_targets = tools.copy()
    policy.torso_target = np.eye(3)
    policy.enter("hold")
    for now in np.arange(0.05, 4.2, 0.05):
        shift = np.array([0.01 * np.sin(2 * np.pi * now) if motion == "small_sway" else 0.04 * now, 0, 0])
        state = dict(
            joints=initial["joints"], pelvis=dict(pos=np.array(initial["pelvis"]["pos"]) + shift, quat=[1, 0, 0, 0])
        )
        seen = CrateEstimate(obj + shift - state["pelvis"]["pos"], orientation, now, None, {})
        command = policy.step(now, seen, state, [0.015] * 2)
        if command.finished:
            break
    assert command.reason == expected


def test_slow_host_does_not_accelerate_physical_motion_or_weaken_camera_freshness():
    fast, state = setup()
    slow, _ = setup()
    for simulation_time in np.arange(0, 2, 0.05):
        a = fast.step(
            simulation_time, estimate(simulation_time, state), state, [0.015] * 2, motion_time=simulation_time
        )
        wall_time = 100 + 4 * simulation_time
        b = slow.step(wall_time, estimate(wall_time, state), state, [0.015] * 2, motion_time=simulation_time)
        np.testing.assert_allclose(a.joints, b.joints, atol=1e-9)
        np.testing.assert_allclose(a.velocity, b.velocity, atol=1e-9)
        assert a.phase == b.phase
        state = dict(joints=a.joints, pelvis=dict(pos=fast.x[:3].copy(), quat=a.root_quaternion))
    # The simulator barely advances, but an image stale in wall time still aborts.
    stale = slow.step(wall_time + 1, estimate(wall_time, state), state, [0.015] * 2, motion_time=simulation_time + 0.01)
    assert stale.finished and stale.reason == "vision_lost"
