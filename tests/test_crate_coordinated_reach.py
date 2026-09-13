import numpy as np
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.coordinated_reach import CoordinatedReach
from locomani.tasks.crate.perception import CrateEstimate


def setup():
    config = CrateTaskConfig.load("configs/tasks/crate_visual_low_reach.json")
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
