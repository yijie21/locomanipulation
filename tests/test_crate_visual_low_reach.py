import numpy as np
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.perception import CrateEstimate
from locomani.tasks.crate.visual_low_reach import SupportFrame, VisualLowReach


def setup():
    config = CrateTaskConfig.load("configs/tasks/crate_visual_low_reach.json")
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.25, 0.5, -0.25
    q[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
    q[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    solver = BimanualIK(scene_ik(), waist_pitch_max=0.52, waist_dofs="all")
    policy = VisualLowReach(solver, {"grasp_points": [[0.1933, 0, 0.06], [-0.1933, 0, 0.06]]}, config, q, [1, 0, 0, 0])
    return policy, q


def visible(now, shift=0.0):
    return CrateEstimate(
        np.array([0.44, shift, -0.40]), Rotation.from_euler("z", 90, degrees=True).as_matrix(), now, 0, {}
    )


def test_bending_stops_on_occlusion_and_does_not_catch_up_after_reacquisition():
    policy, q = setup()
    first = policy.step(0.0, None, q, [1, 0, 0, 0], [0.09] * 2)
    assert first.phase == "observe"
    np.testing.assert_array_equal(first.joints, q)
    policy.step(0.05, visible(0.05), q, [1, 0, 0, 0], [0.09] * 2)
    moved = policy.step(0.1, visible(0.1), q, [1, 0, 0, 0], [0.09] * 2)
    assert moved.phase == "bend", moved.reason
    assert np.max(np.abs(moved.joints - q)) > 0
    paused = policy.step(0.2, None, q, [1, 0, 0, 0], [0.09] * 2)
    resumed = policy.step(0.5, visible(0.5), q, [1, 0, 0, 0], [0.09] * 2)
    np.testing.assert_array_equal(paused.joints, moved.joints)
    np.testing.assert_array_equal(resumed.joints, moved.joints)
    np.testing.assert_array_equal(resumed.root_quaternion, moved.root_quaternion)
    assert policy.step(1.3, None, q, [1, 0, 0, 0], [0.09] * 2).reason == "vision_lost"


def test_visible_placement_changes_low_reach_plan_with_identical_proprioception():
    targets = []
    for shift in (-0.025, 0.025):
        policy, q = setup()
        command = policy.step(0, visible(0, shift), q, [1, 0, 0, 0], [0.09] * 2)
        assert command.phase == "bend", command.reason
        targets.append(policy.posture)
    assert np.linalg.norm(targets[0] - targets[1]) > 0.05


def test_support_frame_recovers_a_rigid_base_rotation_from_foot_encoders_only():
    policy, q = setup()
    frame = SupportFrame(policy.solver.ik, q, [1, 0, 0, 0])
    origin, base = frame.observe(q, [1, 0, 0, 0])
    np.testing.assert_allclose(origin, frame.origin)
    np.testing.assert_allclose(frame.feet(q) @ base.T + origin, frame.anchors)


def test_lift_does_not_call_tracking_delay_an_empty_grasp_but_has_a_deadline():
    policy, q = setup()
    root = np.array([1, 0, 0, 0])
    object_position = policy.frame.origin + visible(0).position
    object_rotation = visible(0).rotation

    def observe(now):
        origin, base = policy.frame.observe(q, root)
        return CrateEstimate(base.T @ (object_position - origin), base.T @ object_rotation, now, 0, {})

    for now in np.arange(0, 20, 0.04):
        command = policy.step(now, observe(now), q, root, [0.016] * 2)
        q, root = command.joints, command.root_quaternion
        assert not command.finished, command.reason
        if command.phase == "lift":
            break
    assert command.phase == "lift"
    started = now
    # The decoder has not yet delivered the requested hand motion. Both jaws
    # still surround the visible bar, so an elapsed 1.5 s alone proves nothing.
    for now in np.arange(started + 0.04, started + 1.8, 0.04):
        command = policy.step(now, observe(now), q, root, [0.016] * 2)
        assert not command.finished, command.reason
    for now in np.arange(now + 0.04, started + 6, 0.04):
        command = policy.step(now, observe(now), q, root, [0.016] * 2)
        if command.finished:
            break
    assert command.finished and command.reason == "probe_not_attached"
