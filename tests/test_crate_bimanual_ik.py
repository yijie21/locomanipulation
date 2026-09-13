import numpy as np
from geometry_fixture import scene_ik

from locomani.tasks.crate.bimanual_ik import BimanualIK


def test_shared_waist_solves_both_targets_without_invalidating_the_first_hand():
    solver = BimanualIK(scene_ik(), waist_pitch_max=0.4)
    target_q = np.zeros(29)
    target_q[14] = 0.3
    target_q[15:22] = [-0.4, 0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
    target_q[22:29] = [-0.4, -0.2, 0.0, 0.8, 0.0, 0.0, 0.0]
    targets, rotations = solver.tool_poses(target_q)
    initial = target_q.copy()
    initial[14] = 0
    result = solver.solve(initial, targets, rotations, iterations=180)
    assert max(result.errors) < 0.003
    assert 0 < result.joints[14] <= 0.4
    np.testing.assert_array_equal(result.joints[:12], initial[:12])
    np.testing.assert_array_equal(result.joints[12:14], [0, 0])


def test_locked_waist_stays_locked_and_unreachable_targets_report_residuals():
    solver = BimanualIK(scene_ik(), waist_pitch_max=0.0)
    q = np.zeros(29)
    q[14] = 0.4
    result = solver.solve(q, [[2, 0.2, 0], [2, -0.2, 0]], [np.eye(3), np.eye(3)], iterations=80)
    assert result.joints[14] == 0
    assert min(result.errors) > 1
    assert np.all(result.joints[solver.indices] <= solver.hi[solver.indices])


def test_all_waist_axes_can_align_both_hands_with_arms_fixed():
    solver = BimanualIK(scene_ik(), waist_pitch_max=0.52, waist_dofs="all")
    target = np.zeros(29)
    target[12:15] = [0.2, -0.12, 0.4]
    target[15:22] = [-0.4, 0.2, 0, 0.8, 0, 0, 0]
    target[22:29] = [-0.4, -0.2, 0, 0.8, 0, 0, 0]
    positions, rotations = solver.tool_poses(target)
    initial = target.copy()
    initial[12:15] = 0
    lo, hi = solver.lo.copy(), solver.hi.copy()
    lo[15:29] = hi[15:29] = target[15:29]
    result = solver.solve(initial, positions, rotations, iterations=180, joint_limits=(lo, hi))
    assert max(result.errors) < 0.003
    np.testing.assert_allclose(result.joints[12:15], target[12:15], atol=0.01)
    physical = solver.ik.model.jnt_range[solver.ik.jids]
    assert np.all(result.joints[12:15] >= physical[12:15, 0])
    assert np.all(result.joints[12:15] <= physical[12:15, 1])
