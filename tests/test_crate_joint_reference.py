import numpy as np
from geometry_fixture import scene_ik

from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.joint_reference import JointReference


def test_target_reversal_is_bounded_and_reference_respects_robot_limits():
    solver = BimanualIK(scene_ik(), waist_pitch_max=0.4)
    q = (solver.lo + solver.hi) / 2
    reference = JointReference(q, solver, speed=0.7, acceleration=2.0)
    previous, velocity = q.copy(), np.zeros(29)
    for tick in range(150):
        target = q.copy()
        target[15] += 0.6 if tick < 50 else -0.6
        current = reference.step(target, 0.02)
        current_velocity = (current - previous) / 0.02
        assert abs(current_velocity[15]) <= 0.7 + 1e-9
        assert abs(current_velocity[15] - velocity[15]) <= 2 * 0.02 + 1e-9
        assert np.all(current[solver.indices] >= solver.lo[solver.indices])
        assert np.all(current[solver.indices] <= solver.hi[solver.indices])
        previous, velocity = current, current_velocity
    assert current[15] < q[15]


def test_acceleration_bound_uses_the_actual_irregular_camera_intervals():
    solver = BimanualIK(scene_ik(), waist_pitch_max=0.4)
    q = (solver.lo + solver.hi) / 2
    reference = JointReference(q, solver, speed=0.7, acceleration=2.0)
    previous, velocity, last_dt = q.copy(), 0.0, None
    for tick in range(90):
        dt = (0.02, 0.05, 0.2, 0.03, 0.3)[tick % 5]
        target = q.copy()
        target[15] += 0.6 if tick < 30 else -0.6
        current = reference.step(target, dt)
        rate = (current[15] - previous[15]) / dt
        interval = dt if last_dt is None else (dt + last_dt) / 2
        assert abs(rate - velocity) <= 2 * interval + 1e-9
        previous, velocity, last_dt = current, rate, dt
