"""WholeBodyIK's analytic Jacobian (jacobian="analytic", TODO 26.5.5): the rotation helpers against scipy, the
Jacobian against central differences of the same residual, and the solve against the numeric default."""

from unittest.mock import patch

import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.primitives import whole_body_ik
from locomani.primitives.whole_body_ik import WholeBodyIK, log_jacobian, rotvec
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig


def supported_state():
    q = np.zeros(29)
    q[[0, 6]], q[[3, 9]], q[[4, 10]] = -0.1, 0.2, -0.1
    q[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
    q[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    return dict(joints=q, pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))


def body(jacobian, **kw):
    return WholeBodyIK(
        scene_ik(),
        supported_state(),
        WholeBodyConfig(),
        tool_offset=[0.1515, 0, 0],
        hand_weights=(0.35, 1),
        com_weight=20,
        jacobian=jacobian,
        **kw,
    )


def targets(b):
    positions = b.tools(b.initial).copy()
    rotations = b.ik.data.xmat[b.hands].reshape(2, 3, 3).copy()
    positions[1] += [0.06, -0.015, -0.06]
    rotations[0] = Rotation.from_rotvec([0.1, -0.2, 0.05]).as_matrix() @ rotations[0]
    return positions, rotations


def test_rotvec_matches_scipy_from_zero_to_near_pi():
    rng = np.random.default_rng(0)
    for angle in (0.0, 1e-9, 1e-4, 0.3, 1.5, 2.9, 3.13, np.pi - 1e-6):
        axis = rng.normal(size=3)
        r = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis))
        np.testing.assert_allclose(rotvec(r.as_matrix()), r.as_rotvec(), atol=1e-9)


def test_log_jacobian_maps_a_small_world_rotation_to_the_rotvec_change():
    rng = np.random.default_rng(1)
    for angle in (0.0, 0.2, 1.4, 2.8):
        axis = rng.normal(size=3)
        r = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis)).as_matrix()
        phi, w, h = rotvec(r), rng.normal(size=3), 1e-6
        moved = Rotation.from_rotvec(h * w).as_matrix() @ r
        np.testing.assert_allclose((rotvec(moved) - phi) / h, log_jacobian(phi) @ w, atol=1e-5)


def captured_problem(b, **solve_kw):
    """The (residual, jacobian, x0) least_squares receives from `solve`."""
    seen = {}

    def fake(fun, x0, jac="2-point", **kw):
        seen.update(fun=fun, jac=jac, x0=x0)

        class Result:
            x = np.asarray(x0, dtype=float)

        return Result()

    with patch.object(whole_body_ik, "least_squares", fake):
        b.solve(*targets(b), **solve_kw)
    return seen


@pytest.mark.parametrize("free_radius", [0.0, 0.004])
def test_the_analytic_jacobian_matches_central_differences_of_the_residual(free_radius):
    b = body("analytic", com_free_radius=free_radius)
    rng = np.random.default_rng(2)
    x = np.clip(b.initial + rng.normal(scale=0.05, size=33), b.lo + 1e-3, b.hi - 1e-3)
    weights = np.full(29, 0.04)
    weights[12:14] = 0.12
    torso = Rotation.from_rotvec([0.0, 0.25, 0.1]).as_matrix()
    seen = captured_problem(
        b,
        posture=b.q0 + 0.1,
        posture_weights=weights,
        torso_rotation=torso,
        continuity_reference=x + 0.01,
        continuity_dt=0.02,
        continuity_weight=0.05,
    )
    fun, jac = seen["fun"], seen["jac"]
    assert callable(jac)
    h = 1e-6
    numeric = np.column_stack([(fun(x + h * e) - fun(x - h * e)) / (2 * h) for e in np.eye(33)])
    analytic = jac(x)
    assert analytic.shape == numeric.shape
    np.testing.assert_allclose(analytic, numeric, atol=2e-5 * max(1.0, np.abs(numeric).max()))


def test_the_numeric_default_hands_least_squares_its_own_differencing():
    seen = captured_problem(body("numeric"))
    assert seen["jac"] == "2-point"


def test_analytic_and_numeric_solves_land_on_the_same_pose():
    poses = {}
    for mode in ("numeric", "analytic"):
        b = body(mode)
        poses[mode] = b.solve(*targets(b), iterations=200)
    a, n = poses["analytic"], poses["numeric"]
    assert max(a.hand_errors) < 0.001 and max(a.foot_errors) < 0.0001
    np.testing.assert_allclose(a.joints, n.joints, atol=2e-3)
    np.testing.assert_allclose(a.position, n.position, atol=1e-3)


def test_an_unknown_jacobian_mode_is_refused():
    with pytest.raises(ValueError, match="jacobian"):
        body("exact")
