from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.streamed_arm import PlannerArm
from locomani.tasks.crate import bimanual_carry as carry
from locomani.tasks.crate.bimanual_ik import BimanualIK
from locomani.tasks.crate.carry_phases import CarryRoundContext, admit_grasp, stand_target


def test_carry_constructor_allows_forward_waist_pitch(monkeypatch):
    ik = scene_ik()
    planner = SimpleNamespace(ik=ik, q=np.zeros(29))
    monkeypatch.setattr(carry, "PlannerArm", lambda *args, **kwargs: planner)
    carry.BimanualArms(SimpleNamespace())
    assert ik.hi[14] > 0.3, "carry still locks forward waist pitch"


def test_carry_deep_targets_are_solved_together(monkeypatch):
    ik = scene_ik()
    planner = SimpleNamespace(ik=ik, q=np.zeros(29))
    monkeypatch.setattr(carry, "PlannerArm", lambda *args, **kwargs: planner)
    p = SimpleNamespace(world_to_root_full=lambda point, state: np.asarray(point))
    arms = carry.BimanualArms(p)
    state = {"joints": np.zeros(29), "pelvis": {"quat": [1, 0, 0, 0]}}
    rotation = Rotation.from_euler("y", 60, degrees=True).as_matrix()
    for depth in (0.44, 0.50):
        targets = {"left": [depth, 0.1933, -0.026], "right": [depth, -0.1933, -0.026]}
        joints = arms.solve(state, targets, dict(left=rotation, right=rotation), iters=220)
        poses, _ = BimanualIK(ik, waist_pitch_max=0.4).tool_poses(joints)
        assert np.max(np.linalg.norm(poses - list(targets.values()), axis=1)) < 0.006
        assert 0 < joints[14] <= 0.4
        assert max(arms.reach_errors(state, targets, dict(left=rotation, right=rotation))) < 0.006


def test_measured_waist_is_used_and_forward_pitch_reaches_planner(monkeypatch):
    ik = scene_ik()
    command = Mock(left_hand=[0, 0], right_hand=[0, 0])
    p = SimpleNamespace(cmd=command, world_to_root_full=lambda point, state: np.asarray(point))
    monkeypatch.setattr(carry, "PlannerArm", lambda *args, **kwargs: PlannerArm(*args, **kwargs, ik=ik))
    arms = carry.BimanualArms(p)
    arms.arm.q = np.zeros(29)
    measured = np.zeros(29)
    measured[12:15] = [0.1, -0.1, 0.3]
    state = {"joints": measured, "pelvis": {"quat": [1, 0, 0, 0]}}
    targets = {"left": [0.44, 0.19, 0], "right": [0.44, -0.19, 0]}
    rotations = dict(left=np.eye(3), right=np.eye(3))
    joints = arms.solve(state, targets, rotations, iters=0, from_measured=True)
    np.testing.assert_allclose(joints[12:15], [0, 0, 0.3])
    arms.arm._q_prev = joints.copy()
    arms.arm.next_t = 0
    arms.arm._face = [1, 0, 0]
    arms.push(joints)
    upper = command.set_upper_body.call_args.args[0]
    assert len(upper) == 17
    np.testing.assert_allclose(upper[:3], [0, 0, 0.3])


def test_carry_respects_caller_limits_and_cannot_reach_arbitrary_depth(monkeypatch):
    ik = scene_ik()
    planner = SimpleNamespace(ik=ik, q=np.zeros(29))
    monkeypatch.setattr(carry, "PlannerArm", lambda *args, **kwargs: planner)
    p = SimpleNamespace(world_to_root_full=lambda point, state: np.asarray(point))
    arms = carry.BimanualArms(p)
    # Wall applies its shoulder restriction after constructing the shared adapter.
    ik.lo[[15, 22]] = -0.2
    targets = {"left": [1.0, 0.19, 0], "right": [1.0, -0.19, 0]}
    state = {"joints": np.zeros(29), "pelvis": {"quat": [1, 0, 0, 0]}}
    joints = arms.solve(state, targets, dict(left=np.eye(3), right=np.eye(3)), iters=220)
    assert np.all(joints[[15, 22]] >= -0.2)
    assert 0 <= joints[14] <= carry.WAIST_PITCH_MAX
    actual, _ = arms.solver.tool_poses(joints)
    assert np.min(np.linalg.norm(actual - list(targets.values()), axis=1)) > 0.3
    assert min(arms.reach_errors(state, targets, dict(left=np.eye(3), right=np.eye(3)))) > 0.3


def test_deeper_crate_stand_target_stays_outside_the_table():
    normal, lateral = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    for centre_x in (0.17, 0.29, 0.5):
        target = stand_target([centre_x, 0, 0.7], normal, lateral, edge_x=0)
        assert target[0] <= -carry.EDGE_FLOOR
        assert target[0] == min(centre_x - carry.STAND_OFF, -carry.EDGE_FLOOR - carry.STAND_TOL)
        assert target[1] == carry.STAND_LAT_BIAS


@pytest.mark.parametrize(
    "depth,edge_clear,lateral,yaw,why",
    [
        (0.50, 0.17, 0, 0, None),
        (1.0, 0.17, 0, 0, "beyond bounded bimanual reach"),
        (0.50, 0.09, 0, 0, "inside the table-edge floor"),
        (0.50, 0.17, 0.05, 0, "lateral"),
        (0.50, 0.17, 0, 21, "outside the reach band"),
    ],
)
def test_far_grasp_requires_reachability_and_a_valid_stand(monkeypatch, depth, edge_clear, lateral, yaw, why):
    ik = scene_ik()
    planner = SimpleNamespace(ik=ik, q=np.zeros(29))
    monkeypatch.setattr(carry, "PlannerArm", lambda *args, **kwargs: planner)
    p = SimpleNamespace(world_to_root_full=lambda point, state: np.asarray(point))
    ctx = CarryRoundContext(
        p=p,
        reset=None,
        meta={},
        rnd=0,
        rng=np.random.default_rng(0),
        log=lambda message: None,
        video_label="reach_test",
        m=dict(edge_clear=edge_clear, lateral=lateral, yaw_err_deg=yaw, stand_err=carry.STAND_OFF - depth),
        arms=carry.BimanualArms(p),
        s={"joints": np.zeros(29), "pelvis": {"quat": [1, 0, 0, 0]}},
        bars={"left": [depth, 0.1933, -0.026], "right": [depth, -0.1933, -0.026]},
        psi=0,
    )
    result = admit_grasp(ctx)
    if why is None:
        assert result is None
        assert max(ctx.m["reach_ik_errors_m"]) < 0.006
    else:
        assert why in result["why"]
