import numpy as np
from geometry_fixture import scene_ik
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.carry_config import CrateCarryConfig
from locomani.tasks.crate.whole_body_ik import plan_low_reach


def test_supported_low_reach_uses_hip_hinge_without_deep_knee_flexion():
    ik = scene_ik()
    q = np.zeros(29)
    q[[0, 6]] = -0.1
    q[[3, 9]] = 0.2
    q[[4, 10]] = -0.1
    q[15:22] = [-0.35, 0.2, 0, 0.8, 0, 0, 0]
    q[22:29] = [-0.35, -0.2, 0, 0.8, 0, 0, 0]
    state = dict(joints=q, pelvis=dict(pos=[-0.69, 0, 0.78], quat=[1, 0, 0, 0]))
    rotation = Rotation.from_euler("y", 60, degrees=True).as_matrix()
    cfg = CrateCarryConfig(reach_control="whole_body", waist_pitch_max=0.52)
    pose = plan_low_reach(
        ik, state, [[-0.35, 0.21, 0.445], [-0.35, -0.17, 0.445]], [rotation, rotation], cfg, tool_offset=[0.1515, 0, 0]
    )
    assert pose.reachable
    assert pose.position[2] >= cfg.pelvis_height_min
    assert Rotation.from_matrix(pose.rotation).as_euler("xyz")[1] > 0.15
    assert np.max(pose.joints[[3, 9]]) <= 0.85
    assert np.max(pose.foot_errors) < 0.005
    physical = ik.model.jnt_range[ik.jids]
    assert np.all(pose.joints >= physical[:, 0])
    assert np.all(pose.joints <= physical[:, 1])
    assert np.linalg.norm(pose.joints[12:14]) > 0.001


def test_unreachable_low_object_is_rejected_by_geometry():
    ik = scene_ik()
    state = dict(joints=np.zeros(29), pelvis=dict(pos=[0, 0, 0.78], quat=[1, 0, 0, 0]))
    pose = plan_low_reach(
        ik,
        state,
        [[2, 0.2, 0.3], [2, -0.2, 0.3]],
        [np.eye(3), np.eye(3)],
        CrateCarryConfig(reach_control="whole_body"),
        tool_offset=[0.1515, 0, 0],
    )
    assert not pose.reachable
    assert min(pose.hand_errors) > 0.5


def test_low_reach_profile_delivers_physical_waist_freedom_and_rejects_mixed_controls(tmp_path):
    import pytest

    from locomani.tasks.crate.config import CrateTaskConfig

    cfg = CrateTaskConfig.load("configs/tasks/crate_low_reach.json")
    path = tmp_path / "task.json"
    cfg.dump(path)
    restored = CrateTaskConfig.load(path)
    assert restored.carry.reach_control == "whole_body"
    assert restored.carry.waist_dofs == "all"
    assert restored.carry.pelvis_height is None
    with pytest.raises(ValueError, match="do not also request"):
        restored.with_overrides(["carry.pelvis_height=0.53"])


def test_whole_body_recording_is_rejected_before_scene_or_stack_use(tmp_path):
    import pytest

    from locomani.tasks.crate.config import CrateTaskConfig
    from locomani.tasks.crate.prepare import prepare_scene

    cfg = CrateTaskConfig.load("configs/tasks/crate_low_reach.json")
    with pytest.raises(SystemExit, match="full-pose VLA action export"):
        prepare_scene(tmp_path / "missing_scene", tmp_path / "out", cfg, "carry", record=True)
    assert not (tmp_path / "out").exists()


def test_full_body_adapter_delivers_config_to_both_control_layers(monkeypatch):
    from types import SimpleNamespace

    from locomani.backends.sonic.streamed_arm import PlannerArm
    from locomani.tasks.crate import bimanual_carry
    from locomani.tasks.crate.carry_whole_body import WholeBodyArms

    ik = scene_ik()
    p = SimpleNamespace(cmd=SimpleNamespace(left_hand=np.zeros(7), right_hand=np.zeros(7)))
    monkeypatch.setattr(bimanual_carry, "PlannerArm", lambda *args, **kwargs: PlannerArm(*args, **kwargs, ik=ik))
    cfg = CrateCarryConfig(reach_control="whole_body", waist_pitch_max=0.52)
    arms = WholeBodyArms(p, cfg)
    assert arms.config == cfg
    assert set([12, 13, 14]).issubset(arms.solver.indices)
    np.testing.assert_allclose(arms.arm.ik.lo[12:15], arms.solver.lo[12:15])
    np.testing.assert_allclose(arms.arm.ik.hi[12:15], arms.solver.hi[12:15])


def test_free_waist_admission_uses_reachability_instead_of_locked_waist_lateral_heuristic(monkeypatch):
    from types import SimpleNamespace

    from locomani.tasks.crate import bimanual_carry as carry
    from locomani.tasks.crate.carry_phases import admit_grasp

    ik = scene_ik()
    monkeypatch.setattr(carry, "PlannerArm", lambda *args, **kwargs: SimpleNamespace(ik=ik, q=np.zeros(29)))
    p = SimpleNamespace(world_to_root_full=lambda point, state: np.asarray(point))
    config = CrateCarryConfig(reach_control="whole_body", waist_pitch_max=0.52)
    arms = carry.BimanualArms(p, config=config)
    for depth, expected in [(0.44, True), (2.0, False)]:
        ctx = SimpleNamespace(
            p=p,
            arms=arms,
            config=config,
            psi=0,
            s=dict(joints=np.zeros(29), pelvis=dict(quat=[1, 0, 0, 0])),
            m=dict(edge_clear=0.2, yaw_err_deg=0, stand_err=-0.01, lateral=0.06),
            bars=dict(left=[depth, 0.13, 0], right=[depth, -0.25, 0]),
            log=lambda message: None,
        )
        result = admit_grasp(ctx)
        assert (result is None) == expected
        assert "reach_ik_errors_m" in ctx.m


def test_low_reach_keeps_grasp_attitude_across_body_and_arm_phases():
    from types import SimpleNamespace

    from locomani.tasks.crate.carry_phases import pitched

    ctx = SimpleNamespace(config=CrateCarryConfig(reach_control="whole_body"), R_stand=np.eye(3))
    np.testing.assert_allclose(pitched(ctx, "left"), Rotation.from_euler("y", 60, degrees=True).as_matrix())


def test_body_reference_arrival_cannot_admit_a_still_moving_robot(monkeypatch):
    from types import SimpleNamespace

    from locomani.tasks.crate import carry_whole_body as body
    from locomani.tasks.crate.whole_body_ik import WholeBodyPose

    now = [0.0]
    monkeypatch.setattr(body, "time", SimpleNamespace(time=lambda: now[0]))

    def state():
        return dict(
            joints=np.zeros(29),
            seq=int(now[0] * 50),
            wall=now[0],
            fallen=False,
            pelvis=dict(pos=[now[0] * 0.1, 0, 0.78], quat=[1, 0, 0, 0]),
        )

    p = SimpleNamespace(obs=state)
    arm = SimpleNamespace(ik=None, q=np.zeros(29), root_quat=[1, 0, 0, 0])

    def push(q, **kwargs):
        now[0] += 0.02

    arms = SimpleNamespace(arm=arm, solve=lambda *a, **kw: np.zeros(29), start=lambda: None, push=push)
    monkeypatch.setattr(body, "WholeBodyArms", lambda *a: arms)
    monkeypatch.setattr(
        body,
        "plan_low_reach",
        lambda *a, **kw: WholeBodyPose(
            np.zeros(29), np.array([0, 0, 0.78]), np.eye(3), np.zeros(2), np.zeros(2), np.zeros(2)
        ),
    )
    ctx = SimpleNamespace(
        p=p,
        config=CrateCarryConfig(reach_control="whole_body"),
        psi=0,
        m={},
        bars=dict(left=np.array([0.4, 0.2, 0.45]), right=np.array([0.4, -0.2, 0.45])),
        n=np.array([1, 0]),
        motion_recording=None,
    )
    result = body._phase_whole_body(ctx)
    assert not result["body_stance_ready"]
    assert "did not settle" in result["why"]


def test_lift_keeps_capture_command_and_bilateral_geometry():
    from types import SimpleNamespace

    from locomani.tasks.crate.bimanual_ik import BimanualIK
    from locomani.tasks.crate.carry_whole_body import PairedLiftReference

    ik = scene_ik()
    solver = BimanualIK(ik, waist_pitch_max=0.52, waist_dofs="all")
    q = np.zeros(29)
    q[12:15] = [0.04, -0.03, 0.5]
    q[15:22] = [-1.4, 0.2, 0, 1.2, 0, 0.15, 0]
    q[22:29] = [-1.4, -0.2, 0, 1.2, 0, 0.15, 0]
    arms = SimpleNamespace(arm=SimpleNamespace(q=q, ik=ik), solver=solver)
    # Physical limits here, including all waist axes, match the real adapter.
    ik.lo[12:15], ik.hi[12:15] = solver.lo[12:15], solver.hi[12:15]
    rotation = Rotation.from_euler("y", 33, degrees=True)
    state = dict(pelvis=dict(quat=rotation.as_quat()[[3, 0, 1, 2]]))
    cfg = CrateCarryConfig(reach_control="whole_body", waist_pitch_max=0.52)
    lift = PairedLiftReference(arms, state, 0.1, cfg)
    np.testing.assert_array_equal(lift.sample(0), q)
    before, before_r = solver.tool_poses(q)
    after, after_r = solver.tool_poses(lift.sample(lift.duration))
    np.testing.assert_allclose((after - before) @ rotation.as_matrix().T, [[0, 0, 0.1]] * 2, atol=0.001)
    np.testing.assert_allclose(after_r, before_r, atol=0.002)
    np.testing.assert_array_equal(lift.target[:12], q[:12])
    times = np.linspace(0, lift.duration, 301)
    velocity = np.diff([lift.sample(t) for t in times], axis=0) / (times[1] - times[0])
    assert np.max(np.abs(velocity)) <= cfg.joint_speed
    assert np.max(np.abs(np.diff(velocity, axis=0))) / (times[1] - times[0]) <= cfg.joint_acceleration


def test_fixed_posture_preference_reduces_redundant_arm_twist_without_locking_waist():
    from locomani.tasks.crate.bimanual_ik import BimanualIK

    solver = BimanualIK(scene_ik(), waist_pitch_max=0.52, waist_dofs="all")
    preferred = np.zeros(29)
    preferred[12:15] = [0.06, -0.08, 0.48]
    preferred[15:22] = [-1.3, 0.3, 0.08, 1.1, -0.12, 0.2, -0.15]
    preferred[22:29] = [-1.3, -0.3, -0.08, 1.1, 0.12, 0.2, 0.15]
    positions, rotations = solver.tool_poses(preferred)
    seed = preferred.copy()
    seed[[17, 19, 24, 26]] += [0.6, 0.6, -0.6, -0.6]
    initial = solver.solve(seed, positions, rotations, iterations=200, rot_weight=0.25).joints
    q = initial.copy()
    for _ in range(80):
        solution = solver.solve(q, positions, rotations, iterations=12, posture=preferred, rot_weight=0.25)
        q = solution.joints
    assert max(solution.errors) < 0.001
    assert np.linalg.norm(q - preferred) < 0.6 * np.linalg.norm(initial - preferred)
    assert set([12, 13, 14]).issubset(solver.indices)
