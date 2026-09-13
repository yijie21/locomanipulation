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

    from locomani.pipeline.run_crate_pilot import parse_args, prepare

    args = parse_args(
        [
            "--expert",
            "carry",
            "--record",
            "--config",
            "configs/tasks/crate_low_reach.json",
            "--scene",
            str(tmp_path / "missing_scene"),
            "--out",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(SystemExit, match="full-pose VLA action export"):
        prepare(args)


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
