import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
from geometry_fixture import scene_ik

from locomani.backends.sonic.primitives import upper_body_from_mj
from locomani.backends.sonic.streamed_arm import StreamedArm
from locomani.pipeline.execution_record import ExecutionRecord
from locomani.primitives.supported_motion import SupportedMotion
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle.policy import make_grasp_policy
from locomani.tasks.bottle.reach_config import ReachConfig
from locomani.tasks.bottle.whole_body_config import WholeBodyConfig


def test_whole_body_configuration_reaches_actual_factory_and_roundtrips():
    from locomani.tasks.bottle.whole_body_policy import WholeBodyGraspPolicy

    task = BottleTaskConfig.load("configs/tasks/bottle_whole_body.json")
    changed = task.with_overrides(["reach.whole_body.hand_speed=0.12"])
    assert changed.reach.whole_body.hand_speed == 0.12
    assert ReachConfig.from_dict(changed.reach.to_dict()) == changed.reach
    args = [object(), object(), changed.reach, object(), ExecutionRecord()]
    with patch.object(WholeBodyGraspPolicy, "__init__", return_value=None) as initialize:
        assert isinstance(make_grasp_policy(*args), WholeBodyGraspPolicy)
    initialize.assert_called_once_with(*args)


@pytest.mark.parametrize(
    "field,value",
    [("joint_speed", 2), ("waist_pitch_max", 1), ("vision_timeout", float("nan")), ("close_seconds", 0.1)],
)
def test_invalid_physical_limits_fail_before_execution(field, value):
    with pytest.raises(ValueError):
        replace(WholeBodyConfig(), **{field: value})


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
    np.testing.assert_array_equal(cmd.set_upper_body.call_args.args[0], upper_body_from_mj(q))
    names = [call[0] for call in cmd.method_calls]
    assert names.index("frame") < names.index("exit_streamed")
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


def test_short_fixed_case_bank_rejects_before_starting_stack(tmp_path, monkeypatch):
    from locomani.pipeline import run_learn_pilot as pilot
    from locomani.pipeline.experiment_cases import save_cases

    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scene_meta.json").write_text("{}")
    bank = tmp_path / "cases.json"
    save_cases(bank, [], scene / "scene_meta.json")
    start = Mock(side_effect=AssertionError("must reject before resource acquisition"))
    monkeypatch.setattr(pilot.stack, "start_stack", start)
    with pytest.raises(ValueError, match="fixed cases must cover"):
        pilot.main(["--scene", str(scene), "--out", str(tmp_path / "out"), "--cases", str(bank)])
    start.assert_not_called()


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
    assert policy.execute(grip_world=None, squat_hint=None, ghost=None) == (True, "lifted")
    policy.arm.stop.assert_called_once_with(preserve_reference=True)
    policy.close.assert_not_called()
    assert policy.p.cmd.reference_observer is policy.previous_observer
