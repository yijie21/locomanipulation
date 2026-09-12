import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.perception import CrateEstimate
from locomani.tasks.crate.visual_config import CrateVisualConfig
from locomani.tasks.crate.visual_policy import VisualCratePolicy

GRIPS = [[0.1933, 0, 0.06], [-0.1933, 0, 0.06]]


def estimate(now):
    return CrateEstimate(np.array([0.35, 0, -0.05]), Rotation.from_euler("z", 90, degrees=True).as_matrix(), now, 0, {})


def test_lost_visual_input_holds_then_exits_without_hidden_target_updates():
    policy = VisualCratePolicy(GRIPS, CrateVisualConfig())
    tools = np.array([[0.27, 0.2, 0.1], [0.27, -0.2, 0.1]])
    rotations = [np.eye(3), np.eye(3)]
    first = policy.step(0.0, estimate(0.0), tools, rotations, [0.09, 0.09])
    waiting = policy.step(0.1, None, tools, rotations, [0.09, 0.09])
    np.testing.assert_array_equal(first.positions, waiting.positions)
    assert waiting.reason == "waiting_for_vision"
    assert not waiting.finished
    stopped = policy.step(0.8, None, tools, rotations, [0.09, 0.09])
    assert stopped.finished and stopped.reason == "vision_lost"


def test_expired_estimate_does_not_count_as_a_new_visual_observation():
    policy = VisualCratePolicy(GRIPS, CrateVisualConfig())
    tools = np.array([[0.27, 0.2, 0.1], [0.27, -0.2, 0.1]])
    policy.step(0.0, estimate(0.0), tools, [np.eye(3)] * 2, [0.09, 0.09])
    stopped = policy.step(1.0, estimate(0.0), tools, [np.eye(3)] * 2, [0.09, 0.09])
    assert stopped.reason == "vision_lost"


def test_visual_limits_validate_before_starting_resources():
    from locomani.tasks.crate.config import CrateTaskConfig

    with pytest.raises(ValueError, match="waist_pitch_max"):
        CrateTaskConfig.from_dict({"visual": {"waist_pitch_max": 0.8}})
    with pytest.raises(ValueError, match="probe_height"):
        CrateTaskConfig.from_dict({"visual": {"probe_height": 0.2}})
    with pytest.raises(ValueError, match="pregrasp_above"):
        CrateTaskConfig.from_dict({"visual": {"pregrasp_above": 0}})
    config = CrateTaskConfig().with_overrides(["visual.waist_pitch_max=0"])
    assert config.visual.waist_pitch_max == 0


def test_visible_displacement_changes_both_hand_references():
    tools = np.array([[0.27, 0.2, 0.1], [0.27, -0.2, 0.1]])
    references = []
    for lateral_shift in (-0.02, 0.02):
        policy = VisualCratePolicy(GRIPS, CrateVisualConfig())
        visible = estimate(0.0)
        visible.position[1] += lateral_shift
        for now in np.arange(0, 0.3, 0.02):
            visible = CrateEstimate(visible.position, visible.rotation, now, 0, {})
            command = policy.step(now, visible, tools, [np.eye(3)] * 2, [0.09, 0.09])
        references.append(command.positions)
    assert np.all(references[1][:, 1] > references[0][:, 1])


def test_completed_control_can_fail_independent_truth_evaluation():
    from locomani.tasks.crate.evaluation import evaluate

    initial = {"objects": {"blue_crate": {"pos": [0, 0, 0]}}}
    final = {
        "objects": {"blue_crate": {"pos": [0, 0, 0.1], "quat": [1, 0, 0, 0], "touching": ["left"]}},
        "fallen": False,
    }
    result = evaluate(initial, final, "completed", lift_height=0.1, max_tilt_deg=15)
    assert not result["ok"] and result["why"] == "evaluation_missing_contact"
    final["objects"]["blue_crate"]["touching"].append("right")
    assert evaluate(initial, final, "completed", lift_height=0.1, max_tilt_deg=15)["ok"]


def closed_policy():
    """Reach the closed state through observable perfect tracking, without simulator truth."""
    policy = VisualCratePolicy(GRIPS, CrateVisualConfig())
    tools = np.array([[0.27, 0.2, 0.1], [0.27, -0.2, 0.1]])
    for now in np.arange(0, 10, 0.05):
        command = policy.step(now, estimate(now), tools, [np.eye(3)] * 2, [0.02, 0.02])
        tools = command.positions.copy()
        if command.phase == "probe":
            return policy, float(now), tools
    raise AssertionError("did not finish observable closure")


def test_loaded_motion_waits_for_lagging_hand():
    policy, now, tools = closed_policy()
    tools[0, 2] -= 0.1
    previous = policy.reference.copy()
    command = policy.step(now + 0.1, estimate(now + 0.1), tools, [np.eye(3)] * 2, [0.02, 0.02])
    np.testing.assert_array_equal(command.positions, previous)


def test_occlusion_during_closure_does_not_cause_catchup_jump():
    policy = VisualCratePolicy(GRIPS, CrateVisualConfig())
    tools = np.array([[0.27, 0.2, 0.1], [0.27, -0.2, 0.1]])
    for now in np.arange(0, 10, 0.05):
        command = policy.step(now, estimate(now), tools, [np.eye(3)] * 2, [0.02, 0.02])
        tools = command.positions
        if 0.1 < command.closure < 0.9:
            break
    before = command.closure
    policy.step(now + 0.2, None, tools, [np.eye(3)] * 2, [0.02, 0.02])
    command = policy.step(now + 0.5, estimate(now + 0.5), tools, [np.eye(3)] * 2, [0.02, 0.02])
    assert command.closure == before
    command = policy.step(now + 0.55, estimate(now + 0.55), tools, [np.eye(3)] * 2, [0.02, 0.02])
    assert command.closure - before == pytest.approx(0.05 / policy.config.close_seconds)


def test_hold_requires_uninterrupted_tracking_and_attachment():
    policy, started, tools = closed_policy()
    now = started
    for now in np.arange(started + 0.05, started + 8, 0.05):
        visible = estimate(now)
        visible.position[2] += np.mean(tools[:, 2] - policy.closed_grips[:, 2])
        command = policy.step(now, visible, tools, [np.eye(3)] * 2, [0.02, 0.02])
        tools = command.positions.copy()
        if command.phase == "hold":
            break
    assert command.phase == "hold"
    # A persistent one-hand lag cannot be accepted by an elapsed hold timer.
    bad_tools = tools.copy()
    bad_tools[0, 2] -= 0.1
    for stamp in np.arange(now + 0.05, now + 3, 0.05):
        visible = estimate(stamp)
        visible.position[2] += policy.config.lift_height
        command = policy.step(stamp, visible, bad_tools, [np.eye(3)] * 2, [0.02, 0.02])
        assert not command.finished
    good_tools = policy.closed_grips + [0, 0, policy.config.lift_height]
    resumed = stamp
    for stamp in np.arange(resumed + 0.05, resumed + 2.3, 0.05):
        visible = estimate(stamp)
        visible.position[2] += policy.config.lift_height
        command = policy.step(stamp, visible, good_tools, [np.eye(3)] * 2, [0.02, 0.02])
        if stamp < resumed + 2:
            assert not command.finished
    assert command.finished and command.reason == "completed"
