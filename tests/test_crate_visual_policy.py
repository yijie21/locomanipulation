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
