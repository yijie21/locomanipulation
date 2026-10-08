import numpy as np
import pytest

from locomani.backends.sonic.sonic_token import expand_hands
from locomani.pipeline.audit_bottle_actions import action_contract, runs


def test_recorded_actions_roundtrip_through_the_deployment_layout():
    tokens = np.tile(np.arange(-16, 16) / 16, (3, 2))
    hands = np.asarray([expand_hands(g) for g in (0.0245, 0.01, -0.02)])
    result = action_contract(tokens, hands[:, 0], hands[:, 1])
    assert result["passed"]
    assert (result["model_state_width"], result["model_action_width"]) == (46, 65)
    assert result["cameras"] == ["ego_view", "right_wrist"]
    hands[1, 1, 1] = 0  # A second independent jaw command cannot fit the 65-D policy.
    assert not action_contract(tokens, hands[:, 0], hands[:, 1])["passed"]


def test_action_audit_rejects_wrong_width_and_reports_off_grid_tokens():
    hands = np.asarray([expand_hands(0.01)])
    with pytest.raises(ValueError):
        action_contract(np.zeros((1, 65)), hands[:, 0], hands[:, 1])
    assert not action_contract(np.full((1, 64), 0.01), hands[:, 0], hands[:, 1])["passed"]
    assert runs([True, True, False, True], 0.02) == [0.04, 0.02]


def test_consumed_hand_labels_replace_stale_pose_subscriber_values():
    from locomani.pipeline.audit_bottle_actions import consumed_hand_contract
    from locomani.pipeline.recorded_hand_actions import install

    left, right = expand_hands(-0.02)

    class Collector:
        latest_proprio_msg = dict(last_left_hand_action=left, last_right_hand_action=right)

        def _add_sonic_pose_features(self, frame):
            frame["teleop.right_hand_joints"] = np.full(7, 0.0245)
            frame["teleop.stream_mode"] = 1
            return 7.0

    install(Collector)
    frame = {}
    assert Collector()._add_sonic_pose_features(frame) == 7.0
    assert frame["teleop.stream_mode"] == 1
    np.testing.assert_allclose(frame["teleop.right_hand_joints"], right)
    receipts = [dict(consumed_hand_action=dict(left=left, right=right))]
    assert consumed_hand_contract(np.array([left]), np.array([right]), receipts)["verified"]
    wrong = np.array([right]).copy()
    wrong[0, :2] = 0.0245
    assert not consumed_hand_contract(np.array([left]), wrong, receipts)["verified"]
    assert not consumed_hand_contract(np.array([left]), wrong, [{}])["verified"]


@pytest.mark.parametrize("invalid", [None, [0] * 6, [float("nan")] * 7])
def test_missing_or_invalid_consumed_commands_do_not_silently_fabricate_labels(invalid):
    from locomani.pipeline.recorded_hand_actions import consumed_hands

    state = dict(last_left_hand_action=[0] * 7)
    if invalid is not None:
        state["last_right_hand_action"] = invalid
    with pytest.raises(ValueError):
        consumed_hands(state)
