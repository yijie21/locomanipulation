"""The real gym prelude must reach token-only deployment without expert IK."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
import grasp_gym as gym
from reach_config import ReachConfig


class ReadyVLAHandoverTests(unittest.TestCase):
    def run_prelude(self, policy, camera=(0.0, 0.0), token_file=None, token_ready=True):
        p, reset = Mock(), Mock()
        p._gym_started = False
        state = {"fallen": False, "pelvis": {"pos": [0.0, 0.0, 0.8]}, "objects": {"water_bottle": {"cam": camera}}}
        p.obs.return_value = state
        p.pelvis_xy.return_value = np.zeros(2)
        p.pelvis_yaw.return_value = 0.0
        p.object_pos.return_value = np.array([1.0, 0.0, 0.7])
        meta = {"objects": {"water_bottle": {"box": {}, "body": "bottle"}}, "surfaces": {"table": {"body": "table"}}}
        with (
            patch.object(gym, "VLA_POLICY", policy),
            patch.dict(os.environ, {"TF_INIT_TOKEN_FILE": token_file or ""}),
            patch.object(gym.time, "sleep"),
            patch.object(gym.upper_ref, "shared"),
            patch.object(gym, "obj_top", return_value=0.8),
            patch.object(gym, "TabletopGrasp") as target,
            patch("ready_grasp_policy.prepare_ready", return_value=False) as ready,
            patch("initial_pose.prepare_token_pose", return_value=token_ready) as token_pose,
        ):
            target.return_value.grip.return_value = [1.0, 0.0, 0.7]
            result = gym._run_round(
                p,
                reset,
                meta,
                "water_bottle",
                {"top": 0.6},
                0.0,
                [0.0, 0.0],
                [0.0, 0.0],
                0.0,
                "right",
                [0.3, -0.2],
                None,
                [0.0, 0.0],
                spawn_off=0.0,
                object_yaw=0.0,
                reach_config=ReachConfig(ready_grasp_baseline=True),
            )
        self.last_token_pose = token_pose
        return result, ready

    def test_vla_reaches_policy_without_encoder_only_ready_command(self):
        policy = Mock(return_value=(True, "token policy reached"))
        result, ready = self.run_prelude(policy)
        self.assertEqual(result, (True, "token policy reached"))
        ready.assert_not_called()
        policy.assert_called_once()

    def test_expert_still_requires_ready_posture(self):
        result, ready = self.run_prelude(None)
        self.assertEqual(result, (False, "[stand] baseline ready posture not delivered"))
        ready.assert_called_once()

    def test_vla_still_rejects_invisible_target_before_policy(self):
        policy = Mock(return_value=(True, "must not run"))
        result, ready = self.run_prelude(policy, camera=None)
        self.assertFalse(result[0])
        self.assertTrue(result[1].startswith("[view]"))
        ready.assert_not_called()
        policy.assert_not_called()

    def test_explicit_token_pose_runs_before_visibility_and_policy(self):
        policy = Mock(return_value=(True, "token policy reached"))
        result, ready = self.run_prelude(policy, token_file="explicit.json")
        self.assertTrue(result[0])
        ready.assert_not_called()
        self.last_token_pose.assert_called_once()
        policy.assert_called_once()

    def test_undelivered_token_pose_cannot_start_policy(self):
        policy = Mock()
        result, ready = self.run_prelude(policy, token_file="explicit.json", token_ready=False)
        self.assertEqual(result, (False, "[stand] token-only ready posture not delivered"))
        policy.assert_not_called()
        ready.assert_not_called()

    def test_explicit_token_pose_does_not_bypass_visibility_gate(self):
        policy = Mock()
        result, _ = self.run_prelude(policy, token_file="explicit.json", camera=None)
        self.assertFalse(result[0])
        self.assertTrue(result[1].startswith("[view]"))
        self.last_token_pose.assert_called_once()
        policy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
