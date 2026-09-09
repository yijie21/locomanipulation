import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
from coupled_grasp_policy import project_squat
from reach_config import ReachConfig


class SquatPredictionTests(unittest.TestCase):
    def test_prediction_accounts_for_measured_retreat_without_changing_input(self):
        c = ReachConfig.load("configs/collection/ready_grasp_measured_prediction.json")
        rot = Rotation.from_euler("z", 1.2)
        s = {"pelvis": {"pos": [1.0, 2.0, 0.8], "quat": rot.as_quat()[[3, 0, 1, 2]].tolist()}}
        original = deepcopy(s)
        pred = project_squat(s, 0.115, c)
        np.testing.assert_allclose(
            np.array(s["pelvis"]["pos"]) - pred["pelvis"]["pos"],
            [*0.08 * rot.apply([1.0, 0.0, 0.0])[:2], 0.1],
            atol=1e-12,
        )
        self.assertEqual(s, original)

    def test_original_config_keeps_pure_vertical_prediction(self):
        s = {"pelvis": {"pos": [1.0, 2.0, 0.8], "quat": [1.0, 0.0, 0.0, 0.0]}}
        np.testing.assert_allclose(project_squat(s, 0.1, ReachConfig())["pelvis"]["pos"], [1.0, 2.0, 0.7])
