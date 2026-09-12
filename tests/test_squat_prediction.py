import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.tasks.bottle.coupled_grasp_policy import project_squat
from locomani.tasks.bottle.reach_config import ReachConfig


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


class PelvisPitchTests(unittest.TestCase):
    """The sign of the pitch the delivered-squat telemetry reports.

    `project_squat` predicts NO pitch, and the squat delivers +16.5 deg median
    (ledger 32.1) -- a number whose sign has to mean "leaning forward", because
    that is what makes the prediction optimistic rather than conservative.
    """

    def test_leaning_forward_is_positive(self):
        from locomani.tasks.bottle.coupled_grasp_policy import _pelvis_pitch_deg

        for deg in (5.0, 16.5, 30.0):
            # Rotation about +y tips the body x-axis DOWNWARD (nose-down).
            q = Rotation.from_euler("y", deg, degrees=True).as_quat()[[3, 0, 1, 2]]
            got = _pelvis_pitch_deg({"pelvis": {"quat": q.tolist()}})
            self.assertAlmostEqual(got, deg, places=6, msg=f"{deg} deg nose-down")
            q = Rotation.from_euler("y", -deg, degrees=True).as_quat()[[3, 0, 1, 2]]
            self.assertAlmostEqual(_pelvis_pitch_deg({"pelvis": {"quat": q.tolist()}}), -deg, places=6)

    def test_upright_and_pure_yaw_are_zero_pitch(self):
        from locomani.tasks.bottle.coupled_grasp_policy import _pelvis_pitch_deg

        self.assertAlmostEqual(_pelvis_pitch_deg({"pelvis": {"quat": [1.0, 0.0, 0.0, 0.0]}}), 0.0, places=9)
        q = Rotation.from_euler("z", 1.2).as_quat()[[3, 0, 1, 2]]
        self.assertAlmostEqual(_pelvis_pitch_deg({"pelvis": {"quat": q.tolist()}}), 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
