import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from locomani.pipeline.experiment_cases import validate
from locomani.tasks.bottle.grasp_gym import sample_object_yaw


class FixedYawTests(unittest.TestCase):
    def test_fixed_yaw_does_not_draw_random_orientation(self):
        with patch("locomani.tasks.bottle.grasp_gym.np.random.uniform") as random:
            self.assertEqual(sample_object_yaw(0.0), 0.0)
            self.assertEqual(sample_object_yaw(-1.2), -1.2)
        random.assert_not_called()

    def test_old_callers_still_sample(self):
        with patch("locomani.tasks.bottle.grasp_gym.np.random.uniform", return_value=0.7) as random:
            self.assertEqual(sample_object_yaw(), 0.7)
        random.assert_called_once()

    def test_invalid_yaw_rejected(self):
        for yaw in (float("nan"), float("inf"), True, "0"):
            with self.assertRaises(ValueError):
                sample_object_yaw(yaw)
        c = dict(
            case_id="a",
            target="water_bottle",
            appearance=42,
            can_xy=[0.1, 0.2],
            stand=[0.3, 0.4],
            rel=[0.2, -0.2],
            psi=1.0,
            spawn_off=1.2,
            spawn_lat=0.01,
            dpsi=0.1,
            squat_hint=0.12,
        )
        self.assertEqual(validate([dict(c, object_yaw=0.0)])[0]["object_yaw"], 0.0)
        with self.assertRaises(ValueError):
            validate([dict(c, object_yaw=float("nan"))])
