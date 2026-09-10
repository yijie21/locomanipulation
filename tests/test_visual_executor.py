import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.scalebfm.executor import ActionExecutor


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.executor = ActionExecutor({"left": (np.array([0, 0.2, -0.1]), np.eye(3))})
        self.row = np.zeros(29)
        self.row[3:9] = self.row[12:18] = [1, 0, 0, 0, 1, 0]
        self.row[18] = 0.4
        self.q = np.array([1.0, 0, 0, 0])

    def test_one_row_covers_two_ticks_in_live_frame(self):
        a, _ = self.executor.step(np.zeros(3), self.q, self.row)
        self.assertFalse(self.executor.needs_action)
        b, _ = self.executor.step(np.array([0.008, 0, 0]), self.q)
        self.assertTrue(self.executor.needs_action)
        np.testing.assert_allclose(b["pelvis"][0] - a["pelvis"][0], np.tile([0.008, 0, 0], (6, 1)))
        self.assertEqual(self.executor.row.dtype, np.float32)

    def test_invalid_submission_does_not_advance_state(self):
        with self.assertRaises(ValueError):
            self.executor.step(np.zeros(3), self.q)
        self.assertEqual(self.executor.tick, 0)
        self.row[0] = np.nan
        with self.assertRaises(ValueError):
            self.executor.step(np.zeros(3), self.q, self.row)
        self.assertEqual(self.executor.tick, 0)

    def test_live_root_displacement_cannot_jump_wrist_command(self):
        a, _ = self.executor.step(np.zeros(3), self.q, self.row)
        b, _ = self.executor.step(np.array([0.10, 0.08, 0]), Rotation.from_euler("z", 0.8).as_quat(scalar_first=True))
        self.assertLessEqual(
            np.linalg.norm(b["right_wrist_yaw_link"][0][0] - a["right_wrist_yaw_link"][0][0]), 0.020000001
        )
        old = Rotation.from_quat(a["right_wrist_yaw_link"][1][0], scalar_first=True)
        new = Rotation.from_quat(b["right_wrist_yaw_link"][1][0], scalar_first=True)
        self.assertLessEqual((new * old.inv()).magnitude(), np.radians(5) + 1e-9)


if __name__ == "__main__":
    unittest.main()
