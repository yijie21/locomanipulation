import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.scalebfm.action import decode_action, encode_action


class VisualActionTests(unittest.TestCase):
    def setUp(self):
        self.hand_rel = {"left": (np.array([0, 0.2, -0.1]), np.eye(3))}
        tau = np.arange(6) * 0.02
        self.targets = {
            "pelvis": (
                np.array([1.1, 2, 0.76]) + tau[:, None] * [0.4, 0.1, 0],
                Rotation.from_euler("z", 0.8 + tau * 0.2).as_quat(scalar_first=True),
            ),
            "right_wrist_yaw_link": (
                np.array([1.4, 1.8, 0.7]) + tau[:, None] * [0.1, -0.2, 0.05],
                (
                    Rotation.from_rotvec(tau[:, None] * [0.2, 0.3, -0.1]) * Rotation.from_euler("xyz", [0.1, -0.2, 0.7])
                ).as_quat(scalar_first=True),
            ),
        }
        self.root = np.array([1.0, 2.0, 0.75])
        self.quat = Rotation.from_euler("xyz", [0.15, -0.2, 0.8]).as_quat(scalar_first=True)

    def test_native_forecast_round_trip_without_future_root_states(self):
        row = encode_action(self.targets, self.root, self.quat, 0.0245)
        targets, grip = decode_action(row, self.root, self.quat, self.hand_rel)
        self.assertAlmostEqual(grip, 0.0245)
        for name, (p, q) in self.targets.items():
            np.testing.assert_allclose(targets[name][0], p, atol=1e-12)
            difference = (
                Rotation.from_quat(targets[name][1], scalar_first=True) * Rotation.from_quat(q, scalar_first=True).inv()
            )
            self.assertLess(difference.magnitude().max(), 1e-12)

    def test_pelvis_roll_pitch_do_not_accumulate_in_yaw_frame(self):
        row = encode_action(self.targets, self.root, self.quat, 0)
        new_quat = Rotation.from_euler("xyz", [-0.3, 0.4, 0.8]).as_quat(scalar_first=True)
        out, _ = decode_action(row, self.root, new_quat, self.hand_rel)
        root_angles = Rotation.from_quat(out["pelvis"][1], scalar_first=True).as_euler("xyz")
        np.testing.assert_allclose(root_angles[:, :2], 0, atol=1e-12)

    def test_global_yaw_translation_do_not_change_action(self):
        row = encode_action(self.targets, self.root, self.quat, 0)
        r = Rotation.from_euler("z", 1.5)
        shift = np.array([-2, 3, 0])
        transformed = {
            name: (r.apply(p) + shift, (r * Rotation.from_quat(q, scalar_first=True)).as_quat(scalar_first=True))
            for name, (p, q) in self.targets.items()
        }
        row2 = encode_action(
            transformed,
            r.apply(self.root) + shift,
            (r * Rotation.from_quat(self.quat, scalar_first=True)).as_quat(scalar_first=True),
            0,
        )
        np.testing.assert_allclose(row2, row, atol=1e-11)

    def test_velocity_survives_constant_body_relative_offset(self):
        first = encode_action(self.targets, self.root, self.quat, 0)
        self.assertGreater(np.linalg.norm(first[18:21]), 0.3)
        out, _ = decode_action(first, self.root, self.quat, self.hand_rel)
        self.assertGreater(np.linalg.norm(out["pelvis"][0][-1] - out["pelvis"][0][0]), 0.03)


if __name__ == "__main__":
    unittest.main()
