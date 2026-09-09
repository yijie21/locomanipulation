import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
from reach_config import ReachConfig
from ready_grasp_policy import terminal_rotation


class LevelBottleTests(unittest.TestCase):
    def test_crouch_pitch_does_not_pitch_entry_into_the_bottle_neck(self):
        root = Rotation.from_euler("zyx", [-1.5, 0.2, 0.1]).as_matrix()
        local = Rotation.from_euler("y", 0.15).as_matrix()
        level = terminal_rotation(root, local, SimpleNamespace(tool_point_grasp=True, horizontal_entry=True))
        self.assertAlmostEqual(level[2, 0], 0.0)
        self.assertAlmostEqual(level[2, 1], 0.0)
        self.assertAlmostEqual(np.linalg.det(level), 1.0)
        np.testing.assert_allclose(
            level[:, 0][:2] / np.linalg.norm(level[:, 0][:2]),
            (root @ local)[:, 0][:2] / np.linalg.norm((root @ local)[:, 0][:2]),
        )

    def test_previous_configs_keep_their_original_attitude(self):
        root = Rotation.from_euler("y", 0.2).as_matrix()
        local = Rotation.from_euler("x", 0.1).as_matrix()
        np.testing.assert_allclose(terminal_rotation(root, local, ReachConfig()), root @ local)
        with self.assertRaises(ValueError):
            ReachConfig(tool_point_grasp=True)
