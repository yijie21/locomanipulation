import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.scalebfm.state import STATE_DIMS, ImageHistory, state_features


class StateTests(unittest.TestCase):
    def test_observation_invariant_to_world_translation_and_yaw(self):
        def build(r, shift):
            return state_features(
                np.arange(29),
                np.zeros(4),
                shift,
                r.as_quat(scalar_first=True),
                shift + r.apply([0.3, -0.2, -0.1]),
                r.as_quat(scalar_first=True),
                r.apply([0.4, 0, 0]),
                r.apply([0, 0, 0.2]),
                r.apply([0.1, 0, 0.1]),
                np.zeros(29),
            )

        a = build(Rotation.identity(), np.zeros(3))
        b = build(Rotation.from_euler("z", 1.2), np.array([100.0, -20, 3]))
        self.assertEqual(sum(STATE_DIMS.values()), 91)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=1e-7)

    def test_history_matches_dataset_minus_ten_frames(self):
        history = ImageHistory()
        for k in range(30):
            out = history.push(k / 25, {"head": np.full((2, 2, 3), k, np.uint8)})
            self.assertTrue(np.all(out["head"][0] == max(0, k - 10)))
            self.assertTrue(np.all(out["head"][1] == k))


if __name__ == "__main__":
    unittest.main()
