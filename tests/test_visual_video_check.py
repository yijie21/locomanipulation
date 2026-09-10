import sys
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from locomani.tasks.bottle_visual.pipeline.follow_video import camera_values, wrapped
from locomani.tasks.bottle_visual.pipeline.video_check import choose_cases
from locomani.tasks.bottle_visual.sampling import DEFAULT_PROFILE


class VideoCheckTests(unittest.TestCase):
    def test_selects_ten_distinct_reproducible_resets_without_policy(self):
        with patch(
            "locomani.tasks.bottle_visual.pipeline.video_check.B.Policy",
            side_effect=AssertionError("No model during selection"),
        ):
            a = choose_cases(DEFAULT_PROFILE, candidate_count=80)
            b = choose_cases(DEFAULT_PROFILE, candidate_count=80)
        self.assertEqual(a, b)
        self.assertEqual(len(a["cases"]), 10)
        self.assertEqual(len({c["seed"] for c in a["cases"]}), 10)
        self.assertEqual(Counter(c["stratum"] for c in a["cases"]), {"high_near": 4, "low_near": 3, "low_deep": 3})
        self.assertEqual({c["edge"] for c in a["cases"]}, {"y+"})
        self.assertEqual(a["physics_calls_during_selection"], 0)

    def test_camera_tracks_position_and_stays_in_front_during_turns(self):
        prior = None
        for yaw in np.linspace(-2 * np.pi, 2 * np.pi, 120):
            pelvis = np.array([yaw, yaw / 2, 0.76])
            prior = camera_values(pelvis, yaw, prior)
            look, view_yaw = prior
            self.assertLessEqual(np.max(abs(look[:2] - pelvis[:2])), 0.02500001)
            self.assertLessEqual(abs(wrapped(view_yaw - yaw)), np.radians(4) + 1e-9)
            self.assertGreater(np.cos(view_yaw - np.radians(80) - yaw), 0)


if __name__ == "__main__":
    unittest.main()
