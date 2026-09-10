import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from locomani.tasks.bottle_visual.sampling import (
    DEFAULT_PROFILE,
    LEGACY_PROFILE,
    profile_edges,
    reset_sampler,
    sample_reset,
    validate_edges,
    validate_reuse,
)


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(
            STAND_D=(0.375, 0.375),
            STAND_BACKSTEP=0.06,
            STAND_LEFT=0.15,
            SPAWN_EDGE=(1.1, 1.3),
            SPAWN_LAT=0.2,
            SPAWN_YAW_ERR=np.radians(20),
        )
        # An off-origin table and an off-centre old bottle expose accidental
        # reliance on world zero or the old nominal bottle coordinates.
        self.sim = SimpleNamespace(
            crate0=np.array([1.84, -2.615, 0.843]),
            smeta={"surfaces": {"A": {"xy": [2.0, -3.0], "flat_half_extent": [0.504, 0.504]}}},
        )

    def test_all_new_resets_stay_in_yplus_middle_support(self):
        for stratum in ("high_near", "low_near", "low_deep"):
            for seed in range(300):
                bottle, _, spawn, yaw, meta = sample_reset(
                    self.runtime, self.sim, np.random.default_rng(seed), "y+", stratum, DEFAULT_PROFILE
                )
                self.assertLessEqual(abs(bottle[0] - 2.0), 0.08)
                depth = -3.0 + 0.504 - bottle[1]
                lo, hi = (0.18, 0.24) if stratum == "low_deep" else (0.12, 0.18)
                self.assertGreaterEqual(depth, lo)
                self.assertLessEqual(depth, hi)
                self.assertGreaterEqual(spawn[1] - (-3.0 + 0.504), 1.1)
                self.assertLessEqual(spawn[1] - (-3.0 + 0.504), 1.3)
                self.assertLessEqual(abs(spawn[0] - bottle[0]), 0.2)
                toward = bottle[:2] - spawn
                error = np.arctan2(
                    np.sin(yaw - np.arctan2(toward[1], toward[0])), np.cos(yaw - np.arctan2(toward[1], toward[0]))
                )
                self.assertLessEqual(abs(error), np.radians(20))
                self.assertEqual(bottle[2], self.sim.crate0[2])
                self.assertEqual(meta["edge"], "y+")
                self.assertEqual(meta["scene_profile"], DEFAULT_PROFILE)
        np.testing.assert_array_equal(self.sim.crate0, [1.84, -2.615, 0.843])

    def test_other_edges_are_rejected_and_default_is_yplus(self):
        self.assertEqual(profile_edges(DEFAULT_PROFILE), ("y+",))
        for edge in ("x-", "x+", "y-"):
            with self.assertRaises(ValueError):
                sample_reset(self.runtime, self.sim, np.random.default_rng(1), edge, "low_near", DEFAULT_PROFILE)
        with self.assertRaises(ValueError):
            validate_edges(DEFAULT_PROFILE, [])

    def test_collection_and_frozen_baseline_resets_match_and_restore_on_error(self):
        original = lambda sim, rng: "old reset"
        self.runtime.sample_scene = original
        expected = sample_reset(self.runtime, self.sim, np.random.default_rng(4001), "y+", "low_near", DEFAULT_PROFILE)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with reset_sampler(self.runtime, "y+", "low_near", DEFAULT_PROFILE):
                actual = self.runtime.sample_scene(self.sim, np.random.default_rng(4001))
                for a, b in zip(actual[:3], expected[:3]):
                    np.testing.assert_array_equal(a, b)
                self.assertEqual(actual[3:], expected[3:])
                raise RuntimeError("interrupted")
        self.assertIs(self.runtime.sample_scene, original)

    def test_legacy_delegates_without_consuming_extra_rng(self):
        self.runtime.sample_scene = lambda sim, rng: rng.random(6)
        expected = np.random.default_rng(17).random(6)
        actual = sample_reset(self.runtime, self.sim, np.random.default_rng(17), "x-", "high_near", LEGACY_PROFILE)
        np.testing.assert_array_equal(actual, expected)

    def test_recovery_cannot_reuse_other_distribution_or_incomplete_cohort(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.json").write_text(json.dumps({"arm": "B"}))
            (root / "run.json").write_text(json.dumps({"status": "complete"}))
            validate_reuse(root, LEGACY_PROFILE)
            with self.assertRaises(ValueError):
                validate_reuse(root, DEFAULT_PROFILE)
            (root / "config.json").write_text(json.dumps({"arm": "B", "scene_profile": DEFAULT_PROFILE}))
            validate_reuse(root, DEFAULT_PROFILE)
            (root / "run.json").write_text(json.dumps({"status": "running"}))
            with self.assertRaises(ValueError):
                validate_reuse(root, DEFAULT_PROFILE)


if __name__ == "__main__":
    unittest.main()
