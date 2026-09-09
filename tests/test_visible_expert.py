"""visible_expert: every expert action is a function of what the image policy sees."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
from reach_config import ReachConfig
from ready_grasp_policy import STANDING_PELVIS_Z, sample_stand, templates
from test_recoverable_grasp_policy import Harness


def visible(**kw):
    return ReachConfig(
        ready_grasp_baseline=True,
        ready_grasp_recovery=True,
        coupled_squat=True,
        timely_capture=True,
        visible_expert=True,
        **kw,
    )


class VisibleConfigTests(unittest.TestCase):
    def test_requires_ready_recovery_controller(self):
        with self.assertRaises(ValueError):
            ReachConfig(visible_expert=True)
        with self.assertRaises(ValueError):
            ReachConfig(ready_grasp_baseline=True, visible_expert=True)
        self.assertTrue(visible().visible_expert)

    def test_excludes_per_episode_draws(self):
        for kw in (dict(visible_brake_depth=0.10),):
            with self.assertRaises(ValueError):
                visible(**kw)

    def test_loads_from_json(self):
        cfg = ReachConfig.load(Path(__file__).resolve().parents[1] / "configs/collection/ready_grasp_visible.json")
        self.assertTrue(cfg.visible_expert and cfg.arrival_advance == 0)


class VisibleStandTests(unittest.TestCase):
    def test_stand_is_a_function_of_grip_height_only(self):
        cfg = visible()
        plan = SimpleNamespace(grip_z=0.683)
        a = sample_stand(plan, np.random.default_rng(1), cfg)
        b = sample_stand(plan, np.random.default_rng(2), cfg)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertEqual(a[1], b[1])
        centre = np.array([a[0][0] + cfg.backshift, a[0][1], 0.683 - (STANDING_PELVIS_Z - a[1])])
        self.assertLess(min(np.linalg.norm(centre - c) for _, c, _ in templates()), 0.015)
        self.assertTrue(0.03 <= a[1] <= cfg.max_drop)

    def test_baseline_still_draws(self):
        cfg = ReachConfig(ready_grasp_baseline=True, ready_grasp_recovery=True)
        plan = SimpleNamespace(grip_z=0.723)
        rels = {tuple(np.round(sample_stand(plan, np.random.default_rng(s), cfg)[0], 4)) for s in range(20)}
        self.assertGreater(len(rels), 1)


class VisibleHarness(Harness):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.config = visible()


class VisibleRecoveryTests(unittest.TestCase):
    def run_policy(self, p):
        with patch("stream_reach.toppled", return_value=False):
            return p.execute(grip_world=np.zeros(3), squat_hint=0.1, ghost=None)

    def test_feasible_stand_grasps_at_its_own_drop(self):
        p = VisibleHarness([True], [True])
        ok, _ = self.run_policy(p)
        self.assertTrue(ok)
        self.assertEqual(p.sequence, ["hold", "assess", "grasp"])

    def test_infeasible_stand_discards_without_repositioning(self):
        p = VisibleHarness([False], [])
        ok, why = self.run_policy(p)
        self.assertFalse(ok)
        self.assertIn("no reposition", why)
        for step in ("choose", "withdraw_then_stand", "move"):
            self.assertNotIn(step, p.sequence)
        self.assertEqual(p.sequence.count("assess"), 1)

    def test_failed_grasp_does_not_retry_elsewhere(self):
        p = VisibleHarness([True, True], [False, True])
        ok, _ = self.run_policy(p)
        self.assertFalse(ok)
        self.assertEqual(p.sequence.count("grasp"), 1)
        self.assertNotIn("move", p.sequence)


if __name__ == "__main__":
    unittest.main()
