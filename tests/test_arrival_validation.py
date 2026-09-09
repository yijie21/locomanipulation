import sys
import unittest
from pathlib import Path

from locomani.tasks.bottle.arrival_validation import ArrivalAdvance, squat_handover
from locomani.tasks.bottle.reach_config import ReachConfig


class ArrivalTests(unittest.TestCase):
    def test_advance_from_first_reachable_not_premature_map_gate(self):
        g = ArrivalAdvance(0.03)
        self.assertFalse(g.stop(0.6, False))
        self.assertFalse(g.stop(0.4, True))
        self.assertFalse(g.stop(0.38, True))
        self.assertTrue(g.stop(0.369, True))
        self.assertEqual(g.first_reachable, 0.4)

    def test_squat_uses_measured_height_and_holds_at_requested_drop(self):
        self.assertEqual(
            squat_handover(dict(ready=False, world_height_error_m=-0.01), 0.07, 0.12), "grip_height_aligned"
        )
        self.assertIsNone(squat_handover(dict(ready=False, world_height_error_m=-0.04), 0.07, 0.12))
        self.assertEqual(
            squat_handover(dict(ready=False, world_height_error_m=-0.04), 0.12, 0.12), "requested_drop_delivered"
        )

    def test_experiment_is_opt_in(self):
        self.assertEqual(ReachConfig().arrival_advance, 0)
        self.assertFalse(ReachConfig().measured_squat_handover)
        with self.assertRaises(ValueError):
            ReachConfig(arrival_advance=0.03)
