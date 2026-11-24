"""The hug lift judged against the goal (bimanual.lift_goal, clearance_now, lift_goal_check): the lowest box corner
over the support top and the upright axis off vertical, as lifted() reads them, with LIFT_MARGIN_M to spare; without
a lifted() in the goal the check is the rise with both hands on, as before."""

from __future__ import annotations

import math
import types
import unittest

import numpy as np

from . import bimanual


class _World:
    def __init__(self, z, pitch_deg=0.0):
        self.meta = {
            "objects": {"Box": {"body": "Box_object", "surface": "Pallet",
                                "box": {"half_extent": [0.2, 0.15, 0.1], "centre": [0.0, 0.0, 0.1]}}},
            "surfaces": {"Pallet": {"top": 0.4, "xy": [0.0, 0.0], "half_extent": [0.6, 0.45, 0.02]}},
        }  # fmt: skip
        self.free = {"Box": 7}
        a = math.radians(pitch_deg) / 2
        self.pose = (np.array([0.0, 0.0, z]), np.array([math.cos(a), 0.0, math.sin(a), 0.0]))  # about y

    def object_pose(self, key):
        return self.pose


class _Motor:
    def __init__(self, hands=("left", "right")):
        self.hands = hands

    def touching(self, body):
        return list(self.hands)


class LiftGoalTest(unittest.TestCase):
    def test_goal_read_from_the_task(self):
        ctx = types.SimpleNamespace(task=types.SimpleNamespace(goal=["held(Box)", "lifted(Box, 0.2, 60.0)"]))
        self.assertEqual(bimanual.lift_goal(ctx, "Box"), (0.2, 60.0))
        self.assertIsNone(bimanual.lift_goal(ctx, "Other"))
        self.assertIsNone(
            bimanual.lift_goal(types.SimpleNamespace(task=types.SimpleNamespace(goal=["held(Box)"])), "Box")
        )
        self.assertIsNone(bimanual.lift_goal(types.SimpleNamespace(), "Box"))

    def test_clearance_level_and_tipped(self):
        low, tilt = bimanual.clearance_now(_World(0.65), "Box")  # bottom face at the origin: 0.25 m over the top
        self.assertAlmostEqual(low, 0.25)
        self.assertAlmostEqual(tilt, 0.0)
        low, tilt = bimanual.clearance_now(_World(0.65, 30.0), "Box")  # the far end down by 0.2 sin 30
        self.assertAlmostEqual(tilt, 30.0, places=6)
        self.assertAlmostEqual(low, 0.25 - 0.2 * math.sin(math.radians(30.0)), places=6)

    def test_check_needs_the_margin_and_the_tilt(self):
        m, goal = _Motor(), (0.2, 60.0)
        need = 0.2 + bimanual.LIFT_MARGIN_M
        ok, v = bimanual.lift_goal_check(_World(0.4 + need + 0.001), m, "Box", 0.4, goal)
        self.assertTrue(ok, v)
        self.assertAlmostEqual(v["need_cm"], 100 * need)
        ok, v = bimanual.lift_goal_check(_World(0.4 + need - 0.01), m, "Box", 0.4, goal)  # held, 1 cm short
        self.assertFalse(ok)
        self.assertAlmostEqual(v["need_cm"] - v["low_cm"], 1.0, places=1)
        ok, v = bimanual.lift_goal_check(_World(0.9, 65.0), m, "Box", 0.4, goal)  # high but tipped past the limit
        self.assertFalse(ok)
        self.assertGreaterEqual(v["tilt_deg"], 60.0)

    def test_no_goal_is_the_rise_check(self):
        ok, v = bimanual.lift_goal_check(_World(0.43), _Motor(), "Box", 0.4, None)
        self.assertTrue(ok)
        self.assertNotIn("low_cm", v)
        ok, _ = bimanual.lift_goal_check(_World(0.43), _Motor(("left",)), "Box", 0.4, None)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
