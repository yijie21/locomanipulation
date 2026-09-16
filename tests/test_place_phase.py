"""Pure-geometry checks for the place phase (tasks/bottle/place_phase.py)."""

from __future__ import annotations

import unittest

import numpy as np

from locomani.tasks.bottle.config import BottleTaskConfig, PlaceConfig
from locomani.tasks.bottle.place_phase import ego_visible, in_container, step_reference, turn_command, yaw_of_quat

IDENT = [1.0, 0.0, 0.0, 0.0]
YAW90 = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]  # the built basket's yaw


class InContainerTests(unittest.TestCase):
    half, floor, rim = (0.12, 0.07), 0.63, 0.772

    def test_centre_inside(self):
        ok, why = in_container([0.0, 0.0, 0.68], [0.0, 0.0, 0.7], IDENT, self.half, self.floor, self.rim)
        self.assertTrue(ok, why)

    def test_yaw_rotates_the_footprint(self):
        # 0.10 m along world y is inside a yaw-90 basket (its long axis lies on y) and outside an unyawed one.
        ok, _ = in_container([0.0, 0.10, 0.68], [0.0, 0.0, 0.7], YAW90, self.half, self.floor, self.rim)
        self.assertTrue(ok)
        ok, why = in_container([0.0, 0.10, 0.68], [0.0, 0.0, 0.7], IDENT, self.half, self.floor, self.rim)
        self.assertFalse(ok)
        self.assertIn("footprint", why)

    def test_resting_on_the_rim_is_not_inside(self):
        ok, why = in_container([0.0, 0.0, 0.80], [0.0, 0.0, 0.7], IDENT, self.half, self.floor, self.rim)
        self.assertFalse(ok)
        self.assertIn("rim", why)

    def test_yaw_of_quat(self):
        self.assertAlmostEqual(yaw_of_quat(YAW90), np.pi / 2, places=6)
        self.assertAlmostEqual(yaw_of_quat(IDENT), 0.0, places=6)


class TurnCommandTests(unittest.TestCase):
    band_x, band_y = (0.26, 0.44), (-0.16, -0.06)

    def test_in_band_stops(self):
        in_band, mv, face, speed = turn_command(0.30, -0.11, 0.3, self.band_x, self.band_y)
        self.assertTrue(in_band)
        self.assertEqual(mv, (0.0, 0.0, 0.0))
        self.assertAlmostEqual(np.arctan2(face[1], face[0]), 0.3)
        self.assertEqual(speed, 0.0)

    def test_goal_to_the_left_turns_left_without_walking(self):
        # Goal ahead-left (x in band): the facing rotates left (positive yaw), no translation.
        in_band, mv, face, speed = turn_command(0.30, 0.20, 0.0, self.band_x, self.band_y)
        self.assertFalse(in_band)
        self.assertGreater(np.arctan2(face[1], face[0]), 0.0)
        self.assertEqual(mv, (0.0, 0.0, 0.0))
        self.assertGreater(speed, 0.0)

    def test_goal_too_far_walks_forward_along_the_facing(self):
        in_band, mv, face, _ = turn_command(0.55, -0.11, 0.0, self.band_x, self.band_y)
        self.assertFalse(in_band)
        self.assertGreater(mv[0] * face[0] + mv[1] * face[1], 0.9)

    def test_goal_too_close_backs_up(self):
        _, mv, face, _ = turn_command(0.20, -0.11, 0.0, self.band_x, self.band_y)
        self.assertLess(mv[0] * face[0] + mv[1] * face[1], -0.9)

    def test_facing_step_is_bounded(self):
        _, _, face, _ = turn_command(0.30, 0.60, 0.0, self.band_x, self.band_y)
        self.assertLessEqual(abs(np.arctan2(face[1], face[0])), 0.35 + 1e-9)


class ReferenceTests(unittest.TestCase):
    def test_bounded_speed_then_arrival(self):
        ref = np.zeros(3)
        steps = 0
        while not np.allclose(ref, [0.0, 0.0, 0.10]):
            ref = step_reference(ref, [0.0, 0.0, 0.10], 0.10, 0.02)
            steps += 1
            self.assertLessEqual(ref[2], 0.10 + 1e-12)
        self.assertEqual(steps, 50)  # 0.10 m at 0.10 m/s in 0.02 s ticks

    def test_ego_gate(self):
        self.assertTrue(ego_visible([0.5, -0.5]))
        self.assertFalse(ego_visible([0.9, 0.0]))
        self.assertFalse(ego_visible(None))


class ConfigTests(unittest.TestCase):
    def test_place_config_round_trips_inside_the_task_config(self):
        cfg = BottleTaskConfig(goal="basket", place=PlaceConfig(band_x=(0.20, 0.28)))
        back = BottleTaskConfig.from_dict(cfg.to_dict())
        self.assertEqual(back, cfg)
        self.assertEqual(back.place.band_x, (0.20, 0.28))

    def test_pick_place_yaml_names_the_goal(self):
        from locomani import conf

        cfg = conf.load_group("task", "pick_place")
        self.assertEqual((cfg.target, cfg.goal), ("soup_can", "basket"))


if __name__ == "__main__":
    unittest.main()
