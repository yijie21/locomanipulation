"""Execution may differ from the preferred low grip without losing capture."""

import sys
import unittest
from pathlib import Path

import numpy as np

from locomani.tasks.bottle.measured_grasp import closure_geometry
from locomani.tasks.bottle.ready_grasp_policy import OFFSET


class CaptureToleranceTests(unittest.TestCase):
    def geometry(self, delta):
        state = {"palms": {"right": {"pos": [0.0, 0.0, 0.7], "quat": [1.0, 0.0, 0.0, 0.0]}}}
        return closure_geometry(state, "right", np.array([0.0, 0.0, 0.7]) + OFFSET + delta)

    def test_slightly_high_and_laterally_offset_capture_is_accepted(self):
        self.assertTrue(self.geometry([-0.006, 0.025, -0.013])["ready"])

    def test_low_grip_is_preference_not_exact_height(self):
        self.assertTrue(self.geometry([0.0, 0.0, -0.010])["ready"])

    def test_actual_misses_remain_rejected(self):
        for delta in ([0.0, 0.030, 0.0], [0.03, 0.0, 0.0], [0.0, 0.0, -0.025]):
            self.assertFalse(self.geometry(delta)["ready"])

    def test_height_correction_uses_same_region_as_closure(self):
        from locomani.tasks.bottle.grasp_capture import height_correction

        self.assertEqual(height_correction(self.geometry([0.0, 0.0, -0.013])), 0.0)
        self.assertLess(height_correction(self.geometry([0.0, 0.0, -0.025])), 0.0)

    def test_planning_alternatives_are_inside_execution_region(self):
        from locomani.tasks.bottle.grasp_capture import planning_points

        preferred = np.array([0.1515, 0.0, 0.7])
        for point in planning_points(preferred):
            state = {"palms": {"right": {"pos": point - OFFSET, "quat": [1.0, 0.0, 0.0, 0.0]}}}
            self.assertTrue(closure_geometry(state, "right", preferred)["ready"])
