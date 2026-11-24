"""The hug's grip height from the collision hull (bimanual.grip_height, beside_window): on straight walls the scan's
top; under a top that narrows, or below a handle cut-out in one wall, the highest band where both walls are at full
width, never below the centre of mass; the widest band when none is; the box's rim without a hull. Imports the module
beside it, so it checks this copy of the package wherever it lives."""

import unittest

import numpy as np

from . import bimanual as B


def block(x0, x1, y0, y1, z0, z1):
    """The 8 corners of an axis-aligned block: one convex collision piece."""
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


U, V, C = np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.zeros(2)  # squeeze along x, approach along y


class GripHeight(unittest.TestCase):
    def setUp(self):
        self.saved = {k: getattr(B, k) for k in ("GRASP_BELOW_RIM_M", "WALL_TOL_M", "SCAN_STEP_M")}
        B.GRASP_BELOW_RIM_M, B.WALL_TOL_M, B.SCAN_STEP_M = 0.045, 0.005, 0.01

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(B, k, v)

    def test_straight_walls_take_the_top_of_the_scan_and_read_each_wall(self):
        g = B.grip_height([block(-0.12, 0.10, -0.05, 0.05, 0.0, 0.20)], C, U, V, 0.0, 0.20, 0.10)
        self.assertEqual(g["how"], "full_width")
        self.assertAlmostEqual(g["z"], 0.20 - 0.045)
        self.assertAlmostEqual(g["left"], 0.10)
        self.assertAlmostEqual(g["right"], -0.12)

    def test_a_top_that_narrows_moves_the_grip_down_to_the_full_width_band(self):
        # full width up to 0.12, then a 2 cm narrower lid to the rim at 0.20
        pieces = [block(-0.15, 0.15, -0.05, 0.05, 0.0, 0.12), block(-0.13, 0.13, -0.05, 0.05, 0.12, 0.20)]
        g = B.grip_height(pieces, C, U, V, 0.0, 0.20, 0.08)
        self.assertEqual(g["how"], "full_width")
        self.assertLessEqual(g["z"] + B.FINGER_HALF_M, 0.12 + 1e-9)  # the pad's band below the step
        self.assertGreater(g["z"], 0.12 - B.FINGER_HALF_M - B.SCAN_STEP_M)  # the highest such band
        self.assertAlmostEqual(g["left"], 0.15)

    def test_a_handle_cut_out_in_one_wall_is_skipped(self):
        # the +x wall has a hole at the jaw's depth from 0.13 to 0.16; the other wall is whole
        pieces = [block(-0.10, 0.08, -0.05, 0.05, 0.0, 0.20), block(0.08, 0.10, -0.05, 0.05, 0.0, 0.13),
                  block(0.08, 0.10, -0.05, 0.05, 0.16, 0.20)]  # fmt: skip
        g = B.grip_height(pieces, C, U, V, 0.0, 0.20, 0.09)
        self.assertLessEqual(g["z"] + B.FINGER_HALF_M, 0.13 + 1e-9)
        self.assertAlmostEqual(g["left"], 0.10)

    def test_never_below_the_centre_of_mass_and_the_widest_band_when_none_is_at_full_width(self):
        # a flared basket: each 1 cm slice 2 mm wider per side than the one below, widest at the rim
        pieces = [block(-0.10 - 0.002 * k, 0.10 + 0.002 * k, -0.05, 0.05, 0.01 * k, 0.01 * (k + 1)) for k in range(15)]
        g = B.grip_height(pieces, C, U, V, 0.0, 0.15, 0.06)
        self.assertEqual(g["how"], "widest")
        self.assertAlmostEqual(g["z"], 0.15 - 0.045)  # the scan's top: the widest of its bands
        tall = [block(-0.15, 0.15, -0.05, 0.05, 0.0, 0.20), block(-0.10, 0.10, -0.05, 0.05, 0.20, 0.30)]
        g = B.grip_height(tall, C, U, V, 0.0, 0.30, 0.19)  # full width only below the centre of mass
        self.assertGreaterEqual(g["z"], 0.19 + B.SCAN_ABOVE_COM_M - 1e-9)

    def test_no_hull_section_at_the_jaws_depth_is_none(self):
        self.assertIsNone(B.grip_height([], C, U, V, 0.0, 0.2, 0.1))
        self.assertIsNone(B.grip_height([block(-0.1, 0.1, 0.2, 0.3, 0.0, 0.2)], C, U, V, 0.0, 0.2, 0.1))

    def test_the_beside_window_follows_the_grip_and_keeps_half_a_thin_objects_depth(self):
        g = dict(depth_half=0.0436, grasp_z=0.50, bottom=0.37, rim=0.60)
        saved = (B.BESIDE_EDGE_M, B.BESIDE_DZ_M, B.BESIDE_LOW_M, B.BESIDE_RIM_M)
        try:
            B.BESIDE_EDGE_M, B.BESIDE_DZ_M, B.BESIDE_LOW_M, B.BESIDE_RIM_M = 0.04, 0.03, 0.02, 0.01
            along, lo, hi = B.beside_window(g)
            self.assertAlmostEqual(along, 0.0218)
            self.assertAlmostEqual(lo, 0.47)
            self.assertAlmostEqual(hi, 0.53)
            along, lo, hi = B.beside_window(dict(g, depth_half=0.16, grasp_z=0.58))
            self.assertAlmostEqual(along, 0.12)
            self.assertAlmostEqual(hi, 0.59)  # clipped under the rim
        finally:
            B.BESIDE_EDGE_M, B.BESIDE_DZ_M, B.BESIDE_LOW_M, B.BESIDE_RIM_M = saved


if __name__ == "__main__":
    unittest.main()
