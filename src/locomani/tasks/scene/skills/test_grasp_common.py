"""The shared stage computations (grasp_common): the lift against the goal and the collision hull's cross-section.
Imports the module beside it, so it checks this copy of the package wherever it lives (a sandbox, a sync candidate)."""

from __future__ import annotations

import math
import types
import unittest

import numpy as np

from . import grasp_common as GC


def block(x0, x1, y0, y1, z0, z1):
    """The 8 corners of an axis-aligned block: one convex collision piece."""
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)


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


class LiftGoal(unittest.TestCase):
    def test_the_goal_is_read_from_the_task(self):
        ctx = types.SimpleNamespace(task=types.SimpleNamespace(goal=["held(Box)", "lifted(Box, 0.2, 60.0)"]))
        self.assertEqual(GC.lift_goal(ctx, "Box"), (0.2, 60.0))
        self.assertIsNone(GC.lift_goal(ctx, "Other"))
        self.assertIsNone(GC.lift_goal(types.SimpleNamespace(task=types.SimpleNamespace(goal=["held(Box)"])), "Box"))
        self.assertIsNone(GC.lift_goal(types.SimpleNamespace(), "Box"))

    def test_clearance_level_and_tipped(self):
        low, tilt = GC.clearance_now(_World(0.65), "Box")  # bottom face at the origin: 0.25 m over the top
        self.assertAlmostEqual(low, 0.25)
        self.assertAlmostEqual(tilt, 0.0)
        low, tilt = GC.clearance_now(_World(0.65, 30.0), "Box")  # the far end down by 0.2 sin 30
        self.assertAlmostEqual(tilt, 30.0, places=6)
        self.assertAlmostEqual(low, 0.25 - 0.2 * math.sin(math.radians(30.0)), places=6)

    def test_the_check_adds_the_goal_to_the_familys_own(self):
        goal, margin = (0.2, 60.0), 0.03
        need = 0.2 + margin
        ok, v = GC.goal_check(_World(0.4 + need + 0.001), "Box", goal, margin, True, dict(rise_cm=20.0))
        self.assertTrue(ok, v)
        self.assertEqual(v["rise_cm"], 20.0)
        self.assertAlmostEqual(v["need_cm"], 100 * need)
        ok, v = GC.goal_check(_World(0.4 + need - 0.01), "Box", goal, margin, True, {})  # held, 1 cm short
        self.assertFalse(ok)
        self.assertAlmostEqual(v["need_cm"] - v["low_cm"], 1.0, places=1)
        ok, v = GC.goal_check(_World(0.9, 65.0), "Box", goal, margin, True, {})  # high but tipped past the limit
        self.assertFalse(ok)
        self.assertGreaterEqual(v["tilt_deg"], 60.0)
        ok, v = GC.goal_check(_World(0.9), "Box", goal, margin, False, {})  # the family's own check failed
        self.assertFalse(ok)

    def test_no_goal_returns_the_familys_check_as_it_came(self):
        values = dict(rise_cm=3.0)
        self.assertEqual(GC.goal_check(_World(0.43), "Box", None, 0.03, True, values), (True, values))
        self.assertNotIn("low_cm", values)

    def test_the_top_up_raises_every_hand_by_the_shortfall_under_the_cap(self):
        v = dict(low_cm=18.0, need_cm=23.0, tilt_deg=5.0)
        self.assertEqual(GC.top_up(v, (0.2, 60.0), [1.0, 1.02], 2.0), [1.05, 1.07])
        self.assertEqual(GC.top_up(v, (0.2, 60.0), [1.0], 1.03), [1.03])  # capped
        self.assertIsNone(GC.top_up(v, (0.2, 60.0), [1.0], 1.004))  # nothing left to raise
        self.assertIsNone(GC.top_up(dict(v, tilt_deg=60.0), (0.2, 60.0), [1.0], 2.0))  # tipped: higher will not help
        self.assertIsNone(GC.top_up(dict(v, low_cm=23.5), (0.2, 60.0), [1.0], 2.0))  # not short


class Section(unittest.TestCase):
    def test_a_cut_is_the_hulls_edge_crossings_and_a_band_adds_the_vertices_between(self):
        (piece,) = GC.prepare([block(-0.1, 0.1, -0.05, 0.05, 0.0, 0.2)])
        c = GC.cut(piece, 0.1)
        self.assertEqual(len(c[:, 0]), len(c))
        self.assertAlmostEqual(c[:, 0].min(), -0.1)
        self.assertAlmostEqual(c[:, 0].max(), 0.1)
        self.assertTrue(np.allclose(c[:, 2], 0.1))
        self.assertEqual(len(GC.cut(piece, 0.3)), 0)
        b = GC.band(piece, 0.15, 0.25)  # the top four corners and the cut at 0.15
        self.assertEqual(len(b[np.abs(b[:, 2] - 0.2) < 1e-9]), 4)
        self.assertEqual(len(b[np.abs(b[:, 2] - 0.25) < 1e-9]), 0)

    def test_a_diamonds_width_follows_its_height(self):
        r, h = 0.3, 0.4  # an octahedron: a square of half-diagonal r at z = 0, apexes at +-h
        diamond = np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, h], [0, 0, -h]], float)
        pieces = GC.prepare([diamond])
        self.assertAlmostEqual(GC.band_width(pieces, -0.5 * h, -0.25 * h, [1.0, 0.0]), 1.5 * r)
        self.assertAlmostEqual(GC.band_width(pieces, -0.1 * h, 0.1 * h, [0.0, 1.0]), 2 * r)
        self.assertEqual(GC.band_width(pieces, 1.1 * h, 1.2 * h, [1.0, 0.0]), 0.0)
        self.assertAlmostEqual(GC.cuts_width(pieces, [0.0, 0.5 * h], [1.0, 0.0]), 2 * r)
        self.assertIsNone(GC.cuts_width(pieces, [2 * h], [1.0, 0.0]))

    def test_the_extent_within_a_strip_reads_the_chords_crossings(self):
        # a wall slanted in plan: wider at y < 0 than at y > 0; within |y| <= 0.01 its x reach is between
        pts = np.array([[-0.2, -0.1, 0.0], [0.2, -0.1, 0.0], [0.1, 0.1, 0.0], [-0.1, 0.1, 0.0]])
        self.assertEqual(GC.extent(pts, (0, 0), (1, 0)), (-0.2, 0.2))
        lo, hi = GC.extent(pts, (0, 0), (1, 0), strip=((0, 1), 0.0, 0.01))
        self.assertAlmostEqual(lo, -0.155)  # the chord from (-0.2, -0.1) to (-0.1, 0.1) at y = -0.01
        self.assertAlmostEqual(hi, 0.155)
        self.assertIsNone(GC.extent(pts, (0, 0), (1, 0), strip=((0, 1), 0.5, 0.01)))
        self.assertIsNone(GC.extent(np.zeros((0, 3)), (0, 0), (1, 0)))

    def test_a_section_extent_is_taken_piece_by_piece(self):
        # two separate blocks: a chord between them would cross the strip's lines where there is no material
        pieces = GC.prepare([block(-0.3, -0.2, -0.05, 0.05, 0.0, 0.2), block(0.2, 0.3, 0.1, 0.15, 0.0, 0.2)])
        self.assertEqual(GC.section_extent(pieces, 0.1, (0, 0), (1, 0)), (-0.3, 0.3))
        self.assertEqual(GC.section_extent(pieces, 0.1, (0, 0), (1, 0), strip=((0, 1), 0.0, 0.02)), (-0.3, -0.2))
        self.assertIsNone(GC.section_extent(pieces, 0.3, (0, 0), (1, 0)))
        lo, hi = GC.section_extent(pieces, 0.1, (0.1, 0), (1, 0))  # from another origin
        self.assertAlmostEqual(lo, -0.4)
        self.assertAlmostEqual(hi, 0.2)

    def test_prepare_keeps_ready_pieces_and_drops_a_lone_point(self):
        ready = GC.prepare([block(0, 1, 0, 1, 0, 1)])
        self.assertEqual(len(GC.prepare(ready + [np.zeros((1, 3))])), 1)
        self.assertIs(GC.prepare(ready)[0], ready[0])

    def test_no_model_gives_no_pieces(self):
        self.assertEqual(GC.pieces(_World(0.5), "Box"), [])
        self.assertEqual(GC.pieces(types.SimpleNamespace(sim=types.SimpleNamespace(m=None)), "Box"), [])


if __name__ == "__main__":
    unittest.main()
