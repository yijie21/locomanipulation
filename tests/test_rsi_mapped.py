"""Vision RSI D2 step 2d: the perceived scene description (locomani.tasks.scene.mapped) -- the blocked cells come back
from the navigator built on it, perceived surfaces replace the truth's, the map's object nodes become the memory."""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.primitives.navigator import build  # noqa: E402
from locomani.tasks.scene import mapped as M  # noqa: E402

META = dict(
    room=dict(bounds=dict(x=[0.0, 4.0], y=[0.0, 4.0]), walls=[[[2.0, 0.0], [2.0, 1.0]]], obstacles=[]),
    surfaces=dict(t=dict(xy=[3.0, 3.0], top=0.75, half_extent=[0.3, 0.2], yaw=0.0, room="r", approach_edges=["+x"])),
    objects=dict(
        cup=dict(free=True, surface="t", init_pos=[3.0, 3.0, 0.8], box=dict(half_extent=[0.04, 0.04, 0.05])),
        lamp=dict(free=False, init_pos=[0.5, 3.5, 0.0]),
    ),
)


def grid_with_box():
    """A 4 x 4 m map: seen free everywhere except a 0.5 x 0.5 m obstacle at (1.0-1.5, 2.0-2.5) and an unseen strip."""
    res, n = 0.05, 80
    free = np.ones((n, n), bool)
    obstacle = np.zeros((n, n), bool)
    obstacle[20:30, 40:50] = True
    free[20:30, 40:50] = False
    free[:, 75:] = False  # never seen
    return dict(origin=np.array([0.0, 0.0]), res=res, obstacle=obstacle, free=free)


class Perceived(unittest.TestCase):
    def test_the_blocked_cells_come_back_through_the_navigator(self):
        g = grid_with_box()
        meta = M.perceived_meta(META, g, [], {}, {}, parts=("grid",))
        occ, _ = build(meta, body_radius=0.35)
        self.assertEqual(meta["room"]["walls"], [])
        self.assertNotIn("lamp", meta["objects"])  # statics are in the grid
        self.assertTrue(occ.occupied[25, 45])  # the obstacle
        self.assertTrue(occ.occupied[10, 78])  # the unseen strip
        self.assertFalse(occ.occupied[10, 10])  # seen free
        self.assertFalse(occ.occupied[40, 10])  # the truth's wall is gone: (2.0, 0.5) seen free

    def test_surfaces_and_objects_take_the_map(self):
        g = grid_with_box()
        found = [dict(xy=[3.02, 2.97], top=0.76, half_extent=[0.31, 0.19], yaw=0.01)]
        meta = M.perceived_meta(META, g, found, {"t": 0}, {"cup": [3.01, 2.99, 0.81]}, parts=("surfaces", "objects"))
        s = meta["surfaces"]["t"]
        self.assertEqual(s["xy"], [3.02, 2.97])
        self.assertAlmostEqual(s["top"], 0.76)
        self.assertIn("-x", s["approach_edges"])  # open floor west of the table
        self.assertNotIn("+y", s["approach_edges"])  # 0.8 m to the unseen strip: no room to stand
        self.assertEqual(meta["objects"]["cup"]["init_pos"], [3.01, 2.99, 0.81])
        self.assertEqual(meta["room"]["walls"], META["room"]["walls"])  # the grid part not asked for
        self.assertEqual(META["surfaces"]["t"]["xy"], [3.0, 3.0])  # the input untouched

    def test_what_the_map_missed_keeps_the_truth_and_is_listed(self):
        meta = M.perceived_meta(META, grid_with_box(), [], {}, {}, parts=("surfaces", "objects"))
        self.assertEqual(meta["perceived"]["surfaces_kept"], ["t"])
        self.assertEqual(meta["perceived"]["objects_kept"], ["cup"])
        with self.assertRaises(ValueError):
            M.perceived_meta(META, grid_with_box(), [], {}, {}, parts=("walls",))


if __name__ == "__main__":
    unittest.main()
