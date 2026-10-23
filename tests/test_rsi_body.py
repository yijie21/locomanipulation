"""TODO 33.1.4 (user 2026-10-23 "继续做 33.1.4，先做预测器"): the body model's reach predictor -- a k-nearest-neighbour
table over the envelope grid, queried in the pelvis heading frame or from a world point and pelvis pose."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "rsi"))

from locomani.tasks.scene import body  # noqa: E402


def _table(path: Path) -> None:
    """A synthetic grid: reaches beyond 0.6 m forward or below 0.3 m miss, the rest land."""
    xs, ys, zs = np.meshgrid(np.arange(0.2, 1.0, 0.1), [-0.05, 0.05, 0.15], np.arange(0.1, 1.5, 0.1), indexing="ij")
    rows = []
    for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel()):
        for att in (0.0, 1.0):
            for hand in (0.0, 1.0):
                rows.append([x, y, z, att, 0.78, hand])
    X = np.array(rows)
    miss = ((X[:, 0] > 0.6) | (X[:, 2] < 0.3)).astype(float)
    err = np.where(miss > 0, 12.0, 1.0)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    np.savez(path, X=((X - mu) / sd).astype(np.float32), y_miss=miss.astype(np.float32), y_err=err.astype(np.float32),
             mu=mu, sd=sd, features=np.array(["x", "y_out", "z", "att", "pelvis", "hand"]), k=np.array(15),
             miss_cm=np.array(3.0), stand_z=np.array(0.78))  # fmt: skip


class Reach(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "table.npz"
        _table(self.path)
        body.table.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_table_separates_the_planted_regions(self):
        near = body.reach(0.45, 0.05, 0.9, "level", None, "right", path=str(self.path))
        far = body.reach(0.85, 0.05, 0.9, "level", None, "right", path=str(self.path))
        low = body.reach(0.45, 0.05, 0.1, "down", 0.78, "left", path=str(self.path))
        self.assertLess(near["p_miss"], 0.2)
        self.assertGreater(far["p_miss"], 0.8)
        self.assertGreater(low["p_miss"], 0.6)  # at the planted border (z < 0.3 misses) the neighbours mix
        self.assertLess(near["err_cm"], far["err_cm"])
        self.assertTrue(near["inside"])
        self.assertEqual((near["n"], near["att"], near["hand"], near["pelvis"]), (15, "level", "right", 0.78))
        # a query far outside the grid says so
        self.assertFalse(body.reach(3.0, 2.0, 4.0, "level", 0.4, "right", path=str(self.path))["inside"])

    def test_world_points_turn_into_the_heading_frame_with_the_outward_sign_per_hand(self):
        q = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))  # yaw +90 deg: the robot faces +y
        pelvis = (1.0, 2.0, 0.78)
        x, y_out, z = body.grid_coords((1.0, 2.5, 0.9), pelvis, q, "right")  # 0.5 m ahead, on the centre line
        self.assertAlmostEqual(x, 0.5, places=6)
        self.assertAlmostEqual(y_out, 0.0, places=6)
        self.assertAlmostEqual(z, 0.9)
        # a point to the robot's right (world +x when facing +y) is outward for the right hand, inward for the left
        xr, out_r, _ = body.grid_coords((1.2, 2.5, 0.9), pelvis, q, "right")
        xl, out_l, _ = body.grid_coords((1.2, 2.5, 0.9), pelvis, q, "left")
        self.assertAlmostEqual(out_r, 0.2, places=6)
        self.assertAlmostEqual(out_l, -0.2, places=6)
        self.assertAlmostEqual(xr, xl)
        r = body.reach_world((1.0, 2.5, 0.9), pelvis, q, "right", pinch=True, path=str(self.path))
        self.assertEqual(r["att"], "down")
        self.assertLess(r["p_miss"], 0.2)

    def test_the_command_line_prints_json(self):
        from contextlib import redirect_stdout
        from io import StringIO

        buf = StringIO()
        with redirect_stdout(buf):
            body.main(["--table", str(self.path), "reach", "--x", "0.45", "--z", "0.9", "--hand", "left"])
        out = json.loads(buf.getvalue())
        self.assertIn("p_miss", out)
        self.assertEqual(out["hand"], "left")


class Fit(unittest.TestCase):
    def test_auc_and_dedupe(self):
        import body_fit as BF

        self.assertAlmostEqual(BF.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])), 0.75)
        self.assertAlmostEqual(BF.auc(np.array([1.0, 1.0, 1.0]), np.array([0, 1, 1])), 0.5)  # all tied: chance
        self.assertTrue(math.isnan(BF.auc(np.array([0.2, 0.3]), np.array([1, 1]))))
        rows = np.array([("a", "h1", "pre", 1.0), ("a", "h1", "pre", 1.0), ("a", "h2", "pre", 1.0), ("b", "h1", "pre", 2.0)],
                        dtype=[("label", "U4"), ("skills", "U4"), ("step", "U4"), ("t0", "f4")])  # fmt: skip
        self.assertEqual(len(BF.dedupe(rows)), 3)


if __name__ == "__main__":
    unittest.main()
