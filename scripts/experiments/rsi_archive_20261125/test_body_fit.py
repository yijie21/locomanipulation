"""The archived body_fit.py (scripts/experiments/rsi_archive_20261125): the AUC and the dedupe of the offline fit."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(REPO / "src")]


class Fit(unittest.TestCase):
    def test_auc_and_dedupe(self):
        import body_fit as BF

        self.assertAlmostEqual(BF.auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])), 0.75)
        self.assertAlmostEqual(BF.auc(np.array([1.0, 1.0, 1.0]), np.array([0, 1, 1])), 0.5)  # all tied: chance
        self.assertTrue(math.isnan(BF.auc(np.array([0.2, 0.3]), np.array([1, 1]))))
        rows = np.array([("a", "h1", "pre", 1.0), ("a", "h1", "pre", 1.0), ("a", "h2", "pre", 1.0), ("b", "h1", "pre", 2.0)],
                        dtype=[("label", "U4"), ("skills", "U4"), ("step", "U4"), ("t0", "f4")])  # fmt: skip
        self.assertEqual(len(BF.dedupe(rows)), 3)


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _bias_table(path: Path, miss) -> None:
    """A synthetic free-space grid where every reach misses by `miss` (heading frame, metres)."""
    xs, ys, zs = np.meshgrid(np.arange(0.2, 0.8, 0.1), [-0.05, 0.05, 0.15], np.arange(0.6, 1.3, 0.1), indexing="ij")
    X = np.array([[x, y, z, a, 0.78, h] for x, y, z in zip(xs.ravel(), ys.ravel(), zs.ravel()) for a in (0, 1, 2)
                  for h in (0, 1)])  # fmt: skip
    mu, sd = X.mean(0), X.std(0) + 1e-9
    np.savez(path, X=((X - mu) / sd).astype(np.float32), E=np.tile(np.asarray(miss, float), (len(X), 1)), mu=mu, sd=sd,
             k=np.array(8))  # fmt: skip
