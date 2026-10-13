"""bfm_brake's walk profile (TODO 24.7): the trapezoid reaches its target at rest, is continuous, and degrades to a
triangle when the run is too short to reach the cruise speed; the open profile cruises forever."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "wbc_compare_20261012"))
from bfm_brake import Profile  # noqa: E402


class WalkProfile(unittest.TestCase):
    def test_trapezoid_ends_at_the_target_at_rest(self):
        p = Profile(0.55, 0.6, 1.3)
        self.assertAlmostEqual(p.vp, 0.55)
        self.assertAlmostEqual(p.s(p.t_end), 1.3)
        self.assertAlmostEqual(p.s(p.t_end + 5.0), 1.3)
        t = np.linspace(0, p.t_end, 2000)
        s = np.array([p.s(x) for x in t])
        self.assertTrue(np.all(np.diff(s) >= -1e-12))  # monotone
        self.assertLess(np.abs(np.diff(s)).max(), 0.55 * (t[1] - t[0]) * 1.01)  # never faster than the cruise speed
        v_end = (p.s(p.t_end) - p.s(p.t_end - 0.01)) / 0.01
        self.assertLess(v_end, 0.01)
        # the deceleration starts v/acc before the end and covers v^2 / 2 acc
        self.assertAlmostEqual(p.t_end - p.t_dec, 0.55 / 0.6)
        self.assertAlmostEqual(1.3 - p.s(p.t_dec), 0.55**2 / 1.2)

    def test_short_run_is_a_triangle(self):
        p = Profile(0.55, 0.6, 0.2)
        self.assertLess(p.vp, 0.55)
        self.assertAlmostEqual(p.vp, np.sqrt(0.6 * 0.2))
        self.assertEqual(p.tc, 0.0)
        self.assertAlmostEqual(p.s(p.t_end), 0.2)
        self.assertAlmostEqual(p.s(p.ta), 0.1)  # the peak sits at half the distance

    def test_open_profile_cruises(self):
        p = Profile(0.35, 0.6, None)
        self.assertTrue(np.isinf(p.t_dec))
        self.assertAlmostEqual(p.s(10.0) - p.s(9.0), 0.35)
        self.assertEqual(p.s(-1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
