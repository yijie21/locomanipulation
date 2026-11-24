"""TODO 49.2 step 6: the RGB-D sensor and its RealSense-like depth noise (perception/rgbd.py, noise.py)."""

import unittest

import numpy as np

from locomani.perception import noise as N
from locomani.perception.rgbd import RgbdSensor
from locomani.perception.sensor import Frame

H, W = 120, 160
K = np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])


def wall(z: float) -> np.ndarray:
    return np.full((H, W), z, np.float32)


def box_on_wall(near: float = 0.8, far: float = 2.0) -> np.ndarray:
    d = wall(far)
    d[40:80, 60:100] = near
    return d


class Realsense(unittest.TestCase):
    def test_noise_grows_with_the_square_of_distance_and_has_holes(self):
        rng = np.random.default_rng(0)
        errs = []
        for z in (0.6, 1.2, 2.4):
            out = N.realsense(wall(z), K, rng)
            ok = np.isfinite(out)
            self.assertGreater(ok.mean(), 0.95)  # a flat wall keeps nearly all its pixels (speckle only)
            self.assertLess(ok.mean(), 1.0)
            errs.append(float(np.std(out[ok] - z)))
        self.assertLess(errs[0], errs[1])
        self.assertLess(errs[1], errs[2])
        self.assertGreater(errs[2] / errs[0], 8.0)  # ~ (2.4 / 0.6)^2 = 16 for pure disparity noise

    def test_nothing_nearer_than_the_minimum_range(self):
        out = N.realsense(wall(0.2), K, np.random.default_rng(1))
        self.assertTrue(np.isnan(out).all())

    def test_a_depth_step_loses_pixels_and_casts_a_shadow_only_with_the_second_imager(self):
        d = box_on_wall()
        rng = np.random.default_rng(2)
        plain = N.realsense(d, K, rng)
        edge = np.zeros((H, W), bool)
        edge[38:82, 58:62] = edge[38:82, 98:102] = True
        self.assertGreater(np.isnan(plain[edge]).mean(), 0.3)
        right = np.roll(d, -int(round(100 * N.BASELINE / 0.8)), axis=1)  # the box seen from 5 cm to the right
        right[:, -7:] = 2.0
        shadowed = N.realsense(d, K, np.random.default_rng(2), depth_right=right)
        band = (slice(42, 78), slice(56, 59))  # just left of the box on the wall: hidden from the right imager
        self.assertGreater(np.isnan(shadowed[band]).mean(), np.isnan(plain[band]).mean())

    def test_same_seed_same_frame_and_nan_input_stays_a_hole(self):
        d = box_on_wall()
        d[0, 0] = np.nan
        a = N.realsense(d, K, np.random.default_rng(3))
        b = N.realsense(d, K, np.random.default_rng(3))
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(a[np.isfinite(a)], b[np.isfinite(b)])
        self.assertTrue(np.isnan(a[0, 0]))


class Sensor(unittest.TestCase):
    def exact(self):
        return Frame(np.zeros((H, W, 3), np.uint8), box_on_wall(), K, np.eye(4), "rgbd", 1.0)

    def test_none_passes_the_rendered_depth_through(self):
        s = RgbdSensor(self.exact, "none")
        f = s.frame()
        np.testing.assert_array_equal(f.depth, box_on_wall())
        self.assertEqual((s.name, f.source), ("rgbd", "rgbd"))

    def test_realsense_degrades_it_and_says_so(self):
        s = RgbdSensor(self.exact, "realsense", seed=4)
        f = s.frame()
        self.assertEqual(f.source, "rgbd:realsense")
        self.assertTrue(np.isnan(f.depth).any())
        self.assertEqual(f.depth.shape, (H, W))

    def test_an_unknown_noise_model_is_refused(self):
        with self.assertRaisesRegex(ValueError, "depth_noise"):
            RgbdSensor(self.exact, "kinect")


if __name__ == "__main__":
    unittest.main()
