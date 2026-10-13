"""probe_cells' attitude frame (TODO 24.6): palm_rotation_at / attitude_of invert each other, the grid's 90-degree
degeneracy is handled, and Envelope judges a candidate by its nearest probed attitude."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "wbc_compare_20261012"))
import probe_cells as C  # noqa: E402


class AttitudeFrame(unittest.TestCase):
    def test_probe_attitude_is_the_grid_origin(self):
        yaw = -1.2
        np.testing.assert_allclose(
            C.palm_rotation_at(yaw, C.PITCH_DOWN_DEG, 0.0, 0.0), C.palm_rotation(yaw), atol=1e-12
        )

    def test_round_trip(self):
        yaw = 0.7
        for att in C.attitudes():
            if att["elev"] == 90:
                continue
            e, y, r = C.attitude_of(C.palm_rotation_at(yaw, att["elev"], att["yaw_dev"], att["roll"]), yaw)
            self.assertAlmostEqual(e, att["elev"], places=6)
            self.assertAlmostEqual((y - att["yaw_dev"] + 180) % 360 - 180, 0.0, places=6)
            self.assertAlmostEqual((r - att["roll"] + 180) % 360 - 180, 0.0, places=6)

    def test_top_down_roll_180_is_yaw_plus_180(self):
        for y in C.YAW_DEVS:
            a = C.attitude_in_heading(90, y, 180)
            b = C.attitude_in_heading(90, y + 180, 0)
            np.testing.assert_allclose(a, b, atol=1e-12)
        self.assertEqual(len(C.attitudes()), (len(C.ELEVATIONS) - 1) * len(C.YAW_DEVS) * len(C.ROLLS) + len(C.YAW_DEVS))

    def test_envelope_nearest_and_reject(self):
        rows = [
            dict(att, deliverable=att["elev"] <= 30 and abs(att["yaw_dev"]) <= 60, err_mean_cm=2.0, rot_mean_deg=3.0)
            for att in C.attitudes()
        ]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "envelope.json"
            f.write_text(json.dumps(dict(thresholds=dict(err_cm=5.0, rot_deg=15.0), attitudes=rows)))
            env = C.Envelope(f)
        yaw = 2.0
        row, dist = env.nearest(C.palm_rotation_at(yaw, 33, -4, 6), yaw)
        self.assertEqual((row["elev"], row["yaw_dev"], row["roll"]), (30, 0, 0))
        self.assertLess(dist, 10.0)
        self.assertIsNone(env.reject(C.palm_rotation_at(yaw, 33, -4, 6), yaw))
        self.assertIn("outside the envelope", env.reject(C.palm_rotation_at(yaw, 60, -90, 180), yaw))
        self.assertIn("unprobed", env.reject(C.palm_rotation_at(yaw, 30, 0, 90), yaw))


if __name__ == "__main__":
    unittest.main()
