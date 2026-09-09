"""A transient that returns to its origin must still fail a drift audit."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free/tools"))
from analyze_posture_stages import analyze


def state(x):
    return dict(pelvis=dict(pos=[x, 0.0, 0.7], yaw=0.0, vel=[0.0, 0.0, 0.0]), fallen=False)


class PostureAnalysisTests(unittest.TestCase):
    def test_peak_drift_survives_return_to_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = [
                dict(event="posture_boundary", stage=name, wall=t, state=state(0))
                for name, t in [("before", 1.0), ("after", 3.0)]
            ]
            (root / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            (root / "privileged.jsonl").write_text(json.dumps(dict(wall=2.0, state=state(0.12))) + "\n")
            s = analyze(root)
            self.assertEqual(s["total"]["xy"], 0.0)
            self.assertAlmostEqual(s["stages"][0]["max_xy"], 0.12)
            self.assertEqual(s["stages"][0]["samples"], 1)

    def test_missing_boundaries_are_not_a_zero_drift_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "events.jsonl").write_text("")
            s = analyze(root)
            self.assertIsNone(s["total"])
            self.assertEqual(s["stages"], [])


if __name__ == "__main__":
    unittest.main()
