import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free/scalebfm"))
from visual_observation import SceneMeasurement, SceneTracker


def measurement(t, x=0.5, visible=True):
    return SceneMeasurement(t, (x, 0, 0.7), 0.62, (0.4, 0, 0.62), (1, 0, 0),
                            (64, 0) if visible else (0, 0),
                            (0.8, 0) if visible else (0, 0))


class VisualObservationTests(unittest.TestCase):
    def test_new_geometry_arrives_only_after_latency(self):
        tracker = SceneTracker()
        tracker.ingest(measurement(0))
        self.assertIsNone(tracker.estimate(0.079).bottle_position)
        self.assertEqual(tracker.estimate(0.08).bottle_position, (0.5, 0, 0.7))
        tracker.ingest(measurement(0.4, x=0.7))
        self.assertEqual(tracker.estimate(0.46).bottle_position, (0.5, 0, 0.7))
        self.assertEqual(tracker.estimate(0.48).bottle_position, (0.7, 0, 0.7))

    def test_hidden_motion_does_not_update_task_information(self):
        a, b = SceneTracker(), SceneTracker()
        for tracker in (a, b):
            tracker.ingest(measurement(0)); tracker.estimate(0.08)
        a.ingest(measurement(0.4, x=-10, visible=False))
        b.ingest(measurement(0.4, x=10, visible=False))
        self.assertEqual(a.estimate(0.48), b.estimate(0.48))
        self.assertFalse(a.estimate(0.5).precision_allowed)
        self.assertEqual(a.estimate(1.0).confidence, 0.0)

    def test_high_rate_truth_updates_are_rejected(self):
        tracker = SceneTracker(); tracker.ingest(measurement(0))
        with self.assertRaises(ValueError):
            tracker.ingest(measurement(0.02, x=0.6))

    def test_reacquisition_restores_precision(self):
        tracker = SceneTracker(); tracker.ingest(measurement(0, visible=False))
        self.assertFalse(tracker.estimate(0.08).precision_allowed)
        tracker.ingest(measurement(0.4))
        self.assertTrue(tracker.estimate(0.48).precision_allowed)

    def test_pixels_and_occlusion_fraction_both_required(self):
        tiny = SceneMeasurement(0, None, None, None, None, (2, 0), (1, 0))
        self.assertFalse(tiny.visible())
        blocked = SceneMeasurement(0, None, None, None, None, (64, 0), (0.1, 0))
        self.assertFalse(blocked.visible())


if __name__ == "__main__":
    unittest.main()
