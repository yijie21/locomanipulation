import json
import sys
import tempfile
import unittest
from pathlib import Path

from locomani.tasks.bottle_visual.pipeline.report import paired, select


def cases(successes, bad_posture=False):
    return [
        dict(
            seed=21001 + i,
            stratum="low_near",
            edge="x-",
            recovery_kind="none",
            ok=i < successes,
            metrics=dict(
                torso_roll_max_deg=30.0 if bad_posture else 8.0,
                torso_roll_above15_s=0.0,
                waist_yaw_max_deg=12.0,
                phase_max_abs_deg={"vla": [0, 0, 8, 25, 12, 0, 0]},
            ),
        )
        for i in range(60)
    ]


class VisualReportTest(unittest.TestCase):
    def test_registered_tie_prefers_posture_then_simplicity_and_locks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arm, count, bad in [("A", 30, True), ("B", 29, False), ("C", 30, False)]:
                for seed in (17, 29):
                    path = root / f"selection_{arm}_s{seed}"
                    path.mkdir()
                    (path / "summary.json").write_text(json.dumps({"cases": cases(count, bad)}))
            self.assertEqual(select(root)["winner"], "B")
            self.assertEqual(select(root)["winner"], "B")
            # Altered post-selection evidence cannot quietly change the lock.
            path = root / "selection_C_s17/summary.json"
            path.write_text(json.dumps({"cases": cases(50)}))
            with self.assertRaises(ValueError):
                select(root)

    def test_pairing_and_scene_cluster_uncertainty(self):
        a, b = cases(60), cases(0)
        report = paired([a, a], [b, b])
        self.assertEqual(report["mean_success_rate_difference"], 1.0)
        self.assertEqual(report["scene_bootstrap_95_interval"], [1.0, 1.0])
        self.assertEqual(report["per_seed"][0]["candidate_only"], 60)
        mismatched = cases(0)
        mismatched[0]["seed"] = 999
        with self.assertRaises(ValueError):
            paired([a, a], [mismatched, b])


if __name__ == "__main__":
    unittest.main()
