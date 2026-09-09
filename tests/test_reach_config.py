import json
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from _sim_stubs import Clock, Robot

from locomani.backends.sonic import stream_reach as sr
from locomani.tasks.bottle.reach_config import DEFAULT_REACH, ReachConfig


class ReachConfigTests(unittest.TestCase):
    def test_default_keeps_old_band(self):
        self.assertEqual(DEFAULT_REACH.approach_band(), (0.18, 0.27))

    def test_invalid_calibration_fails_before_actuation(self):
        for args in [
            dict(post_max=0.25),
            dict(post_min=float("nan")),
            dict(transport_guard="false"),
            dict(sweep_joint_step=0.0),
            dict(max_drop=0.2),
        ]:
            with self.assertRaises(ValueError):
                ReachConfig(**args)

    def test_load_ignores_retired_keys_with_a_warning(self):
        repo = Path(__file__).resolve().parents[1]
        path = repo / "configs/collection/ready_grasp_visible.json"
        data = json.loads(path.read_text())
        self.assertIn("brake_preplan", data)  # committed historical snapshots still carry it
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = ReachConfig.load(path)
        self.assertFalse(hasattr(cfg, "brake_preplan"))
        self.assertTrue(any("brake_preplan" in str(w.message) for w in caught))

    def test_official_protocol_configs_still_load(self):
        repo = Path(__file__).resolve().parents[1]
        for path in (
            repo / "configs/collection/ready_grasp_visible.json",
            repo / "outputs/experiments/ready07d_200_20260908/reach_config.json",
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.assertTrue(ReachConfig.load(path).ready_grasp_baseline)

    def test_recovery_false_is_retired_before_any_actuation(self):
        # ready_grasp_recovery=False (the old non-recovery approach/crouch/
        # servo tail) was retired in refactor step 2, family 2 (2026-09-09):
        # _execute_grasp now fails fast instead of ever building an approach
        # band from pre_min/pre_max, but it still records the config snapshot
        # and never touches the arm.
        config = ReachConfig(pre_min=0.20, pre_max=0.24, azimuth_deg=25.0, approach_timeout=0.1)
        with patch.object(sr, "time", Clock()), patch.object(sr, "make_arm") as factory:
            result = sr.execute_grasp(Robot(), "water_bottle", [0.6, 0.0, 0.7], None, config=config)
        self.assertFalse(result.ok)
        self.assertEqual(result.execution["metrics"]["reach_config"]["azimuth_deg"], 25.0)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
