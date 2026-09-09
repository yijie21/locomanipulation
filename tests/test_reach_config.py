import json
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
import stream_reach as sr
from reach_config import DEFAULT_REACH, ReachConfig
from test_grasp_transition import Clock, Robot


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

    def test_real_approach_uses_explicit_bounds(self):
        config = ReachConfig(pre_min=0.20, pre_max=0.24, azimuth_deg=25.0, approach_timeout=0.1)
        with patch.object(sr, "time", Clock()), patch.object(sr, "make_arm") as factory:
            result = sr.execute_grasp(Robot(), "water_bottle", [0.6, 0.0, 0.7], None, config=config)
        self.assertEqual(result.execution["transition"]["pre_band"], [0.20, 0.24])
        self.assertEqual(result.execution["metrics"]["reach_config"]["azimuth_deg"], 25.0)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
