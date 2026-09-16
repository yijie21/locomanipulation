import json
import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from _sim_stubs import Clock, Robot

from locomani.tasks.bottle import execution as sr
from locomani.tasks.bottle.reach_config import DEFAULT_REACH, ReachConfig, require_expert_calibration


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
            # the 07d-200 collection's reach_config.json, kept as a tracked fixture: unit tests
            # must not read outputs/ (2026-09-10, the data directory was lost)
            repo / "tests/fixtures/reach_config_ready07d_200.json",
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


class CalibrationGateTests(unittest.TestCase):
    """The startup refusal, and where it must NOT fire (2026-09-10)."""

    def test_expert_run_without_a_calibration_is_refused_at_startup(self):
        with self.assertRaises(SystemExit) as caught:
            require_expert_calibration(None, ReachConfig())
        self.assertIn("ready_grasp_recovery=False", str(caught.exception))
        self.assertIn("task=bottle_visible", str(caught.exception))

    def test_the_message_names_the_file_when_one_was_given(self):
        with self.assertRaises(SystemExit) as caught:
            require_expert_calibration(None, ReachConfig(), "configs/collection/ready_grasp_baseline.json")
        self.assertIn("ready_grasp_baseline.json", str(caught.exception))

    def test_a_runnable_calibration_passes(self):
        require_expert_calibration(None, ReachConfig(strategy="recovery"))

    def test_a_task_config_never_recalibrates_a_vla_round(self):
        import tempfile

        from locomani import conf
        from locomani.tasks.bottle.cli_args import parse_args

        profile = Path(tempfile.mkdtemp()) / "bottle_visible.json"
        conf.load_group("task", "bottle_visible").dump(profile)
        args = parse_args(["--task-config", str(profile)], vla=True)
        self.assertEqual(args.reach_config_obj, DEFAULT_REACH)
        self.assertEqual(args.spawn_yaw_err, 20.0)
        expert = parse_args(["--task-config", str(profile)])
        self.assertTrue(expert.reach_config_obj.ready_grasp_recovery)

    def test_a_vla_round_is_never_refused(self):
        # Closed-loop eval returns from _p_handover before execute_grasp, and
        # eval_closed_loop.py passes --reach-config only when asked; refusing
        # here would block every eval that did not name a calibration.
        require_expert_calibration(object(), ReachConfig())


if __name__ == "__main__":
    unittest.main()
