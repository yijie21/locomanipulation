"""Attempt isolation, first-failure preservation, and recorder lifecycle."""

import inspect
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from _sim_stubs import Clock, Robot

from locomani.pipeline.execution_record import ExecutionRecord
from locomani.tasks.bottle import execution as sr
from locomani.tasks.bottle import grasp_gym as gym


class ExecutionTests(unittest.TestCase):
    def test_wall_clock_jump_cannot_reverse_elapsed_time(self):
        mono, wall = [5.0], [100.0]
        record = ExecutionRecord(clock=lambda: mono[0], wall_clock=lambda: wall[0])
        record.enter("approach")
        mono[0] += 2
        wall[0] -= 20
        record.enter("posture")
        record.fail("not delivered")
        record.enter("cleanup")
        record.fail("cleanup error")
        snapshot = record.snapshot()
        self.assertEqual(snapshot["events"][1]["elapsed_s"], 2.0)
        self.assertEqual(snapshot["first_failure"]["stage"], "posture")
        record.transition["later"] = True
        self.assertNotIn("later", snapshot["transition"])
        json.dumps(snapshot, allow_nan=False)

    def test_recovery_false_failure_is_owned_and_never_starts_arm(self):
        # DEFAULT_REACH has ready_grasp_recovery=False; that mode is retired
        # (refactor step 2, family 2, 2026-09-09), so this fails immediately
        # instead of ever reaching the (now-deleted) approach/crouch/servo
        # tail -- but the failure is still owned by ExecutionRecord.fail(),
        # and the arm is still never touched.
        with patch.object(sr, "time", Clock()), patch.object(sr, "make_arm") as factory:
            result = sr.execute_grasp(Robot(), "water_bottle", [0.6, 0.0, 0.7], None)
        self.assertFalse(result.ok)
        self.assertIn("ready_grasp_recovery", result.execution["first_failure"]["reason"])
        self.assertIsNone(result.visibility)
        factory.assert_not_called()

    def run_round(self, robot):
        return gym.run_round(
            robot, None, {}, "water_bottle", {}, None, None, None, 0.0, "right", None, None, None, record=True
        )

    def test_failed_attempt_discards_and_next_attempt_has_no_stale_metrics(self):
        signature = inspect.signature(gym._run_round)
        calls = []

        def task(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            record = bound["execution"]
            calls.append(record)
            if len(calls) == 1:
                bound["rec"]["on"] = True
                record.enter("grasp.up")
                record.transition["reach_started"] = True
                record.metrics["ego_visibility"] = 0.9
                record.fail("bottle toppled")
                return False, "[grasp] bottle toppled"
            return False, "[view] target invisible"

        robot = SimpleNamespace(cmd=Mock())
        with patch.object(gym, "_run_round", side_effect=task):
            failed = self.run_round(robot)
            rejected = self.run_round(robot)
        robot.cmd.key.assert_called_once_with("x")
        self.assertEqual(tuple(failed), (False, "[grasp] bottle toppled", True))
        self.assertEqual(failed.execution["first_failure"]["reason"], "bottle toppled")
        self.assertEqual(rejected.execution["transition"], {})
        self.assertEqual(rejected.execution["metrics"], {})
        self.assertFalse(rejected.produced)

    def test_exception_discards_open_recording_and_propagates(self):
        signature = inspect.signature(gym._run_round)

        def task(*args, **kwargs):
            signature.bind(*args, **kwargs).arguments["rec"]["on"] = True
            raise RuntimeError("connection lost")

        robot = SimpleNamespace(cmd=Mock())
        with patch.object(gym, "_run_round", side_effect=task):
            with self.assertRaisesRegex(RuntimeError, "connection lost"):
                self.run_round(robot)
        robot.cmd.key.assert_called_once_with("x")


if __name__ == "__main__":
    unittest.main()
