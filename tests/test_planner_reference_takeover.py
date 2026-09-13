"""An already frozen Planner reference must survive explicit same-channel takeover."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from locomani.backends.sonic.primitives import READY_UPPER, upper_body_from_mj
from locomani.tasks.bottle.execution import PlannerArm


class PlannerReferenceTests(unittest.TestCase):
    def arm(self):
        st = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
        cmd = SimpleNamespace(set_upper_body=Mock(), set_upper_body_vel=Mock())
        arm = PlannerArm.__new__(PlannerArm)
        arm.p = SimpleNamespace(obs=lambda: st, cmd=cmd, _facing=lambda s: (1.0, 0.0, 0.0))
        arm.hold = Mock()
        return arm, cmd, st

    def test_explicit_reference_is_preserved_and_velocity_starts_at_zero(self):
        arm, cmd, st = self.arm()
        reference = READY_UPPER.copy()
        arm.start(upper_reference=reference)
        np.testing.assert_allclose(cmd.set_upper_body.call_args.args[0], reference)
        np.testing.assert_array_equal(arm.q[:12], st["joints"][:12])
        np.testing.assert_allclose(upper_body_from_mj(arm._q_prev), reference)
        np.testing.assert_array_equal(cmd.set_upper_body_vel.call_args.args[0], np.zeros(17))
        reference[:] = 99
        self.assertLess(np.max(np.abs(arm.q)), 10)

    def test_default_entry_retains_measured_seed(self):
        arm, cmd, st = self.arm()
        arm.start()
        np.testing.assert_allclose(cmd.set_upper_body.call_args.args[0], upper_body_from_mj(st["joints"]))
        arm.hold.assert_called_once_with(0.6)

    def test_settled_owner_can_take_over_without_another_wait(self):
        arm, cmd, _ = self.arm()
        arm.start(upper_reference=READY_UPPER, settle_seconds=0)
        arm.hold.assert_called_once_with(0)
        np.testing.assert_allclose(cmd.set_upper_body.call_args.args[0], READY_UPPER)

    def test_immediate_unowned_or_invalid_hold_is_rejected_before_publish(self):
        for seconds in [0, -1, float("nan"), float("inf")]:
            arm, cmd, _ = self.arm()
            with self.assertRaises(ValueError):
                arm.start(settle_seconds=seconds)
            cmd.set_upper_body.assert_not_called()

    def test_invalid_reference_rejected_before_publishing(self):
        for reference in [np.zeros(16), np.full(17, np.nan)]:
            arm, cmd, _ = self.arm()
            with self.assertRaises(ValueError):
                arm.start(upper_reference=reference)
            cmd.set_upper_body.assert_not_called()


if __name__ == "__main__":
    unittest.main()
