"""Planner-to-planner handover must not apply the VLA heading reset."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free/tools"))
import diag_crouched_handover as diag


class BeforeArmConstruction(Exception):
    pass


class PlannerHandoverTests(unittest.TestCase):
    def check_mode(self, planner_active):
        state = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
        state["objects"]["water_bottle"]["quat"] = [1.0, 0.0, 0.0, 0.0]
        cmd = SimpleNamespace(
            heading_offset=0.7, set_upper_body=Mock(), set_upper_body_vel=Mock(), exit_streamed=Mock()
        )
        robot = SimpleNamespace(obs=lambda: state, cmd=cmd, pelvis_yaw=lambda st: st["pelvis"]["yaw"])
        with patch.object(diag.sr, "PlannerArm", side_effect=BeforeArmConstruction), patch.object(diag.time, "sleep"):
            with self.assertRaises(BeforeArmConstruction):
                diag.in_place_expert(
                    robot,
                    {},
                    "water_bottle",
                    SimpleNamespace(event=Mock()),
                    {},
                    SimpleNamespace(pause=Mock()),
                    planner_active=planner_active,
                )
        return cmd, state

    def test_planner_handover_preserves_calibrated_heading(self):
        cmd, _ = self.check_mode(True)
        self.assertEqual(cmd.heading_offset, 0.7)
        cmd.exit_streamed.assert_not_called()

    def test_vla_handover_retains_original_channel_switch(self):
        cmd, state = self.check_mode(False)
        self.assertEqual(cmd.heading_offset, state["pelvis"]["yaw"])
        self.assertEqual(cmd.exit_streamed.call_count, 2)


if __name__ == "__main__":
    unittest.main()
