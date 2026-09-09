"""A crouched controller is borrowed without another takeover or owned cleanup."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free/tools"))
import diag_crouched_handover as diag


class ContinuousHandoverTests(unittest.TestCase):
    def test_rejected_plan_does_not_restart_or_stop_borrowed_arm(self):
        import adaptive_grasp

        st = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
        st["objects"]["water_bottle"]["quat"] = [1.0, 0.0, 0.0, 0.0]
        cmd = SimpleNamespace(
            heading_offset=0.7, set_upper_body=Mock(), set_upper_body_vel=Mock(), exit_streamed=Mock()
        )
        p = SimpleNamespace(
            obs=lambda: st,
            cmd=cmd,
            pelvis_yaw=lambda s: s["pelvis"]["yaw"],
            pelvis_xy=lambda s: np.array(s["pelvis"]["pos"][:2]),
        )
        arm = SimpleNamespace(
            p=p,
            side="right",
            mode=diag.agent_server.LocomotionMode.IDEL_SQUAT,
            height=0.63,
            start=Mock(),
            hold=Mock(),
            stop=Mock(),
            guard=None,
        )
        meta = dict(
            surfaces={"A": {"top": 0.62}},
            objects={"water_bottle": dict(surface="A", init_pos=[0, 0, 0.70], box={"half_extent": [0.02, 0.02, 0.08]})},
        )
        with (
            patch.object(diag.sr, "PlannerArm", side_effect=AssertionError("second arm constructed")),
            patch.object(adaptive_grasp, "plan", return_value=SimpleNamespace(ok=False, why="unreachable")) as plan,
        ):
            result = diag.in_place_expert(
                p,
                meta,
                "water_bottle",
                SimpleNamespace(event=Mock()),
                {},
                SimpleNamespace(pause=Mock()),
                planner_active=True,
                active_arm=arm,
            )
        self.assertEqual(result, (False, "unreachable"))
        arm.start.assert_not_called()
        arm.hold.assert_not_called()
        arm.stop.assert_not_called()
        cmd.set_upper_body.assert_not_called()
        cmd.set_upper_body_vel.assert_not_called()
        cmd.exit_streamed.assert_not_called()
        self.assertEqual(cmd.heading_offset, 0.7)
        self.assertEqual(arm.height, 0.63)
        np.testing.assert_allclose(plan.call_args.args[1], st["palms"]["right"]["pos"])


if __name__ == "__main__":
    unittest.main()
