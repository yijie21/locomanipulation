import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
from execution_record import ExecutionRecord
from recoverable_grasp_policy import RecoverableGraspPolicy


class RegionStopTests(unittest.TestCase):
    def policy(self, feasible):
        p = object.__new__(RecoverableGraspPolicy)
        state = {"fallen": False, "pelvis": {"pos": [0.0, 0.0, 0.8]}}
        finished = [False]
        p.p = SimpleNamespace(
            obs=lambda: state,
            pelvis_yaw=lambda s: 0.0,
            cmd=SimpleNamespace(frame=Mock(side_effect=lambda **kw: finished.__setitem__(0, True))),
        )
        p.execution = ExecutionRecord()
        p.execution.metrics["recovery"] = {}
        p.config = SimpleNamespace(recovery_region_stop=True)
        p.expired = lambda: finished[0]
        p.floor_clear = lambda *a, **kw: True
        p.walk_tick = lambda s: True
        p.retreat_clear = lambda *a: True
        p.pose_clear = lambda *a: True
        p.retracted = False
        p.select_drop = Mock(return_value=0.1 if feasible else None)
        p.hold = Mock()
        return p

    def test_feasible_region_stops_at_three_cm_without_chasing_point(self):
        p = self.policy(True)
        self.assertTrue(p.move_to_region((np.array([0.03, 0.0]), 0.0, 0.1)))
        p.select_drop.assert_called_once()
        p.p.cmd.frame.assert_not_called()
        self.assertEqual(p.execution.metrics["recovery"]["motions"][0]["stop_reason"], "feasible_grasp_region")

    def test_nearby_but_infeasible_pose_does_not_pass_region_gate(self):
        p = self.policy(False)
        self.assertFalse(p.move_to_region((np.array([0.03, 0.0]), 0.0, 0.1)))
        p.select_drop.assert_called_once()
        p.p.cmd.frame.assert_called_once()

    def test_actual_feasible_grasp_does_not_require_nearness_to_the_hint(self):
        p = self.policy(True)
        self.assertTrue(p.move_to_region((np.array([0.12, 0.0]), 0.3, 0.1)))
        p.select_drop.assert_called_once()
        p.p.cmd.frame.assert_not_called()
