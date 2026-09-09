import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src/teleop_free"), str(Path(__file__).resolve().parent)]
from types import SimpleNamespace
from unittest.mock import Mock, patch

from coupled_grasp_policy import CoupledGraspPolicy, screen_motion
from execution_record import ExecutionRecord
from reach_config import ReachConfig
from ready_grasp_policy import OFFSET
from recoverable_grasp_policy import RecoverableGraspPolicy
from test_transport_guard import TransportGuardTests
from transport_guard import TransportGuard


class CoupledTests(unittest.TestCase):
    def test_descent_collision_rejected_even_when_static_arm_is_clear(self):
        fixture = TransportGuardTests()
        q = -0.5
        ik, state = fixture.setup_scene(q=q, target=(np.cos(q), np.sin(q), -0.13))
        guard = TransportGuard(ik, "bottle", "target")
        self.assertIsNone(screen_motion(guard, state, [q], 0.0, [1.0, 0.0, 0.0]))
        original = deepcopy(state)
        self.assertIsNotNone(screen_motion(guard, state, [q], -0.06, [1.0, 0.0, 0.0]))
        self.assertEqual(state, original)

    def test_strategy_is_opt_in(self):
        self.assertFalse(ReachConfig().coupled_squat)
        with self.assertRaises(ValueError):
            ReachConfig(coupled_squat=True)
        config = ReachConfig.load("configs/collection/ready_grasp_coupled.json")
        self.assertTrue(config.coupled_squat)
        self.assertEqual(config.arrival_advance, 0.03)

    def cached_policy(self, target_shift=0.0, blocked=False):
        p = object.__new__(CoupledGraspPolicy)
        p.side = "right"
        p._prepared = (0.1, np.zeros(29), np.eye(3))
        p.execution = ExecutionRecord()
        p.execution.metrics["brake_planning"] = [dict(reused=False)]
        p.floor_clear = lambda *a, **kw: True
        p.guard = SimpleNamespace(
            hands=["right", "left", "body"], check=lambda *a, **kw: SimpleNamespace(clear=not blocked)
        )
        p.arm = SimpleNamespace(ik=SimpleNamespace(fk=Mock(return_value=(np.zeros(3), np.eye(3)))))
        p.task = SimpleNamespace(grip=lambda s: OFFSET + np.array([target_shift, 0.0, 0.7]))
        state = dict(
            joints=np.zeros(29),
            pelvis=dict(pos=[0.0, 0.0, 0.8], quat=[1.0, 0.0, 0.0, 0.0]),
            palms={"right": dict(pos=[0.0, 0.0, 0.8], quat=[1.0, 0.0, 0.0, 0.0])},
        )
        return p, state

    def test_prepared_candidate_reuses_ik_after_current_geometry_check(self):
        p, state = self.cached_policy()
        with patch.object(RecoverableGraspPolicy, "select_drop") as fallback:
            self.assertEqual(p.select_drop(state, 0.1), 0.1)
            fallback.assert_not_called()
        self.assertTrue(p.execution.metrics["brake_planning"][0]["reused"])

    def test_target_motion_or_new_obstacle_invalidates_prepared_candidate(self):
        for shift, blocked in ((0.05, False), (0.0, True)):
            p, state = self.cached_policy(shift, blocked)
            with patch.object(RecoverableGraspPolicy, "select_drop", return_value=None) as fallback:
                self.assertIsNone(p.select_drop(state, 0.1))
                fallback.assert_called_once()
            self.assertFalse(p.execution.metrics["brake_planning"][0]["reused"])

    def test_region_probe_can_reuse_candidate_without_brake_planning_log(self):
        p, state = self.cached_policy()
        p.execution.metrics.pop("brake_planning")
        p.execution.enter("recovery_reposition")
        with patch.object(RecoverableGraspPolicy, "select_drop") as fallback:
            self.assertEqual(p.select_drop(state, 0.1), 0.1)
            fallback.assert_not_called()

    def test_region_candidate_survives_braking_for_fresh_geometry_revalidation(self):
        p, _ = self.cached_policy()
        p.config = SimpleNamespace(recovery_region_stop=True)
        prepared = p._prepared
        with patch.object(RecoverableGraspPolicy, "brake", return_value=True):
            self.assertTrue(p.brake())
        self.assertIs(p._prepared, prepared)

    def test_legacy_braking_still_starts_without_a_cached_candidate(self):
        p, _ = self.cached_policy()
        p.config = SimpleNamespace(recovery_region_stop=False)
        with patch.object(RecoverableGraspPolicy, "brake", return_value=True):
            p.brake()
        self.assertIsNone(p._prepared)

    def test_pre_alignment_failure_still_forbids_blind_stand_retry(self):
        p = object.__new__(CoupledGraspPolicy)
        p.config = SimpleNamespace()
        p.execution = ExecutionRecord()
        p.finish_cleanup = Mock()
        p.coupled_transport = Mock(return_value=(False, "wrist path blocked"))
        p.finish(grip_world=None, squat_hint=0.1, ghost=None)
        self.assertFalse(p.execution.metrics["measured_grasp"][-1]["can_reposition"])
