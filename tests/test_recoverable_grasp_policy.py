import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
from execution_record import ExecutionRecord
from reach_config import ReachConfig
from recoverable_grasp_policy import RecoverableGraspPolicy


class Harness(RecoverableGraspPolicy):
    def __init__(self, assess, finishes, contact=False, retract=True):
        self.config = ReachConfig(ready_grasp_baseline=True, ready_grasp_recovery=True)
        self.execution = ExecutionRecord()
        self.execution.metrics["recovery"] = {"attempts": []}
        self.sequence = []
        self.assessments = iter(assess)
        self.finishes = iter(finishes)
        self.contact = contact
        self.retract = retract
        self.deadline = None
        self.state = {"fallen": False, "pelvis": {"pos": [0, 0, 0.8]}}
        self.arm = SimpleNamespace(mode=0, height=-1.0)
        self.p = SimpleNamespace(
            obs=lambda: self.state,
            _facing=lambda _: [1.0, 0.0, 0.0],
            cmd=SimpleNamespace(set_upper_body_vel=lambda _: None),
        )
        self.task = SimpleNamespace(target=SimpleNamespace(key="bottle"), grip=lambda s: np.zeros(3))

    def brake(self):
        return True

    def select_drop(self, state, hint):
        return hint if self.assess(state, hint) else None

    def hold(self):
        self.sequence.append("hold")

    def assess(self, state, drop):
        self.sequence.append("assess")
        return next(self.assessments)

    def finish(self, **kwargs):
        self.sequence.append("grasp")
        if self.contact:
            self.execution.enter("grasp.in")
        return next(self.finishes), "test result"

    def withdraw_and_stand(self):
        self.sequence.append("withdraw_then_stand")
        return self.retract

    def choose_stand(self, state):
        self.sequence.append("choose")
        return np.zeros(2), 0.0, 0.1

    def move_to_region(self, goal):
        self.sequence.append("move")
        return True


class RecoveryFlowTests(unittest.TestCase):
    def run_policy(self, p):
        with patch("stream_reach.toppled", return_value=False):
            return p.execute(grip_world=np.zeros(3), squat_hint=0.1, ghost=None)

    def test_small_error_grasps_without_reposition(self):
        p = Harness([True], [True])
        self.assertTrue(self.run_policy(p)[0])
        self.assertEqual(p.sequence, ["hold", "assess", "grasp"])

    def test_large_error_recovers_and_rechecks_before_grasp(self):
        p = Harness([False, True], [True])
        self.assertTrue(self.run_policy(p)[0])
        self.assertEqual(
            p.sequence, ["hold", "assess", "choose", "withdraw_then_stand", "choose", "move", "assess", "grasp"]
        )

    def test_post_squat_failure_retracts_before_walking(self):
        p = Harness([True, True], [False, True])
        self.assertTrue(self.run_policy(p)[0])
        self.assertLess(p.sequence.index("withdraw_then_stand"), p.sequence.index("move"))
        self.assertEqual(p.sequence.count("grasp"), 2)

    def test_contact_failure_never_repositions(self):
        p = Harness([True], [False], contact=True)
        self.assertFalse(self.run_policy(p)[0])
        self.assertNotIn("move", p.sequence)
        self.assertNotIn("withdraw_then_stand", p.sequence)

    def test_unsafe_retraction_never_stands_or_walks(self):
        p = Harness([False], [], retract=False)
        self.assertFalse(self.run_policy(p)[0])
        self.assertNotIn("move", p.sequence)

    def test_attempt_budget_bounds_unreachable_states(self):
        p = Harness([False] * 3, [])
        self.assertFalse(self.run_policy(p)[0])
        self.assertEqual(p.sequence.count("move"), 2)
        self.assertEqual(p.sequence.count("assess"), 3)

    def test_opt_in_does_not_change_baseline(self):
        self.assertFalse(ReachConfig(ready_grasp_baseline=True).ready_grasp_recovery)
        with self.assertRaises(ValueError):
            ReachConfig(ready_grasp_recovery=True)
        with self.assertRaises(ValueError):
            ReachConfig(recovery_attempts=2.5)


if __name__ == "__main__":
    unittest.main()


class RecoveryGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import json

        import paths
        from arm_ik import ArmIK
        from primitives import STANDING_PELVIS_Z, Primitives
        from ready_grasp_policy import ready_q
        from transport_guard import TransportGuard

        meta = json.loads(Path(paths.SCENE_META).read_text())
        obj = meta["objects"]["water_bottle"]
        xyz = np.asarray(obj["init_pos"]).copy()
        xyz[1] = 0.46
        cls.state = dict(
            joints=ready_q(),
            fallen=False,
            pelvis=dict(pos=[0.04, 0.80, STANDING_PELVIS_Z], quat=[2**-0.5, 0, 0, -(2**-0.5)], yaw=-np.pi / 2),
            objects={"water_bottle": dict(pos=xyz.tolist(), quat=[1, 0, 0, 0])},
            palms={sd: dict(hand_q=[0.024, 0.024]) for sd in ("left", "right")},
        )
        robot = object.__new__(Primitives)
        robot.obs = lambda: cls.state
        task = SimpleNamespace(
            target=SimpleNamespace(key="water_bottle", side="right"),
            grip=lambda state: np.array([xyz[0], xyz[1], meta["surfaces"]["A"]["top"] + 0.10]),
            transport_guard=lambda ik, cfg: TransportGuard(
                ik, "water_bottle", obj["body"], [meta["surfaces"]["A"]["body"]], include_body=True
            ),
        )
        with patch("stream_reach.PlannerArm", side_effect=lambda *a, **kw: SimpleNamespace(ik=ArmIK())):
            cls.policy = RecoverableGraspPolicy(
                robot, task, ReachConfig(ready_grasp_baseline=True, ready_grasp_recovery=True), None, ExecutionRecord()
            )

    def test_far_near_and_lateral_stops_have_feasible_corrections(self):
        from copy import deepcopy

        p = self.policy
        for xy, axis, sign in [([0.04, 0.91], 1, -1), ([0.04, 0.70], 1, 1), ([0.12, 0.80], 0, -1)]:
            state = deepcopy(self.state)
            state["pelvis"]["pos"][:2] = xy
            goal = p.choose_stand(state)
            self.assertIsNotNone(goal)
            self.assertGreater(sign * (goal[0][axis] - xy[axis]), 0.01)
            state["pelvis"]["pos"][:2] = list(goal[0])
            yaw = goal[1]
            state["pelvis"]["yaw"] = yaw
            state["pelvis"]["quat"] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            self.assertTrue(p.assess(state, goal[2]))

    def test_current_geometry_can_select_height_without_moving(self):
        self.assertIsNotNone(self.policy.select_drop(self.state, 0.1))

    def test_table_interior_is_never_a_stand(self):
        self.assertFalse(self.policy.floor_clear([0.0, 0.0]))

    def test_candidate_preserves_measured_tilt(self):
        from copy import deepcopy

        from primitives import Primitives
        from scipy.spatial.transform import Rotation

        state = deepcopy(self.state)
        original = Rotation.from_euler("ZYX", [-np.pi / 2, 0.04, -0.03]).as_matrix()
        state["pelvis"]["quat"] = Rotation.from_matrix(original).as_quat()[[3, 0, 1, 2]].tolist()
        state["pelvis"]["pos"][1] = 0.86
        goal = self.policy.choose_stand(state)
        self.assertIsNotNone(goal)
        state["pelvis"]["pos"][:2] = list(goal[0])
        rotated = Rotation.from_euler("z", goal[1] + np.pi / 2).as_matrix() @ original
        state["pelvis"]["yaw"] = goal[1]
        state["pelvis"]["quat"] = Rotation.from_matrix(rotated).as_quat()[[3, 0, 1, 2]].tolist()
        self.assertTrue(self.policy.assess(state, goal[2]))


class RecoveryIntegrationTests(unittest.TestCase):
    def test_dispatch_does_not_force_old_narrow_band(self):
        from unittest.mock import Mock

        import stream_reach

        controller = SimpleNamespace(execute=Mock(return_value=(True, "")))
        cfg = ReachConfig(ready_grasp_baseline=True, ready_grasp_recovery=True)
        with patch.object(stream_reach, "_creep_to_band") as old:
            result = stream_reach.execute_grasp(None, "bottle", [0, 0, 0], None, coordinated=controller, config=cfg)
            self.assertTrue(result.ok)
            old.assert_not_called()
            controller.execute.assert_called_once()

    def test_failed_attempt_cleanup_preserves_crouched_hold(self):
        import inspect
        from unittest.mock import Mock

        import grasp_gym as gym

        hold = dict(mode=4, height=0.61, speed=0.0, movement=[0.0, 0.0, 0.0], facing=[1.0, 0.0, 0.0])
        signature = inspect.signature(gym._run_round)

        def fail(*args, **kwargs):
            bound = signature.bind(*args, **kwargs).arguments
            bound["rec"]["on"] = True
            bound["execution"].metrics["terminal_hold"] = hold
            return False, "no safe recovery"

        robot = SimpleNamespace(cmd=Mock())
        with patch.object(gym, "_run_round", side_effect=fail):
            result = gym.run_round(
                robot,
                None,
                {},
                "bottle",
                {},
                None,
                None,
                None,
                0.0,
                "right",
                None,
                None,
                None,
                record=True,
                reach_config=ReachConfig(ready_grasp_baseline=True, ready_grasp_recovery=True),
            )
        self.assertFalse(result.ok)
        robot.cmd.frame.assert_called_once_with(**hold)
        robot.cmd.hold.assert_called_once_with(2.0, **hold)
        robot.cmd.key.assert_called_once_with("x")


class RecoveryRetractionTests(unittest.TestCase):
    def test_retraction_moves_both_palms_toward_body(self):
        from arm_ik import ArmIK
        from ready_grasp_policy import ready_q
        from recoverable_grasp_policy import retracted_q

        ik = ArmIK()
        for side in ("left", "right"):
            ready, _ = ik.fk(ready_q(), side)
            back, _ = ik.fk(retracted_q(), side)
            self.assertGreater(ready[0] - back[0], 0.025)

    def test_time_budget_prevents_another_attempt(self):
        p = Harness([], [])
        with patch.object(p, "expired", return_value=True), patch("stream_reach.toppled", return_value=False):
            ok, why = p.execute(grip_world=np.zeros(3), squat_hint=0.1, ghost=None)
        self.assertFalse(ok)
        self.assertIn("time budget", why)
        self.assertNotIn("move", p.sequence)


class CloseStopGeometryTests(RecoveryGeometryTests):
    def test_close_stop_uses_ik_and_clearance_instead_of_navigation_margin(self):
        from copy import deepcopy

        from tabletop_grasp import TabletopGrasp, TabletopTarget

        p = self.policy
        obj = p.meta["objects"]["water_bottle"]
        task = TabletopGrasp(
            p.p,
            TabletopTarget(
                "water_bottle", "right", 0.72, p.meta["surfaces"]["A"]["top"], obj["box"], below_center=0.01
            ),
        )
        state = deepcopy(self.state)
        state["pelvis"]["pos"][:2] = [0.08, p.meta["surfaces"]["A"]["half_extent"][1] + 0.156]
        self.assertFalse(p.floor_clear(state["pelvis"]["pos"][:2]))
        with patch.object(p, "task", task):
            self.assertIsNotNone(p.select_drop(state, 0.10))
            state["pelvis"]["pos"][1] = 0.0
            self.assertIsNone(p.select_drop(state, 0.10))

    def test_close_stop_can_only_retreat_outward(self):
        p = self.policy
        edge = p.meta["surfaces"]["A"]["half_extent"][1]
        self.assertTrue(p.retreat_clear([0.08, edge + 0.156], [0.08, edge + 0.186]))
        self.assertFalse(p.retreat_clear([0.08, edge + 0.156], [0.08, edge + 0.126]))
        self.assertFalse(p.retreat_clear([0.08, edge + 0.156], [0.10, edge + 0.156]))
        self.assertFalse(p.retreat_clear([0.08, edge + 0.04], [0.08, edge + 0.07]))

    def test_body_table_collision_is_screened_even_with_arms_clear(self):
        from copy import deepcopy

        from ready_grasp_policy import ready_q

        state = deepcopy(self.state)
        state["pelvis"]["pos"] = [0.0, 0.0, self.policy.meta["surfaces"]["A"]["top"]]
        result = self.policy.guard.check(state, ready_q(), "body")
        self.assertFalse(result.clear)
