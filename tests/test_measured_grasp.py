import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from locomani.tasks.bottle.measured_grasp import CaptureStability, closure_clear, closure_geometry
from locomani.tasks.bottle.ready_grasp_policy import OFFSET, PRE_GAP


class CaptureGeometryTests(unittest.TestCase):
    def state(self):
        return dict(
            joints=np.zeros(29),
            palms={"right": dict(pos=[0.0, 0.0, 0.7], quat=[1.0, 0.0, 0.0, 0.0], hand_q=[0.0245, 0.0245])},
        )

    def test_seated_object_can_close_without_visiting_pregrasp(self):
        state = self.state()
        target = np.asarray(state["palms"]["right"]["pos"]) + OFFSET
        self.assertTrue(closure_geometry(state, "right", target)["ready"])
        # The old waypoint gate rejects this same physically seated pose.
        pre = target - OFFSET - np.array([PRE_GAP, 0.0, 0.0])
        self.assertGreater(np.linalg.norm(pre - state["palms"]["right"]["pos"]), 0.025)

    def test_near_fingertips_is_not_seated(self):
        state = self.state()
        target = np.asarray(state["palms"]["right"]["pos"]) + OFFSET + [0.05, 0.0, 0.0]
        self.assertFalse(closure_geometry(state, "right", target)["ready"])

    def test_lateral_and_height_misses_do_not_close(self):
        state = self.state()
        for delta in ([0.0, 0.030, 0.0], [0.0, 0.0, 0.025]):
            target = np.asarray(state["palms"]["right"]["pos"]) + OFFSET + delta
            self.assertFalse(closure_geometry(state, "right", target)["ready"])

    def test_geometry_is_world_pose_invariant(self):
        from scipy.spatial.transform import Rotation

        state = self.state()
        rot = Rotation.from_euler("z", 1.2)
        state["palms"]["right"]["pos"] = [1.0, 2.0, 0.7]
        state["palms"]["right"]["quat"] = rot.as_quat()[[3, 0, 1, 2]].tolist()
        target = np.asarray(state["palms"]["right"]["pos"]) + rot.apply(OFFSET)
        self.assertTrue(closure_geometry(state, "right", target)["ready"])

    def test_closure_sweep_checks_intermediate_fingers_without_mutating_state(self):
        state = self.state()
        seen = []

        def check(trial, q, phase):
            seen.append(trial["palms"]["right"]["hand_q"][0])
            return "table blocks closure" if trial["palms"]["right"]["hand_q"][0] < 0.01 else None

        policy = SimpleNamespace(side="right", reference_check=check)
        self.assertEqual(closure_clear(policy, state), "table blocks closure")
        self.assertGreater(len(seen), 1)
        self.assertEqual(state["palms"]["right"]["hand_q"], [0.0245, 0.0245])

    def test_close_handover_rejects_unseated_object(self):
        from locomani.backends.sonic.stream_reach import _servo_continuous

        state = self.state()
        p = SimpleNamespace(obs=lambda: state, object_pos=lambda *_: np.array([0.3, 0.0, 0.7]))
        plan = SimpleNamespace(rot=np.eye(3), wrist=np.array([0.0, 0.0, 0.7]))
        ok, why = _servo_continuous(
            p,
            None,
            "bottle",
            plan,
            "right",
            close_from_current=True,
            reference_guard=Mock(),
            grip_target=lambda s: np.array([0.3, 0.0, 0.7]),
        )
        self.assertFalse(ok)
        self.assertIn("geometry not delivered", why)

    def test_seated_capture_reaches_close_without_ik_or_pregrasp(self):
        self.assert_seated_capture(np.zeros(3))

    def test_safe_nonpreferred_capture_closes_within_execution_interval(self):
        self.assert_seated_capture(np.array([0.018, 0.018, 0.018]))

    def assert_seated_capture(self, delta):
        from unittest.mock import patch

        from locomani.pipeline.execution_record import ExecutionRecord
        from locomani.tasks.bottle.measured_grasp import run

        state = self.state()
        state["fallen"] = False
        state["pelvis"] = {"quat": [1.0, 0.0, 0.0, 0.0], "pos": [0.0, 0.0, 0.8]}
        target = np.asarray(state["palms"]["right"]["pos"]) + OFFSET + delta
        state["objects"] = {"bottle": {"pos": target, "touching": []}}
        clock = [0.0]

        def push(q):
            clock[0] += 0.05

        arm = SimpleNamespace(height=0.6, q=np.zeros(29), _push=push, ik=Mock())
        arm.ik.solve.side_effect = AssertionError("Safe seated grasp should close, not chase precision")
        robot = SimpleNamespace(obs=lambda: state, object_pos=lambda *_: target, world_to_root_full=lambda xyz, s: xyz)
        policy = SimpleNamespace(
            p=robot,
            arm=arm,
            execution=ExecutionRecord(),
            side="right",
            config=SimpleNamespace(timely_capture=True),
            task=SimpleNamespace(target=SimpleNamespace(key="bottle", surface_top=0.6), grip=lambda s: target),
            ramp=Mock(),
            expired=lambda: False,
            reference_check=Mock(return_value=None),
            guard=Mock(),
        )
        with (
            patch("locomani.tasks.bottle.measured_grasp.templates", return_value=[(None, target, np.eye(3))]),
            patch("locomani.tasks.bottle.measured_grasp.closure_clear", return_value=None),
            patch("locomani.tasks.bottle.measured_grasp.time.monotonic", side_effect=lambda: clock[0]),
            patch("locomani.backends.sonic.stream_reach.toppled", return_value=False),
            patch("locomani.backends.sonic.stream_reach._servo_continuous", return_value=(True, "")) as servo,
        ):
            self.assertTrue(run(policy, None)[0])
        arm.ik.solve.assert_not_called()
        self.assertTrue(servo.call_args.kwargs["close_from_current"])
        self.assertLessEqual(clock[0], 0.4)
        self.assertEqual([e["stage"] for e in policy.execution.events], ["grasp.local_align", "grasp.capture_ready"])


class CaptureStabilityTests(unittest.TestCase):
    def test_one_finger_or_wrist_touch_is_not_a_bilateral_grasp(self):
        from locomani.tasks.bottle.grasp_capture import bilateral_dex1_contact

        for links in ([], ["R.wrist_yaw"], ["R.dex1_finger_1"], ["R.dex1_finger_1", "L.dex1_finger_2"]):
            self.assertFalse(bilateral_dex1_contact(dict(touch_links=links), "right"))
        self.assertTrue(bilateral_dex1_contact(dict(touch_links=["R.dex1_finger_1", "R.dex1_finger_2"]), "right"))

    def test_threshold_noise_does_not_postpone_closure(self):
        gate = CaptureStability()
        decisions = [gate.update(i * 0.02, True, 0.034 if i % 2 == 0 else 0.037) for i in range(17)]
        self.assertTrue(any(decisions))

    def test_hysteresis_never_closes_while_fast_or_unsafe(self):
        for safe, speed in ((False, 0.01), (True, 0.05)):
            gate = CaptureStability()
            for i in range(14):
                gate.update(i * 0.02, True, 0.01)
            self.assertFalse(gate.update(0.28, safe, speed))
            self.assertFalse(gate.update(0.30, True, 0.01))

    def test_long_observation_gap_cannot_satisfy_dwell(self):
        gate = CaptureStability()
        self.assertFalse(gate.update(0.0, True, 0.01))
        self.assertFalse(gate.update(0.4, True, 0.01))


class SharedTargetExecutionTests(unittest.TestCase):
    def test_plan_feasible_level_orientation_can_continue_guarded_alignment(self):
        self.check_level_alignment_candidate(None, True)

    def test_relaxed_template_error_never_bypasses_collision_rejection(self):
        self.check_level_alignment_candidate("wrist would hit bottle", False)

    def check_level_alignment_candidate(self, rejection, should_push):
        from unittest.mock import patch

        from locomani.pipeline.execution_record import ExecutionRecord
        from locomani.tasks.bottle.measured_grasp import run

        state = CaptureGeometryTests().state()
        state.update(fallen=False, pelvis={"pos": [0.0, 0.0, 0.8], "quat": [1.0, 0.0, 0.0, 0.0]})
        target = np.asarray(state["palms"]["right"]["pos"]) + OFFSET - [0.04, 0.0, 0.0]
        state["objects"] = {"bottle": {"pos": target, "touching": []}}
        pushed = []
        # Residuals re-solved from candidate_02's recorded failed IK target,
        # trimmed just under the 0.15 rad bound (the level-bottle experiment's
        # lenient tolerance that used to admit 0.150909 rad here is retired).
        arm = SimpleNamespace(
            height=0.6,
            q=np.zeros(29),
            _push=lambda q: pushed.append(q),
            ik=SimpleNamespace(solve=Mock(return_value=(np.zeros(29), 0.002271, 0.149))),
        )
        robot = SimpleNamespace(obs=lambda: state, object_pos=lambda *_: target, world_to_root_full=lambda xyz, s: xyz)
        policy = SimpleNamespace(
            p=robot,
            arm=arm,
            execution=ExecutionRecord(),
            side="right",
            config=SimpleNamespace(timely_capture=True),
            task=SimpleNamespace(target=SimpleNamespace(key="bottle"), grip=lambda s: target),
            ramp=Mock(),
            expired=lambda: bool(pushed),
            reference_check=Mock(return_value=rejection),
        )
        with (
            patch("locomani.tasks.bottle.measured_grasp.templates", return_value=[(None, target, np.eye(3))]),
            patch("locomani.backends.sonic.stream_reach.toppled", return_value=False),
        ):
            run(policy, None)
        self.assertEqual(bool(pushed), should_push)
        policy.reference_check.assert_called_once()

    def test_handover_uses_live_grip_even_if_body_center_is_seated(self):
        from locomani.backends.sonic.stream_reach import _servo_continuous

        state = CaptureGeometryTests().state()
        center = np.asarray(state["palms"]["right"]["pos"]) + OFFSET
        p = SimpleNamespace(obs=lambda: state, object_pos=lambda *_: center)
        plan = SimpleNamespace(rot=np.eye(3), wrist=np.array([0.0, 0.0, 0.7]))
        target = Mock(return_value=center + [0.0, 0.0, -0.03])
        ok, why = _servo_continuous(
            p, None, "bottle", plan, "right", close_from_current=True, reference_guard=Mock(), grip_target=target
        )
        self.assertFalse(ok)
        self.assertIn("geometry not delivered", why)
        target.assert_called_once_with(state)

    def test_near_target_commands_arm_retraction_without_base_movement(self):
        from unittest.mock import patch

        from locomani.pipeline.execution_record import ExecutionRecord
        from locomani.tasks.bottle.measured_grasp import run

        state = CaptureGeometryTests().state()
        state.update(fallen=False, pelvis={"pos": [0.0, 0.0, 0.8], "quat": [1.0, 0.0, 0.0, 0.0]})
        target = np.asarray(state["palms"]["right"]["pos"]) + OFFSET - [0.04, 0.0, 0.0]
        state["objects"] = {"bottle": {"pos": target, "touching": []}}
        pushed = []
        arm = SimpleNamespace(
            height=0.6,
            q=np.zeros(29),
            _push=lambda q: pushed.append(q),
            ik=SimpleNamespace(solve=Mock(return_value=(np.zeros(29), 0.0, 0.0))),
        )
        robot = SimpleNamespace(
            obs=lambda: state, object_pos=lambda *_: target, world_to_root_full=lambda xyz, s: xyz, cmd=Mock()
        )
        policy = SimpleNamespace(
            p=robot,
            arm=arm,
            execution=ExecutionRecord(),
            side="right",
            config=SimpleNamespace(),
            task=SimpleNamespace(target=SimpleNamespace(key="bottle"), grip=lambda s: target),
            ramp=Mock(),
            expired=lambda: bool(pushed),
            reference_check=lambda *args: None,
        )
        with (
            patch("locomani.tasks.bottle.measured_grasp.templates", return_value=[(None, target, np.eye(3))]),
            patch("locomani.backends.sonic.stream_reach.toppled", return_value=False),
        ):
            run(policy, None)
        commanded = arm.ik.solve.call_args.args[1]
        self.assertLess(commanded[0], state["palms"]["right"]["pos"][0])
        self.assertLessEqual(np.linalg.norm(commanded - state["palms"]["right"]["pos"]), 0.002501)
        self.assertEqual(len(pushed), 1)
        self.assertEqual(robot.cmd.mock_calls, [])
