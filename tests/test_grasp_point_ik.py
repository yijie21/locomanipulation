import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.sonic.primitives import Primitives
from locomani.primitives.arm_ik import LAST_IK, ArmIK
from locomani.primitives.g1_joints import ARM_MJ
from locomani.tasks.bottle.grasp_capture import OFFSET
from locomani.tasks.bottle.ready_grasp_policy import ready_q


class GraspPointIKTests(unittest.TestCase):
    def test_closing_servo_does_not_apply_the_tool_offset_twice(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from locomani.backends.sonic.primitives import HAND_OPEN
        from locomani.backends.sonic.stream_reach import _servo_continuous
        from locomani.pipeline.execution_record import ExecutionRecord

        palm = np.array([0.0, 0.0, 0.7])
        target = palm + OFFSET
        state = dict(
            fallen=False,
            joints=np.zeros(29),
            pelvis=dict(pos=[0.0, 0.0, 0.8], quat=[1.0, 0.0, 0.0, 0.0], yaw=0.0),
            palms={"right": dict(pos=palm, quat=[1.0, 0.0, 0.0, 0.0], hand_q=[0.0245, 0.0245])},
            objects={"bottle": dict(pos=target, quat=[1.0, 0.0, 0.0, 0.0], touching=[], touch_links=[])},
        )
        robot = SimpleNamespace(
            obs=lambda: state,
            state=SimpleNamespace(latest=lambda _: state),
            object_pos=lambda *_: target.copy(),
            world_to_root_full=lambda p, s: p,
            log=Mock(),
        )
        solver = Mock(return_value=(np.zeros(29), 0.0, 0.0))

        def push(q):
            state["fallen"] = True  # stop after examining one real servo tick

        arm = SimpleNamespace(
            q=np.zeros(29), hands={"right": HAND_OPEN.copy()}, ik=SimpleNamespace(solve=solver), _push=push
        )
        plan = SimpleNamespace(rot=np.eye(3), wrist=palm)
        _servo_continuous(
            robot,
            arm,
            "bottle",
            plan,
            "right",
            close_from_current=True,
            grip_target=lambda s: target,
            reference_guard=lambda *a: None,
            execution=ExecutionRecord(),
            grasp_point_control=True,
        )
        solver.assert_called_once()
        np.testing.assert_allclose(solver.call_args.args[1], target, atol=1e-8)
        np.testing.assert_allclose(solver.call_args.kwargs["point_offset"], OFFSET)

    def test_orientation_residual_does_not_become_a_lever_arm_position_error(self):
        ik = ArmIK()
        q = ready_q()
        q[22], q[25] = 0.1, 0.15
        palm, actual_rotation = ik.fk(q, "right")
        target = palm + actual_rotation @ OFFSET
        desired_rotation = actual_rotation @ Rotation.from_euler("y", 0.25).as_matrix()
        # Two position DOFs remain; the exact desired attitude is deliberately
        # unavailable. This reproduces the constrained comfortable-arm solve.
        for j in ARM_MJ["right"]:
            if j not in (22, 25):
                ik.lo[j] = ik.hi[j] = q[j]
        solved, error, rotation_error = ik.solve(
            "right", target, desired_rotation, q_mj29=q, iters=200, rot_weight=0.01, point_offset=OFFSET
        )
        wrist, rotation = ik.fk(solved, "right")
        self.assertGreater(rotation_error, 0.1)
        self.assertLess(np.linalg.norm(wrist + rotation @ OFFSET - target), 0.003)
        self.assertAlmostEqual(error, np.linalg.norm(wrist + rotation @ OFFSET - target))
        # The sidecar's existing IK field remains a wrist pose, not a tool
        # center silently mislabelled as a wrist command.
        np.testing.assert_allclose(LAST_IK["right"][0], wrist, atol=1e-8)
        np.testing.assert_allclose(Primitives._rot(LAST_IK["right"][1]), rotation, atol=1e-8)

    def test_zero_offset_matches_the_existing_wrist_solver(self):
        ik = ArmIK()
        q = ready_q()
        palm, rotation = ik.fk(q, "right")
        target = palm + [0.01, 0.0, -0.01]
        old = ik.solve("right", target, rotation, q_mj29=q, iters=30)
        point = ik.solve("right", target, rotation, q_mj29=q, iters=30, point_offset=np.zeros(3))
        np.testing.assert_allclose(old[0], point[0], atol=1e-8)
        np.testing.assert_allclose(old[1:], point[1:], atol=1e-8)
