"""Exercise transport execution against tracking error and guard rejection."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from locomani.primitives.g1_joints import ARM_MJ
from locomani.tasks.bottle.diag import diagnostic_transport as dt
from locomani.tasks.bottle.transport_guard import SweepResult

ROOT = Path(__file__).resolve().parents[1]


class TransportExecutionTests(unittest.TestCase):
    def setup_case(self):
        state = json.loads((ROOT / "tests/fixtures/crouched_pregrasp.json").read_text())
        q = np.asarray(state["joints"]).copy()
        goal = q.copy()
        goal[ARM_MJ["right"][0]] += 0.2
        path = SimpleNamespace(ok=True, waypoints=[q, goal])
        arm = SimpleNamespace(q=q.copy(), ik=SimpleNamespace(lo=np.full(29, -4.0), hi=np.full(29, 4.0)), _push=Mock())
        p = SimpleNamespace(obs=lambda: state)
        return state, goal, path, arm, p

    def test_preserved_reference_limits_the_first_transport_command(self):
        state, goal, path, arm, p = self.setup_case()
        arm.q[ARM_MJ["right"][0]] -= 0.3
        before = arm.q.copy()
        sent = []

        class FirstCommand(Exception):
            pass

        def push(q, **kwargs):
            self.assertIs(kwargs.get("observed"), state, "transport must reuse its guard observation")
            sent.append(q.copy())
            raise FirstCommand()

        arm._push = push
        guard = SimpleNamespace(check=lambda *args: SweepResult(True, float("inf")))
        with (
            patch.object(dt, "plan_transport", return_value=(path, {"ok": True})),
            patch.object(dt.sr, "toppled", return_value=False),
        ):
            with self.assertRaises(FirstCommand):
                dt.execute_transport(
                    p,
                    arm,
                    SimpleNamespace(),
                    guard,
                    SimpleNamespace(event=Mock()),
                    "water_bottle",
                    preserve_reference=True,
                )
        self.assertLessEqual(np.max(np.abs(sent[0] - before)), 0.5 * dt.sr.DT + 1e-9)

    def test_feedback_cannot_bypass_guard(self):
        state, goal, path, arm, p = self.setup_case()
        count = 0
        cap = SimpleNamespace(event=Mock())

        def check(st, command, side):
            nonlocal count
            count += 1
            # Allow nominal movement, reject the first feedback extension.
            return SweepResult(
                bool(command[ARM_MJ["right"][0]] <= goal[ARM_MJ["right"][0]]), 0.002, "arm", "target", 1.0, 1
            )

        guard = SimpleNamespace(check=check)
        with (
            patch.object(dt, "plan_transport", return_value=(path, {"ok": True})),
            patch.object(dt.sr, "toppled", return_value=False),
        ):
            ok, why = dt.execute_transport(
                p,
                arm,
                SimpleNamespace(wrist=np.array(state["palms"]["right"]["pos"]) + [1.0, 0.0, 0.0], rot=np.eye(3)),
                guard,
                cap,
                "water_bottle",
                feedback=True,
            )
        self.assertFalse(ok)
        self.assertIn("clearance", why)
        self.assertEqual(arm._push.call_count, count - 1)
        self.assertTrue(
            all(c.args[0][ARM_MJ["right"][0]] <= goal[ARM_MJ["right"][0]] for c in arm._push.call_args_list)
        )

    def test_unresponsive_robot_times_out_despite_compensated_reference(self):
        state, goal, path, arm, p = self.setup_case()
        cap = SimpleNamespace(event=Mock())
        guard = SimpleNamespace(check=lambda *a: SweepResult(True, float("inf")))
        now = [0.0]

        def tick():
            now[0] += 0.01
            return now[0]

        with (
            patch.object(dt, "plan_transport", return_value=(path, {"ok": True})),
            patch.object(dt.sr, "toppled", return_value=False),
            patch.object(dt.time, "monotonic", side_effect=tick),
        ):
            ok, why = dt.execute_transport(
                p,
                arm,
                SimpleNamespace(wrist=np.array(state["palms"]["right"]["pos"]) + [1.0, 0.0, 0.0], rot=np.eye(3)),
                guard,
                cap,
                "water_bottle",
                feedback=True,
            )
        self.assertFalse(ok)
        self.assertIn("tracking timeout", why)
        names = [c.args[0] for c in cap.event.call_args_list]
        self.assertNotIn("transport_path_delivered", names)
        commands = np.array([c.args[0] for c in arm._push.call_args_list])
        self.assertLessEqual(np.max(np.abs(np.diff(commands, axis=0))), 0.5 * dt.sr.DT + 1e-9)
        joint = ARM_MJ["right"][0]
        self.assertLessEqual(commands[:, joint].max(), goal[joint] + 0.35 + 1e-9)
        fixed = [i for i in range(29) if i not in ARM_MJ["right"]]
        np.testing.assert_allclose(
            commands[:, fixed], np.broadcast_to(np.asarray(state["joints"])[fixed], commands[:, fixed].shape)
        )

    def test_final_acceptance_uses_delivered_pose_not_unique_ik_joints(self):
        state, goal, path, arm, p = self.setup_case()
        rot = dt.Primitives._rot(state["palms"]["right"]["quat"])
        col = 0 if dt.sr._HAND == "dex1" else 1
        grasp = SimpleNamespace(
            rot=rot, wrist=np.asarray(state["palms"]["right"]["pos"]) + rot[:, col] * dt.sr.SAFE_GAP
        )
        cap = SimpleNamespace(event=Mock())
        guard = SimpleNamespace(check=lambda *a: SweepResult(True, float("inf")))
        with (
            patch.object(dt, "plan_transport", return_value=(path, {"ok": True})),
            patch.object(dt.sr, "toppled", return_value=False),
        ):
            ok, why = dt.execute_transport(p, arm, grasp, guard, cap, "water_bottle")
        self.assertTrue(ok, why)
        self.assertGreater(abs(goal[ARM_MJ["right"][0]] - state["joints"][ARM_MJ["right"][0]]), 0.12)
        self.assertEqual(cap.event.call_args.args[0], "transport_path_delivered")

    def test_position_and_joint_arrival_do_not_hide_wrong_orientation(self):
        state, goal, path, arm, p = self.setup_case()
        path.waypoints[-1] = np.asarray(state["joints"]).copy()
        actual = dt.Primitives._rot(state["palms"]["right"]["quat"])
        angle = 0.5
        yaw = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        rot = yaw @ actual
        col = 0 if dt.sr._HAND == "dex1" else 1
        grasp = SimpleNamespace(
            rot=rot, wrist=np.asarray(state["palms"]["right"]["pos"]) + rot[:, col] * dt.sr.SAFE_GAP
        )
        cap = SimpleNamespace(event=Mock())
        guard = SimpleNamespace(check=lambda *a: SweepResult(True, float("inf")))
        with (
            patch.object(dt, "plan_transport", return_value=(path, {"ok": True})),
            patch.object(dt.sr, "toppled", return_value=False),
            patch.object(dt.time, "monotonic", side_effect=iter(range(100))),
        ):
            ok, why = dt.execute_transport(p, arm, grasp, guard, cap, "water_bottle")
        self.assertFalse(ok)
        self.assertIn("tracking timeout", why)


if __name__ == "__main__":
    unittest.main()
