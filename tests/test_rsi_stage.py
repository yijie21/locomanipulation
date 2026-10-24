"""TODO 33.16: the step runner (src/locomani/tasks/scene/stage.py) on fake motors, no simulation."""

from __future__ import annotations

import unittest

import numpy as np

from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene import stage as S


class FakeMotor:
    def __init__(self):
        self.t, self.fell_at, self.trace = 0.0, None, []
        self.jaw = np.zeros(3)
        self.pel, self.yaw_v = np.zeros(3), 0.0
        self.opening, self.between, self.touch = 0.0, set(), {}

    def pelvis(self):
        return self.pel.copy(), np.array([1.0, 0, 0, 0])

    def yaw(self):
        return self.yaw_v

    def jaw_centre(self, side):
        return self.jaw.copy(), np.eye(3)

    def jaw_opening(self, side):
        return self.opening

    def between_fingers(self, side):
        return self.between

    def touching(self, body):
        return self.touch.get(body, set())


class FakeCtx:
    def __init__(self, motor):
        self.motor, self.log = motor, []

    def note(self, name, out, **extra):
        self.log.append(dict(step=name, ok=out.ok, why=out.why, **out.info, **extra))
        return out


def step(m, seconds=2.0, ok=True):
    def run():
        t0 = m.t
        m.t += seconds
        return Outcome(ok, "", t0, m.t, {"jaw_err_cm": 1.0})

    return run


class Stage(unittest.TestCase):
    def setUp(self):
        self.m = FakeMotor()
        self.ctx = FakeCtx(self.m)

    def test_retry_spends_the_shared_budget(self):
        tries = []

        def check(out):
            return len(tries) >= 2, dict(n=len(tries))

        def on_fail(attempt, values):
            tries.append(attempt)
            return step(self.m, 3.0)

        out = S.stage(self.ctx, "pregrasp", step(self.m), check, on_fail)
        self.assertTrue(out.ok)
        self.assertEqual([r["attempt"] for r in self.ctx.log], [0, 1, 2])
        b = self.ctx.stage_budget
        self.assertEqual((b.retries, b.extra_s), (2, 6.0))
        out2 = S.stage(self.ctx, "entry", step(self.m), lambda o: (False, {}), lambda a, v: step(self.m))
        self.assertFalse(out2.ok)
        self.assertIn("budget spent", out2.why)
        self.assertEqual(b.retries, 3)  # one more retry, then the round's budget (3) is gone

    def test_no_retry_when_on_fail_gives_none(self):
        out = S.stage(self.ctx, "grip", step(self.m), lambda o: (False, dict(opening_cm=0.01)), lambda a, v: None)
        self.assertFalse(out.ok)
        self.assertIn("no retry", out.why)
        self.assertEqual(len(self.ctx.log), 1)
        self.assertEqual(self.ctx.log[0]["check"], dict(opening_cm=0.01))

    def test_a_fall_ends_the_step(self):
        def falls():
            self.m.fell_at = 4.2
            return Outcome(False, "fell", 0.0, 4.2)

        out = S.stage(self.ctx, "walk", falls, lambda o: (False, {}), lambda a, v: step(self.m))
        self.assertFalse(out.ok)
        self.assertIn("fell at 4.20", out.why)
        self.assertEqual(len(self.ctx.log), 1)

    def test_check_values_are_logged_with_the_motor_info(self):
        S.stage(self.ctx, "lift", step(self.m), lambda o: (True, dict(rise_cm=9.8)))
        row = self.ctx.log[0]
        self.assertEqual(
            (row["step"], row["check"], row["attempt"], row["jaw_err_cm"]), ("lift", {"rise_cm": 9.8}, 0, 1.0)
        )

    def test_no_check_keeps_the_motor_verdict(self):
        self.assertFalse(S.stage(self.ctx, "walk", step(self.m, ok=False)).ok)
        self.assertTrue(S.stage(self.ctx, "walk", step(self.m, ok=True)).ok)

    def test_track_ratio_from_a_synthetic_trace(self):
        for k in range(11):  # a reach commanding the right wrist 20 cm forward; it moves 2 cm
            ref = np.array([[0, 0, 0], [0.02 * k, 0, 0]])
            act = np.array([[0, 0, 0], [0.002 * k, 0, 0]])
            self.m.trace.append(dict(t=0.02 * k, primitive="reach", ref_wrists=ref, wrists=act))
        self.assertAlmostEqual(S.track_ratio(self.m, "right"), 0.1)
        self.m.jaw = np.array([0.0, 0.0, 0.0])
        ok, values = S.reach_check(self.m, "right", [0.0, 0.0, 0.01])
        self.assertFalse(ok)  # within 3 cm, yet the arm did not follow its command
        self.assertEqual(values["track_ratio"], 0.1)
        self.assertIsNone(S.track_ratio(self.m, "left"))  # the left wrist was not commanded

    def test_stance_grip_and_lift_checks(self):
        self.m.pel, self.m.yaw_v = np.array([1.03, 0.0, 0.78]), np.radians(5.0)
        ok, v = S.stance_check(self.m, [1.0, 0.0], 0.0)
        self.assertTrue(ok)
        self.assertEqual((v["stop_err_cm"], v["yaw_err_deg"]), (3.0, 5.0))
        self.m.opening, self.m.between = 0.0001, set()
        self.assertFalse(S.grip_check(self.m, "right", 7)[0])
        self.m.opening, self.m.between = 0.065, {7}
        self.assertTrue(S.grip_check(self.m, "right", 7)[0])

        class World:
            def object_pose(self, key):
                return np.array([0.0, 0.0, 0.93]), np.array([1.0, 0, 0, 0])

        self.assertTrue(S.lift_check(World(), "x", 0.85)[0])
        self.assertFalse(S.lift_check(World(), "x", 0.92)[0])

    def test_bimanual_grip_and_lift_checks(self):
        # TODO 38.5: both hands' fingers on the body; the lift holds only while both stay on
        self.m.opening, self.m.touch = 0.05, {7: {"left"}}
        ok, v = S.bimanual_grip_check(self.m, 7)
        self.assertFalse(ok)
        self.assertEqual((v["touching"], v["opening_cm"]), (["left"], [5.0, 5.0]))
        self.m.touch = {7: {"right", "left"}}
        self.assertTrue(S.bimanual_grip_check(self.m, 7)[0])

        class World:
            free = {"x": 7}

            def object_pose(self, key):
                return np.array([0.0, 0.0, 0.93]), np.array([1.0, 0, 0, 0])

        ok, v = S.bimanual_lift_check(World(), self.m, "x", 0.85)
        self.assertTrue(ok)
        self.assertEqual((v["rise_cm"], v["touching"]), (8.0, ["left", "right"]))
        self.m.touch = {7: {"right"}}
        self.assertFalse(S.bimanual_lift_check(World(), self.m, "x", 0.85)[0])
        self.assertFalse(S.bimanual_lift_check(World(), self.m, "x", 0.92)[0])


if __name__ == "__main__":
    unittest.main()
