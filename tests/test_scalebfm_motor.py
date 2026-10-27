"""The ScaleBFM motor layer (TODO 27.2.1, backends/scalebfm/motor.py): reset seats the feet, the stationary reference
pins both ankles in mode 4 while a walk runs in mode 2, the reference blends are continuous and come to rest, the
jaw-centre servo is capped, and the walk's speed profile ends exactly at its distance. The policy here is a stand-in
that returns the default joint targets (no network); the live behaviour is scripts/rsi/
motor_check.py."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from geometry_fixture import DIRECTORY

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.motor import (
    ANKLE,
    MODE_STAND,
    MODE_WALK,
    WRIST,
    Motor,
    MotorConfig,
    Pose,
    Profile,
    lowest_foot_point,
    turn_seconds,
    yaw_quat,
)


class StandInPolicy:
    def __init__(self, meta):
        self.meta = meta
        self.sel = meta["selected_body_names"]
        self.fidx = list(meta["future_idx"])
        self.F = len(self.fidx)
        self.mode = torch.tensor([2], dtype=torch.long)
        self.targets = []

    def reset(self, st):
        pass

    def __call__(self, st, targets):
        self.targets.append(targets)
        return np.asarray(self.meta["default_dof_pos"], dtype=float)


def motor(config=None):
    meta = json.loads(Path(R.META).read_text())
    sim = R.Sim(meta, backend=ScaleBfmBackendConfig(scene_dir=str(DIRECTORY)))
    m = Motor(sim, StandInPolicy(meta), config or MotorConfig(), model_xml=str(DIRECTORY / "scene_locomani.xml"))
    m.reset((-1.5, 0.0), 0.0, warm_up=0)
    return m


class ResetAndModes(unittest.TestCase):
    def test_reset_seats_the_feet_5_mm_above_the_floor(self):
        m = motor()
        low = lowest_foot_point(m.sim.m, m.sim.d, m.foot_bodies)
        self.assertAlmostEqual(low, 0.005, places=6)

    def test_standing_pins_both_ankles_in_mode_4(self):
        m = motor()
        self.assertEqual(m.mode, MODE_STAND)
        self.assertEqual(int(m.pol.mode[0]), MODE_STAND)
        fut = m._futures()
        for side in ("left", "right"):
            pos, quat = m.sim.body(ANKLE[side])
            np.testing.assert_allclose(fut[ANKLE[side]][0], np.repeat(pos[None], m.pol.F, 0))
            np.testing.assert_allclose(fut[ANKLE[side]][1], np.repeat(quat[None], m.pol.F, 0))

    def test_a_tick_steps_the_sim_and_records_a_row(self):
        m = motor()
        self.assertTrue(m.tick())
        self.assertEqual(m.k, 1)
        self.assertEqual(len(m.trace), 1)
        self.assertEqual(len(m.pol.targets), 1)
        self.assertIn(WRIST["right"], m.pol.targets[0])

    def test_mode_change_reaches_the_policy(self):
        m = motor()
        m._set_mode(MODE_WALK)
        self.assertEqual(int(m.pol.mode[0]), MODE_WALK)


class References(unittest.TestCase):
    def test_blend_starts_and_ends_on_its_poses_at_rest(self):
        m = motor()
        a = m.ref(m.t)
        b = Pose(a.pos + [0.0, 0.0, -0.1], yaw_quat(0.3), a.joints + 0.2, np.zeros(29))
        at, total = m._blend([a, b], [1.0])
        self.assertEqual(total, 1.0)
        np.testing.assert_allclose(at(m.t).joints, a.joints)
        np.testing.assert_allclose(at(m.t + 1.0).joints, b.joints)
        np.testing.assert_allclose(at(m.t + 1.0).velocity, 0.0)
        np.testing.assert_allclose(at(m.t).velocity, 0.0, atol=1e-12)
        mid = at(m.t + 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(mid.quat)), 1.0, places=9)
        np.testing.assert_allclose(mid.joints, a.joints + 0.1)

    def test_servo_lead_is_capped_per_axis(self):
        m = motor(MotorConfig(servo_ki=1000.0, servo_cap=0.05))
        m.servo_on["right"] = True
        m.servo_target["right"] = m.jaw_centre("right")[0] + np.array([1.0, -1.0, 0.5])
        for _ in range(5):
            m._futures()
        np.testing.assert_allclose(m.servo[("pos", "right")], [0.05, -0.05, 0.05])


class WalkProfile(unittest.TestCase):
    def test_profile_rests_exactly_at_its_distance(self):
        for dist in (0.05, 0.3, 2.0):
            p = Profile(0.25, 0.6, dist)
            s = [p.s(t) for t in np.linspace(0, p.t_end + 1, 400)]
            self.assertAlmostEqual(s[-1], dist, places=12)
            self.assertTrue(np.all(np.diff(s) >= -1e-12))
            self.assertLessEqual(max(np.diff(s)) / ((p.t_end + 1) / 399), 0.25 + 1e-6)

    def test_turn_peak_rate(self):
        T = turn_seconds(np.pi, 0.8)
        self.assertAlmostEqual(1.5 * np.pi / T, 0.8, places=9)
        self.assertEqual(turn_seconds(0.01, 0.8), 0.6)


if __name__ == "__main__":
    unittest.main()


class HandProgram(unittest.TestCase):
    def test_hand_drives_the_wrist_keypoint_with_its_lead(self):
        m = motor()
        m.tick()
        jaw, rot = np.array([0.0, -0.2, 0.9]), np.eye(3)
        m.hand("right", jaw, rot, velocity=np.array([0.1, 0.0, 0.0]))
        fut = m._futures()
        lead = np.array([j * R.DT for j in m.pol.fidx])[:, None] * np.array([0.1, 0.0, 0.0])
        np.testing.assert_allclose(fut[WRIST["right"]][0], (jaw - R.OFF)[None] + lead)
        m.hand("right")
        self.assertNotIn("right", m.hand_override)

    def test_commanded_jaw_is_the_last_ticks_wrist_plus_offset(self):
        m = motor()
        m.tick()
        p, q = m.last_fut[WRIST["left"]]
        np.testing.assert_allclose(m.commanded_jaw("left"), p[0] + R.mat(q[0]) @ R.OFF)


class ReachCompensation(unittest.TestCase):
    """TODO 33.1.4 (2): the `reach_offset` hook moves a one-hand reach's command; the outcome is judged against the
    target the skill asked for."""

    def _reach(self, m, asked, landed):
        """One right-hand reach with the IK, the blend and the ticking stubbed out: the commanded jaw target, and the
        outcome when the jaw ends at `landed`."""
        seen, jaw = [], {"left": np.array([0.0, 0.2, 0.9]), "right": np.array([0.0, -0.2, 0.9])}

        def solve(solver, tp, tr, **kw):
            seen.append(np.array(tp))
            return SimpleNamespace(hand_errors=(0.0, 0.0))

        def run(name, update, timeout):
            jaw["right"] = landed
            return update(m.t + 5.0)

        m.reach_note = None
        with (
            mock.patch.object(m, "_solver", lambda **kw: None),
            mock.patch.object(m, "_solve", solve),
            mock.patch.object(m, "_pose_of", lambda pose, yaw: None),
            mock.patch.object(m, "_blend", lambda poses, durations: (lambda t: None, 1.0)),
            mock.patch.object(m, "_stand", lambda pose: None),
            mock.patch.object(m, "jaw_centre", lambda s: (jaw[s], np.eye(3))),
            mock.patch.object(m, "_run", run),
        ):
            out = m.reach("right", asked, np.eye(3))
        return seen[-1][1], out

    def test_the_command_moves_by_the_offset_and_the_error_stays_against_the_asked_target(self):
        m = motor()
        m.tick()
        asked, off = np.array([0.3, -0.25, 0.85]), np.array([0.02, -0.01, 0.005])
        m.reach_offset = lambda side, pos, rot, pelvis_p, pelvis_q, pelvis_z: off
        commanded, out = self._reach(m, asked, landed=asked.copy())
        np.testing.assert_allclose(commanded, asked - off)
        self.assertTrue(out.ok)
        self.assertEqual(out.info["jaw_err_cm"], 0.0)
        self.assertEqual(out.info["comp_offset_cm"], [2.0, -1.0, 0.5])

    def test_no_offset_leaves_the_command_alone(self):
        m = motor()
        m.tick()
        asked = np.array([0.3, -0.25, 0.85])
        for hook in (None, lambda *a: None, lambda *a: 1 / 0):
            m.reach_offset = hook
            commanded, out = self._reach(m, asked, landed=asked + np.array([0.0, 0.0, 0.03]))
            np.testing.assert_allclose(commanded, asked)
            self.assertNotIn("comp_offset_cm", out.info)
            self.assertEqual(out.info["jaw_err_cm"], 3.0)
        self.assertIn("ZeroDivisionError", out.info["comp_error"])
        self.assertFalse(MotorConfig().reach_compensation)


class ReachRoute(unittest.TestCase):
    """TODO 40: under `reach_route` a reach with via points follows the planned path to its straight solve's goal, or
    to its last via point and then the last leg straight; without a plan it keeps the straight legs."""

    def setUp(self):
        self.m = motor()
        self.m.tick()
        q = np.array([1.0, 0.0, 0.0, 0.0])
        self.poses = [
            Pose(np.array([0.0, 0.0, 0.78 - 0.01 * i]), q, np.full(29, 0.1 * i), np.zeros(29)) for i in range(3)
        ]
        self.durations, self.leg_ends = [1.0, 0.5], [1, 2]
        self.calls = []

    def _hook(self, refuse=()):
        def plan(start, goal, pelvis_p, pelvis_q):
            self.calls.append((np.array(goal), np.array(pelvis_p)))
            if any(np.allclose(goal, r) for r in refuse):
                return None
            return np.linspace(start, goal, 6)

        plan.last = {"ok": True}
        return plan

    def test_the_route_runs_to_the_goal_no_faster_than_the_legs(self):
        m = self.m
        m.reach_route = self._hook()
        t0 = m.t
        at, total, info = m._routed(self.poses, self.durations, self.leg_ends)
        self.assertEqual(info["to"], "goal")
        self.assertGreaterEqual(total, 1.5)
        np.testing.assert_allclose(self.calls[0][1], self.poses[2].pos)  # planned with the pelvis at the goal
        np.testing.assert_allclose(at(t0).joints, self.poses[0].joints)
        np.testing.assert_allclose(at(t0 + total).joints, self.poses[2].joints)
        np.testing.assert_allclose(at(t0 + total).pos, self.poses[2].pos)

    def test_a_refused_goal_routes_to_the_last_via_point_and_keeps_the_last_leg(self):
        m = self.m
        m.reach_route = self._hook(refuse=[self.poses[2].joints])
        t0 = m.t
        at, total, info = m._routed(self.poses, self.durations, self.leg_ends)
        self.assertEqual(info["to"], "last_via")
        self.assertEqual(len(info["tries"]), 2)
        route_total = total - 0.5
        self.assertGreaterEqual(route_total, 1.0)
        np.testing.assert_allclose(at(t0 + route_total).joints, self.poses[1].joints)
        np.testing.assert_allclose(at(t0 + total).joints, self.poses[2].joints)

    def test_no_plan_keeps_the_straight_legs(self):
        m = self.m
        for hook in (self._hook(refuse=[p.joints for p in self.poses]), lambda *a: 1 / 0):
            m.reach_route = hook
            t0 = m.t
            at, total, info = m._routed(self.poses, self.durations, self.leg_ends)
            self.assertIsNone(info["to"])
            self.assertEqual(total, 1.5)
            np.testing.assert_allclose(at(t0 + 1.0).joints, self.poses[1].joints)
        self.assertIn("ZeroDivisionError", info["tries"][0]["error"])
        self.assertFalse(MotorConfig().reach_route)

    def test_a_pose_goal_route_ends_in_its_own_arm(self):
        """TODO 40.4: a route the hook planned to the wrist pose ends where its path ends, not at the IK goal."""
        m = self.m
        own = np.full(29, 0.7)

        def plan(start, goal, pelvis_p, pelvis_q, **kw):
            self.calls.append(kw)
            return np.linspace(start, own, 5)

        plan.last = {"ok": True, "goal": "pose"}
        m.reach_route = plan
        jaws = [(np.zeros(3), np.eye(3)), (np.ones(3), np.eye(3))]
        t0 = m.t
        at, total, info = m._routed(self.poses, self.durations, self.leg_ends, jaws, approach=False)
        np.testing.assert_allclose(at(t0 + total + 1.0).joints, own)
        np.testing.assert_allclose(self.calls[0]["jaw"][0], np.ones(3))  # the goal's jaw goes with the goal's cut
        self.assertTrue(self.calls[0]["pose_ok"])
        self.assertEqual((MotorConfig().reach_route_scope, MotorConfig().reach_route_pose_goal), ("via", False))
