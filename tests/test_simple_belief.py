"""TODO 49.3: the Belief's two switches. self_pose=odometry builds Body + JawFeed and turns the stop servo off; controller_root=odometry
builds OdoRoot (which sets pol.root_source). All four combinations, with fakes (no simulator)."""

import dataclasses
import types
import unittest
from unittest import mock

import numpy as np

from locomani.backends.scalebfm.motor import MotorConfig
from locomani.bench.simple import body


class Fake:
    built = []

    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        obj = types.SimpleNamespace(kind=self.name, args=a)
        Fake.built.append(self.name)
        return obj


def make(odo, root_odo=None, settle=0.7):
    Fake.built = []
    mo = types.SimpleNamespace(
        cfg=dataclasses.replace(MotorConfig(), settle_s=settle), pol=types.SimpleNamespace(root_source=None)
    )
    sim = types.SimpleNamespace(d=types.SimpleNamespace(qpos=np.array([0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0])))
    fakes = dict(Body=Fake("Body"), JawFeed=Fake("JawFeed"), OdoRoot=Fake("OdoRoot"))
    with (
        mock.patch.object(body, "Body", fakes["Body"]),
        mock.patch.object(body.odometry, "OdoRoot", fakes["OdoRoot"]),
        mock.patch.object(body.odometry, "JawFeed", fakes["JawFeed"]),
    ):
        b = body.Belief(sim, mo, odo) if root_odo is None else body.Belief(sim, mo, odo, root_odo)
    return b, mo, list(Fake.built)


class Combinations(unittest.TestCase):
    def test_odometry_odometry_is_the_old_single_flag(self):
        b, mo, built = make(True)
        self.assertEqual(built, ["Body", "OdoRoot", "JawFeed"])  # the old order
        self.assertEqual((b.odo, b.root_odo), (True, True))
        self.assertEqual((mo.cfg.stop_ki, mo.cfg.stop_timeout), (0.0, 0.7))
        b2, _, built2 = make(True, True)
        self.assertEqual(built2, built)

    def test_truth_truth_builds_nothing_and_keeps_the_stop_servo(self):
        b, mo, built = make(False)
        self.assertEqual(built, [])
        self.assertEqual((b.odo, b.root_odo), (False, False))
        self.assertEqual((mo.cfg.stop_ki, mo.cfg.stop_timeout), (MotorConfig().stop_ki, MotorConfig().stop_timeout))
        self.assertFalse(hasattr(b, "feed"))

    def test_odometry_pose_with_truth_root_has_feed_no_odo_root(self):
        b, mo, built = make(True, False)
        self.assertEqual(built, ["Body", "JawFeed"])
        self.assertNotIn("OdoRoot", built)
        self.assertEqual((mo.cfg.stop_ki, mo.cfg.stop_timeout), (0.0, 0.7))
        self.assertTrue(hasattr(b, "feed"))
        self.assertIsNone(mo.pol.root_source)

    def test_truth_pose_with_odometry_root_has_odo_root_no_feed_and_the_stop_servo_on(self):
        b, mo, built = make(False, True)
        self.assertEqual(built, ["Body", "OdoRoot"])
        self.assertNotIn("JawFeed", built)
        self.assertFalse(hasattr(b, "feed"))
        self.assertEqual((mo.cfg.stop_ki, mo.cfg.stop_timeout), (MotorConfig().stop_ki, MotorConfig().stop_timeout))

    def test_truth_pose_belief_is_the_simulator(self):
        b, _, _ = make(False, True)
        b.start()
        p, yaw = b.pelvis()
        self.assertTrue(np.allclose(p, [0.0, 0.0, 0.8]) and yaw == 0.0)
        self.assertFalse(b.anchored)


class WalkWithoutOdometry(unittest.TestCase):
    def test_a_truth_pose_walk_ties_the_belief_to_the_simulator_and_reports_no_error(self):
        b, _, _ = make(False, True)
        b.start()
        calls = []
        b.mo.walk_to = lambda xy, yaw, **kw: calls.append(("walk", xy, yaw))
        b.mo.hold = lambda s: calls.append(("hold", s))
        b.sim.d.qpos[:3] = [1.0, 2.0, 0.8]
        err = b.walk([1.0, 2.0], 0.0)
        self.assertEqual(err, (0.0, 0.0))
        self.assertTrue(np.allclose(b.pb, b.pt) and b.yb == b.yt)
        self.assertEqual(calls[0][0], "walk")
        self.assertIsNone(getattr(b.mo, "stop_xy", None))


if __name__ == "__main__":
    unittest.main()
