"""Collision-checked detour with physical MuJoCo geometry, not a fake predicate."""

import sys
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from locomani.tasks.bottle.transport_guard import TransportGuard
from locomani.tasks.bottle.transport_planner import JointTransportPlanner

XML = """<mujoco><worldbody>
<body name="pelvis"><freejoint/><geom size=".01" mass="1" contype="0" conaffinity="0"/>
<body name="right_shoulder_pitch_link"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
<geom type="sphere" size=".04" mass="1"/></body></body>
<body name="target"><freejoint/><geom type="sphere" size=".05" mass=".1"/></body>
</worldbody></mujoco>"""


class PlannerTests(unittest.TestCase):
    def setup_case(self):
        model = mujoco.MjModel.from_xml_string(XML)
        ik = SimpleNamespace(model=model, base_qadr=0, qadr=np.array([7, 8]))
        guard = TransportGuard(ik, "bottle", "target", joint_step=0.02)
        st = dict(
            joints=[-0.2, 0.0],
            pelvis=dict(pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]),
            palms={"right": dict(hand_q=[])},
            objects={"bottle": dict(pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])},
        )
        return guard, st

    def test_find_detour_when_direct_route_hits_object(self):
        guard, st = self.setup_case()
        goal = np.array([0.2, 0.0])
        self.assertFalse(guard.check(st, goal, "right").clear)
        planner = JointTransportPlanner(guard, [0, 1], [-0.4, -0.4], [0.4, 0.4], seed=7, step=0.08, timeout=2.0)
        path = planner.plan(st, goal, "right")
        self.assertTrue(path.ok, path.reason)
        self.assertFalse(path.direct)
        self.assertGreater(len(path.waypoints), 2)
        np.testing.assert_array_equal(path.waypoints[0], st["joints"])
        np.testing.assert_array_equal(path.waypoints[-1], goal)
        for a, b in zip(path.waypoints, path.waypoints[1:]):
            local = deepcopy(st)
            local["joints"] = a
            self.assertTrue(guard.check(local, b, "right").clear)

    def test_search_budget_failure_never_returns_an_unchecked_route(self):
        guard, st = self.setup_case()
        planner = JointTransportPlanner(guard, [0, 1], [-0.4, -0.4], [0.4, 0.4], max_nodes=2)
        path = planner.plan(st, [0.2, 0.0], "right")
        self.assertFalse(path.ok)
        self.assertEqual(path.reason, "search budget exhausted")
        self.assertEqual(path.waypoints, [])

    def test_unsafe_initial_state_is_not_silently_repaired(self):
        guard, st = self.setup_case()
        st["joints"] = [0.0, 0.0]
        planner = JointTransportPlanner(guard, [0, 1], [-0.4, -0.4], [0.4, 0.4])
        path = planner.plan(st, [0.2, 0.0], "right")
        self.assertFalse(path.ok)
        self.assertEqual(path.reason, "initial posture lacks clearance")
        self.assertEqual(path.waypoints, [])


if __name__ == "__main__":
    unittest.main()
