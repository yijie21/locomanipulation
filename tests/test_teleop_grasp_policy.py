"""TeleopGraspPolicy: the stand is screened geometrically, never rejected on an IK solve."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.usefixtures("bottle_geometry")

sys.path[:0] = [str(Path(__file__).resolve().parent)]

from locomani import profiles
from locomani.pipeline.execution_record import ExecutionRecord
from locomani.tasks.bottle.reach_config import ReachConfig
from locomani.tasks.bottle.ready_grasp_policy import OFFSET
from locomani.tasks.bottle.teleop_grasp_policy import TeleopGraspPolicy, level_jaw


def _policy(*, floor=True, clear=True, grip_shift=(0.0, 0.0, 0.0)):
    p = object.__new__(TeleopGraspPolicy)
    p.side = "right"
    p.execution = ExecutionRecord()
    p.execution.metrics["teleop_grasp"] = {}
    p.config = profiles.load("task", "bottle_visible_teleop").reach
    p.floor_clear = lambda *a, **kw: floor
    p.guard = SimpleNamespace(hands=["right", "left", "body"], check=lambda *a, **kw: SimpleNamespace(clear=clear))
    # an IK that FAILS everything: the teleop policy must not consult it
    p.arm = SimpleNamespace(ik=SimpleNamespace(solve=lambda *a, **kw: (np.zeros(29), 9.0, 9.0)))
    # a far grasp point, outside any arm-only envelope
    p.task = SimpleNamespace(grip=lambda s: OFFSET + np.array([0.6, -0.3, 0.72]) + np.asarray(grip_shift))
    p.p = SimpleNamespace(
        world_to_root_full=lambda pt, s: np.asarray(pt) - np.asarray(s["pelvis"]["pos"]),
    )
    state = dict(
        joints=np.zeros(29),
        pelvis=dict(pos=[0.0, 0.0, 0.79], quat=[1.0, 0.0, 0.0, 0.0], yaw=0.0),
        palms={"right": dict(pos=[0.1, -0.2, 0.7], quat=[1.0, 0.0, 0.0, 0.0])},
    )
    return p, state


class AssessTests(unittest.TestCase):
    def test_config_switch_is_opt_in_and_guarded(self):
        self.assertFalse(ReachConfig().teleop_arm)
        with self.assertRaises(ValueError):
            ReachConfig.from_dict({"schema_version": 1, "teleop_arm": True})
        self.assertTrue(profiles.load("task", "bottle_visible_teleop").reach.teleop_arm)

    def test_level_jaw_keeps_the_entry_axis_and_zeroes_the_tilt(self):
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_euler("xyz", [0.2, -0.3, 0.9]).as_matrix()
        lev = level_jaw(rot)
        np.testing.assert_allclose(lev[:, 0], rot[:, 0], atol=1e-12)
        self.assertAlmostEqual(float(lev[2, 1]), 0.0, places=12)
        np.testing.assert_allclose(lev @ lev.T, np.eye(3), atol=1e-12)
        p, state = _policy()
        self.assertTrue(p.assess(state, 0.10))
        self.assertLess(abs(float(p.coupled_rotation[2, 1])), 1e-9)
        self.assertGreater(p.execution.metrics["teleop_grasp"]["template_jaw_tilt"], 0.1)

    def test_far_target_is_not_rejected_on_the_solve(self):
        p, state = _policy()
        self.assertTrue(p.assess(state, 0.10))
        self.assertEqual(p.coupled_rotation.shape, (3, 3))
        self.assertNotIn("coupled_terminal_ik", p.execution.metrics.get("prediction_rejections", {}))

    def test_geometric_screens_still_reject(self):
        p, state = _policy(floor=False)
        self.assertFalse(p.assess(state, 0.10))
        self.assertIn("coupled_base_floor", p.execution.metrics["prediction_rejections"])
        p, state = _policy(clear=False)
        self.assertFalse(p.assess(state, 0.10))
        self.assertIn("coupled_initial_clearance", p.execution.metrics["prediction_rejections"])
        p, state = _policy()
        self.assertFalse(p.assess(state, 0.30))
        self.assertIn("coupled_squat_out_of_bounds", p.execution.metrics["prediction_rejections"])


if __name__ == "__main__":
    unittest.main()
