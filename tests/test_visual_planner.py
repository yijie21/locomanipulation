import sys
import unittest
from pathlib import Path

import numpy as np

from locomani.tasks.bottle_visual.observation import SceneEstimate
from locomani.tasks.bottle_visual.planner import GraspConfig, RobotObservation, VisualGraspPlanner


def scene(x=0.5, visible=True):
    return SceneEstimate((x, 0, 0.702), 0.62, (0.44, 0, 0.62), (1, 0, 0), 0, 0.9, visible)


def robot(x=-1.0, pad=(0, -0.2, 0.6), grip=0.0245):
    return RobotObservation(
        np.array([x, 0, 0.76]), np.array([1.0, 0, 0, 0]), np.array(pad), np.array([1.0, 0, 0, 0]), np.zeros(3), grip
    )


class VisualPlannerTests(unittest.TestCase):
    def test_waiting_does_not_start_reach_while_far_away(self):
        planner = VisualGraspPlanner()
        for _ in range(500):
            self.assertEqual(planner.decide(scene(), robot()).phase, "walk")
        self.assertEqual(planner.decide(scene(), robot(x=0)).phase, "reach")

    def test_new_visible_bottle_position_changes_reach_target(self):
        a, b = VisualGraspPlanner(), VisualGraspPlanner()
        aa = a.decide(scene(0.5), robot(0))
        bb = b.decide(scene(0.54), robot(0))
        self.assertAlmostEqual(bb.pad_target[0] - aa.pad_target[0], 0.04)

    def test_blackout_pauses_precision_motion(self):
        planner = VisualGraspPlanner()
        planner.decide(scene(), robot(0))
        self.assertTrue(planner.decide(scene(visible=False), robot(0)).pause)

    def test_lift_needs_visible_bottle_motion_not_elapsed_calls(self):
        planner = VisualGraspPlanner()
        planner.phase = "lift"
        planner.lift_reference_z = 0.702
        planner.lift_anchor = np.array([0.5, 0, 0.8])
        for _ in range(100):
            self.assertEqual(planner.decide(scene(), robot(0, grip=-0.023)).phase, "lift")
        raised = SceneEstimate((0.5, 0, 0.8), 0.62, (0.44, 0, 0.62), (1, 0, 0), 1, 0.9, True)
        self.assertEqual(planner.decide(raised, robot(0, grip=-0.023)).phase, "carry")

    def test_carry_clearance_responds_to_visible_slip_only(self):
        planner = VisualGraspPlanner(GraspConfig(carry_clearance=0.12))
        planner.phase = "carry"
        planner.lift_reference_z = 0.702
        planner.carry_anchor = np.array([0.4, 0, 0.8])
        hidden = planner.decide(scene(visible=False), robot(0, pad=(0.4, 0, 0.8), grip=-0.023))
        self.assertTrue(hidden.pause)
        self.assertAlmostEqual(hidden.pad_target[2], 0.8)
        visible = planner.decide(scene(), robot(0, pad=(0.4, 0, 0.8), grip=-0.023))
        self.assertAlmostEqual(visible.pad_target[2], 0.84)

    def test_lift_uses_observed_pregrasp_height_for_tall_bottle(self):
        planner = VisualGraspPlanner()
        planner.phase = "entry"
        before = SceneEstimate((0.5, 0, 0.85), 0.62, (0.44, 0, 0.62), (1, 0, 0), 0, 0.9, True)
        self.assertEqual(planner.decide(before, robot(0, pad=(0.5, 0, 0.83))).phase, "close")
        self.assertEqual(planner.decide(before, robot(0, pad=(0.5, 0, 0.83), grip=-0.023)).phase, "lift")
        raised = SceneEstimate((0.5, 0, 0.94), 0.62, (0.44, 0, 0.62), (1, 0, 0), 0.4, 0.9, True)
        self.assertEqual(planner.decide(raised, robot(0, pad=(0.5, 0, 0.92), grip=-0.023)).phase, "carry")

    def test_entry_wrist_rotation_agrees_with_locked_root_heading(self):
        from scipy.spatial.transform import Rotation

        planner = VisualGraspPlanner(GraspConfig(facing="edge", pitch_feedforward_deg=0))
        planner.phase = "reach"
        own = robot(0.185, pad=(0.4, 0, 0.682))
        own.pelvis_position[1] = 0.15
        own.pelvis_quaternion[:] = Rotation.from_euler("z", 0.2).as_quat(scalar_first=True)
        intent = planner.decide(scene(), own)
        self.assertEqual(intent.phase, "entry")
        wrist_yaw = Rotation.from_quat(intent.wrist_quaternion, scalar_first=True).as_euler("xyz")[2]
        self.assertAlmostEqual(wrist_yaw, intent.yaw)


if __name__ == "__main__":
    unittest.main()
