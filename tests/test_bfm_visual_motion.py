import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/teleop_free/scalebfm'))
from visual_motion import GraspMotion
from visual_planner import GraspIntent, RobotObservation


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.robot = RobotObservation(np.array([0., 0, .76]), np.array([1., 0, 0, 0]),
                                      np.array([.3, -.15, .7]), np.array([1., 0, 0, 0]), np.zeros(3), .0245)
        self.rel = {side: (np.array([.05, y, -.1]), np.eye(3)) for side, y in [('left', .2), ('right', -.2)]}
        self.intent = GraspIntent('reach', np.zeros(2), 0., np.array([.35, -.15, .7]), self.robot.right_quaternion, .0245)

    def test_preview_length_does_not_advance_committed_reference(self):
        a, b = GraspMotion(self.robot, self.rel), GraspMotion(self.robot, self.rel)
        for _ in range(30):
            short, _ = a.targets(self.intent, self.robot, future_count=2)
            long, _ = b.targets(self.intent, self.robot, future_count=6)
            for name in short:
                np.testing.assert_allclose(short[name][0], long[name][0][:2])
                np.testing.assert_allclose(short[name][1], long[name][1][:2])

    def test_blackout_holds_wrist_despite_changed_precision_target(self):
        motion = GraspMotion(self.robot, self.rel, integral_gain=1.)
        for _ in range(150):
            old, _ = motion.targets(self.intent, self.robot)
        hidden = GraspIntent('reach', np.zeros(2), 0., np.array([.1, .3, .9]), self.robot.right_quaternion, .0245, pause=True)
        for _ in range(20):
            held, _ = motion.targets(hidden, self.robot)
            np.testing.assert_allclose(held['right_wrist_yaw_link'][0][0], old['right_wrist_yaw_link'][0][0])

    def test_delayed_plant_keeps_walking_but_arrival_and_blackout_stop(self):
        from dataclasses import replace
        motion = GraspMotion(self.robot, self.rel)
        intent = replace(self.intent, stand_xy=np.array([.2, 0.]))
        for _ in range(500):
            targets, _ = motion.targets(intent, self.robot)
        self.assertGreater(targets['pelvis'][0][-1, 0] - targets['pelvis'][0][0, 0], .005)
        paused = replace(intent, pause=True)
        for _ in range(50):
            targets, _ = motion.targets(paused, self.robot)
        np.testing.assert_allclose(targets['pelvis'][0][-1], targets['pelvis'][0][0])
        arrived = replace(self.robot, pelvis_position=np.array([.2, 0., .76]))
        for _ in range(200):
            targets, _ = motion.targets(intent, arrived)
        np.testing.assert_allclose(targets['pelvis'][0][-1], targets['pelvis'][0][0], atol=1e-5)


if __name__ == '__main__':
    unittest.main()
