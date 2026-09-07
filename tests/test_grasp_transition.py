"""Control-flow regression: an unsuccessful approach must never crouch."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/teleop_free'))
import stream_reach as sr


class Clock:
    def __init__(self): self.t = 0.
    def time(self): return self.t
    def sleep(self, dt): self.t += dt


class Robot:
    hold_yaw = None
    def __init__(self, x=.6, fallen=False):
        self.x, self.fallen = x, fallen
        self.frames = []
        self.cmd = SimpleNamespace(frame=lambda **kw: self.frames.append(kw))
    def obs(self):
        return {'fallen': self.fallen,
                'pelvis': {'vel': [0., 0., 0.], 'pos': [0., 0., .8], 'yaw': 0.},
                'palms': {'right': {'pos': [0., 0., .9], 'quat': [1., 0., 0., 0.]}}}
    def world_to_root_full(self, point, state): return np.array([self.x, 0., -.07])
    world_to_root = world_to_root_full
    def pelvis_yaw(self, state): return 0.
    def pelvis_xy(self, state): return np.zeros(2)
    def log(self, text): pass


class ApproachTests(unittest.TestCase):
    def test_random_stand_radius_cannot_change_caller_transition_gate(self):
        gates = []
        for radius in [.26, .39]:
            with patch.object(sr, 'time', Clock()):
                sr.grasp_streamed(Robot(), 'water_bottle', np.array([.6, 0., .7]), None,
                                  plan_r=radius)
            gates.append((sr.LAST_TRANSITION['pre_band'], sr.LAST_TRANSITION['post_band']))
        self.assertEqual(gates[0], gates[1])

    def test_same_geometry_ignores_hidden_target(self):
        results = []
        for target in [[.33, -.16], [.26, -.09]]:
            robot = Robot(x=.23)
            with patch.object(sr, 'time', Clock()):
                result = sr._creep_to_band(robot, np.array([.23, 0., .7]), 'right', target_rel=target)
            self.assertEqual(result.status, sr.ApproachStatus.READY)
            self.assertTrue(all(x['movement'] == (0., 0., 0.) for x in robot.frames))
            results.append((result, robot.frames))
        self.assertEqual(results[0], results[1])

    def test_predicted_post_band_has_margin(self):
        lo, hi = sr.approach_depth_band(.32)
        self.assertGreaterEqual(lo + .08, sr.GRIP_X_MIN + .01)
        self.assertLessEqual(hi + .08, .32 - .01)
        with self.assertRaises(ValueError): sr.approach_depth_band(.25)

    def test_timeout_is_not_ready(self):
        with patch.object(sr, 'time', Clock()):
            result = sr._creep_to_band(Robot(), np.array([.6, 0., .7]), 'right', max_s=.1)
        self.assertEqual(result.status.value, 'timeout')

    def test_fall_is_not_ready(self):
        with patch.object(sr, 'time', Clock()):
            result = sr._creep_to_band(Robot(fallen=True), np.array([.6, 0., .7]), 'right')
        self.assertEqual(result.status.value, 'fallen')

    def test_grasp_does_not_start_arm_after_timeout(self):
        # Exercise the real caller and approach loop, not a mocked result.
        with patch.object(sr, 'time', Clock()), patch.object(sr, 'make_arm') as make_arm:
            ok, why = sr.grasp_streamed(Robot(), 'water_bottle', np.array([.6, 0., .7]), None)
        self.assertFalse(ok)
        self.assertIn('timeout', why)
        make_arm.assert_not_called()

    def test_post_crouch_drift_cannot_start_reaching(self):
        robot = Robot(x=.22)
        with patch.object(sr, 'time', Clock()), patch.object(sr, 'make_arm') as factory, \
             patch.object(sr, '_servo_continuous') as servo, patch.dict(sr.os.environ, {'GYM_ONE_SHOT':'1'}):
            def drift(*args, **kw):
                robot.x = .44
                return .12
            factory.return_value.squat.side_effect = drift
            ok, why = sr.grasp_streamed(robot, 'water_bottle', np.array([.22, 0., .7]), None)
        self.assertFalse(ok)
        self.assertIn('band correction', why)
        servo.assert_not_called()

    def test_undelivered_crouch_cannot_start_reaching(self):
        with patch.object(sr, 'time', Clock()), patch.object(sr, 'make_arm') as factory, \
             patch.object(sr, '_servo_continuous') as servo:
            factory.return_value.squat.return_value = 0.
            ok, why = sr.grasp_streamed(Robot(x=.22), 'water_bottle', np.array([.22, 0., .7]), None)
        self.assertFalse(ok)
        self.assertIn('crouch not delivered', why)
        servo.assert_not_called()


if __name__ == '__main__': unittest.main()
