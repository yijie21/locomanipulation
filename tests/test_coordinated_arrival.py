"""Configuration isolation and measured-to-command clearance of arrival."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from reach_config import ReachConfig, DEFAULT_REACH
from coordinated_arrival import CoordinatedArrival
from upper_ref import UpperRamp
from primitives import Primitives


class CoordinatedArrivalTests(unittest.TestCase):
    def test_opt_in_and_strict_boolean(self):
        self.assertFalse(DEFAULT_REACH.coordinated_arrival)
        cfg = ReachConfig.load(Path(__file__).resolve().parents[1]/'configs/collection/coordinated_arrival.json')
        self.assertTrue(cfg.coordinated_arrival)
        self.assertEqual(cfg.approach_band(), DEFAULT_REACH.approach_band())
        with self.assertRaises(ValueError):
            ReachConfig(coordinated_arrival='false')

    def test_final_approach_rejection_stops_base(self):
        import stream_reach as sr
        from test_grasp_transition import Robot, Clock
        robot = Robot()
        callback = Mock(return_value=False)
        with patch.object(sr, 'time', Clock()):
            result = sr._creep_to_band(robot, [.6, 0., .7], 'right', update_upper=callback)
        self.assertEqual(result.status, sr.ApproachStatus.BLOCKED)
        callback.assert_called_once()

    def test_preparation_exception_releases_velocity_and_motion(self):
        c = CoordinatedArrival.__new__(CoordinatedArrival)
        c._finish = Mock(side_effect=RuntimeError('IK failed'))
        c.abort = Mock()
        with self.assertRaises(RuntimeError):
            c.finish(grip_world=[0, 0, 0], squat_hint=None, ghost=None)
        c.abort.assert_called_once()

    def controller(self, clear):
        c = CoordinatedArrival.__new__(CoordinatedArrival)
        c.side='right'; c.error=None; c.last_t=0.
        c.reference=np.array([.2,-.2,.8]); c.rotation_reference=np.eye(3); c.rotation_bias=np.zeros(3)
        c.p=SimpleNamespace(world_to_root_full=lambda p,s: np.asarray(p))
        c.task=SimpleNamespace(target=SimpleNamespace(surface_top=.62, grip_z=.72))
        c.plan=SimpleNamespace(wrist=np.array([.3,-.2,.8]),rot=np.eye(3))
        c.ramp=UpperRamp(Mock(),np.zeros(17))
        c.arm=SimpleNamespace(q=np.zeros(29),ik=SimpleNamespace(solve=Mock(return_value=(np.ones(29),0.,None))))
        result=SimpleNamespace(clear=clear,link='arm',obstacle='table',to_dict=lambda:{'clear':clear})
        c.guard=SimpleNamespace(check=Mock(return_value=result))
        c.stats=dict(checks=0,rejected=None,nav_ticks=0)
        state=dict(joints=np.zeros(29),pelvis=dict(pos=[0,0,0],quat=[1,0,0,0]),
                   palms={'right':dict(pos=[.2,-.2,.8],quat=[1,0,0,0])})
        return c,state

    def test_collision_rejects_before_command_mutation(self):
        c,s=self.controller(False)
        c.step(s,walking=True)
        self.assertIn('clearance',c.error)
        np.testing.assert_array_equal(c.ramp.q,np.zeros(17))
        self.assertEqual(c.stats['nav_ticks'],0)

    def test_both_arms_checked_and_reference_slew_preserved(self):
        c,s=self.controller(True)
        c.step(s,walking=True)
        self.assertIsNone(c.error)
        self.assertEqual([x.args[2] for x in c.guard.check.call_args_list],['right','left'])
        self.assertLessEqual(np.max(np.abs(c.ramp.q)),c.ramp.max_step)
        np.testing.assert_array_equal(c.arm.q[22:29],c.ramp.q[4::2])


if __name__=='__main__':unittest.main()
