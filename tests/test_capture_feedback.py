import sys
import unittest
from pathlib import Path
from copy import deepcopy
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from capture_feedback import CaptureFeedback
from grasp_capture import OFFSET
from reach_config import ReachConfig


class FeedbackTests(unittest.TestCase):
    def fixture(self):
        state = dict(seq=1,wall=1.,pelvis=dict(pos=[0.,0.,.8]),
                     palms={'right':dict(pos=[0.,0.,.7],quat=[1.,0.,0.,0.])})
        target = OFFSET+np.array([0.,0.,.7])
        return state,target,CaptureFeedback(state,'right',lambda _:target)

    def test_base_sway_does_not_drag_the_world_reference(self):
        state,target,feedback = self.fixture()
        aim,_,_ = feedback.update(state,.02)
        state['pelvis']['pos'][0] += .02
        state['palms']['right']['pos'][0] += .01
        state.update(seq=2,wall=1.02)
        next_aim,advance,_ = feedback.update(state,.02)
        np.testing.assert_allclose(aim,next_aim)
        self.assertFalse(advance)

    def test_borderline_capture_has_bounded_interior_preference(self):
        state,target,_ = self.fixture()
        target[2] += .018
        feedback = CaptureFeedback(state,'right',lambda _:target)
        aim,_,_ = feedback.update(state,.02)
        self.assertAlmostEqual(aim[2]-.7,.006)

    def test_stale_observation_cannot_advance_closing(self):
        state,_,feedback = self.fixture()
        feedback.update(state,.02)
        self.assertFalse(feedback.update(state,.02)[1])
        state.update(seq=2,wall=1.02)
        self.assertTrue(feedback.update(state,.02)[1])

    def test_actual_wrist_rotation_is_compensated_at_the_grasp_center(self):
        from scipy.spatial.transform import Rotation
        state,target,feedback = self.fixture()
        rotation = Rotation.from_euler('y',.1)
        state['palms']['right']['quat'] = rotation.as_quat()[[3,0,1,2]].tolist()
        aim,_,_ = feedback.update(state,.02)
        np.testing.assert_allclose(aim+rotation.apply(OFFSET),target,atol=1e-6)

    def test_stationary_grasp_center_during_wrist_rotation_can_keep_closing(self):
        from scipy.spatial.transform import Rotation
        state,target,feedback = self.fixture()
        feedback.update(state,.02)
        rotation = Rotation.from_euler('y',.1)
        state['palms']['right']['quat'] = rotation.as_quat()[[3,0,1,2]].tolist()
        state['palms']['right']['pos'] = (target-rotation.apply(OFFSET)).tolist()
        state.update(seq=2,wall=1.02)
        self.assertTrue(feedback.update(state,.02)[1])

    def test_knocked_object_is_not_chased(self):
        state,target,feedback = self.fixture()
        feedback.update(state,.02)
        target[0] += .08
        state.update(seq=2,wall=1.02)
        aim,advance,_ = feedback.update(state,.02)
        self.assertLessEqual(np.linalg.norm(aim-feedback.palm),.010001)
        self.assertFalse(advance)
        for i in range(50):
            state.update(seq=i+3,wall=1.04+i*.02)
            _,_,lost=feedback.update(state,.02)
        self.assertTrue(lost)

    def test_new_strategy_is_explicit_and_legacy_unchanged(self):
        self.assertFalse(ReachConfig.load('configs/collection/ready_grasp_coupled.json').continuous_capture)
        self.assertTrue(ReachConfig.load('configs/collection/ready_grasp_continuous.json').continuous_capture)
        with self.assertRaises(ValueError):ReachConfig(continuous_capture=True)
