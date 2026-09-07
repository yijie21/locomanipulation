"""Real robot FK with a biased actuator model; no claims of WBC equivalence."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import time
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from arm_ik import ArmIK
from pose_feedback import PoseFeedback


class PoseFeedbackTests(unittest.TestCase):
    def test_corrects_measured_pose_error_with_biased_actuator(self):
        ik=ArmIK()
        state=json.loads((Path(__file__).parent/'fixtures/crouched_pregrasp.json').read_text())
        q=np.asarray(state['joints'])
        desired_pos,desired_rot=ik.fk(q,'right')
        controller=PoseFeedback(ik,q,'right')
        ref=q.copy();bias=np.zeros(29);bias[26]=.15;bias[27]=.12
        for _ in range(800):
            actual_pos,actual_rot=ik.fk(ref+bias,'right')
            candidate,metrics=controller.step(ref,actual_pos,actual_rot,desired_pos,desired_rot,.02)
            self.assertLessEqual(np.max(np.abs(candidate-ref)),.5*.02+1e-9)
            np.testing.assert_array_equal(candidate[:22],q[:22])
            self.assertLessEqual(np.max(np.abs(candidate-q)),.35+1e-9)
            ref=candidate
        self.assertLess(metrics['position_error'],.01)
        self.assertLess(metrics['rotation_error'],.05)

    def test_nonfinite_target_is_rejected(self):
        ik=ArmIK();q=np.zeros(29);controller=PoseFeedback(ik,q,'right')
        with self.assertRaises(ValueError):
            controller.step(q,np.zeros(3),np.eye(3),np.array([np.nan,0,0]),np.eye(3),.02)

    def test_live_guard_rejects_before_publishing(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free/tools'))
        from diagnostic_pose_servo import execute_pose_feedback
        from transport_guard import SweepResult
        state=json.loads((Path(__file__).parent/'fixtures/crouched_pregrasp.json').read_text())
        state['wall']=time.time()
        arm=SimpleNamespace(ik=ArmIK(),q=np.asarray(state['joints']),side='right',_push=Mock())
        robot=SimpleNamespace(obs=lambda:{**state, "wall":time.time()})
        guard=SimpleNamespace(check=lambda *a:SweepResult(False,.001,'arm','table',0.,1))
        ok,reason=execute_pose_feedback(robot,arm,np.asarray(state['palms']['right']['pos'])+.1,np.eye(3),guard,Mock())
        self.assertFalse(ok);self.assertEqual(reason,'clearance rejected')
        arm._push.assert_not_called()

    def test_rotation_error_at_pi_does_not_vanish(self):
        from pose_feedback import rotation_vector
        error=rotation_vector(np.diag([1.,-1.,-1.]),np.eye(3))
        self.assertAlmostEqual(np.linalg.norm(error),np.pi)

    def test_transport_handover_preserves_original_drift_anchor(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free/tools'))
        from diagnostic_pose_servo import execute_pose_feedback
        origin=json.loads((Path(__file__).parent/'fixtures/crouched_pregrasp.json').read_text())
        state=deepcopy(origin);state['wall']=time.time();state['pelvis']['pos'][0]+=.05
        arm=SimpleNamespace(ik=ArmIK(),q=np.asarray(state['joints']),side='right',_push=Mock())
        robot=SimpleNamespace(obs=lambda:{**state, "wall":time.time()});guard=SimpleNamespace(check=Mock())
        ok,reason=execute_pose_feedback(robot,arm,np.zeros(3),np.eye(3),guard,Mock(),origin_state=origin)
        self.assertFalse(ok);self.assertEqual(reason,'base drift exceeded 4 cm')
        arm._push.assert_not_called();guard.check.assert_not_called()

    def test_offhand_guard_reuses_current_observation(self):
        import stream_reach as sr
        state=json.loads((Path(__file__).parent/'fixtures/crouched_pregrasp.json').read_text())
        state['wall']=time.time()
        state['palms']['left']['pos']=(np.array(state['objects']['water_bottle']['pos'])+[1.,0.,0.]).tolist()
        robot=SimpleNamespace(cmd=SimpleNamespace(left_hand=np.zeros(7),right_hand=np.zeros(7)),
            obs=Mock(side_effect=AssertionError('second blocking observation')),
            object_pos=lambda key,st:np.array(st['objects'][key]['pos']))
        arm=sr.PlannerArm(robot,'right');arm.guard='water_bottle'
        q=np.array(state['joints'])
        np.testing.assert_array_equal(arm._guard_left(q.copy(),observed=state),q)
        robot.obs.assert_not_called()


if __name__=='__main__':unittest.main()
