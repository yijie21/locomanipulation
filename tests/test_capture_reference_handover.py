import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from stream_reach import _servo_continuous
from primitives import HAND_OPEN
from grasp_capture import OFFSET


class CaptureReferenceHandoverTests(unittest.TestCase):
    def fixture(self):
        base=np.array([1.,2.,.8]); Rb=Rotation.from_euler('z',.9).as_matrix()
        pm=base+Rb@np.array([.17,-.18,-.08]); Rm=Rb@Rotation.from_euler('y',.2).as_matrix()
        target=pm+Rm@OFFSET
        reference=pm+np.array([.012,.006,-.015])
        rotation=Rotation.from_euler('x',.1).as_matrix()@Rm
        state=dict(fallen=False,joints=np.zeros(29),pelvis=dict(pos=base,quat=Rotation.from_matrix(Rb).as_quat()[[3,0,1,2]],yaw=.9),palms={'right':dict(pos=pm,quat=Rotation.from_matrix(Rm).as_quat()[[3,0,1,2]],hand_q=[.0245,.0245])},objects={'bottle':dict(pos=target,quat=[1.,0.,0.,0.],touching=[],touch_links=[])})
        robot=SimpleNamespace(obs=lambda:state,state=SimpleNamespace(latest=lambda _:state),object_pos=lambda *_:target.copy(),world_to_root_full=lambda p,s:Rb.T@(p-base),log=Mock())
        solver=Mock(return_value=(np.zeros(29),0.,0.))
        def push(q):state['fallen']=True
        arm=SimpleNamespace(q=np.zeros(29),hands={'right':HAND_OPEN.copy()},ik=SimpleNamespace(solve=solver,fk=Mock(return_value=(Rb.T@(reference-base),Rb.T@rotation))),_push=push)
        plan=SimpleNamespace(rot=Rm,wrist=pm)
        return robot,arm,plan,target,reference,rotation,base,Rb

    def test_first_close_tick_keeps_previous_command_under_tracking_error(self):
        robot,arm,plan,target,reference,rotation,base,Rb=self.fixture()
        _servo_continuous(robot,arm,'bottle',plan,'right',close_from_current=True,
            grip_target=lambda s:target,reference_guard=lambda *a:None,
            preserve_capture_reference=True)
        arm.ik.solve.assert_called_once()
        np.testing.assert_allclose(arm.ik.solve.call_args.args[1],Rb.T@(reference-base),atol=1e-8)
        np.testing.assert_allclose(arm.ik.solve.call_args.args[2],Rb.T@rotation,atol=1e-8)

    def test_legacy_default_still_anchors_measured_wrist(self):
        robot,arm,plan,target,reference,rotation,base,Rb=self.fixture()
        _servo_continuous(robot,arm,'bottle',plan,'right',close_from_current=True,
            grip_target=lambda s:target,reference_guard=lambda *a:None)
        np.testing.assert_allclose(arm.ik.solve.call_args.args[1],Rb.T@(plan.wrist-base),atol=1e-8)
        np.testing.assert_allclose(arm.ik.solve.call_args.args[2],Rb.T@plan.rot,atol=1e-8)
        arm.ik.fk.assert_not_called()

    def test_handover_does_not_bypass_actual_capture_geometry(self):
        robot,arm,plan,target,*_=self.fixture()
        ok,why=_servo_continuous(robot,arm,'bottle',plan,'right',close_from_current=True,
            grip_target=lambda s:target+[.1,0.,0.],reference_guard=lambda *a:None,
            preserve_capture_reference=True)
        self.assertFalse(ok);self.assertIn('geometry not delivered',why)
        arm.ik.fk.assert_not_called();arm.ik.solve.assert_not_called()
