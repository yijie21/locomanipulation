"""Real model FK/IK checks for the comfortable collection support."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from reach_config import ReachConfig, DEFAULT_REACH
from ready_grasp_policy import (templates, ready_q, sample_stand, OFFSET,
                                PRE_GAP, WRIST_LIMIT, ReadyGraspPolicy)
from arm_ik import ArmIK
from primitives import STANDING_PELVIS_Z


class ReadyBaselineTests(unittest.TestCase):
    def test_selection_is_explicit_and_exclusive(self):
        self.assertFalse(DEFAULT_REACH.ready_grasp_baseline)
        path=Path(__file__).resolve().parents[1]/'configs/collection/ready_grasp_baseline.json'
        cfg=ReachConfig.load(path)
        self.assertTrue(cfg.ready_grasp_baseline)
        with self.assertRaises(ValueError):
            ReachConfig(ready_grasp_baseline=True, coordinated_arrival=True)
        with self.assertRaises(ValueError):
            ReachConfig(ready_grasp_baseline='true')

    def test_ready_hands_are_in_front_with_neutral_wrists(self):
        ik=ArmIK(); q=ready_q()
        left,_=ik.fk(q,'left');right,_=ik.fk(q,'right')
        self.assertTrue(.10 < left[0] < .22 and .10 < right[0] < .22)
        self.assertAlmostEqual(left[1],-right[1],places=3)
        np.testing.assert_array_equal(q[19:22],0.)
        np.testing.assert_array_equal(q[26:29],0.)

    def test_sampling_is_near_fk_support_and_rejects_unsupported_height(self):
        cfg=ReachConfig(ready_grasp_baseline=True)
        rng=np.random.default_rng(5)
        for _ in range(100):
            rel,drop=sample_stand(SimpleNamespace(grip_z=.723),rng,cfg)
            centre=np.array([rel[0]+cfg.backshift, rel[1], .723-(STANDING_PELVIS_Z-drop)])
            self.assertLess(min(np.linalg.norm(centre-c) for _,c,_ in templates()),.015)
            self.assertTrue(.03 <= drop <= cfg.max_drop)
        with self.assertRaises(ValueError):
            sample_stand(SimpleNamespace(grip_z=1.5),rng,cfg)

    def test_short_path_has_comfortable_ik_on_real_robot(self):
        ik=ArmIK()
        ik.lo[26:29]=-WRIST_LIMIT;ik.hi[26:29]=WRIST_LIMIT
        ik.lo[[22,25]]=-.65;ik.hi[[22,25]]=.65
        ik.lo[24]=-.5;ik.hi[24]=.5
        for _,centre,rot in templates():
            wrist=centre-rot@OFFSET
            pre=wrist-rot[:,0]*PRE_GAP
            q=ready_q()
            origin,_=ik.fk(q,'right')
            self.assertLess(np.linalg.norm(pre-origin),.20)
            for target in [origin+(pre-origin)*u for u in np.linspace(.1,1.,10)]+[pre+(wrist-pre)*u for u in np.linspace(.1,1.,10)]:
                q,pos_err,rot_err=ik.solve('right',target,rot,q_mj29=q,use_waist=False)
                self.assertLess(pos_err,.02)
                self.assertLess(rot_err,.15)
                self.assertLessEqual(np.abs(q[26:29]).max(),WRIST_LIMIT+1e-6)

    def test_intended_target_contact_does_not_disable_table_guard(self):
        from test_transport_guard import TransportGuardTests, XML
        from transport_guard import TransportGuard
        import mujoco
        helper=TransportGuardTests()
        ik,state=helper.setup_scene(q=0.)
        guard=TransportGuard(ik,'bottle','target')
        self.assertFalse(guard.check(state,[0.],'right').clear)
        self.assertTrue(guard.check(state,[0.],'right',allow_target_contact=True).clear)
        xml=XML.replace('</worldbody>', '<body name="table" pos="1 0 0"><geom size=".04"/></body></worldbody>')
        model=mujoco.MjModel.from_xml_string(xml)
        ik=SimpleNamespace(model=model,data=mujoco.MjData(model),base_qadr=0,qadr=np.array([7]))
        guard=TransportGuard(ik,'bottle','target',('table',))
        hit=guard.check(state,[0.],'right',allow_target_contact=True)
        self.assertFalse(hit.clear)
        self.assertEqual(hit.obstacle,'table')

    def test_close_but_laterally_misaligned_cannot_enter_grasp(self):
        from unittest.mock import patch
        from test_grasp_transition import Robot, Clock
        import stream_reach as sr
        robot=Robot(x=.26)
        robot.world_to_root=lambda point,state:np.array([.26,-.10,0.])
        with patch.object(sr,'time',Clock()):
            result=sr._creep_to_band(robot,[0,0,0],'right',x_min=.24,x_max=.29,
                                    target_lateral=-.19,max_s=.1)
        self.assertEqual(result.status,sr.ApproachStatus.TIMEOUT)
        self.assertTrue(all(np.linalg.norm(frame['movement'])==0 for frame in robot.frames))
        self.assertGreater(robot.frames[0]['facing'][1],0.)

    def test_pregrasp_gap_is_shared_with_actual_servo_handover(self):
        import stream_reach as sr
        class EnteredServo(Exception):
            pass
        def latest(_):
            raise EnteredServo()
        state={'palms':{'right':{'pos':[0.,0.,.8],'quat':[1.,0.,0.,0.]}}}
        robot=SimpleNamespace(obs=lambda:state,object_pos=lambda key,s:np.array([.22,0.,.8]),
                              state=SimpleNamespace(latest=latest))
        plan=SimpleNamespace(rot=np.eye(3),wrist=np.array([PRE_GAP,0.,.8]))
        old=sr._servo_continuous(robot,None,'bottle',plan,'right',start_at_pregrasp=True)
        self.assertFalse(old[0])
        with self.assertRaises(EnteredServo):
            sr._servo_continuous(robot,None,'bottle',plan,'right',start_at_pregrasp=True,
                                 pregrasp_gap=PRE_GAP)
        with self.assertRaises(ValueError):
            sr._servo_continuous(robot,None,'bottle',plan,'right',pregrasp_gap=.01)

    def test_extreme_wrist_is_rejected_even_during_closure(self):
        policy=ReadyGraspPolicy.__new__(ReadyGraspPolicy)
        q=ready_q();q[28]=.6
        self.assertIn('wrist',policy.reference_check({},q,'close'))


if __name__=='__main__':unittest.main()


class LowerGripSamplingTests(unittest.TestCase):
    def test_lowered_grip_can_sample_without_exceeding_squat_limit(self):
        cfg = ReachConfig(ready_grasp_baseline=True, ready_grasp_recovery=True)
        for seed in range(3):
            rel, drop = sample_stand(SimpleNamespace(grip_z=.6914), np.random.default_rng(seed), cfg)
            self.assertTrue(np.isfinite(rel).all())
            self.assertLessEqual(drop, cfg.max_drop)
            self.assertGreaterEqual(drop, .03)
