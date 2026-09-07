"""Opt-in joint-path transport for live expert takeover diagnostics only."""
from copy import copy, deepcopy
import time
import numpy as np
import mujoco
from arm_ik import LAST_IK
from g1_joints import ARM_MJ
from primitives import Primitives
from transport_planner import JointTransportPlanner
import stream_reach as sr


def pose_error(state, grasp, side):
    entry_col = 0 if sr._HAND == 'dex1' else 1
    pre = np.asarray(grasp.wrist) - np.asarray(grasp.rot)[:, entry_col] * sr.SAFE_GAP
    position = float(np.linalg.norm(np.asarray(state['palms'][side]['pos']) - pre))
    actual = Primitives._rot(state['palms'][side]['quat'])
    angle = float(np.arccos(np.clip((np.trace(np.asarray(grasp.rot) @ actual.T)-1)/2, -1., 1.)))
    return position, angle


def plan_transport(arm,state,grasp,guard,side='right'):
    # Offline IK must not overwrite the action-label sidecar or live IK data.
    ik=copy(arm.ik);ik.data=mujoco.MjData(ik.model)
    original=deepcopy(LAST_IK)
    entry_col = 0 if sr._HAND == 'dex1' else 1
    goal_world=np.asarray(grasp.wrist)-np.asarray(grasp.rot)[:,entry_col]*sr.SAFE_GAP
    root=Primitives._rot(state['pelvis']['quat']);base=np.asarray(state['pelvis']['pos'])
    final_depth=float((root.T@(np.asarray(grasp.wrist)-base))[0])
    if final_depth < .05:
        return None,dict(ok=False,reason='final wrist behind existing forward limit',final_wrist_depth=final_depth)
    try:
        _,final_err,final_angle=ik.solve(side,root.T@(np.asarray(grasp.wrist)-base),root.T@grasp.rot,
            q_mj29=np.asarray(state['joints']),use_waist=True,iters=120)
        goal,err,angular=ik.solve(side,root.T@(goal_world-base),root.T@grasp.rot,
            q_mj29=np.asarray(state['joints']),use_waist=True,iters=120)
    finally:
        LAST_IK.clear();LAST_IK.update(original)
    if final_err>.025 or final_angle>.25:
        return None,dict(ok=False,reason='final grasp IK residual',position_error=float(final_err),rotation_error=float(final_angle))
    if err>.025 or angular>.25:
        return None,dict(ok=False,reason='pregrasp IK residual',position_error=float(err),rotation_error=float(angular))
    indices=list(ARM_MJ[side])+[12,13,14]
    planner=JointTransportPlanner(guard,indices,ik.lo,ik.hi,seed=17,timeout=1.5)
    path=planner.plan(state,goal,side)
    return path,dict(ok=path.ok,reason=path.reason,direct=path.direct,
                     checks=path.checks,elapsed_s=path.elapsed_s,
                     waypoint_count=len(path.waypoints),
                     waypoints=[q.tolist() for q in path.waypoints],
                     position_error=float(err),rotation_error=float(angular))


def execute_transport(p,arm,grasp,guard,cap,key,side='right',feedback=False,pose_feedback=False,preserve_reference=False):
    if feedback and pose_feedback:raise ValueError('joint and pose feedback are separate experiments')
    start=p.obs();path,details=plan_transport(arm,start,grasp,guard,side)
    cap.event('transport_path_planned',state=start,**details)
    if path is None or not path.ok:return False,details['reason']
    baseline=np.asarray(start['pelvis']['pos']);object0=np.asarray(start['objects'][key]['pos'])
    indices=list(ARM_MJ[side])+[12,13,14]
    # Execute sampled references through the same planner channel. Recheck
    # each from the current measurement; a static plan never overrides a
    # newly detected collision or base/target displacement.
    if not preserve_reference:
        arm.q=np.asarray(start['joints']).copy()
    cap.event('transport_reference_seed',preserved=preserve_reference,reference=arm.q.copy(),state=start)
    nominal=arm.q.copy()
    correction=np.zeros_like(arm.q)
    correction_peak=0.
    deadline=time.monotonic()+25.
    for n,waypoint in enumerate(path.waypoints[1:]):
        settled=0
        while time.monotonic()<deadline:
            st=p.obs()
            if st['fallen'] or sr.toppled(st,key):return False,'fell or target toppled during planned transport'
            if np.linalg.norm(np.asarray(st['pelvis']['pos'])-baseline)>.04:
                return False,'base moved outside static transport assumption'
            if np.linalg.norm(np.asarray(st['objects'][key]['pos'])-object0)>.02:
                return False,'target moved outside static transport assumption'
            delta=waypoint[indices]-nominal[indices]
            if pose_feedback and n == len(path.waypoints)-2 and np.max(np.abs(delta))<.015:
                from diagnostic_pose_servo import execute_pose_feedback
                entry_col = 0 if sr._HAND == 'dex1' else 1
                target = np.asarray(grasp.wrist)-np.asarray(grasp.rot)[:,entry_col]*sr.SAFE_GAP
                moved, reason = execute_pose_feedback(p,arm,target,np.asarray(grasp.rot),guard,cap.event,
                    timeout=max(0.,deadline-time.monotonic()),speed=.2,origin_state=start)
                if not moved:return False,reason
                break
            nominal[indices]+=np.clip(delta,-.5*sr.DT,.5*sr.DT)
            measured=np.asarray(st['joints'])
            if feedback and np.max(np.abs(delta))<.015:
                # Experimental bounded integral correction of the active
                # arm's measured joint error. Waist/base are not integrated.
                joints=ARM_MJ[side]
                correction[joints]=np.clip(correction[joints]+.5*sr.DT*(waypoint[joints]-measured[joints]),-.35,.35)
                correction_peak=max(correction_peak,float(np.max(np.abs(correction))))
            command=nominal.copy()
            command[indices]=np.clip((nominal+correction)[indices],arm.ik.lo[indices],arm.ik.hi[indices])
            # Limit corrected references as well as nominal path progress.
            command=arm.q+np.clip(command-arm.q,-.5*sr.DT,.5*sr.DT)
            checked=command.copy();checked[:12]=st['joints'][:12]
            result=guard.check(st,checked,side)
            if not result.clear:
                cap.event('transport_path_rejected',segment=n,state=st,collision=result.to_dict())
                return False,'planned transport rejected by live clearance check'
            previous_reference=arm.q.copy()
            push_started=time.time()
            arm.q=command;arm._push(arm.q,observed=st)
            if preserve_reference:
                cap.event('transport_reference_tick',reference=command,previous_reference=previous_reference,
                          observation_wall=st['wall'],push_started=push_started,
                          max_step=float(np.max(np.abs(command[indices]-previous_reference[indices]))))
            measured=np.asarray(st['joints'])
            # Intermediate joints define the screened route. At the final
            # waypoint the task requires palm pose, not one redundant IK solution.
            if n == len(path.waypoints)-2:
                position, angle = pose_error(st, grasp, side)
                arrived = position < .045 and angle < .25
            else:
                arrived = np.max(np.abs(measured[indices]-waypoint[indices])) < .12
            if np.max(np.abs(delta))<.015 and arrived:
                settled+=1
                if settled>=3:break
            else:settled=0
        else:
            cap.event('transport_tracking_timeout',state=p.obs(),feedback=feedback,correction_peak=correction_peak,goal=waypoint)
            return False,'planned transport tracking timeout'
        cap.event('transport_waypoint_reached',segment=n,state=p.obs())
    state=p.obs()
    err,angular=pose_error(state,grasp,side)
    ok=err<.045 and angular<.25
    cap.event('transport_path_delivered',ok=ok,position_error=err,rotation_error=angular,state=state,feedback=feedback,pose_feedback=pose_feedback,correction_peak=correction_peak)
    return ok,'' if ok else 'planned pregrasp was not delivered'
