"""One explicit tabletop stance, without VLA or robot pinning.

The grid names a requested standing stop after a scripted short approach,
not the delivered crouched geometry.
Run each case with a fresh deploy. Every rejection is retained.
"""
import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import paths
from primitives import Primitives, LocomotionMode, StreamMode
from privileged_pub import EpisodeControlClient
from stream_reach import PlannerArm
import stream_reach as sr
import upper_ref
from primitives import READY_UPPER
from transition_capture import Capture
from diag_crouched_handover import in_place_expert


class SetupRejected(Exception):
    pass


def geometry(state,key='water_bottle'):
    base=np.asarray(state['pelvis']['pos']);yaw=state['pelvis']['yaw']
    rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    relative=rot.T@(np.asarray(state['objects'][key]['pos'][:2])-base[:2])
    return dict(depth=float(relative[0]),lateral=float(relative[1]),pelvis_z=float(base[2]),yaw=float(yaw))


def wait_for_settle(p, yaw, timeout=6., arm=None):
    """Fresh low-speed states for .6 s; a borrowed arm keeps its squat request."""
    deadline=time.monotonic()+timeout;since=None;previous=None
    while time.monotonic()<deadline:
        state=p.obs();stamp=state['wall']
        if time.time()-stamp>.2:return False,state,'stale settle observation'
        if state['fallen'] or sr.toppled(state,'water_bottle'):return False,state,'unsafe settle state'
        stable=False
        if previous is not None and stamp>previous['wall']:
            dyaw=state['pelvis']['yaw']-previous['pelvis']['yaw']
            yaw_rate=abs(np.arctan2(np.sin(dyaw),np.cos(dyaw)))/(stamp-previous['wall'])
            height_ok=arm is None or abs(state['pelvis']['pos'][2]-(arm.height+sr.SQUAT_HEIGHT_BIAS))<.03
            stable=np.linalg.norm(state['pelvis']['vel'][:3])<.03 and yaw_rate<.08 and height_ok
        if stable:
            if since is None:since=stamp
            if stamp-since>=.6:return True,state,'measured posture settled'
        else:since=None
        previous=state
        if arm is None:
            p.cmd.frame(mode=LocomotionMode.IDLE,facing=(np.cos(yaw),np.sin(yaw),0.))
        else:
            arm._push(arm.q,observed=state)
    return False,p.obs(),'posture settle timeout'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cases',type=Path,required=True)
    ap.add_argument('--case',type=int,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--posture-only',action='store_true',help='Measure arm takeover, squat and a 2 s hold; skip grasp')
    ap.add_argument('--preserve-reference',action='store_true',help='Opt-in same-channel takeover from the current upper-body command')
    ap.add_argument('--no-squat-track',action='store_true',help='Diagnostic ablation: freeze the upper reference during squat')
    ap.add_argument('--settle-before-arm',action='store_true',help='Require .6 s measured low speed before takeover (6 s budget)')
    ap.add_argument('--continuous-grasp',action='store_true',help='Measured stand/crouch settle, preserve reference, reuse arm through grasp')
    a=ap.parse_args();cases=json.loads(a.cases.read_text());case=cases[a.case]
    if a.continuous_grasp:
        if a.posture_only or a.no_squat_track:ap.error('continuous grasp cannot be combined with posture-only or no-squat-track')
        a.settle_before_arm=True;a.preserve_reference=True
    if a.out.exists():ap.error('refusing overwrite')
    meta=json.loads(Path(paths.SCENE_META).read_text())
    yaw=float(case['yaw']);rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    object_xy=np.array([-.45,0.])
    robot_xy=object_xy-rot@np.array([case['depth'],case['lateral']])
    staging_xy=robot_xy-rot@np.array([.65,0.])
    cap=Capture(a.out.resolve());cap.instrument()
    p=Primitives(meta=meta,verbose=False);reset=EpisodeControlClient();arm=None
    result=(False,'setup not completed')
    cap.event('experiment',case=case,robot_xy=robot_xy,staging_xy=staging_xy,object_xy=object_xy,protocol_version=4,posture_only=a.posture_only,preserve_reference=a.preserve_reference,squat_track=not a.no_squat_track,settle_before_arm=a.settle_before_arm,continuous_grasp=a.continuous_grasp,scene_dir=paths.SIM_DIR,mode='scripted short approach, no VLA')
    def boundary(stage,state):
        cap.event('posture_boundary',stage=stage,state=state,reference=p.cmd.upper_body,
                  heading_offset=p.cmd.heading_offset)
    cap.event('round_start',round=case['case'])
    try:
        reset.pin_arms();p.cmd.reset_pose()
        reset.reset(objects={'water_bottle':{'dxy':(object_xy-meta['objects']['water_bottle']['init_pos'][:2]).tolist(),'yaw':0.}},
                    robot={'dxy':(staging_xy-meta['robot_start']['xy']).tolist(),'yaw':yaw},label='stance_grid')
        time.sleep(.3)
        p.cmd.stream_mode=StreamMode.PLANNER_VR_3PT;p.cmd.upper_body=None
        p.cmd.set_upper_body_vel(np.zeros(17));p.cmd.start_control()
        p.cmd.hold(5.,mode=LocomotionMode.IDLE,facing=(np.cos(yaw),np.sin(yaw),0.))
        state=p.obs();cap.event('staging_delivered',state=state,geometry=geometry(state))
        if state['fallen']:raise SetupRejected('free-space staging stand failed')
        # Clean deploy auto-starts before this episode reset. The idle hold
        # exposes its inherited heading offset; calibrate outside the table.
        p.cmd.heading_offset=float(np.arctan2(np.sin(state['pelvis']['yaw']-yaw),np.cos(state['pelvis']['yaw']-yaw)))
        if not p.turn_to(yaw,tol=.16,timeout=20.):raise SetupRejected('staging heading alignment failed')
        state=p.obs();cap.event('heading_calibrated',state=state,geometry=geometry(state),heading_offset=p.cmd.heading_offset)

        tap=upper_ref.shared()
        if not tap.wait(3.):raise RuntimeError('planner reference unavailable')
        q_free=tap.upper17()
        if q_free is None:raise RuntimeError('planner reference stale')
        ramp=upper_ref.UpperRamp(p.cmd,q_free)
        p.cmd.stream_mode=StreamMode.PLANNER_FROZEN_UPPER_BODY
        for _ in range(150):
            ramp.push(READY_UPPER)
            p.cmd.frame(mode=LocomotionMode.IDLE,facing=(np.cos(yaw),np.sin(yaw),0.))
            time.sleep(.02)
        ramp.settle()
        approached=p.walk_to(robot_xy,stop_dist=.025,timeout=25.,face_target=False,face_yaw=yaw)
        p.cmd.hold(.4,mode=LocomotionMode.IDLE,facing=(np.cos(yaw),np.sin(yaw),0.))
        state=p.obs();cap.event('standing_delivered',state=state,geometry=geometry(state),approached=approached)

        xy_error=float(np.linalg.norm(np.asarray(state['pelvis']['pos'][:2])-robot_xy))
        heading_error=abs(np.arctan2(np.sin(state['pelvis']['yaw']-yaw),np.cos(state['pelvis']['yaw']-yaw)))
        cap.event('stance_offset',xy_error=xy_error,heading_error=heading_error,nominal_stance_met=xy_error<=.06)
        if state['fallen']:result=(False,'approach fell')
        elif not approached or heading_error>np.radians(10):result=(False,'approach not delivered')
        elif sr.toppled(state,'water_bottle'):result=(False,'target toppled before squat')
        else:
            if a.settle_before_arm:
                boundary('before_settle',state)
                settled,state,why=wait_for_settle(p,yaw)
                cap.event('stand_settle_exit',ok=settled,reason=why,state=state,geometry=geometry(state))
                if not settled:raise SetupRejected(why)
                error=state['pelvis']['yaw']-yaw
                if abs(np.arctan2(np.sin(error),np.cos(error)))>np.radians(10):
                    raise SetupRejected('settled heading outside 10 degrees')
            boundary('before_arm_start',state)
            arm=PlannerArm(p,'right',send_vel=True)
            arm.start(upper_reference=p.cmd.upper_body if a.preserve_reference else None)
            boundary('after_arm_start',p.obs())
            arm.squat(.13,track=not a.no_squat_track)
            state=p.obs();cap.event('crouch_delivered',state=state,geometry=geometry(state))
            boundary('after_squat',state)
            if state['fallen']:result=(False,'crouch fell')
            elif sr.toppled(state,'water_bottle'):result=(False,'target toppled during crouch')
            elif state['pelvis']['pos'][2]>.73:result=(False,'crouch height not delivered')
            elif a.posture_only:
                arm.hold(2.)
                state=p.obs();boundary('after_hold',state)
                result=(not state['fallen'] and not sr.toppled(state,'water_bottle'),'posture measured; not a grasp success')
            elif a.continuous_grasp:
                settled,state,why=wait_for_settle(p,yaw,arm=arm)
                cap.event('crouch_settle_exit',ok=settled,reason=why,state=state,geometry=geometry(state),height_command=arm.height)
                if not settled:raise SetupRejected(why)
                boundary('after_hold',state)
                # Borrow the same controller. This caller keeps cleanup ownership.
                result=in_place_expert(p,meta,'water_bottle',cap,{},SimpleNamespace(pause=lambda:None),
                    transport_strategy='planned',transport_pose_feedback=True,planner_active=True,active_arm=arm)
            else:
                # Same expert transport/feedback/grasp implementation as the
                # previous handover diagnostic; no second squat or teleport.
                arm=None  # Cleanup ownership transfers to the expert helper.
                result=in_place_expert(p,meta,'water_bottle',cap,{},SimpleNamespace(pause=lambda:None),
                    transport_strategy='planned',transport_pose_feedback=True,planner_active=True)
    except SetupRejected as exc:
        result=(False,str(exc))
    except Exception as exc:
        result=(False,'exception: '+str(exc))
        cap.event('diagnostic_exception',error=repr(exc))
        raise
    finally:
        if arm is not None:arm.stop()
        cap.event('round_end',result=result)
        (a.out/'result.json').write_text(json.dumps(dict(case=case,result=result),indent=2)+'\n')
        cap.event('experiment_end');cap.close()
    print('[stance-grid]',case,result,flush=True)


if __name__=='__main__':main()
