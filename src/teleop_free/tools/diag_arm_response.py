"""Diagnostic only: fixed upper reference and wrist pulses in free space.

No robot pinning, grasp, VLA, or exporter recording. Run each condition with
fresh deploy state. The fixed reference is shared; measured poses are not assumed
identical across runs. A local response measurement is not a global inverse.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths
from g1_joints import MUJOCO_JOINT_NAMES
from primitives import Primitives, LocomotionMode, StreamMode
from privileged_pub import EpisodeControlClient
from stream_reach import PlannerArm
from transport_guard import TransportGuard


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--posture', choices=['standing', 'crouched'], required=True)
    ap.add_argument('--amplitude', type=float, default=.12)
    ap.add_argument('--pose-speed', type=float, default=.5, help='Diagnostic feedback reference speed cap (rad/s)')
    ap.add_argument('--pose-feedback', action='store_true', help='After baseline hold, try guarded fixed-world palm feedback')
    ap.add_argument('--hold-only', action='store_true', help='Eight-second fixed-reference validation without pulses')
    a = ap.parse_args()
    if not 0 < a.pose_speed <= .5:ap.error('--pose-speed must be in (0, .5]')
    if a.pose_feedback and not a.hold_only:ap.error('--pose-feedback requires --hold-only')
    if not 0 < a.amplitude <= .15:ap.error('amplitude must be in (0, .15] rad')
    reference = np.asarray(json.loads(a.reference.read_text())['q'], dtype=float)
    if reference.shape != (29,) or not np.isfinite(reference).all():ap.error('invalid reference')
    a.out.mkdir(parents=True, exist_ok=False)
    meta = json.loads(Path(paths.REPO, 'outputs/scene/sim/scene_meta.json').read_text())
    p = Primitives(meta=meta, verbose=False)
    reset = EpisodeControlClient()
    reset.pin_arms()  # Release, never pin the test robot.
    p.cmd.reset_pose()
    reset.reset(robot={'dxy': [0., 3.], 'yaw': 0.}, label='arm_response')
    time.sleep(.3)
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.set_upper_body_vel(np.zeros(17))
    p.cmd.start_control()
    p.cmd.hold(5., mode=LocomotionMode.IDLE, facing=(1., 0., 0.))
    if p.obs()['fallen']:raise RuntimeError('initial stand failed')
    arm = PlannerArm(p, 'right', send_vel=True)
    arm.start()
    if a.posture == 'crouched':
        arm.mode = LocomotionMode.IDEL_SQUAT
        arm.height = .64
        arm.hold(5.)
    guard = TransportGuard(arm.ik, 'water_bottle', meta['objects']['water_bottle']['body'],
                           [s['body'] for s in meta['surfaces'].values()])
    target = arm.q.copy()
    target[12:] = reference[12:]
    if np.any(target[12:] < arm.ik.lo[12:]) or np.any(target[12:] > arm.ik.hi[12:]):
        raise RuntimeError('reference outside controller bounds')
    events = [];origin = np.asarray(p.obs()['pelvis']['pos'])
    began = time.monotonic()
    with (a.out/'trace.jsonl').open('w') as trace:
        def phase(label, goal, hold):
            events.append(dict(phase=label,wall=time.time(),goal=goal.tolist()))
            print('[arm-response]',label,flush=True)
            settled = None
            deadline = time.monotonic()+15.+hold
            while time.monotonic() < deadline:
                st=p.obs()
                if st['fallen']:raise RuntimeError('robot fell')
                if np.linalg.norm(np.asarray(st['pelvis']['pos'])-origin)>.15:
                    raise RuntimeError('base drift exceeded 15 cm diagnostic bound')
                if time.time()-st['wall']>.2:raise RuntimeError('stale privileged measurement')
                command=arm.q+np.clip(goal-arm.q,-.5*.02,.5*.02)
                checked=command.copy();checked[:12]=st['joints'][:12]
                collision=guard.check(st,checked,'right')
                if not collision.clear:raise RuntimeError('clearance rejected: '+str(collision.to_dict()))
                command_wall=time.time()
                arm.q=command;arm._push(command)
                # One blocking measurement per tick. A second p.obs() halves
                # the command rate. Preserve the pre-command timestamp.
                after=st
                trace.write(json.dumps(dict(phase=label,wall=command_wall,command_wall=command_wall,
                    elapsed=time.monotonic()-began,reference=arm._q_prev.tolist(),
                    upper_velocity=p.cmd.upper_body_vel.tolist(),state=after))+'\n')
                if np.max(np.abs(goal[12:]-command[12:]))<1e-5:
                    if settled is None:settled=time.monotonic()
                    if time.monotonic()-settled>=hold:return
            raise RuntimeError('reference ramp deadline exceeded')
        error=None
        try:
            phase('baseline',target,8. if a.hold_only else 5.)
            if a.pose_feedback:
                from diagnostic_pose_servo import execute_pose_feedback
                state=p.obs()
                root=Primitives._rot(state['pelvis']['quat'])
                pos,rot=arm.ik.fk(target,'right')
                world_pos=np.asarray(state['pelvis']['pos'])+root@pos
                world_rot=root@rot
                with (a.out/'feedback.jsonl').open('w') as feedback_trace:
                    def emit(event,**data):
                        feedback_trace.write(json.dumps(dict(event=event,wall=time.time(),**data))+'\n')
                    ok,reason=execute_pose_feedback(p,arm,world_pos,world_rot,guard,emit,speed=a.pose_speed)
                print('[arm-response] feedback',ok,reason,flush=True)
                if not ok:raise RuntimeError(reason)

            for joint in ([] if a.hold_only else [26,27,28]):
                for sign in [1,-1]:
                    goal=target.copy();goal[joint]+=sign*a.amplitude
                    if not arm.ik.lo[joint] <= goal[joint] <= arm.ik.hi[joint]:
                        raise RuntimeError('pulse outside controller bounds')
                    phase(f'{MUJOCO_JOINT_NAMES[joint]}:{sign:+d}',goal,3.)
                    phase('baseline',target,2.)
        except Exception as exc:
            error=str(exc)
            raise
        finally:
            (a.out/'run.json').write_text(json.dumps(dict(posture=a.posture,amplitude=a.amplitude,
                reference_source=str(a.reference),reference=target.tolist(),events=events,error=error,
                pose_feedback=a.pose_feedback,pose_speed=a.pose_speed,schema_version=2,measurement_timing='pre_command',hold_only=a.hold_only),indent=2)+'\n')
            # Return control to the measured posture; no corrective motion or pin.
            arm.stop()
    print('[arm-response] complete',a.out,flush=True)


if __name__=='__main__':main()
