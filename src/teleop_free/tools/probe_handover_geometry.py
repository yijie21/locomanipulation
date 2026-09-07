"""Read-only IK checks at recorded stalled/handover poses, not rollouts."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import adaptive_grasp
import grasping
import stream_reach as sr
from primitives import Primitives
from collect_episode import obj_top


def probe(st, meta, key='water_bottle'):
    pel=np.array(st['pelvis']['pos']);rot=Primitives._rot(st['pelvis']['quat'])
    yaw=st['pelvis']['yaw'];grip=np.array(st['objects'][key]['pos'])
    grip[2]=obj_top(meta,key)-grasping.GRIP_BELOW_RIM
    yawrot=Primitives._rot([np.cos(yaw/2),0,0,np.sin(yaw/2)])
    full=rot.T@(grip-pel);flat=yawrot.T@(grip-pel)
    surface=meta['surfaces'][meta['objects'][key].get('surface','A')]
    anchor=pel+np.array([.31*np.sin(yaw),-.31*np.cos(yaw),.44])
    d=grip[:2]-pel[:2];d/=max(np.linalg.norm(d),1e-6)
    plan=adaptive_grasp.plan(grip,st['palms']['right']['pos'],
        Primitives._rot(st['palms']['right']['quat']),surface['top'],
        box=meta['objects'][key]['box'],obj_rot=Primitives._rot(st['objects'][key]['quat']),
        azimuth_from=anchor,avoid_dir=np.r_[d,0],avoid_sign=1.)
    result=dict(pelvis=pel.tolist(),grip_full=full.tolist(),grip_yaw=flat.tolist(),
                plan_ok=plan.ok,plan_why=plan.why)
    result['upright_target'] = not sr.toppled(st,key)
    if not result['upright_target']:
        result['note']='Object already toppled: not an intact upright-grasp feasibility test'
        return result
    if not plan.ok:return result
    # Reuse the real controller's joint limits, without starting/publishing.
    dummy=SimpleNamespace(cmd=SimpleNamespace(left_hand=np.zeros(7),right_hand=np.zeros(7)))
    arm=sr.PlannerArm(dummy,'right');q=np.array(st['joints'])
    pre=np.asarray(plan.wrist)-np.asarray(plan.rot)[:,0]*sr.SAFE_GAP
    for name,w in [('above',pre+[0,0,sr.STAGE_UP]),('pregrasp',pre),('grasp',plan.wrist)]:
        target=rot.T@(w-pel);target_rot=rot.T@plan.rot
        sol,err,angular=arm.ik.solve('right',target,target_rot,q_mj29=q.copy(),use_waist=True,iters=80)
        result[name]=dict(root_target=target.tolist(),position_error_m=float(err),rotation_error_rad=float(angular))
    result['note']='Static IK only; no collision/path/dynamic feasibility certificate'
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path)
    ap.add_argument('--meta',type=Path,required=True);a=ap.parse_args()
    meta=json.loads(a.meta.read_text());result=[];round_i=-1
    for line in (a.capture/'events.jsonl').read_text().splitlines():
        r=json.loads(line)
        if r['event']=='round_start':round_i+=1
        if r['event']=='vla_suffix_exit' and r.get('reason')=='success':continue
        if r['event'] in ['handover_before','handover_after','vla_suffix_exit']:
            result.append(dict(round=round_i,event=r['event'],wall=r['wall'],**probe(r['state'],meta)))
    (a.capture/'geometry_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
