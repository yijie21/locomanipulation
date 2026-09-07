"""Offline static approach grid; never a physical success-rate benchmark.

Sweeps tabletop height, yaw-frame target depth/lateral offset and body yaw.
Checks measured-shape clearance and sampled joint paths to IK waypoints. The
servo has live replanning, tracking bias and a different orientation schedule;
this grid screens candidate poses, it does not simulate that feedback policy.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from arm_ik import ArmIK
from primitives import Primitives
from tabletop_grasp import TabletopTarget,TabletopGrasp
from reach_config import DEFAULT_REACH
from transport_guard import TransportGuard
import stream_reach as sr


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True)
    ap.add_argument('--meta',type=Path,required=True)
    ap.add_argument('--xml',required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--shoulder-pitch-offset',type=float,default=0.)
    ap.add_argument('--elbow-offset',type=float,default=0.)
    args=ap.parse_args()
    if args.out.exists():raise SystemExit(f'Refusing overwrite: {args.out}')
    original=json.loads(args.state.read_text());meta=json.loads(args.meta.read_text())
    key='water_bottle';obj=meta['objects'][key];surface=meta['surfaces'][obj.get('surface','A')]
    ik=ArmIK(args.xml)
    for j,(lo,hi) in {12:(-.35,.35),13:(-.15,.15),14:(-.05,.20)}.items():ik.lo[j],ik.hi[j]=lo,hi
    guard=TransportGuard(ik,key,obj['body'],[s['body'] for s in meta['surfaces'].values()])
    table=guard.model.body(surface['body']).id;table_z=float(guard.model.body_pos[table,2])
    rows=[];started=time.perf_counter()
    for height in [.50,float(surface['top']),.78]:
      delta=height-surface['top'];guard.model.body_pos[table,2]=table_z+delta
      for depth in [.14,.20,.26,.32,.38,.44,.50]:
       for lateral in [-.40,-.28,-.16,-.04,.08]:
        for yaw_delta in [-15.,0.,15.]:
         st=deepcopy(original);q=np.asarray(st['joints']).copy()
         q[22]=np.clip(q[22]+args.shoulder_pitch_offset,ik.lo[22],ik.hi[22])
         q[25]=np.clip(q[25]+args.elbow_offset,ik.lo[25],ik.hi[25])
         st['joints']=q.tolist()
         st['objects'][key]['pos'][2]+=delta
         yaw=original['pelvis']['yaw']+np.deg2rad(yaw_delta)
         quat=[np.cos(yaw/2),0.,0.,np.sin(yaw/2)];rot=Primitives._rot(quat)
         pos=np.asarray(st['pelvis']['pos']).copy()
         pos[:2]=np.asarray(st['objects'][key]['pos'][:2])-(rot@np.array([depth,lateral,0.]))[:2]
         st['pelvis'].update(pos=pos.tolist(),quat=quat,yaw=yaw)
         for side in ['left','right']:
          wrist,rotation=ik.fk(q,side)
          st['palms'][side]['pos']=(rot@wrist+pos).tolist()
          from arm_ik import _mat2quat
          st['palms'][side]['quat']=_mat2quat(rot@rotation)
         robot=SimpleNamespace(obs=lambda:st,object_pos=lambda k,s:np.asarray(s['objects'][k]['pos']),
             pelvis_xy=lambda s:np.asarray(s['pelvis']['pos'][:2]),pelvis_yaw=lambda s:s['pelvis']['yaw'])
         # Use the same grip offset constant as production, not an assumed one.
         import grasping
         target=TabletopTarget(key,'right',height+obj['height']-grasping.GRIP_BELOW_RIM,height,obj['box'])
         strategy=TabletopGrasp(robot,target)
         plan=strategy.replan(st['palms']['right']['pos'],Primitives._rot(st['palms']['right']['quat']))
         row=dict(table_top=height,depth=depth,lateral=lateral,yaw_delta=yaw_delta,
                  depth_cone=bool(DEFAULT_REACH.post_min<=depth<=DEFAULT_REACH.post_max
                    and abs(np.degrees(np.arctan2(lateral,depth)))<=DEFAULT_REACH.azimuth_deg),
                  plan_ok=plan.ok,accepted=False)
         if not plan.ok:row['reason']='plan: '+plan.why
         else:
          pre=np.asarray(plan.wrist)-np.asarray(plan.rot)[:,0]*sr.SAFE_GAP
          row['reason']='clear_static_waypoints'
          for phase,point in [('up',pre+[0,0,sr.STAGE_UP]),('pregrasp',pre)]:
           solved,err,angular=ik.solve('right',rot.T@(point-pos),rot.T@plan.rot,q_mj29=q,use_waist=True,iters=80)
           if err>.025 or angular>.25:
            row.update(reason='ik_residual',phase=phase,position_error=float(err),rotation_error=float(angular));break
           local=deepcopy(st);local['joints']=q.tolist()
           check=guard.check(local,solved,'right')
           if not check.clear:
            row.update(reason='transport_clearance',phase=phase,collision=check.to_dict());break
           q=solved
          else:
           solved,err,angular=ik.solve('right',rot.T@(np.asarray(plan.wrist)-pos),rot.T@plan.rot,q_mj29=q,use_waist=True,iters=80)
           if err>.025 or angular>.25:
            row.update(reason='grasp_ik_residual',position_error=float(err),rotation_error=float(angular))
           else:row['accepted']=row['depth_cone']
         rows.append(row)
    result=dict(protocol='Static joint-space waypoint screen; not closed-loop servo or full-body rollout',
                state=str(args.state),shoulder_pitch_offset=args.shoulder_pitch_offset,
                elbow_offset=args.elbow_offset,config=DEFAULT_REACH.to_dict(),elapsed_s=time.perf_counter()-started,rows=rows)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    from collections import Counter
    print(len(rows),'candidates',dict(Counter(r['reason'] for r in rows)), 'accepted',sum(r['accepted'] for r in rows))


if __name__=='__main__':main()
