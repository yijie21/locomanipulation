"""Requested versus delivered stance, with failures kept in the denominator."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def geometry(st):
    yaw=st['pelvis']['yaw'];d=np.asarray(st['objects']['water_bottle']['pos'])-st['pelvis']['pos']
    c,s=np.cos(yaw),np.sin(yaw)
    return dict(depth=float(c*d[0]+s*d[1]),lateral=float(-s*d[0]+c*d[1]),pelvis_z=st['pelvis']['pos'][2],yaw=yaw)


def analyze(root):
    rows=[]
    for folder in sorted(root.glob('planner_case_*')):
        if not (folder/'result.json').exists():continue
        row=json.loads((folder/'result.json').read_text());events=[json.loads(l) for l in (folder/'events.jsonl').read_text().splitlines()]
        start=end=transport_end=None;origin=None;handover_before=None;standing=None;feedback_times=[];row['pose_feedback_started']=False;row['transport_started']=False
        for e in events:
            kind=e['event']
            if kind in ['staging_delivered','heading_calibrated','standing_delivered','crouch_delivered','handover_after','transport_start']:
                row[kind]=geometry(e['state'])
            if kind=='stance_offset':row['stance_offset']={k:v for k,v in e.items() if k not in ['wall','event']}
            if kind=='standing_delivered':standing=e['state']
            if kind=='crouch_delivered' and standing is not None:
                delta=np.asarray(e['state']['pelvis']['pos'])-standing['pelvis']['pos']
                yaw_delta=e['state']['pelvis']['yaw']-standing['pelvis']['yaw']
                row['crouch_change']=dict(xy_drift=float(np.linalg.norm(delta[:2])),z_change=float(delta[2]),
                    yaw_change=float(np.arctan2(np.sin(yaw_delta),np.cos(yaw_delta))))
            if kind=='handover_before':handover_before=e['state']
            if kind=='handover_after':
                row['handover_geometry_preserved']=e['geometry_preserved']
                if handover_before is not None:
                    delta=e['state']['pelvis']['yaw']-handover_before['pelvis']['yaw']
                    row['handover_yaw_change']=float(np.arctan2(np.sin(delta),np.cos(delta)))
            if kind=='transport_start':
                row['transport_started']=True;start=e['wall'];origin=np.array(e['state']['pelvis']['pos'])
            if kind=='expert_servo_exit':end=e['wall']
            if kind=='pose_feedback_start':row['pose_feedback_started']=True
            if kind=='pose_feedback_tick':feedback_times.append(e['wall'])
            if kind=='pose_feedback_exit':row['pose_feedback_result']={k:v for k,v in e.items() if k not in ['state']}
            if kind=='transport_path_planned':row['path']={k:v for k,v in e.items() if k not in ['state','waypoints']}
            if kind=='transport_path_delivered':
                row['delivered']={k:v for k,v in e.items() if k!='state'}
                transport_end=e['wall']
            if kind=='expert_hold_exit':row['held']=e['held']
        metrics=dict(samples=0,max_base_drift=0.,transport_max_base_drift=0.,min_up=1.,fallen_samples=0)
        if start is not None and end is not None:
            for line in (folder/'privileged.jsonl').open():
                e=json.loads(line)
                if not start<=e['wall']<=end:continue
                st=e['state'];w,x,y,z=st['objects']['water_bottle']['quat']
                metrics['samples']+=1
                metrics['max_base_drift']=max(metrics['max_base_drift'],float(np.linalg.norm(np.asarray(st['pelvis']['pos'])-origin)))
                if e['wall']<=(transport_end if transport_end is not None else end):
                    metrics['transport_max_base_drift']=max(metrics['transport_max_base_drift'],float(np.linalg.norm(np.asarray(st['pelvis']['pos'])-origin)))
                metrics['min_up']=min(metrics['min_up'],1-2*(x*x+y*y))
                metrics['fallen_samples']+=bool(st['fallen'])
        row['expert_interval']=metrics
        if len(feedback_times)>1:
            periods=np.diff(feedback_times)*1000
            row['feedback_cadence_ms']=dict(median=float(np.median(periods)),p95=float(np.percentile(periods,95)),ticks=len(feedback_times))
        row['capture_status']=json.loads((folder/'capture_status.json').read_text())
        row['capture_complete']=bool(events and events[-1]['event']=='experiment_end' and
            not row['capture_status']['dropped'] and not row['capture_status']['errors'] and not row['capture_status']['writer_alive'])
        rows.append(row)
    result=dict(completed=len(rows),successes=sum(bool(r['result'][0]) for r in rows),
        nominal_stance_met=sum(r.get('stance_offset',{}).get('nominal_stance_met',False) for r in rows),
        crouch_completed=sum('crouch_delivered' in r for r in rows),transport_delivered=sum('delivered' in r for r in rows),
        captures_complete=sum(r['capture_complete'] for r in rows),
        transport_started=sum(r['transport_started'] for r in rows),feedback_started=sum(r['pose_feedback_started'] for r in rows),
        reasons=dict(Counter(r['result'][1] for r in rows)),rows=rows)
    (root/'grid_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root',type=Path);a=ap.parse_args();analyze(a.root)
