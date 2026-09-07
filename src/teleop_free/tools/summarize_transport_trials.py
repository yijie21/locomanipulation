"""Separate actual interventions from VLA prefix successes/timeouts."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(root):
    rows=[];current=None
    for line in (root/'events.jsonl').read_text().splitlines():
        event=json.loads(line);kind=event['event']
        if kind=='round_start':
            current=dict(round=len(rows),intervened=False,transport_started=False)
            rows.append(current)
        if current is None:continue
        if kind=='first_policy_observation':current['first_observation']=event['state']
        if kind=='handover_after':
            current.update(intervened=True,handover=event['state'],xy_drift=event['xy_drift'],
                           z_drift=event['z_drift'],geometry_preserved=event['geometry_preserved'])
        if kind=='transport_start':current.update(transport_started=True,strategy=event['strategy'],start=event['state'])
        if kind=='transport_path_planned':current['path']={k:v for k,v in event.items() if k not in ['state','waypoints']}
        if kind=='transport_path_delivered':
            current['path_delivered']=event['ok']
            current['delivery']={k:v for k,v in event.items() if k!='state'}
        if kind=='transport_tracking_timeout':
            current['tracking_timeout']={k:v for k,v in event.items() if k!='state'}
        if kind=='transport_path_rejected':current['path_rejection']=event['collision']
        if kind=='pose_feedback_start':current['pose_feedback_started']=True
        if kind=='pose_feedback_exit':current['pose_feedback_result']={k:v for k,v in event.items() if k!='state'}
        if kind=='expert_servo_exit':
            current['servo_result']=event['result'];current['execution']=event.get('execution')
        if kind=='expert_hold_exit':current['held']=event['held']
        if kind=='round_end':current['result']=event['result']
    result=dict(rounds=rows,completed=sum('result' in r for r in rows),
                successes=sum(bool(r.get('result',[False])[0]) for r in rows),
                interventions=sum(r['intervened'] for r in rows),
                held=sum(r.get('held',False) for r in rows),
                reasons=dict(Counter(r['result'][1] for r in rows if 'result' in r)))
    (root/'transport_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k!='rounds'}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('capture',nargs='+',type=Path);a=ap.parse_args()
    for root in a.capture:print(root.name,json.dumps(summarize(root)))

if __name__=='__main__':main()
