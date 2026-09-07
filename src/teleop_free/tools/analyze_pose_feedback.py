"""Summarize measured fixed-world feedback outcomes, including early aborts."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pose_feedback import rotation_vector
from primitives import Primitives


def analyze(root):
    events=[json.loads(l) for l in (root/'feedback.jsonl').read_text().splitlines()]
    start=events[0];end=events[-1]
    if start['event']!='pose_feedback_start' or end['event']!='pose_feedback_exit':raise ValueError('incomplete feedback capture')
    target=np.asarray(start['target_pos']);rot=np.asarray(start['target_rot'])
    base=np.asarray(start.get('origin',start['state']['pelvis']['pos']));ticks=[]
    for row in events:
        if 'state' not in row:continue
        st=row['state'];palm=st['palms']['right']
        ticks.append(dict(t=row['wall']-start['wall'],
            position_error=float(np.linalg.norm(target-palm['pos'])),
            rotation_error=float(np.linalg.norm(rotation_vector(rot,Primitives._rot(palm['quat'])))),
            base_drift=float(np.linalg.norm(np.asarray(st['pelvis']['pos'])-base))))
    refs=[r for r in events if r['event']=='pose_feedback_tick']
    result=dict(ok=end['ok'],reason=end['reason'],initial=ticks[0],final=ticks[-1],
        best_rotation_error=min(r['rotation_error'] for r in ticks),
        max_base_drift=max(r['base_drift'] for r in ticks),
        max_excursion=max((r['excursion'] for r in refs),default=0.),
        clipped_ticks=sum(r['clipped'] for r in refs),sent_ticks=len(refs),
        command_period_median=float(np.median(np.diff([r['wall'] for r in refs]))) if len(refs)>1 else None,
        ticks=ticks)
    (root/'feedback_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(root.name,json.dumps({k:v for k,v in result.items() if k!='ticks'}))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('capture',type=Path,nargs='+');ap.add_argument('--require-success',action='store_true');a=ap.parse_args()
    success=True
    for root in a.capture:success &= analyze(root)['ok']
    if a.require_success and not success:raise SystemExit(1)
