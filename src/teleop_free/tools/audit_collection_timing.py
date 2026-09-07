"""Audit expert transition timing; this is not a learned-policy chunk rollout."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(row, trace):
    events = {e['stage']: e['wall_time'] for e in row['execution']['events']}
    metrics = row['execution']['metrics']
    result = dict(case_id=row['case_id'], ok=row['ok'],
                  brake_planning=metrics.get('brake_planning', []))
    for label, a, b in (
            ('brake_to_squat_s','recovery_brake','grasp.coupled_squat'),
            ('squat_to_align_s','grasp.coupled_squat','grasp.local_align'),
            ('align_to_close_s','grasp.local_align','grasp.close')):
        result[label] = events[b]-events[a] if a in events and b in events else None
    result['safe_to_close_s'] = [m['alignment_s']-m['first_safe_elapsed_s']
        for m in metrics.get('measured_grasp', []) if m.get('alignment_s') is not None
        and m.get('first_safe_elapsed_s') is not None]
    # Observe command changes in every 0.4-second expert prefix, starting
    # every 0.04 seconds. Small tracking noise is not counted as motion.
    result.update(capture_reached='grasp.close' in events,
                  timing_complete=False, prefix_windows=[], quiet_prefix_count=0)
    if 'recovery_brake' not in events:
        return result
    # Failed attempts are essential to the stall audit. They have no close
    # event; use their terminal failure instead of silently omitting them.
    # Use the first braking event so recovery loops cannot hide earlier waits.
    brake_start = next(e['wall_time'] for e in row['execution']['events']
                       if e['stage'] == 'recovery_brake')
    endpoint = next((e for e in row['execution']['events']
                     if e['wall_time'] >= brake_start
                     and e['stage'] in ('grasp.close', 'failed', 'succeeded')), None)
    if endpoint is None:
        return result
    end = endpoint['wall_time']
    result['timing_endpoint'] = endpoint['stage']
    # A truncated sidecar is not evidence that the unobserved interval was
    # active or quiet. Only count windows with their own sufficient coverage.
    result['timing_complete'] = bool(trace and trace[0]['wall'] <= brake_start+.1
                                    and trace[-1]['wall'] >= end-.1)
    trace = [r for r in trace if brake_start <= r['wall'] <= end+.4]
    gaps = np.diff([r['wall'] for r in trace])
    result['max_trace_gap_s'] = float(np.max(gaps)) if len(gaps) else None
    result['timing_complete'] &= bool(len(gaps) and np.all((gaps > 0) & (gaps <= .1)))
    windows = []
    for start in np.arange(brake_start, end, .04):
        rows = [r for r in trace if start <= r['wall'] <= start+.4]
        if (len(rows)<2 or rows[-1]['wall']-rows[0]['wall'] < .32
                or np.max(np.diff([r['wall'] for r in rows])) > .1):
            continue
        upper = np.asarray([r['u17'] for r in rows if 'u17' in r])
        hands = np.asarray([r['hands']['right'] for r in rows
                            if isinstance(r.get('hands'),dict) and 'right' in r['hands']])
        commands = [r['cmd'] for r in rows if 'cmd' in r]
        squat = [c['height'] for c in commands if c['mode']==4]
        change = bool(len(upper)>1 and np.max(np.ptp(upper,axis=0))>.01)
        change |= bool(len(hands)>1 and np.max(np.ptp(hands,axis=0))>.001)
        change |= bool(squat and (len(squat)!=len(commands) or np.ptp(squat)>.003))
        windows.append(dict(start_after_brake_s=float(start-brake_start),
                            command_changes=change))
    result['prefix_windows'] = windows
    result['quiet_prefix_count'] = sum(not w['command_changes'] for w in windows)
    if 'grasp.coupled_squat' in events:
        rows = [r for r in trace if r['wall'] >= events['grasp.coupled_squat']]
        if rows:
            hit = next((r for r in rows if r['pelvis'][2] < rows[0]['pelvis'][2]-.005), None)
            result['squat_command_to_5mm_delivery_s'] = (
                hit['wall']-events['grasp.coupled_squat'] if hit else None)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (args.root/'manifest.jsonl').read_text().splitlines()]
    results = []
    for row in rows:
        if row['case_id']=='initialization_warmup':continue
        files = list((args.root/'pose_sidecar').glob(
            f'*_round_{row["round"]:03d}_attempt_{row["attempt"]}.jsonl'))
        if len(files)!=1:raise ValueError(f'Expected one trace for {row["case_id"]}')
        trace = [json.loads(l) for l in files[0].read_text().splitlines()]
        results.append(audit(row,trace))
    report = dict(scope='Expert command prefixes; not VLA predictions or closed-loop proof',
                  action_fps=25, chunk_steps=40, replan_s=.4, records=results)
    (args.root/'timing_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='prefix_windows'} for r in results],indent=2))


if __name__=='__main__':main()
