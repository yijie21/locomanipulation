"""Report accepted coverage, frame mix and physical-attempt cost, offline only."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from probe_visual_expert import atomic_json


def summarize(root):
    report = {}
    for arm in 'ABC':
        cohort = root / f'cohort_{arm}_v2'
        if not (cohort / 'summary.json').exists():
            report[arm] = {'status':'not_complete'}
            continue
        run = json.loads((cohort / 'run.json').read_text())
        if run['status'] != 'complete':
            report[arm] = {'status':run['status']}
            continue
        arm_report = dict(status='complete', physical_attempts=run['physical_attempts_this_cohort'],
            prior_retired_attempts=run['prior_retired_attempts'], total_accounted_attempts=run['total_accounted_attempts'])
        used = set()
        for split, expected in [('train',200),('val',20)]:
            inputs = [Path(p) for p in json.loads((cohort / f'{split}_inputs.json').read_text())]
            if len(inputs) != expected or len(set(inputs)) != expected or used.intersection(inputs):
                raise ValueError(f'{arm}/{split}: episode quota, uniqueness or split separation failed')
            used.update(inputs)
            counts, frames = Counter(), defaultdict(int)
            lengths = []
            for case in inputs:
                result = json.loads((case / 'result.json').read_text())
                with np.load(case / 'student.npz') as data:
                    n = len(data['action'])
                    if n < 40:
                        raise ValueError(f'No full40-row training chunk: {case}')
                kind = result.get('recovery_kind','none')
                for key in (f'stratum:{result["stratum"]}',f'recovery:{kind}'):
                    counts[key] += 1; frames[key] += n
                lengths.append(n)
            attempts = json.loads((cohort / f'{split}_attempts.json').read_text())
            arm_report[split] = dict(episodes=len(inputs), frames=sum(lengths),
                mean_episode_seconds=float(np.mean(lengths)/25),
                min_episode_frames=min(lengths), max_episode_frames=max(lengths),
                effective_training_starts=sum(n-39 for n in lengths),
                episode_mix=dict(counts), frame_mix=dict(frames),
                new_attempts=len(attempts), new_failures=sum(not r['accepted'] for r in attempts),
                measured_new_attempt_wall_seconds=sum(r['wall_s'] for r in attempts),
                unknown_interrupted_wall_costs=sum(r.get('wall_cost_unknown',False) for r in attempts))
        report[arm] = arm_report
    atomic_json(root / 'cohort_data_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(Path(args.root)),indent=2))
