"""Registered paired VLA selection and confirmation statistics."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from probe_visual_expert import atomic_json

TRAIN_SEEDS = (17, 29)


def load_arm(root, arm, stage):
    return [json.loads((root / f'{stage}_{arm}_s{seed}' / 'summary.json').read_text())['cases'] for seed in TRAIN_SEEDS]


def posture_bad(case):
    m = case['metrics']
    pitch = max(v[3] for v in m['phase_max_abs_deg'].values())
    upright_violation = (case['stratum'] == 'high_near' and case.get('close_pitch_median_deg') is not None
                         and case['close_pitch_median_deg'] > 15)
    return m['torso_roll_max_deg'] > 20 or m['torso_roll_above15_s'] > .2 or m['waist_yaw_max_deg'] > 25 or pitch > 40 or upright_violation


def summarize(cases):
    counts = [sum(c['ok'] for c in run) for run in cases]
    strata = {}
    for key in ('high_near', 'low_near', 'low_deep'):
        strata[key] = [dict(ok=sum(c['ok'] for c in run if c['stratum'] == key), count=sum(c['stratum'] == key for c in run)) for run in cases]
    groups = {}
    for kind in ('none', 'stop_short', 'wrist_offset', 'blackout', 'object_shift'):
        groups[kind] = [dict(ok=sum(c['ok'] for c in run if c['recovery_kind'] == kind), count=sum(c['recovery_kind'] == kind for c in run)) for run in cases]
    return dict(successes_by_training_seed=counts, count_per_seed=len(cases[0]), mean_success_count=float(np.mean(counts)),
        mean_success_rate=float(np.mean(counts) / len(cases[0])), strata=strata, recovery_groups=groups,
        mean_posture_violations=float(np.mean([sum(posture_bad(c) for c in run) for run in cases])),
        stationary_without_close_by_seed=[sum(c.get('stationary_without_close',False) for c in run) for run in cases],
        mean_visibility_fraction=float(np.mean([c.get('visibility_fraction',0.) for run in cases for c in run])),
        median_case_inference_ms=float(np.median([c['inference_median_ms'] for run in cases for c in run if c.get('inference_median_ms') is not None]))
            if any(c.get('inference_median_ms') is not None for run in cases for c in run) else None)


def paired(candidate, baseline):
    results, differences = [], []
    for seed, a, b in zip(TRAIN_SEEDS, candidate, baseline):
        keys = lambda rows: [(r['seed'],r['stratum'],r['edge'],r['recovery_kind']) for r in rows]
        if keys(a) != keys(b):
            raise ValueError('Paired evaluation scenes differ')
        x, y = np.array([r['ok'] for r in a]), np.array([r['ok'] for r in b])
        win, loss = int((x & ~y).sum()), int((~x & y).sum())
        differences.append(x.astype(float) - y)
        results.append(dict(training_seed=seed, candidate_only=win, baseline_only=loss,
            mcnemar_exact_p=float(binomtest(win, win + loss).pvalue) if win + loss else 1.))
    # Resample scenes jointly across both train seeds. Two training runs on the
    # same scene do not create twice as many independent scene observations.
    per_scene = np.mean(differences, axis=0)
    rng = np.random.default_rng(20260908)
    samples = per_scene[rng.integers(0, len(per_scene), size=(20000, len(per_scene)))].mean(1)
    return dict(per_seed=results, mean_success_rate_difference=float(per_scene.mean()),
                scene_bootstrap_95_interval=np.quantile(samples,[.025,.975]).tolist())


def select(root):
    arms = {arm:load_arm(root, arm, 'selection') for arm in 'ABC'}
    if any(len(run) != 60 for runs in arms.values() for run in runs):
        raise ValueError('Selection requires60 cases per arm and training seed')
    summaries = {arm:summarize(runs) for arm,runs in arms.items()}
    best = max(v['mean_success_count'] for v in summaries.values())
    tied = [arm for arm in arms if best - summaries[arm]['mean_success_count'] <= 1.]
    winner = min(tied, key=lambda arm:(summaries[arm]['mean_posture_violations'], 'ABC'.index(arm)))
    report = dict(winner=winner, arms=summaries, tie_candidates=tied,
                  paired_vs_A={arm:paired(runs, arms['A']) for arm,runs in arms.items() if arm != 'A'},
                  rule='Mean task success; within1/60 prefer fewer posture violations, then simpler arm. Final3000 only.')
    path = root / 'selection_locked.json'
    if path.exists() and json.loads(path.read_text()) != report:
        raise ValueError('Locked selection differs; do not retune using confirmation')
    atomic_json(path, report)
    return report


def confirm(root):
    selection = json.loads((root / 'selection_locked.json').read_text())
    winner = selection['winner']
    arms = {arm:load_arm(root, arm, 'confirmation') for arm in sorted(set(['A',winner]))}
    if any(len(run) != 100 for runs in arms.values() for run in runs):
        raise ValueError('Confirmation requires100 cases per model')
    report = dict(winner=winner, arms={arm:summarize(runs) for arm,runs in arms.items()},
                  paired_vs_A=paired(arms[winner], arms['A']))
    atomic_json(root / 'confirmation_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    print(json.dumps((confirm if args.confirmation else select)(Path(args.root)), indent=2))
