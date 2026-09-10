"""Bounded, restartable collection of the preregistered A/B/C cohorts."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import torch

import run_bottle as B
from audit_visual_case import audit_case
from probe_visual_expert import atomic_json
from run_visual_bottle import run_case
from visual_baseline import load_frozen, run_baseline
from visual_planner import GraspConfig
from visual_recording import RawEpisodeRecorder
from visual_scene import STRATA, make_sim, scene_directory
from visual_sensor import BottleProxySensor
from visual_provenance import verify_runtime_assets
from visual_sampling import DEFAULT_PROFILE, PROFILES, profile_edges, validate_reuse

EXPERIMENT = Path(__file__).resolve().parents[3] / 'outputs/experiments/scalebfm_visual_goal_20260908'
STRATUM_ORDER = ['high_near', 'low_near', 'low_deep']
RECOVERIES = ['stop_short', 'wrist_offset', 'blackout', 'object_shift']
EXPERT_RUN_OPTIONS = dict(max_s=24., native_codec=True, action_hz=25,
                          integral_gain=1., reference_style='direct')


def expert_config():
    """The same current expert settings for collection and ten-video inspection."""
    return GraspConfig(stand_distance=.375, pre_tolerance=.035, lift_height=.12,
        carry_clearance=.12, edge_clearance=.22, high_table_front=.20,
        pitch_feedforward_deg=0., facing='edge')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--arm', choices=['A', 'B', 'C'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--scene-profile', choices=PROFILES, default=DEFAULT_PROFILE)
    parser.add_argument('--normal-cohort', help='Completed B cohort reused by C')
    parser.add_argument('--train-only', action='store_true', help='Collect the200 training episodes without an extra validation split')
    parser.add_argument('--pilot', action='store_true', help='12 separate interface cases; excluded from training')
    parser.add_argument('--prior-attempts', type=int, default=0, help='Physical attempts already spent by this arm in retired cohorts')
    args = parser.parse_args()
    if not 0 <= args.prior_attempts <= 880:
        parser.error('Prior physical attempts must be within the880-attempt arm budget')
    if args.train_only and (args.arm != 'B' or args.pilot):
        parser.error('--train-only requires the ordinary B expert and cannot be combined with --pilot')
    if args.arm == 'C' and not args.normal_cohort:
        parser.error('C requires its frozen B normal-demonstration source')
    if args.arm == 'C' and args.pilot:
        parser.error('Recovery qualification uses the dedicated12-case runner')
    if args.arm == 'C':
        validate_reuse(args.normal_cohort, args.scene_profile)
    edges = profile_edges(args.scene_profile)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(B.__file__), Path(B.R.__file__),
             Path(__file__).with_name('run_visual_bottle.py'), Path(__file__).with_name('audit_visual_case.py'),
             Path(__file__).with_name('probe_visual_expert.py'), *Path(__file__).parent.glob('visual_*.py')]
    cfg = vars(args) | dict(torch=torch.__version__, policy_device='cuda', grip_kp=1., grip_kd=.025,
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    cfg['runtime_assets'] = verify_runtime_assets()
    saved = out / 'config.json'
    if saved.exists() and json.loads(saved.read_text()) != cfg:
        raise ValueError('Source/config changed; use another run ID and preserve prior attempts')
    atomic_json(saved, cfg)
    (out / 'source').mkdir(exist_ok=True)
    for path in files:
        shutil.copy2(path, out / 'source' / path.name)
    status_path = out / 'run.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv)
    status.update(status='running', pid=os.getpid()); atomic_json(status_path, status)
    torch.set_num_threads(1)
    B.R.GRIP_KP, B.R.GRIP_KD = 1., .025
    policy = B.Policy(device='cuda')
    legacy = load_frozen(EXPERIMENT / 'baseline_source') if args.arm == 'A' else None
    config = expert_config()
    environments = {}
    splits = [('pilot', 1401, 12, 12)] if args.pilot else [('train', 4001, 800, 200), ('val', 6001, 80, 20)]
    if args.arm == 'C':
        splits = [('train', 8001, 400, 50), ('val', 9001, 40, 5)]
    if args.train_only:
        splits = splits[:1]
    reports = {}
    physical_count = len(list(out.glob('*/seed*/attempt.json')))
    try:
        for split, first_seed, max_attempts, target in splits:
            accepted, attempts, reused = [], [], []
            if args.arm == 'C':
                source = Path(args.normal_cohort) / f'{split}_inputs.json'
                normal = json.loads(source.read_text())
                needed = 150 if split == 'train' else 15
                if len(normal) < needed:
                    raise ValueError('B cohort has insufficient completed normal episodes')
                reused = normal[:needed]
            for i in range(max_attempts):
                seed = first_seed + i
                edge = edges[(i // 3 + i % 3) % len(edges)]
                stratum = STRATUM_ORDER[i % 3]
                recovery = RECOVERIES[(i // 3) % 4] if args.arm == 'C' else 'none'
                case = out / split / f'seed{seed}_{edge}'; case.mkdir(parents=True, exist_ok=True)
                attempt_path = case / 'attempt.json'
                # Recover a completely recorded simulation whose final metadata
                # commit was interrupted. Never replay its physics to recover it.
                if (not (case / 'result.json').exists() and (case / 'simulation_result.json').exists()
                        and all((case / name).exists() for name in ('student.npz','ego_view.mp4','right_wrist.mp4'))):
                    result = json.loads((case / 'simulation_result.json').read_text())
                    directory = scene_directory(out / 'scene_assets', stratum)
                    result.update(stratum=stratum, scene_directory=str(directory.resolve()))
                    attempt = json.loads(attempt_path.read_text())
                    result.setdefault('wall_s', max(0., (case / 'student.npz').stat().st_mtime - datetime.fromisoformat(attempt['started_utc']).timestamp()))
                    atomic_json(case / 'result.json', result)
                if (case / 'result.json').exists():
                    if not (case / 'student.npz').exists():
                        raise RuntimeError(f'Completed simulation with incomplete recording; inspect {case}')
                    result = json.loads((case / 'result.json').read_text())
                    if result.get('stratum') != stratum or not result.get('scene_directory'):
                        raise RuntimeError(f'Incomplete scene metadata: {case}')
                else:
                    if attempt_path.exists():
                        # A started physical attempt consumes budget even when
                        # interrupted. Preserve it and advance to the next seed.
                        attempt = json.loads(attempt_path.read_text())
                        attempt.update(status='interrupted', classified_utc=datetime.now(timezone.utc).isoformat())
                        atomic_json(attempt_path, attempt)
                        attempts.append(dict(seed=seed, edge=edge, stratum=stratum, recovery=recovery,
                            accepted=False, expert_ok=False, why='infrastructure interruption', case=str(case.resolve()),
                            wall_s=0., wall_cost_unknown=True, audit_gates={}))
                        atomic_json(out / f'{split}_attempts.json', attempts)
                        continue
                    if any(case.iterdir()):
                        raise RuntimeError(f'Unaccounted artifacts without attempt marker: {case}')
                    if args.prior_attempts + physical_count + (165 if args.arm == 'C' else 0) >= 880:
                        break
                    atomic_json(attempt_path, dict(status='running', seed=seed, split=split,
                        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
                    physical_count += 1
                    directory = scene_directory(out / 'scene_assets', stratum)
                    if stratum not in environments:
                        sim = make_sim(policy, directory)
                        environments[stratum] = sim, BottleProxySensor(sim)
                    sim, sensor = environments[stratum]
                    B.EDGE_INBOARD = STRATA[stratum][1]
                    recording = RawEpisodeRecorder(sim, case)
                    start = time.time()
                    try:
                        if legacy:
                            result = run_baseline(legacy, sim, policy, sensor, seed, edge, stratum, directory, case, recording,
                                                  scene_profile=args.scene_profile)
                        else:
                            result = run_case(sim, policy, sensor, seed, edge, case, config, **EXPERT_RUN_OPTIONS,
                                recording=recording, recovery_kind=recovery,
                                scene_profile=args.scene_profile, stratum=stratum)
                    finally:
                        recording.close()
                    result.update(stratum=stratum, scene_directory=str(directory.resolve()), wall_s=time.time() - start)
                    atomic_json(case / 'result.json', result)
                if attempt_path.exists():
                    attempt = json.loads(attempt_path.read_text())
                    attempt.update(status='complete', ended_utc=datetime.now(timezone.utc).isoformat(), wall_s=result['wall_s'])
                    atomic_json(attempt_path, attempt)
                audit_path = case / 'audit.json'
                audit = json.loads(audit_path.read_text()) if audit_path.exists() else audit_case(case)
                atomic_json(audit_path, audit)
                keep = (result['ok'] and audit['gates']['initial_visibility']) if args.arm == 'A' else audit['accepted']
                attempts.append(dict(seed=seed, edge=edge, stratum=stratum, recovery=recovery,
                                     accepted=bool(keep), expert_ok=result['ok'], why=result['why'], case=str(case.resolve()),
                                     wall_s=result['wall_s'], audit_gates=audit['gates']))
                if keep:
                    accepted.append(str(case.resolve()))
                atomic_json(out / f'{split}_attempts.json', attempts)
                atomic_json(out / f'{split}_inputs.json', reused + accepted)
                print(json.dumps(dict(arm=args.arm, split=split, seed=seed, stratum=stratum, ok=bool(keep),
                    why=result['why'], accepted=len(accepted), target=target, attempts=len(attempts))), flush=True)
                if len(accepted) >= target:
                    break
            reports[split] = dict(accepted=len(accepted), target=target, reused=len(reused), attempts=len(attempts),
                accepted_strata=dict(Counter(r['stratum'] for r in attempts if r['accepted'])),
                wall_s=sum(r['wall_s'] for r in attempts), complete=len(accepted) >= target)
            reports[split]['unknown_interrupted_wall_costs'] = sum(r.get('wall_cost_unknown', False) for r in attempts)
            reports[split]['prior_retired_attempts'] = args.prior_attempts
            reports[split]['physical_attempts_this_cohort'] = physical_count
            atomic_json(out / 'summary.json', reports)
    finally:
        for _, sensor in environments.values():
            sensor.close()
    status.update(status='complete' if all(r['complete'] for r in reports.values()) else 'attempt_budget_exhausted',
                  ended_utc=datetime.now(timezone.utc).isoformat(), physical_attempts_this_cohort=physical_count,
                  prior_retired_attempts=args.prior_attempts,
                  total_accounted_attempts=args.prior_attempts + physical_count + (165 if args.arm == 'C' else 0))
    atomic_json(status_path, status)
    if status['status'] != 'complete':
        raise SystemExit('Collection attempt budget exhausted before the required quotas')


if __name__ == '__main__':
    main()
