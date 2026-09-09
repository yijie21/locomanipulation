#!/usr/bin/env python3
"""Behaviour gate for the refactor (docs/handoff/refactor-plan.md, decision 2).

The online pipeline is not bit-deterministic (wall-clock loops, async deploy), so
the gate compares DISTRIBUTIONS of a short fixed-seed run against a recorded
baseline, with tolerances set by the measured seed variance. Two arms:

  bottle  single-arm ready-grasp collection (grasp_gym manifests, 1 batch of 10)
  crate   bimanual carry pilot (bimanual_carry manifests, 10 rounds)

Record a baseline (once, before touching main code):
    python scripts/gate.py record bottle "outputs/experiments/<run>/collect/manifest_*.jsonl" tests/gate/baseline_bottle.json
    python scripts/gate.py record crate  outputs/videos/wall_pilot/manifest.jsonl               tests/gate/baseline_crate.json

Produce the candidate run with the SAME commands the baseline used (see the
`source` field of the baseline JSON), then:
    python scripts/gate.py compare bottle "<new manifests glob>" tests/gate/baseline_bottle.json [--json result.json]
Exit 0 = PASS, 1 = FAIL. Paste the printed table into the PR.

Tolerances (decision 2): yield +-0.15 absolute; depth / stand medians +-0.01 m;
squat drop +-0.005 m; per-stage duration medians +-30 %; the SET of stages and
the SET of (normalised) failure reasons may not gain new members; the majority
approach-exit reason must match.
"""
from __future__ import annotations
import argparse, glob, json, re, statistics, sys
from collections import Counter

TOL = dict(yield_abs=.15, depth_m=.01, drop_m=.005, duration_rel=.30, angle_deg=1.5)


def _rows(pattern):
    files = sorted(glob.glob(pattern)) or ([pattern] if pattern.endswith('.jsonl') else [])
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        sys.exit(f'[gate] no manifest rows under {pattern}')
    return rows


def _norm_reason(why):
    # "[stand] could not turn -12 deg ..." -> "[stand] could not turn N deg ..."; numbers are noise
    return re.sub(r'[-+]?\d+(\.\d+)?', 'N', why or '')[:70]


def _median(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and x == x]
    return round(statistics.median(xs), 4) if xs else None


def metrics_bottle(rows):
    ok = [r for r in rows if r.get('ok') and r.get('produced', True)]
    stages, exits, ends, drops, durs = Counter(), Counter(), [], [], {}
    for r in ok:
        ex = r.get('execution') or {}
        ev = ex.get('events') or []
        names = [e['stage'] for e in ev]
        stages.update(set(names))
        for a, b in zip(ev, ev[1:]):
            durs.setdefault(a['stage'], []).append(b['elapsed_s'] - a['elapsed_s'])
        for e in ev:
            if e['stage'] == 'recovery_attempt' and e.get('drop') is not None:
                drops.append(e['drop'])
        for a in (ex.get('metrics') or {}).get('arrival_measurements') or []:
            exits[a.get('reason')] += 1
            ends.append(a.get('end_depth_m'))
    return dict(
        rounds=len(rows), ok=len(ok), yield_=round(len(ok) / len(rows), 3),
        stages=sorted(stages), exit_majority=(exits.most_common(1)[0][0] if exits else None),
        exits=dict(exits), end_depth_m=_median(ends), squat_drop_m=_median(drops),
        stage_s={k: _median(v) for k, v in durs.items() if len(v) >= 3},
        failure_reasons=sorted({_norm_reason(r.get('why')) for r in rows if not r.get('ok')}),
        seconds=_median([r.get('seconds') for r in rows]))


def metrics_crate(rows):
    ok = [r for r in rows if r.get('ok')]
    return dict(
        rounds=len(rows), ok=len(ok), yield_=round(len(ok) / len(rows), 3),
        exit_majority=Counter(r.get('approach_exit') for r in rows).most_common(1)[0][0],
        close_triggers=dict(Counter(r.get('close_trigger') for r in ok)),
        stand_off_m=_median([r.get('stand_off_meas') for r in rows]),
        yaw_err_deg=_median([abs(r['yaw_err_deg']) for r in rows if 'yaw_err_deg' in r]),
        depth_m=_median([r.get('depth') for r in rows]),
        stage_s=dict(reach=_median([r.get('reach_s') for r in ok]), descend=_median([r.get('descend_s') for r in ok])),
        failure_reasons=sorted({_norm_reason(r.get('why')) for r in rows if not r.get('ok')}),
        seconds=_median([r.get('seconds') for r in rows]),
        deploy_restarts=sum(int(r.get('deploy_restarts') or 0) for r in rows))


METRICS = dict(bottle=metrics_bottle, crate=metrics_crate)


def compare(base, cand):
    checks = []
    def add(name, ok, detail): checks.append((name, bool(ok), detail))
    add('yield', abs(cand['yield_'] - base['yield_']) <= TOL['yield_abs'], f"{base['yield_']} -> {cand['yield_']}")
    for key, tol in (('end_depth_m', 'depth_m'), ('stand_off_m', 'depth_m'), ('depth_m', 'depth_m'),
                     ('squat_drop_m', 'drop_m'), ('yaw_err_deg', 'angle_deg')):
        if key in base and base[key] is not None and cand.get(key) is not None:
            add(key, abs(cand[key] - base[key]) <= TOL[tol], f"{base[key]} -> {cand[key]}")
    add('exit_majority', cand.get('exit_majority') == base.get('exit_majority'),
        f"{base.get('exit_majority')} -> {cand.get('exit_majority')}")
    if 'stages' in base:
        new = sorted(set(cand['stages']) - set(base['stages'])); gone = sorted(set(base['stages']) - set(cand['stages']))
        add('stages', not new and not gone, f"new {new} missing {gone}")
    new_f = sorted(set(cand['failure_reasons']) - set(base['failure_reasons']))
    add('failure_reasons', not new_f, f"new {new_f}")
    for k, v in base.get('stage_s', {}).items():
        c = cand.get('stage_s', {}).get(k)
        if v and c is not None:
            add(f'stage_s.{k}', abs(c - v) <= TOL['duration_rel'] * v, f"{v} -> {c}")
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('record', 'compare'))
    ap.add_argument('arm', choices=tuple(METRICS))
    ap.add_argument('manifests', help='manifest .jsonl path or glob')
    ap.add_argument('baseline', help='baseline JSON to write (record) or read (compare)')
    ap.add_argument('--json', default=None, help='compare: also write the result here')
    args = ap.parse_args()
    m = METRICS[args.arm](_rows(args.manifests))
    if args.mode == 'record':
        json.dump(dict(arm=args.arm, source=args.manifests, metrics=m), open(args.baseline, 'w'), indent=1)
        print(json.dumps(m, indent=1)); print(f'[gate] baseline -> {args.baseline}'); return
    base = json.load(open(args.baseline))['metrics']
    checks = compare(base, m)
    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    verdict = 'PASS' if all(c[1] for c in checks) else 'FAIL'
    print(f'[gate] {args.arm}: {verdict} ({sum(c[1] for c in checks)}/{len(checks)})')
    if args.json:
        json.dump(dict(arm=args.arm, verdict=verdict, candidate=m, checks=[dict(name=n, ok=o, detail=d) for n, o, d in checks]),
                  open(args.json, 'w'), indent=1)
    sys.exit(0 if verdict == 'PASS' else 1)


if __name__ == '__main__':
    main()
