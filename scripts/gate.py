#!/usr/bin/env python3
"""Behaviour gate for the refactor (docs/handoff/refactor-plan.md, decision 2).

The online pipeline is not bit-deterministic (wall-clock loops, async deploy), so
the gate compares DISTRIBUTIONS of a short fixed-seed run against a recorded
baseline, with tolerances set by the measured seed variance. Two arms:

  bottle  single-arm ready-grasp collection (grasp_gym manifests, 1 batch of 10)
  simple  frozen single-arm simple expert (fixed bank regression8_lowcom, 8 rounds)
  carry   crate record line (tasks/crate/expert.py --record manifests, 20 rounds)
  crate   bimanual wall pilot (bimanual_wall manifests, 10 rounds)

The crate geometry/timing fields are emitted by wall_phases*.py. Carry manifests
do not provide them; missing metrics must not be interpreted as verified checks.

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

import argparse
import glob
import json
import re
import statistics
import sys
from collections import Counter

TOL = dict(yield_abs=0.15, depth_m=0.01, drop_m=0.005, duration_rel=0.30, angle_deg=1.5, duration_floor_s=0.05)
# The simple expert's stop is set by the planner's gait at the brake: three batches of the same code
# gave stop medians 0.147-0.157 m (stage4_wrap8n/8o/8p), so its depth tolerance is 2 cm, and its
# rare last-resort phases (retreat) may be absent from a candidate without being a regression.
# A reason ABSENT from a baseline of n rounds is not evidence that it cannot happen: by the rule of
# three its rate could still be as high as 3/n at 95%. So the check below is on the SHARE of
# candidate rounds whose failure the baseline cannot explain, not on whether such a reason exists.
#
# Calibrated 2026-09-21 on seven runs of ONE expert revision over one fixed bank and seed (42
# ordered pairs), against four revisions the ledger records as behaving differently:
#
#   criterion                    A/A false alarms      regressions caught
#   any new reason (until now)   29/42  69%            5/5
#   share > 3/n_base              1/42   2%            3/5
#   share > 2/n_base              3/42   7%            4/5
#
# The old test barely discriminated: it fired on 69% of identical-code pairs against 75% of
# genuinely different ones. Six failure families appeared across those seven runs, four of them in
# a single run each, so a set observed over eight rounds cannot enumerate the tail and a
# set-difference test is close to a coin toss. 3 is the rule of three, not the value that scores
# best here; `yield` already catches all five regressions by itself, so what this check is left to
# add is the case yield cannot see -- the same success rate reached by failing a new way -- and at
# n=8 it can only see that when it is large.
RULE_OF_THREE = 3.0

ARM_TOL = dict(simple=dict(depth_m=0.02), carry=dict(depth_m=0.03))
ARM_STAGES_NEW_ONLY = {"simple"}


def _rows(pattern):
    files = sorted(glob.glob(pattern)) or ([pattern] if pattern.endswith(".jsonl") else [])
    rows = []
    for f in files:
        rows += [json.loads(l) for l in open(f) if l.strip()]
    if not rows:
        sys.exit(f"[gate] no manifest rows under {pattern}")
    return rows


def _norm_reason(why):
    # "[stand] could not turn -12 deg ..." -> "[stand] could not turn N deg ..."; numbers are noise
    return re.sub(r"[-+]?\d+(\.\d+)?", "N", why or "")[:70]


def _norm_side(why):
    # the side of a body/table contact is stance noise, not behaviour
    return _norm_reason(re.sub(r"left|right", "S", why or ""))


def _median(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and x == x]
    return round(statistics.median(xs), 4) if xs else None


def metrics_bottle(rows):
    ok = [r for r in rows if r.get("ok") and r.get("produced", True)]
    stages, exits, ends, drops, durs = Counter(), Counter(), [], [], {}
    for r in ok:
        ex = r.get("execution") or {}
        ev = ex.get("events") or []
        names = [e["stage"] for e in ev]
        stages.update(set(names))
        for a, b in zip(ev, ev[1:]):
            durs.setdefault(a["stage"], []).append(b["elapsed_s"] - a["elapsed_s"])
        for e in ev:
            if e["stage"] == "recovery_attempt" and e.get("drop") is not None:
                drops.append(e["drop"])
        for a in (ex.get("metrics") or {}).get("arrival_measurements") or []:
            exits[a.get("reason")] += 1
            ends.append(a.get("end_depth_m"))
    return dict(
        rounds=len(rows),
        ok=len(ok),
        yield_=round(len(ok) / len(rows), 3),
        stages=sorted(stages),
        exit_majority=(exits.most_common(1)[0][0] if exits else None),
        exits=dict(exits),
        end_depth_m=_median(ends),
        squat_drop_m=_median(drops),
        stage_s={k: _median(v) for k, v in durs.items() if len(v) >= 3},
        failure_reasons=sorted({_norm_reason(r.get("why")) for r in rows if not r.get("ok")}),
        failure_counts=dict(Counter(_norm_reason(r.get("why")) for r in rows if not r.get("ok"))),
        seconds=_median([r.get("seconds") for r in rows]),
    )


def metrics_crate(rows):
    ok = [r for r in rows if r.get("ok")]
    return dict(
        rounds=len(rows),
        ok=len(ok),
        yield_=round(len(ok) / len(rows), 3),
        exit_majority=Counter(r.get("approach_exit") for r in rows).most_common(1)[0][0],
        close_triggers=dict(Counter(r.get("close_trigger") for r in ok)),
        stand_off_m=_median([r.get("stand_off_meas") for r in rows]),
        yaw_err_deg=_median([abs(r["yaw_err_deg"]) for r in rows if "yaw_err_deg" in r]),
        depth_m=_median([r.get("depth") for r in rows]),
        stage_s=dict(reach=_median([r.get("reach_s") for r in ok]), descend=_median([r.get("descend_s") for r in ok])),
        failure_reasons=sorted({_norm_reason(r.get("why")) for r in rows if not r.get("ok")}),
        failure_counts=dict(Counter(_norm_reason(r.get("why")) for r in rows if not r.get("ok"))),
        seconds=_median([r.get("seconds") for r in rows]),
        deploy_restarts=sum(int(r.get("deploy_restarts") or 0) for r in rows),
    )


def metrics_simple(rows):
    """The frozen single-arm `reach.strategy=simple` expert (docs/handoff/infra-plan-20260916.md, decision 7)."""
    ok = [r for r in rows if r.get("ok")]
    phases, edges, prep, switch, entry_z = Counter(), [], [], 0, []
    for r in rows:
        m = ((r.get("execution") or {}).get("metrics") or {}).get("simple") or {}
        phases.update({p["phase"] for p in m.get("phases") or []})
        if (m.get("plan") or {}).get("pelvis_edge_m") is not None:
            edges.append(m["plan"]["pelvis_edge_m"])
        if (m.get("prepare_walk") or {}).get("seconds") is not None:
            prep.append(m["prepare_walk"]["seconds"])
        switch += int(bool(((m.get("takeover") or {}).get("switch") or {}).get("confirmed")))
        for e in m.get("entries") or []:
            if e.get("offset_in_palm"):
                entry_z.append(abs(e["offset_in_palm"][2]))
    return dict(
        rounds=len(rows),
        ok=len(ok),
        yield_=round(len(ok) / len(rows), 3),
        stages=sorted(phases),
        end_depth_m=_median(edges),
        prepare_walk_s=_median(prep),
        switch_confirmed=switch,
        entry_height_err_m=_median(entry_z),
        failure_reasons=sorted({_norm_side(r.get("why")) for r in rows if not r.get("ok")}),
        failure_counts=dict(Counter(_norm_side(r.get("why")) for r in rows if not r.get("ok"))),
        seconds=_median([r.get("seconds") for r in rows]),
    )


def metrics_carry(rows):
    """The crate record line (`tasks/crate/expert.py --record` manifests: rise, tilt, stand attempts)."""
    ok = [r for r in rows if r.get("ok")]
    return dict(
        rounds=len(rows),
        ok=len(ok),
        yield_=round(len(ok) / len(rows), 3),
        produced=sum(int(bool(r.get("produced"))) for r in rows),
        rise_m=_median([r.get("rise") for r in ok]),
        tilt_deg=_median([r.get("tilt_deg") for r in ok]),
        end_depth_m=_median([r.get("initial_edge_distance") for r in rows]),
        stand_attempts=_median([r.get("stand_attempts") for r in rows]),
        failure_reasons=sorted({_norm_side(r.get("why")) for r in rows if not r.get("ok")}),
        failure_counts=dict(Counter(_norm_side(r.get("why")) for r in rows if not r.get("ok"))),
        seconds=_median([r.get("seconds") for r in rows]),
    )


METRICS = dict(bottle=metrics_bottle, crate=metrics_crate, simple=metrics_simple, carry=metrics_carry)


def compare(base, cand, arm=None):
    checks = []
    tol = TOL | ARM_TOL.get(arm, {})

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    # One-sided: a yield DROP beyond the tolerance is a regression; a higher yield is not
    # (2026-09-10: an 8/9 batch tripped the symmetric check against a 20/30 baseline).
    add("yield", cand["yield_"] >= base["yield_"] - tol["yield_abs"], f"{base['yield_']} -> {cand['yield_']}")
    for key, tol_key in (
        ("end_depth_m", "depth_m"),
        ("stand_off_m", "depth_m"),
        ("depth_m", "depth_m"),
        ("squat_drop_m", "drop_m"),
        ("yaw_err_deg", "angle_deg"),
    ):
        if key in base and base[key] is not None and cand.get(key) is not None:
            add(key, abs(cand[key] - base[key]) <= tol[tol_key], f"{base[key]} -> {cand[key]}")
    add(
        "exit_majority",
        cand.get("exit_majority") == base.get("exit_majority"),
        f"{base.get('exit_majority')} -> {cand.get('exit_majority')}",
    )
    if "stages" in base:
        new = sorted(set(cand["stages"]) - set(base["stages"]))
        gone = sorted(set(base["stages"]) - set(cand["stages"]))
        add("stages", not new and (not gone or arm in ARM_STAGES_NEW_ONLY), f"new {new} missing {gone}")
    new_f = sorted(set(cand["failure_reasons"]) - set(base["failure_reasons"]))
    counts = cand.get("failure_counts") or {}
    unexplained = sum(counts.get(r, 1) for r in new_f)
    share = unexplained / cand["rounds"] if cand.get("rounds") else 0.0
    bound = RULE_OF_THREE / base["rounds"] if base.get("rounds") else 1.0
    add(
        "failure_reasons",
        share <= bound,
        f"new {new_f} in {unexplained}/{cand.get('rounds')} rounds ({share:.2f} vs bound {bound:.2f})",
    )
    for k, v in base.get("stage_s", {}).items():
        c = cand.get("stage_s", {}).get(k)
        if v and c is not None:
            if v < tol["duration_floor_s"]:
                continue  # millisecond-scale stage markers: their jitter is not behaviour
            add(f"stage_s.{k}", abs(c - v) <= tol["duration_rel"] * v, f"{v} -> {c}")
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("record", "compare"))
    ap.add_argument("arm", choices=tuple(METRICS))
    ap.add_argument("manifests", help="manifest .jsonl path or glob")
    ap.add_argument("baseline", help="baseline JSON to write (record) or read (compare)")
    ap.add_argument("--json", default=None, help="compare: also write the result here")
    args = ap.parse_args()
    m = METRICS[args.arm](_rows(args.manifests))
    if args.mode == "record":
        json.dump(dict(arm=args.arm, source=args.manifests, metrics=m), open(args.baseline, "w"), indent=1)
        print(json.dumps(m, indent=1))
        print(f"[gate] baseline -> {args.baseline}")
        return
    base = json.load(open(args.baseline))["metrics"]
    checks = compare(base, m, args.arm)
    width = max(len(c[0]) for c in checks)
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    verdict = "PASS" if all(c[1] for c in checks) else "FAIL"
    print(f"[gate] {args.arm}: {verdict} ({sum(c[1] for c in checks)}/{len(checks)})")
    if args.json:
        json.dump(
            dict(
                arm=args.arm, verdict=verdict, candidate=m, checks=[dict(name=n, ok=o, detail=d) for n, o, d in checks]
            ),
            open(args.json, "w"),
            indent=1,
        )
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
