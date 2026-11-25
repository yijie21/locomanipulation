"""TODO 27.10.1 (plan docs/handoff/scene-rsi-plan-20261016.md §12.3): how often is the checker right when it puts a
task nobody ran into bucket 1? The bucket-1 count of the 2830 templates rose 41 -> 97 after write-back 7 because
two measured put-ins widened the measured ranges, and the checker extrapolates inside them. This samples tasks the
checker puts in bucket 1 whose goal no family ever ran, stratified by goal kind (the predicates it uses), plus a
control sample from bucket 2, and runs each `--per-task` times with the current catalog, so bucket 1's precision is
a measured number.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/calibrate.py sample cal_r2 \\
        --runs outputs/scene_rsi_20261016/capability/feasibility/upright_{round2,round1,demo1,templates_2830,templates}
    PYTHONPATH=src $PY scripts/rsi/family.py draw cal_r2 sealed <pool>
    PYTHONPATH=src $PY scripts/rsi/batch.py cal_r2 <pool> src/locomani/tasks/scene/skills <rounds>
    PYTHONPATH=src $PY scripts/rsi/calibrate.py report cal_r2 <rounds>

`sample` writes the tasks to outputs/scene_rsi_20261016_hidden/calibration/<name>/ (tasks.jsonl, sample.json) and
the family file families/<name>.json (one pool, `sealed`, from --pool-seed). Goals a family or an earlier sample
ran are left out. `report` writes report.json beside them: per task the success rate; bucket 1's precision (tasks
at >= BUCKET1_RATE among those with at least MIN_N rounds) and mean success with Wilson intervals, per goal kind and
per source; the same for the bucket-2 control. cal_r3 (after the checker's carry_single.creep rule, plan §12.3) is
drawn from the creep_* runs with --pool-seed 8100000.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(HERE.parent), str(REPO / "src")]

from gate import wilson  # noqa: E402

from locomani.scene_discovery.predicates import parse  # noqa: E402

CAL_SEED = 8_000_000  # no pool's block comes near it (rounds: 1,1x0,000 + 1,000,000 per round); cal_r3 8,100,000
OUT = REPO / "outputs" / "scene_rsi_20261016_hidden" / "calibration"
SCENE = "outputs/scene_discovery/adt_apartment_mars"
BUCKET1_RATE = 0.50
MIN_N = 4


def norm(goal) -> tuple:
    return tuple(sorted(str(parse(g)) for g in goal))


def kind(goal) -> str:
    return "+".join(sorted({parse(g).name for g in goal}))


def measured_goals(name: str = "") -> set:
    """Every goal a family or an earlier calibration sample ran (its tasks' goals): measured, not predicted. The
    sample `name` itself is left out, so it can be drawn again."""
    out = set()
    for f in sorted((HERE.parent / "families").glob("*.json")):
        fam = json.loads(f.read_text())
        if fam["name"] == name or fam.get("backend") or not fam.get("tasks_file"):
            continue  # a second backend's goals are not scene goals (norm() would not parse them)
        rows = {json.loads(x)["id"]: json.loads(x) for x in (REPO / fam["tasks_file"]).read_text().splitlines() if x.strip()}  # fmt: skip
        out |= {norm(rows[t]["goal"]) for t in fam["tasks"] if t in rows}
    return out


def allocate(sizes: dict, n: int) -> dict:
    """n spread over strata by size, at least one per stratum while n lasts, largest remainder (ties by name)."""
    keys = sorted(sizes)
    total = sum(sizes.values())
    if total == 0 or n <= 0:
        return {k: 0 for k in keys}
    want = {k: n * sizes[k] / total for k in keys}
    out = {k: min(sizes[k], max(1, math.floor(want[k]))) for k in keys}
    while sum(out.values()) > n:  # more strata than n: drop the smallest strata's one
        k = min((k for k in keys if out[k] > 0), key=lambda k: (sizes[k], k))
        out[k] -= 1
    rest = sorted(keys, key=lambda k: (-(want[k] - math.floor(want[k])), k))
    while sum(out.values()) < min(n, total):
        for k in rest:
            if out[k] < sizes[k] and sum(out.values()) < n:
                out[k] += 1
    return out


def sample(name: str, runs: list[Path], n1: int, n2: int, per_task: int, seed: int, pool_seed: int = CAL_SEED) -> dict:
    measured = measured_goals(name)
    seen, cands = set(), {1: collections.defaultdict(list), 2: collections.defaultdict(list)}
    for run in runs:
        for line in (run / "tasks.jsonl").read_text().splitlines():
            t = json.loads(line)
            g = norm(t["goal"])
            if t.get("invalid") or t["bucket"] not in (1, 2) or g in measured or g in seen:
                continue
            seen.add(g)
            cands[t["bucket"]][kind(t["goal"])].append(dict(t, source=run.name))
    rng = np.random.default_rng(seed)
    picked = []
    for bucket, n in ((1, n1), (2, n2)):
        alloc = allocate({k: len(v) for k, v in cands[bucket].items()}, n)
        for k in sorted(alloc):
            rows = cands[bucket][k]
            for i in sorted(rng.choice(len(rows), size=alloc[k], replace=False)):
                picked.append(rows[int(i)])
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for t in picked:
        tid = f"b{t['bucket']}_{t['id']}"
        tasks.append(tid)
        t["cal_id"] = tid
    (out / "tasks.jsonl").write_text("".join(json.dumps(dict(id=t["cal_id"], goal=t["goal"], room=t.get("room"))) + "\n"
                                             for t in picked))  # fmt: skip
    summary = dict(name=name, runs=[str(r) for r in runs], seed=seed, per_task=per_task, measured_goals=len(measured),
                   candidates={b: {k: len(v) for k, v in sorted(c.items())} for b, c in cands.items()},
                   picked=[dict(id=t["cal_id"], bucket=t["bucket"], kind=kind(t["goal"]), source=t["source"],
                                goal=t["goal"]) for t in picked])  # fmt: skip
    (out / "sample.json").write_text(json.dumps(summary, indent=1))
    fam = dict(name=name, about=f"checker calibration (plan §12.3): {n1} bucket-1 and {n2} bucket-2 tasks no family ran",
               scene=SCENE, tasks_file=str((out / "tasks.jsonl").relative_to(REPO)), tasks=tasks, new_skills=[],
               pools={"sealed": {"seed": pool_seed, "n": per_task * len(tasks)}})  # fmt: skip
    (HERE.parent / "families" / f"{name}.json").write_text(json.dumps(fam, indent=1) + "\n")
    print(f"[cal] {len(picked)} tasks ({sum(t['bucket'] == 1 for t in picked)} bucket 1) of candidates "
          f"{summary['candidates']}; measured goals left out {len(measured)}")  # fmt: skip
    return summary


def summarize(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["n"] >= MIN_N]
    good = sum(r["rate"] >= BUCKET1_RATE for r in scored)
    ok, n = sum(r["ok"] for r in scored), sum(r["n"] for r in scored)
    return dict(tasks=len(scored), at_bucket1_rate=good, share=round(good / len(scored), 3) if scored else None,
                share_ci=wilson(good, len(scored)), rounds=n, success=round(ok / n, 3) if n else None,
                success_ci=wilson(ok, n), unscored=[r["id"] for r in rows if r["n"] < MIN_N])  # fmt: skip


def report(name: str, rounds: Path) -> dict:
    out = OUT / name
    picked = {t["id"]: t for t in json.loads((out / "sample.json").read_text())["picked"]}
    per = collections.defaultdict(lambda: dict(ok=0, n=0, fell=0, dropped=0, why=collections.Counter()))
    for p in sorted(rounds.glob("*.json")):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text())
        x = per[r["task"]]
        x["ok"] += bool(r["ok"])
        x["n"] += 1
        x["fell"] += bool(r.get("fell"))
        x["dropped"] += bool(r.get("dropped"))
        if not r["ok"]:
            x["why"][(r.get("plan_why") or r.get("why") or "")[:80]] += 1
    rows = []
    for tid, t in picked.items():
        x = per.get(tid, dict(ok=0, n=0, fell=0, dropped=0, why=collections.Counter()))
        rows.append(dict(id=tid, bucket=t["bucket"], kind=t["kind"], source=t["source"], ok=x["ok"], n=x["n"],
                         rate=round(x["ok"] / x["n"], 3) if x["n"] else 0.0, fell=x["fell"], dropped=x["dropped"],
                         why=dict(x["why"].most_common(3))))  # fmt: skip
    rep = dict(name=name, rounds=str(rounds), bucket1_rate=BUCKET1_RATE, min_n=MIN_N, tasks=rows)
    for b in (1, 2):
        rb = [r for r in rows if r["bucket"] == b]
        rep[f"bucket{b}"] = summarize(rb)
        rep[f"bucket{b}_by_kind"] = {
            k: summarize([r for r in rb if r["kind"] == k]) for k in sorted({r["kind"] for r in rb})
        }
        rep[f"bucket{b}_by_source"] = {s: summarize([r for r in rb if r["source"] == s]) for s in sorted({r["source"] for r in rb})}  # fmt: skip
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    for b in (1, 2):
        print(f"[cal] bucket {b}: {rep[f'bucket{b}']}")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("name")
    s.add_argument("--runs", type=Path, nargs="+", required=True, help="scene_feasibility.py output dirs, by priority")
    s.add_argument("--n1", type=int, default=24)
    s.add_argument("--n2", type=int, default=12)
    s.add_argument("--per-task", type=int, default=8)
    s.add_argument("--seed", type=int, default=20261019)
    s.add_argument("--pool-seed", type=int, default=CAL_SEED, help="the pool's first start seed (one block per sample)")
    r = sub.add_parser("report")
    r.add_argument("name")
    r.add_argument("rounds", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "sample":
        sample(a.name, a.runs, a.n1, a.n2, a.per_task, a.seed, a.pool_seed)
    else:
        report(a.name, a.rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
