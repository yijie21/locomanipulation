"""TODO 27.2.4 (plan docs/handoff/scene-rsi-plan-20261016.md D7, D8; the warehouse gate's rules, frozen): a candidate
against the incumbent on a family's acceptance pool, paired by start. Code decides; no model reads the per-round files.

A candidate is a directory holding `skills/` (a snapshot of the skills package) and `candidate.json` ({"set": {...}},
the parameter overrides on top of that code). The gate copies it into its own tree (`<gate>/<n>_<hash>/`), so later
edits cannot change what was judged; the hash covers the code and the overrides. The incumbent is the last accepted
candidate (the first one: `--seed-incumbent`, the repository's skills).

Steps, stopping at the first rejection:
1. static: the skills package loads and the overrides apply; the hash was never rejected; it is not the incumbent.
2. sentinel: SENTINELS incumbent rounds re-run must reproduce its stored outcomes (ok and end time) -- the motor
   layer and the scene are shared code, a change there must not move the incumbent.
3. stage 1: STAGE1 starts drawn from every task in proportion to its share of the pool (`stage1_labels`); reject
   when the candidate's net wins (b - c) <= 0. (Until 2026-10-18 the first STAGE1 labels by name: about half of a
   multi-task family's tasks, so a candidate that changed only a later task was rejected without one round of it.)
4. stage 2: the rest of the pool.
5. verdict: accept iff b - c >= MIN_NET, one-sided exact McNemar p <= ALPHA, falls <= incumbent + 1,
   drops <= incumbent + 1 -- or, against an incumbent with no success on the pool, drops <= half the candidate's
   successes when that is more (`drop_limit`; user 2026-10-18: a family starting from zero has no drops to compare).
On acceptance the candidate's rounds become the incumbent's. The verdict JSON is the only output a loop role reads.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/gate.py pick_single --gate <dir> --pool <accept pool> \\
        --candidate <dir> [--procs 12 --gpus 0,1] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from math import comb, sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import batch as B  # noqa: E402
from scene_round import load_skills, skills_hash  # noqa: E402

STAGE1 = 30
SENTINELS = 5
MIN_NET = 3
ALPHA = 0.05
SEED_SKILLS = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"


def mcnemar_one_sided(b: int, c: int) -> float:
    n = b + c
    return 1.0 if n == 0 else sum(comb(n, i) for i in range(b, n + 1)) / 2**n


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """The 95% Wilson score interval of k successes in n: every success rate a report gives carries one (plan
    §12, TODO 27.10.5; 8 sealed starts per task leave 4/8 anywhere in 0.22-0.78)."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def candidate_hash(cand: Path) -> str:
    spec = json.loads((cand / "candidate.json").read_text()) if (cand / "candidate.json").exists() else {}
    h = hashlib.sha256((skills_hash(cand / "skills") + json.dumps(spec.get("set") or {}, sort_keys=True)).encode())
    return h.hexdigest()[:12]


def snapshot(src: Path, dst: Path) -> None:
    """A candidate directory copied (skills without caches)."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src / "skills", dst / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    spec = (src / "candidate.json").read_text() if (src / "candidate.json").exists() else '{"set": {}}'
    (dst / "candidate.json").write_text(spec)


def stage1_labels(labels: list[str], n: int = STAGE1) -> list[str]:
    """`n` of the pool's labels, every task (`<task>_s<seed>`) by its share of the pool: largest remainder, ties to
    the task first by name; within a task the first by name."""
    by: dict[str, list[str]] = {}
    for x in sorted(labels):
        by.setdefault(x.rsplit("_s", 1)[0], []).append(x)
    quota = {t: n * len(v) / len(labels) for t, v in by.items()}
    take = {t: int(q) for t, q in quota.items()}
    for t in sorted(by, key=lambda t: (-(quota[t] - take[t]), t))[: n - sum(take.values())]:
        take[t] += 1
    return sorted(x for t, v in by.items() for x in v[: take[t]])


def drop_limit(t: dict) -> int:
    """The most drops a candidate may have on the pool (verdict step 5)."""
    limit = t["incumbent_dropped"] + 1
    return max(limit, t["candidate_ok"] // 2) if t["incumbent_ok"] == 0 else limit


def overrides(cand: Path) -> dict:
    return json.loads((cand / "candidate.json").read_text()).get("set") or {}


def tally(cand: dict, inc: dict) -> dict:
    keys = sorted(set(cand) & set(inc))
    b = sum(cand[k]["ok"] and not inc[k]["ok"] for k in keys)
    c = sum(inc[k]["ok"] and not cand[k]["ok"] for k in keys)
    why = {}
    for k in keys:
        if not cand[k]["ok"]:
            w = (cand[k].get("plan_why") or cand[k]["why"] or "?")[:50]
            why[w] = why.get(w, 0) + 1
    return dict(n=len(keys), candidate_ok=sum(cand[k]["ok"] for k in keys), incumbent_ok=sum(inc[k]["ok"] for k in keys),
                b=b, c=c, p_one_sided=round(mcnemar_one_sided(b, c), 5),
                candidate_fell=sum(cand[k]["fell"] for k in keys), incumbent_fell=sum(inc[k]["fell"] for k in keys),
                candidate_dropped=sum(cand[k]["dropped"] for k in keys),
                incumbent_dropped=sum(inc[k]["dropped"] for k in keys),
                candidate_why=dict(sorted(why.items(), key=lambda kv: -kv[1])))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("--gate", type=Path, required=True, help="the gate's directory (hidden from the loop's roles)")
    ap.add_argument("--pool", type=Path, required=True, help="the acceptance pool")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--dry-run", action="store_true", help="static checks only")
    a = ap.parse_args(argv)
    gate = a.gate
    gate.mkdir(parents=True, exist_ok=True)
    ledger = gate / "ledger.jsonl"
    history = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    inc_state = gate / "incumbent.json"
    if not inc_state.exists():  # the seed: the repository's skills, no overrides
        seed = gate / "incumbent_seed"
        snapshot(_seed_dir(gate), seed)
        inc_state.write_text(json.dumps(dict(dir=str(seed), rounds=str(gate / "incumbent_seed_rounds"), since=0)))
    inc = json.loads(inc_state.read_text())
    labels = B.pool_labels(a.pool)
    h = candidate_hash(a.candidate)
    verdict = dict(candidate=str(a.candidate), hash=h, started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    verdict["call"] = len([x for x in history if not x.get("dry_run")]) + 1

    def finish(accepted: bool, reason: str, **extra) -> int:
        verdict.update(accepted=accepted, reason=reason, dry_run=a.dry_run, **extra)
        verdict["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with ledger.open("a") as fh:
            fh.write(json.dumps(verdict) + "\n")
        print(json.dumps(verdict, indent=1))
        return 0 if accepted else 1

    # 1. static
    try:
        pkg = load_skills(a.candidate / "skills", "gate_static_skills")
        pkg.apply_set(pkg, overrides(a.candidate))
    except Exception as e:
        return finish(False, f"the candidate does not load: {type(e).__name__}: {e}")
    if any(x["hash"] == h and not x["accepted"] and not x.get("dry_run") for x in history):
        return finish(False, "this exact candidate was rejected before")
    if h == candidate_hash(Path(inc["dir"])):
        return finish(False, "the candidate is the incumbent")
    if len(labels) < STAGE1 + 1:
        return finish(False, f"acceptance pool has {len(labels)} starts")
    if a.dry_run:
        return finish(False, "dry run: static checks passed")
    work = gate / f"{verdict['call']:02d}_{h}"
    snapshot(a.candidate, work / "cand")
    inc_dir, inc_rounds = Path(inc["dir"]), Path(inc["rounds"])
    log = work / "run.log"
    jobs = [
        B.Job(inc_dir / "skills", inc_rounds, overrides(inc_dir)),  # the incumbent's own rounds, once per incumbent
        B.Job(inc_dir / "skills", work / "sentinel", overrides(inc_dir), tuple(labels[:SENTINELS])),
        B.Job(work / "cand" / "skills", work / "candidate", overrides(work / "cand"), tuple(stage1_labels(labels))),
    ]
    B.run_jobs(jobs, a.family, a.pool, a.procs, log, a.gpus)
    inc_out = B.outcomes(inc_rounds)
    again = B.outcomes(work / "sentinel")
    drift = [k for k in again if (again[k]["ok"], again[k].get("t_end")) != (inc_out[k]["ok"], inc_out[k].get("t_end"))]
    if drift:
        return finish(False, f"sentinel: incumbent rounds no longer reproduce ({drift}); the pairing is void")
    t1 = tally(B.outcomes(work / "candidate"), inc_out)
    if t1["b"] - t1["c"] <= 0:
        return finish(False, f"stage 1: net wins {t1['b'] - t1['c']} <= 0 on {t1['n']} pairs", stage1=t1)
    B.run_jobs([B.Job(work / "cand" / "skills", work / "candidate", overrides(work / "cand"))], a.family, a.pool,
               a.procs, log, a.gpus)  # fmt: skip
    t = tally(B.outcomes(work / "candidate"), inc_out)
    fails = []
    if t["b"] - t["c"] < MIN_NET:
        fails.append(f"net wins {t['b'] - t['c']} < {MIN_NET}")
    if t["p_one_sided"] > ALPHA:
        fails.append(f"one-sided McNemar p {t['p_one_sided']} > {ALPHA}")
    if t["candidate_fell"] > t["incumbent_fell"] + 1:
        fails.append(f"falls {t['candidate_fell']} > {t['incumbent_fell']} + 1")
    if t["candidate_dropped"] > drop_limit(t):
        fails.append(f"drops {t['candidate_dropped']} > {drop_limit(t)}")
    if fails:
        return finish(False, "; ".join(fails), stage1=t1, result=t)
    inc_state.write_text(
        json.dumps(dict(dir=str(work / "cand"), rounds=str(work / "candidate"), since=verdict["call"]))
    )
    return finish(True, "accepted", stage1=t1, result=t)


def _seed_dir(gate: Path) -> Path:
    """The repository's skills as a candidate directory (no overrides)."""
    d = gate / "_seed_src"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(SEED_SKILLS, d / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (d / "candidate.json").write_text('{"set": {}}')
    return d


if __name__ == "__main__":
    raise SystemExit(main())
