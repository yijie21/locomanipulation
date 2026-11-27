"""TODO 27.2.4 (plan docs/handoff/scene-rsi-plan-20261016.md D7, D8; the warehouse gate's rules, frozen): a candidate
against the incumbent on a family's acceptance pool, paired by start. Code decides; no model reads the per-round files.

A candidate is a directory holding `skills/` (a snapshot of the skills package) and `candidate.json` ({"set": {...}},
the parameter overrides on top of that code). The gate copies it into its own tree (`<gate>/<n>_<hash>/`), so later
edits cannot change what was judged; the hash covers the code and the overrides. The incumbent is the last accepted
candidate (the first one: `--seed-incumbent`, the repository's skills).

Steps, stopping at the first rejection:
1. static: the skills package loads and the overrides apply; the hash was never rejected; it is not the incumbent.
2. sentinel: SENTINELS incumbent rounds re-run must reproduce its stored outcomes (ok and end time) -- the motor
   layer and the scene are shared code, a change there must not move the incumbent. One start per task (`sentinel_labels`).
   A drift is a void call (user 2026-11-27, from main's TODO 54 A4: "void": true, like an incomplete pairing), not a
   rejection: the candidate's hash is not blacklisted and the loop counts no gate call. Until then the hash was
   "rejected before" for good and the call used up one of the loop's gate calls. The loop stops abnormally after
   DRIFT_STREAK drift voids in a row (loop.stop_reason): the simulation is deterministic (TODO 38.2), so a drift
   means the bed or its environment changed under the run.
3. stage 1: STAGE1 starts drawn from every task in proportion to its share of the pool (`stage1_labels`); reject
   when the candidate's net wins (b - c) <= 0. (Until 2026-10-18 the first STAGE1 labels by name: about half of a
   multi-task family's tasks, so a candidate that changed only a later task was rejected without one round of it.)
4. stage 2: the rest of the pool.
5. verdict (`verdict_fails`): accept iff b - c >= MIN_NET and c <= b / 3 (LOSS_SHARE), falls <= incumbent + 1,
   drops <= incumbent + 1 -- or, against an incumbent with no success on the pool, drops <= half the candidate's
   successes when that is more (`drop_limit`; user 2026-10-18: a family starting from zero has no drops to compare).
   The one-sided exact McNemar p is still computed and reported, no longer required (TODO 33.8, gate rule 2, user
   2026-10-22: early research; until then p <= 0.05 was required, which alone needs 5:0, so MIN_NET never bound and a
   change that fixes one object of a 4-start-per-object pool could never pass). Every verdict line records the gate
   rule it was judged under (`rule`, GATE_RULE); "rejected before" counts only rejections under the present rule, so a
   candidate rejected under the old one may be judged again.
Pairing must be complete (TODO 36, review 2026-10-22): when either arm lacks a round for a requested start (a worker
died, the incumbent's own round is missing) the call ends with a void verdict (accepted false, "void": true, reason
"pairing incomplete: ..."), never a verdict on fewer pairs. A crash still leaves no ledger line (the loop reads that from
the exit code and the hash).
A tally also carries `per_class` (TODO 33.15): wins and losses by the round records' failure class, reported, not judged.
On a pack family (gso_pack_all.md D15, user 2026-11-07) a win is a start the candidate reaches a higher progress stage
on (`stage` in the round record, 0-4: none, picked, over the opening, in the container, the goal) and a loss a lower
one, so a change that moves starts along the chain can be accepted before any start completes it; `candidate_ok`, the
stops' 80 % and the sealed pool still count stage 4 alone. Rounds without a stage are judged on ok / fail as before.
On acceptance the candidate's rounds become the incumbent's. The verdict JSON is the only output a loop role reads.

`--non-inferior` (rule `sync`; scripts/rsi/alternate.py, user 2026-11-18): another family's code merged into this
family's skills package. Not written for this family's tasks, so it need not win: no stage-1 cut, the whole pool at
once, accepted iff the candidate loses no more starts than it wins (c <= b; `non_inferior_fails`). The static checks,
the sentinel and complete pairing are as above. "Rejected before" counts only rejections under the same rule.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    PYTHONPATH=src $PY scripts/rsi/gate.py pick_single --gate <dir> --pool <accept pool> \\
        --candidate <dir> [--procs 12 --gpus 0,1] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
LOSS_SHARE = 3  # c <= b / LOSS_SHARE
GATE_RULE = 2  # 1: b - c >= 3 and p <= 0.05 (to 2026-10-22); 2: b - c >= 3 and c <= b / 3 (TODO 33.8)
SYNC_RULE = "sync"  # --non-inferior: c <= b (scripts/rsi/alternate.py)
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


def sentinel_labels(labels: list[str], n: int = SENTINELS) -> list[str]:
    """TODO 36 (review 2026-10-22): the incumbent re-run's starts, one per task (tasks by name, the first label of each),
    up to `n`; a family with fewer tasks than `n` gets a second start of each task in turn. (Until 2026-10-22 the first
    `n` labels by name: all of one task.) The sentinel only compares the incumbent with its own stored rounds, so which
    starts it takes changes what drift it can see, not any verdict's rule."""
    by: dict[str, list[str]] = {}
    for x in sorted(labels):
        by.setdefault(x.rsplit("_s", 1)[0], []).append(x)
    out, i = [], 0
    while len(out) < min(n, len(labels)):
        out += [v[i] for t, v in sorted(by.items()) if i < len(v)][: n - len(out)]
        i += 1
    return sorted(out)


def incomplete(labels, *arms: dict) -> list[str]:
    """The labels some arm has no round for."""
    return sorted(x for x in labels if any(x not in arm for arm in arms))


def drop_limit(t: dict) -> int:
    """The most drops a candidate may have on the pool (verdict step 5)."""
    limit = t["incumbent_dropped"] + 1
    return max(limit, t["candidate_ok"] // 2) if t["incumbent_ok"] == 0 else limit


def verdict_fails(t: dict) -> list[str]:
    """Why a full-pool tally `t` is rejected (verdict step 5); empty when it is accepted."""
    fails = []
    if t["b"] - t["c"] < MIN_NET:
        fails.append(f"net wins {t['b'] - t['c']} < {MIN_NET}")
    if t["c"] * LOSS_SHARE > t["b"]:
        fails.append(f"losses {t['c']} > wins {t['b']} / {LOSS_SHARE}")
    if t["candidate_fell"] > t["incumbent_fell"] + 1:
        fails.append(f"falls {t['candidate_fell']} > {t['incumbent_fell']} + 1")
    if t["candidate_dropped"] > drop_limit(t):
        fails.append(f"drops {t['candidate_dropped']} > {drop_limit(t)}")
    return fails


def non_inferior_fails(t: dict) -> list[str]:
    """Why a full-pool tally is rejected under --non-inferior (rule `sync`): it loses more starts than it wins."""
    return [f"losses {t['c']} > wins {t['b']}"] if t["c"] > t["b"] else []


def overrides(cand: Path) -> dict:
    return json.loads((cand / "candidate.json").read_text()).get("set") or {}


def stage_of(r: dict) -> int:
    """A round's progress for the paired comparison (gso_pack_all.md D15, user 2026-11-07): its `stage` (a pack
    round, 0-4: none, picked, over the opening, in the container, the goal), else 4 for a success and 0 for a failure,
    so a family without stages is judged on ok / fail as before."""
    st = r.get("stage")
    return int(st) if st is not None else (4 if r["ok"] else 0)


def stage_tally(cand: dict, inc: dict, keys: list[str]) -> dict | None:
    """The stages behind b and c when any round carries one: wins, losses, ties, each arm's mean stage and, per task,
    [incumbent mean, candidate mean]. Reported in the verdict; the rule judges b and c."""
    if not any(r.get("stage") is not None for k in keys for r in (cand[k], inc[k])):
        return None
    sa, si = {k: stage_of(cand[k]) for k in keys}, {k: stage_of(inc[k]) for k in keys}
    per: dict[str, list] = {}
    for k in keys:
        per.setdefault(cand[k].get("task") or k.rsplit("_s", 1)[0], []).append((si[k], sa[k]))
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
    return dict(wins=sum(sa[k] > si[k] for k in keys), losses=sum(sa[k] < si[k] for k in keys),
                ties=sum(sa[k] == si[k] for k in keys), candidate_mean=mean(list(sa.values())),
                incumbent_mean=mean(list(si.values())),
                per_task={t: [mean([x[0] for x in v]), mean([x[1] for x in v])] for t, v in sorted(per.items())})  # fmt: skip


def per_class(cand: dict, inc: dict, keys: list[str]) -> dict:
    """TODO 33.15 (plan section 3.5): the tally by failure class (`failure_class` in the round records; a record
    without one counts as "unknown"). A win (b) counts under the class the incumbent failed with -- what the candidate
    improved --, a loss (c) under the class the candidate fails with -- what it broke; `cand_fail` / `inc_fail` count
    each arm's failures by class. A win or loss is a higher or lower stage (stage_of; ok / fail without stages).
    Reported in the verdict, not judged."""
    out: dict[str, dict] = {}

    def row(k: str) -> dict:
        return out.setdefault(k, dict(b=0, c=0, cand_fail=0, inc_fail=0))

    cls = lambda r: r.get("failure_class") or "unknown"  # noqa: E731
    for k in keys:
        a, i = cand[k], inc[k]
        if not a["ok"]:
            row(cls(a))["cand_fail"] += 1
        if not i["ok"]:
            row(cls(i))["inc_fail"] += 1
        if stage_of(a) > stage_of(i):
            row(cls(i))["b"] += 1
        elif stage_of(i) > stage_of(a):
            row(cls(a))["c"] += 1
    return dict(sorted(out.items()))


def tally(cand: dict, inc: dict) -> dict:
    """The paired comparison over the shared starts: b = starts the candidate reaches a higher stage on, c = a lower
    one (stage_of: a pack round's 0-4 progress, else ok / fail), `candidate_ok` / `incumbent_ok` the successes (stage
    4), `stage` the stages behind b and c (stage_tally; None when no round carries one)."""
    keys = sorted(set(cand) & set(inc))
    b = sum(stage_of(cand[k]) > stage_of(inc[k]) for k in keys)
    c = sum(stage_of(cand[k]) < stage_of(inc[k]) for k in keys)
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
                candidate_why=dict(sorted(why.items(), key=lambda kv: -kv[1])),
                per_class=per_class(cand, inc, keys), stage=stage_tally(cand, inc, keys))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("--gate", type=Path, required=True, help="the gate's directory (hidden from the loop's roles)")
    ap.add_argument("--pool", type=Path, required=True, help="the acceptance pool")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("RSI_PROCS", "12")))
    ap.add_argument("--gpus", default=os.environ.get("RSI_GPUS", "0,1"))
    ap.add_argument("--dry-run", action="store_true", help="static checks only")
    ap.add_argument("--non-inferior", action="store_true", help="rule `sync`: accept iff c <= b on the whole pool")
    a = ap.parse_args(argv)
    rule = SYNC_RULE if a.non_inferior else GATE_RULE
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
    verdict = dict(candidate=str(a.candidate), hash=h, rule=rule,
                   started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))  # fmt: skip
    verdict["call"] = len([x for x in history if not x.get("dry_run")]) + 1

    def void(stage: str, miss: list[str], **extra) -> int:
        """TODO 36: a call whose pairing is incomplete judges nothing: a ledger line, rejected and void."""
        return finish(
            False, f"pairing incomplete: {stage} lacks {len(miss)} start(s): {', '.join(miss)}", void=True, **extra
        )

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
    if any(x["hash"] == h and not x["accepted"] and not x.get("dry_run") and not x.get("void")
           and x.get("rule", 1) == rule for x in history):  # fmt: skip
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
        B.Job(inc_dir / "skills", work / "sentinel", overrides(inc_dir), tuple(sentinel_labels(labels))),
        B.Job(work / "cand" / "skills", work / "candidate", overrides(work / "cand"),
              None if a.non_inferior else tuple(stage1_labels(labels))),  # --non-inferior: the whole pool at once
    ]  # fmt: skip
    B.run_jobs(jobs, a.family, a.pool, a.procs, log, a.gpus)
    inc_out = B.outcomes(inc_rounds)
    again = B.outcomes(work / "sentinel")
    miss = incomplete(sentinel_labels(labels), inc_out, again)
    if miss:
        return void("sentinel or incumbent", miss)
    drift = [k for k in again if (again[k]["ok"], again[k].get("t_end")) != (inc_out[k]["ok"], inc_out[k].get("t_end"))]
    if drift:
        why = f"sentinel: incumbent rounds no longer reproduce ({drift}); the pairing is void"
        return finish(False, why, void=True)
    if a.non_inferior:
        miss = incomplete(labels, inc_out, B.outcomes(work / "candidate"))
        if miss:
            return void("final", miss)
        t = tally(B.outcomes(work / "candidate"), inc_out)
        fails = non_inferior_fails(t)
        if fails:
            return finish(False, "; ".join(fails), result=t)
        inc_state.write_text(
            json.dumps(dict(dir=str(work / "cand"), rounds=str(work / "candidate"), since=verdict["call"]))
        )
        return finish(True, "accepted (non-inferior)", result=t)
    s1 = stage1_labels(labels)
    miss = incomplete(s1, inc_out, B.outcomes(work / "candidate"))
    if miss:
        return void("stage 1", miss)
    t1 = tally(B.outcomes(work / "candidate"), inc_out)
    if t1["b"] - t1["c"] <= 0:
        return finish(False, f"stage 1: net wins {t1['b'] - t1['c']} <= 0 on {t1['n']} pairs", stage1=t1)
    B.run_jobs([B.Job(work / "cand" / "skills", work / "candidate", overrides(work / "cand"))], a.family, a.pool,
               a.procs, log, a.gpus)  # fmt: skip
    miss = incomplete(labels, inc_out, B.outcomes(work / "candidate"))
    if miss:
        return void("final", miss, stage1=t1)
    t = tally(B.outcomes(work / "candidate"), inc_out)
    fails = verdict_fails(t)
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
