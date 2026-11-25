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
3. stage 1: STAGE1 starts drawn from every task in proportion to its share of the pool (`stage1_labels`); reject
   when the candidate's net wins (b - c) < 0 (TODO 54: it was <= 0). (Until 2026-10-18 the first STAGE1 labels by name: about half of a
   multi-task family's tasks, so a candidate that changed only a later task was rejected without one round of it.)
4. stage 2: the rest of the pool.
5. verdict (verdict.decide, TODO 54): with caps["p_max"] accept iff the one-sided sign test p <= p_max (net >= MIN_NET
   is not read there: with p_max it can never bind), else (scene) b - c >= MIN_NET and c <= b / 3 (LOSS_SHARE); and falls <= incumbent + 1,
   drops <= incumbent + 1 -- or, against an incumbent with no success on the pool, drops <= half the candidate's
   successes when that is more (`drop_limit`; user 2026-10-18: a family starting from zero has no drops to compare).
   With caps["alpha_task"] (the benchmark families; 0.05 here, 0.10 at the check half): a task whose losses are
   significant -- one-sided exact sign test on that task's discordant pairs, Holm-corrected across the tasks -- rejects,
   so a gain on one task cannot pay for a real loss on another while a small, insignificant loss no longer rejects (it
   was a fixed "net <= -2", too loose for a task that flips half its starts and never binding for one that flips none).
   `per_task` in the tally gives each task's wins and losses, `task_tests` its p values. A scene family has no such cap.
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
On acceptance the candidate's rounds become the incumbent's. The verdict JSON is the only output a loop role reads.

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
from math import sqrt
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(REPO / "src")]

import backend as BK  # noqa: E402
import batch as B  # noqa: E402
import verdict as V  # noqa: E402
from scene_round import load_skills, skills_hash  # noqa: E402

STAGE1 = 30
SENTINELS = 5
MIN_NET = V.MIN_NET
LOSS_SHARE = V.LOSS_SHARE  # c <= b / LOSS_SHARE (scene families)
GATE_RULE = 3  # 3: TODO 54, the per-task sign test and stage 1 on net < 0; 1: b - c >= 3 and p <= 0.05 (to 2026-10-22); 2: b - c >= 3 and c <= b / 3 (TODO 33.8)
SEED_SKILLS = REPO / "src" / "locomani" / "tasks" / "scene" / "skills"


mcnemar_one_sided = V.sign_p  # TODO 54: the rule lives in verdict.py


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


drop_limit = V.drop_limit


def verdict_fails(
    t: dict, min_net: int = MIN_NET, p_max: float | None = None, alpha_task: float | None = None,
    loss_share: int = LOSS_SHARE,
) -> list[str]:  # fmt: skip
    """Why a full-pool tally `t` is rejected (verdict step 5); empty when it is accepted. The rule is verdict.decide's
    (TODO 54): with `p_max` the sign test's one-sided p must not exceed it and `min_net` is not read (it can never bind
    there); a scene family (no p_max) needs net >= `min_net` and losses of at most a `loss_share`th of the wins.
    `alpha_task`: a task whose losses are significant (one-sided exact sign test on its discordant pairs, Holm across
    the tasks) rejects the candidate whatever the other tasks gained."""
    return V.decide(t, "gate", dict(min_net=min_net, p_max=p_max, alpha_task=alpha_task, loss_share=loss_share))[
        "reasons"
    ]


def task_of(label: str, rnd: dict | None = None) -> str:
    """A round's task: the record's `task`, else the label's part before `_s<n>`."""
    return (rnd or {}).get("task") or label.rsplit("_s", 1)[0]


def gate_params(caps: dict) -> dict:
    """The family's caps the verdict reads (verdict.py): p_max, min_net (scene only), alpha_task."""
    return dict(p_max=caps.get("p_max"), min_net=caps.get("min_net", MIN_NET), alpha_task=caps.get("alpha_task"),
                loss_share=caps.get("loss_share", LOSS_SHARE))  # fmt: skip


def per_task(cand: dict, inc: dict, keys: list[str]) -> dict:
    """Wins (b) and losses (c) of the candidate against the incumbent, by task (the per-task test's input)."""
    return V.counts(cand, inc, keys)["per_task"]


def sentinel_drift(again: dict, inc_out: dict, exact: bool = True) -> list[str]:
    """The sentinel labels whose re-run no longer reproduces the stored round: `ok` and `t_end` for a scene family;
    `ok` alone when the backend's caps say sentinel_exact is False (official reruns do not always repeat their end
    time)."""
    if exact:
        return [
            k for k in again if (again[k]["ok"], again[k].get("t_end")) != (inc_out[k]["ok"], inc_out[k].get("t_end"))
        ]
    return [k for k in again if again[k]["ok"] != inc_out[k]["ok"]]


def overrides(cand: Path) -> dict:
    return json.loads((cand / "candidate.json").read_text()).get("set") or {}


def per_class(cand: dict, inc: dict, keys: list[str]) -> dict:
    """TODO 33.15 (plan section 3.5): the tally by failure class (`failure_class` in the round records; a record
    without one counts as "unknown"). A win (b) counts under the class the incumbent failed with -- what the candidate
    fixed --, a loss (c) under the class the candidate fails with -- what it broke; `cand_fail` / `inc_fail` count each
    arm's failures by class. Reported in the verdict, not judged."""
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
        if a["ok"] and not i["ok"]:
            row(cls(i))["b"] += 1
        elif i["ok"] and not a["ok"]:
            row(cls(a))["c"] += 1
    return dict(sorted(out.items()))


def tally(cand: dict, inc: dict) -> dict:
    keys = sorted(set(cand) & set(inc))
    why = {}
    for k in keys:
        if not cand[k]["ok"]:
            w = (cand[k].get("plan_why") or cand[k]["why"] or "?")[:50]
            why[w] = why.get(w, 0) + 1
    return dict(V.counts(cand, inc, keys), candidate_why=dict(sorted(why.items(), key=lambda kv: -kv[1])),
                per_class=per_class(cand, inc, keys))  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("family")
    ap.add_argument("--gate", type=Path, required=True, help="the gate's directory (hidden from the loop's roles)")
    ap.add_argument("--pool", type=Path, required=True, help="the acceptance pool")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("RSI_PROCS", "12")))
    ap.add_argument("--gpus", default=os.environ.get("RSI_GPUS", "0,1"))
    ap.add_argument("--dry-run", action="store_true", help="static checks only")
    a = ap.parse_args(argv)
    gate = a.gate
    gate.mkdir(parents=True, exist_ok=True)
    ledger = gate / "ledger.jsonl"
    history = [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    caps = BK.caps(a.family)  # a scene family: today's values (backend.SCENE_CAPS)
    inc_state = gate / "incumbent.json"
    if not inc_state.exists():  # the seed: the repository's skills, no overrides
        seed = gate / "incumbent_seed"
        snapshot(_seed_dir(gate, BK.layout(a.family)["skills"]), seed)
        inc_state.write_text(json.dumps(dict(dir=str(seed), rounds=str(gate / "incumbent_seed_rounds"), since=0)))
    inc = json.loads(inc_state.read_text())
    labels = B.pool_labels(a.pool)
    h = candidate_hash(a.candidate)
    verdict = dict(candidate=str(a.candidate), hash=h, rule=GATE_RULE, alpha_task=caps.get("alpha_task"),
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
        be = BK.of(a.family)  # a second backend loads its own package (no import of the bed tree here)
        pkg = (be.load_skills(a.candidate / "skills", "gate_static_skills", a.family) if be
               else load_skills(a.candidate / "skills", "gate_static_skills"))  # fmt: skip
        pkg.apply_set(pkg, overrides(a.candidate))
    except Exception as e:
        return finish(False, f"the candidate does not load: {type(e).__name__}: {e}")
    if any(x["hash"] == h and not x["accepted"] and not x.get("dry_run") and not x.get("void")
           and x.get("rule", 1) == GATE_RULE for x in history):  # fmt: skip
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
        B.Job(work / "cand" / "skills", work / "candidate", overrides(work / "cand"), tuple(stage1_labels(labels))),
    ]
    # caps["repeat"] k > 1 (a backend whose official reruns differ): the sentinel and each candidate round would run k
    # times here and `ok` would be their majority; not implemented, k is 1 for every family so far.
    B.run_jobs(jobs, a.family, a.pool, a.procs, log, a.gpus, where="gate")
    inc_out = B.outcomes(inc_rounds)
    again = B.outcomes(work / "sentinel")
    miss = incomplete(sentinel_labels(labels), inc_out, again)
    if miss:
        return void("sentinel or incumbent", miss)
    drift = sentinel_drift(again, inc_out, caps["sentinel_exact"])
    if len(drift) > caps.get("sentinel_flips", 0):  # a noisy bed's caps tolerate that many flips of the sentinel
        # TODO 54 (review verify.md 8): void=True, as `void()` sets it. Without it the hash was blacklisted for good and
        # the call counted against MAX_GATE_CALLS: an environment fault judged a candidate nobody had measured.
        return finish(
            False, f"sentinel: incumbent rounds no longer reproduce ({drift}); the pairing is void", void=True
        )
    s1 = stage1_labels(labels)
    miss = incomplete(s1, inc_out, B.outcomes(work / "candidate"))
    if miss:
        return void("stage 1", miss)
    t1 = tally(B.outcomes(work / "candidate"), inc_out)
    s1v = V.decide(
        t1, "stage1"
    )  # TODO 54: only a net loss rejects (it was net <= 0: 3 starts a task cannot tell 0 from -1)
    if not s1v["decision"]:
        return finish(False, f"stage 1: {'; '.join(s1v['reasons'])} on {t1['n']} pairs", stage1=t1)
    # loop.gate_candidate (caps stage1_first, 2026-11-23) starts the candidate's diagnostic rounds and the early review
    # only once this file exists: a call stage 1 rejects costs only stage 1
    (work / "stage1_passed.json").write_text(json.dumps(t1))
    B.run_jobs([B.Job(work / "cand" / "skills", work / "candidate", overrides(work / "cand"))], a.family, a.pool,
               a.procs, log, a.gpus, where="gate")  # fmt: skip
    miss = incomplete(labels, inc_out, B.outcomes(work / "candidate"))
    if miss:
        return void("final", miss, stage1=t1)
    t = tally(B.outcomes(work / "candidate"), inc_out)
    v = V.decide(t, "gate", gate_params(caps))
    if not v["decision"]:
        return finish(False, "; ".join(v["reasons"]), stage1=t1, result=dict(t, task_tests=v["per_task"]))
    inc_state.write_text(
        json.dumps(dict(dir=str(work / "cand"), rounds=str(work / "candidate"), since=verdict["call"]))
    )
    return finish(True, "accepted", stage1=t1, result=dict(t, task_tests=v["per_task"]))


def _seed_dir(gate: Path, skills: Path = SEED_SKILLS) -> Path:
    """The repository's skills (a second backend's package: `skills`) as a candidate directory (no overrides)."""
    d = gate / "_seed_src"
    if d.exists():
        shutil.rmtree(d)
    shutil.copytree(skills, d / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (d / "candidate.json").write_text('{"set": {}}')
    return d


if __name__ == "__main__":
    raise SystemExit(main())
