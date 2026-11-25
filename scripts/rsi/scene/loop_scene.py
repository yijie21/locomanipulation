"""The scene line's part of the inner loop (the plug-in behind `S.H`): the Manager flow, infeasibility claims and parks, reopens, design probes, diagnostic redraw, attribution, replays."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent), str(_HERE.parents[2] / "src")]
import audit as AU  # noqa: E402
import backend as BK  # noqa: E402
import batch as B  # noqa: E402
import cascade as CS  # noqa: E402
import design as DS  # noqa: E402
import diag_pool as DP  # noqa: E402
import drafts as DR  # noqa: E402
import family as F  # noqa: E402
import ledger as LG  # noqa: E402
import records as RC  # noqa: E402
import roles as RL  # noqa: E402
import segments as SG  # noqa: E402
import state as S  # noqa: E402
from gate import candidate_hash, overrides  # noqa: E402

# TODO 33.17 (plan section 3.4, from DexAgent's diagnose_stage3_failure): what a failure class's next step usually is,
# shown to the Manager after a rejection (repair_notes); the class comes from scripts/rsi/failure_class.py (33.15)
CLASS_SUGGEST = {
    "too_wide": "the object does not fit the open jaw on the side presented: scripts/rsi/graspability.py (rotate the "
    "approach to a side that fits, a rim pinch, a top pinch)",
    "side_reach": "the hand did not get to the grasp: a nearer stance, or re-approach through stage() ([scene-stage])",
    "pinch_reach": "the top pinch's wrist attitude was not delivered: a twist the arm delivers, a lower pre-grasp",
    "walk_off": "the walk stopped off the stance: re-walk through stage() with stance_check, or a stance with more "
    "clearance",
    "knocked": "the object moved before the grip: a raised-hand approach, a path clear of it",
    "slip": "gripped, then lost: grip_check / lift_check and a re-grip through stage()",
    "fell": "the robot fell: a controller or motor-layer limit, an escalation ([escalate]), not a skill fix",
    # the two-hand classes (TODO 38.11, [scene-probe]): side_reach, knocked, slip and fell read as above
    "one_hand_short": "one hand did not get to the box: the stance is off-centre or too far for that arm, or the two "
    "targets are not symmetric about the body; a re-approach of that hand through stage() (bimanual_grip_check)",
    "no_lift": "both hands on the box, it never rose: not enough squeeze (motor.squeeze inward_m), the lift too small "
    "or too fast, or the hands too low on the walls; bimanual_lift_check and a re-squeeze through stage()",
    # the strict box goal's classes (TODO 38.19): lifted(X, clearance, tilt) judged over the round's last window
    "low_lift": "the box rose but its lowest corner never cleared the platform by the goal's clearance: a bigger lift "
    "(stand up fully, raise the hands more), checked by the measured lowest corner before the window starts",
    "tilted": "the box's bottom tipped past the goal's limit: the two hands at unequal heights, or the grasp off the "
    "box's centre of mass along the walls; level the hands while lifting and check the tilt",
    "dropped_hold": "the whole goal held at some moment, then failed inside the judging window: the box slid in the "
    "hands or the body sagged; keep pressing (squeeze), stop moving before the window, or a re-grip",
}
MAX_REOPEN_RUN = 2  # TODO 33.14 (D11): Manager `reopen` decisions in a row at most


MAX_ATTEMPTS = 3  # closed-loop attempts per infeasibility claim
SKIP_CHECKS = ("interface_kept", "safety_checks_kept", "changed_code_reached", "evidence_not_refuting")  # TODO 33.17


ATTRIBUTE_CLASSES = ("fell_walk",)  # TODO 33.20: failure classes the driver attributes to a layer (isolate.CLASSES)
ATTRIBUTE_MIN = 2  # ... once the incumbent fails at least this many diagnostic rounds of the class


UNPARKED_MIN = 0.5  # D13.4's line counts only when the starts outside the parked tasks are this share of the pool
REDRAW_SEED = (
    5000  # a redrawn acceptance pool ([scene-redraw], redraw_accept) takes the family's acceptance seed + this
)
CLAIM_SCHEMA = {  # D13.2: an infeasibility claim
    "type": "object",
    "required": ["id", "task", "kind", "starts", "actions_assumed", "bound", "exits", "refuted_by", "attempts"],
    "properties": {
        "id": {"type": "string"},
        "task": {"type": "string"},
        "kind": {"enum": ["current_skills", "physical"]},
        "starts": {"type": "array", "items": {"type": "string"}},
        "actions_assumed": {"type": "string"},
        "bound": {
            "type": "object",
            "required": ["statement", "check"],
            "properties": {"statement": {"type": "string"}, "check": {"type": "string"}},
        },
        "exits": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["capability", "how"],
                "properties": {"capability": {"type": "string"}, "how": {"type": "string"}},
            },
        },
        "no_exit_reason": {"type": "string"},
        "refuted_by": {"type": "string"},
        "attempts": {
            "type": "array",
            "maxItems": MAX_ATTEMPTS,
            "items": {
                "type": "object",
                "required": ["name", "set"],
                "properties": {"name": {"type": "string"}, "set": {"type": "object"}},
            },
        },
    },
}
MANAGER_SCHEMA = {
    "type": "object",
    "required": ["decision", "reason", "summary_zh"],
    "properties": {
        "decision": {"enum": ["engineer", "reopen", "review", "stop"]},
        "hypothesis": {"type": "string"},
        "hypotheses": {"type": "array", "items": {"type": "string"}},  # TODO 32.9: one per Engineer, best first
        "reopen_claim": {"type": "string"},  # 27.8.7: the park whose exit the Engineer builds
        "base": {"type": "string"},  # TODO 33.8: a candidate_pool hash the Engineers start from
        # change 1: hypotheses whose best probed set the driver builds into a candidate itself (kind "tuned")
        "tune": {"type": "array", "items": {"type": "string"}},
        "skip": {  # TODO 33.17: a forced hypothesis not built, with the check that fails ({"H2": "interface_kept"})
            "type": "object",
            "additionalProperties": {"enum": list(SKIP_CHECKS)},
        },
        "repeats": {  # TODO 33.14: this iteration's hypotheses that are earlier iterations' mechanisms ("NN:id")
            "type": "array",
            "items": {
                "type": "object",
                "required": ["now", "same_as"],
                "properties": {"now": {"type": "string"}, "same_as": {"type": "array", "items": {"type": "string"}}},
            },
        },
        "focus": {"type": "string"},
        "reason": {"type": "string"},
        "escalation": {"type": "string"},
        "infeasible": {"type": "string"},
        "gap_proposals": {  # D13.6: skills too big for one Engineer call, for the outer loop
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "capability", "why", "tasks"],
                "properties": {
                    "name": {"type": "string"},
                    "capability": {"type": "string"},
                    "why": {"type": "string"},
                    "tasks": {"type": "array", "items": {"type": "string"}},
                    "est_cost": {"type": "string"},
                },
            },
        },
        "summary_zh": {"type": "string"},
    },
}


def _live_pool() -> tuple[Path, list[str], list[str]]:
    """(the redrawn acceptance pool's path, the tasks it holds, the task-wide parked tasks) -- the path is the
    family's own pool when the parks leave at least UNPARKED_MIN of it (or nothing)."""
    base = S.HIDDEN / "pool_accept"
    labels = B.pool_labels(base) if base.exists() else []
    parked = sorted({p["task"] for p in LG.jload(S.STATE / "parked.json", {}).values() if p["scope"] == "task"})
    live = sorted({LG.task_of(x) for x in labels} - set(parked))
    keep = [x for x in labels if LG.task_of(x) in live]
    if not parked or not keep or len(keep) >= UNPARKED_MIN * len(labels):
        return base, live, parked
    return S.HIDDEN / f"pool_accept_{hashlib.sha256(json.dumps(live).encode()).hexdigest()[:8]}", live, parked


def redraw_accept() -> None:
    """Rule 26 (user 2026-10-19, plan §12): once task-wide parks cover more than 1 - UNPARKED_MIN of the acceptance
    pool, a pool of the same size drawn afresh (seeds + REDRAW_SEED) from the tasks they leave -- r2_in_single had 36
    of its 60 starts on parked tasks, and its candidates were 3-5 net wins better four times, never significant. A
    new set of parked tasks draws a new pool; the gate runs the incumbent on starts it has not seen. Parked starts
    still count in the sealed and diagnostic pools."""
    pool, live, parked = _live_pool()
    if pool.exists() or BK.of(S.FAMILY):  # a backend's pools are named by ids, not by a seed a redraw could offset
        return
    fam = F.load(S.FAMILY)
    fam["tasks"], fam["rows"] = live, {t: fam["rows"][t] for t in live}
    spec = fam["pools"]["accept"]
    fam["pools"]["accept"] = dict(spec, seed=int(spec["seed"]) + REDRAW_SEED)
    res = F.draw(fam, "accept", pool)
    LG.ledger(
        dict(step="redraw_accept", tasks=live, parked_tasks=parked, n=res["written"], unavailable=res["unavailable"])
    )


def claim_problems(c: dict, diag: list[str]) -> list[str]:
    """What a claim lacks (D13.2): the actions it assumed, a hard constraint code can recompute (a check file), exits
    or -- for a physical claim -- why there are none, what would refute it, 1..MAX_ATTEMPTS attempts. Any problem
    and the claim is refused."""
    out = []
    if not c.get("starts"):
        out.append("no starts")
    bad = [x for x in c.get("starts") or [] if x not in diag or LG.task_of(x) != c.get("task")]
    if bad:
        out.append(f"not diagnostic starts of {c.get('task')}: {bad}")
    if len((c.get("actions_assumed") or "").strip()) < 10:
        out.append("actions_assumed is empty")
    b = c.get("bound") or {}
    if not (b.get("statement") or "").strip():
        out.append("bound.statement is empty")
    chk = Path(b.get("check") or "")
    if chk.suffix != ".py" or not chk.is_file():
        out.append(f"bound.check {b.get('check')!r} is not a Python file")
    exits = [e for e in c.get("exits") or [] if (e.get("capability") or "").strip() and (e.get("how") or "").strip()]
    if not exits and not (c.get("kind") == "physical" and (c.get("no_exit_reason") or "").strip()):
        out.append("no exits (only a physical claim may give no_exit_reason instead)")
    if not (c.get("refuted_by") or "").strip():
        out.append("refuted_by is empty")
    n = len([a for a in c.get("attempts") or [] if a.get("set")])
    if not 1 <= n <= MAX_ATTEMPTS:
        out.append(f"{n} attempts with a non-empty set (need 1-{MAX_ATTEMPTS})")
    return out


def claim_id(c: dict) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(c.get("id") or "claim"))[:60]


def verify_claims(claims: list[dict], inc: dict, it_dir: Path) -> list[dict]:
    """D13.3: a claim parks its starts only if its hard constraint holds on every one of them (claim_check.py, run by
    the driver on the family's diagnostic starts of that task) and none of its attempts -- the incumbent plus the
    attempt's overrides, closed loop on the claimed starts -- succeeds."""
    diag = B.pool_labels(S.LOOP / "pool_diag")
    meta = S.REPO / F.load(S.FAMILY)["scene"] / "scene_meta.json"
    res, jobs, owner = [], [], []
    for c in claims:
        r = dict(id=claim_id(c), task=c.get("task"), kind=c.get("kind"), problems=claim_problems(c, diag))
        res.append(r)
        if r["problems"]:
            continue
        d = it_dir / "claims" / r["id"]
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy(c["bound"]["check"], d / "check.py")  # the version that was judged
        task_starts = [x for x in diag if LG.task_of(x) == c["task"]]
        cmd = [str(S.SIM_PY), str(S.HERE / "scene" / "claim_check.py"), str(d / "check.py"), str(meta),
               *[str(S.LOOP / "pool_diag" / f"{x}.json") for x in task_starts]]  # fmt: skip
        try:
            out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=300,
                                 env=dict(os.environ, PYTHONPATH=str(S.REPO / "src")))  # fmt: skip
            r["bound"] = json.loads(out.stdout.strip().splitlines()[-1])
        except (subprocess.TimeoutExpired, ValueError, IndexError) as e:
            r["bound"] = {}
            r["problems"].append(f"bound.check did not run: {type(e).__name__}")
            continue
        for k, a in enumerate([a for a in c["attempts"] if a.get("set")][:MAX_ATTEMPTS]):
            jobs.append(B.Job(Path(inc["dir"]) / "skills", d / f"attempt_{k}", {**inc["set"], **a["set"]},
                              tuple(c["starts"])))  # fmt: skip
            owner.append((r, a.get("name") or f"attempt {k}"))
    if jobs:
        B.run_jobs(jobs, S.FAMILY, S.LOOP / "pool_diag", S.PROCS, it_dir / "prober.log", S.GPUS)
    for (r, name), job in zip(owner, jobs):
        out = B.outcomes(job.out)
        whys = collections.Counter((x.get("plan_why") or x["why"] or "?")[:60] for x in out.values() if not x["ok"])
        r.setdefault("attempts", {})[name] = dict(n=len(out), ok=sorted(k for k, x in out.items() if x["ok"]),
                                                  missing=sorted(set(job.labels) - set(out)), why=dict(whys))  # fmt: skip
    for r, c in zip(res, claims):
        if r["problems"]:
            r["parked"] = False
            continue
        r["bound_ok"] = all(r["bound"].get(x, {}).get("holds") for x in c["starts"])
        atts = list(r.get("attempts", {}).values())
        r["loop_ok"] = bool(atts) and all(not a["ok"] and not a["missing"] for a in atts)
        r["parked"] = r["bound_ok"] and r["loop_ok"]
        r["scope"] = "task" if set(c["starts"]) == {x for x in diag if LG.task_of(x) == c["task"]} else "starts"
    return res


def park(results: list[dict], claims: list[dict], n: int) -> None:
    """Record the verified claims as parks (a claim with the same id replaces the old one) and every other claim as
    refused (a part missing) or unproven (the check or the attempts said otherwise)."""
    parked = LG.jload(S.STATE / "parked.json", {})
    for r, c in zip(results, claims):
        if r.get("parked"):
            parked[r["id"]] = dict(iter=n, task=c["task"], kind=c["kind"], scope=r["scope"], starts=sorted(c["starts"]),
                                   statement=c["bound"]["statement"], actions_assumed=c["actions_assumed"],
                                   exits=c.get("exits") or [], no_exit_reason=c.get("no_exit_reason"),
                                   refuted_by=c["refuted_by"], user_confirmed=False, since=LG.now())  # fmt: skip
            LG.ledger(dict(step="park", iter=n, claim=r["id"], task=c["task"], kind=c["kind"], scope=r["scope"],
                        starts=sorted(c["starts"])))  # fmt: skip
        else:
            LG.ledger(dict(step="claim_refused" if r["problems"] else "claim_unproven", iter=n, claim=r["id"],
                        task=r.get("task"), problems=r["problems"], bound_ok=r.get("bound_ok"),
                        loop_ok=r.get("loop_ok"), attempts=r.get("attempts")))  # fmt: skip
    LG.jdump(S.STATE / "parked.json", parked)


def parked_labels(parked: dict, labels) -> dict:
    """{label: claim id} for the labels a park covers (a task-wide park: every start of its task)."""
    out = {}
    for cid, p in parked.items():
        for x in labels:
            if (p["scope"] == "task" and LG.task_of(x) == p["task"]) or x in p["starts"]:
                out[x] = cid
    return out


def unpark(base: Path, n: int) -> list[str]:
    """D13.5: a parked start the incumbent now finishes reopens its claim, recorded as wrong. `base` holds the
    incumbent's diagnostic rounds, run at the start of every iteration (so after every accepted skill)."""
    parked = LG.jload(S.STATE / "parked.json", {})
    out = B.outcomes(base)
    wins: dict = {}
    for x, cid in parked_labels(parked, out).items():
        if out[x]["ok"]:
            wins.setdefault(cid, []).append(x)
    for cid, won in sorted(wins.items()):
        p = parked.pop(cid)
        LG.ledger(dict(step="unpark", iter=n, claim=cid, task=p["task"], won=sorted(won), claim_wrong=True))
    if wins:
        LG.jdump(S.STATE / "parked.json", parked)
    return sorted(wins)


def reopen_test(n: int, it_dir: Path, cand: Path, cid: str) -> bool:
    """TODO 27.8.7: a candidate built for park `cid`'s exit runs on that park's diagnostic starts only, before any gate
    call. Returns whether it finished any of them; None finished: the candidate is dropped. The park stays either way
    (TODO 36 (review 2026-10-22): 32.12's three reopen candidates all finished a start, all were rejected, and the
    parks were gone): `settle_reopen` lifts it only once the gate has accepted this candidate."""
    parked = LG.jload(S.STATE / "parked.json", {})
    p = parked.get(cid)
    out_dir = it_dir / "reopen" / cid
    if p is not None:
        B.run_jobs([B.Job(cand / "skills", out_dir, overrides(cand), tuple(p["starts"]))], S.FAMILY, S.LOOP / "pool_diag",
                   S.PROCS, it_dir / "prober.log", S.GPUS)  # fmt: skip
    out = B.outcomes(out_dir) if p is not None else {}
    won = reopen_won(it_dir, cid)
    whys = collections.Counter((x.get("plan_why") or x["why"] or "?")[:60] for x in out.values() if not x["ok"])
    LG.ledger(dict(step="reopen_test", iter=n, claim=cid, n=len(out), ok=won, why=dict(whys),
                error=None if p is not None else "no such park"))  # fmt: skip
    return bool(won)


def reopen_won(it_dir: Path, cid: str) -> list[str]:
    """The park's starts the reopen candidate finished (reopen_test's rounds)."""
    return sorted(k for k, x in B.outcomes(it_dir / "reopen" / cid).items() if x["ok"])


def settle_reopen(n: int, cid: str, won: list[str], accepted: bool) -> None:
    """The gate's verdict on a reopen candidate: accepted -> the park lifts (`unpark`, by reopen; the claim was about
    the present skills and stands, so it is not recorded as wrong); rejected or void -> `reopen_rejected`, the park
    stays (TODO 36 (review 2026-10-22))."""
    parked = LG.jload(S.STATE / "parked.json", {})
    p = parked.get(cid)
    task = p["task"] if p else None
    if accepted and p is not None:
        parked.pop(cid)
        LG.jdump(S.STATE / "parked.json", parked)
        LG.ledger(dict(step="unpark", iter=n, claim=cid, task=task, won=won, claim_wrong=False, by="reopen"))
    elif not accepted:
        LG.ledger(dict(step="reopen_rejected", iter=n, claim=cid, task=task, won=won))


def unparked_view(rounds: Path) -> dict | None:
    """D13.4's second number on the acceptance pool: successes on the starts outside the task-wide parks (a park of
    some starts names diagnostic starts and says nothing about a hidden pool). None without such parks or rounds."""
    tasks = sorted({p["task"] for p in LG.jload(S.STATE / "parked.json", {}).values() if p["scope"] == "task"})
    labels = B.pool_labels(CS.accept_pool())
    out = {k: v for k, v in B.outcomes(rounds).items() if v.get("rule", 1) == B.SUCCESS_RULE}
    if not tasks or not labels or any(x not in out for x in labels):
        return None
    keep = [x for x in labels if LG.task_of(x) not in tasks]
    return dict(ok=sum(out[x]["ok"] for x in keep), n=len(keep), parked_tasks=tasks,
                all_ok=sum(out[x]["ok"] for x in labels), all_n=len(labels))  # fmt: skip


def gap_candidates() -> dict:
    """D13.7: the parks' exits by capability, most tasks first -- what each would reopen."""
    out: dict = {}
    for cid, p in LG.jload(S.STATE / "parked.json", {}).items():
        for e in p.get("exits") or []:
            g = out.setdefault(e["capability"].strip().lower(), dict(tasks=set(), starts=0, claims=[], how=[]))
            g["tasks"].add(p["task"])
            g["starts"] += len(p["starts"])
            g["claims"].append(cid)
            g["how"].append(e["how"][:300])
    ranked = sorted(out.items(), key=lambda kv: (-len(kv[1]["tasks"]), -kv[1]["starts"], kv[0]))
    return {k: dict(v, tasks=sorted(v["tasks"])) for k, v in ranked}


def reaudit(n: int) -> None:
    """Iteration n's review, if the audit voided it, read again under the present audit before its hypotheses are
    re-probed (TODO 32.13): a hit, or a transcript gone, refuses the re-probe."""
    rev = [e for e in LG.entries() if e.get("step") == "review" and e.get("iter") == n]
    if not rev or not rev[-1].get("void"):
        return
    sessions = rev[-1].get("sessions") or [rev[-1]["session"]]
    if any(AU.transcript(s) is None for s in sessions):
        raise SystemExit(f"iteration {n}'s review transcript is gone: it cannot be audited again")
    leaks = sorted({x for s in sessions for x in AU.transcript_leaks(s)})
    if leaks:
        raise SystemExit(f"iteration {n}'s review still names off-limits paths ({', '.join(leaks)})")
    LG.ledger(dict(step="reaudit", iter=n, sessions=sessions, was=rev[-1].get("leaks"), leaks=[]))


def attribute(inc: dict, base: Path, n: int, it_dir: Path, stub: bool) -> dict | None:
    """TODO 33.20 (plan section 1.4, D20): whose failure is it. When the incumbent fails ATTRIBUTE_MIN or more
    diagnostic rounds of a class in ATTRIBUTE_CLASSES, each one's failing command is run again alone from its start
    (scripts/rsi/isolate.py; cached per incumbent in diag/attribution_<hash>, which the roles may read). `motor` -- the
    command alone fails the same way -- is the motor layer's or the controller's limit: an escalation row
    (loop/escalations.jsonl, [escalate]) and a body-model constraint (body/constraints.jsonl); a skill may route around
    it, not fix it. `skill` -- alone it does not fail -- is the skill's. The summary goes to the Reviewer and the
    Manager."""
    if stub or BK.of(S.FAMILY):  # isolate.py replays a scene round command by command; a backend has no such replay
        return None
    fails = [r for r in B.outcomes(base).values() if not r.get("ok")]
    count = collections.Counter(r.get("failure_class") for r in fails)
    classes = [c for c in ATTRIBUTE_CLASSES if count[c] >= ATTRIBUTE_MIN]
    if not classes:
        return None
    out = S.LOOP / "diag" / f"attribution_{inc['hash']}"
    cmd = [str(S.SIM_PY), str(S.HERE / "scene" / "isolate.py"), S.FAMILY, "--pool", str(S.LOOP / "pool_diag"), "--rounds", str(base),
           "--out", str(out), *[x for c in classes for x in ("--class", c)]]  # fmt: skip
    env = dict(os.environ, PYTHONPATH="src", OMP_NUM_THREADS="2", NUMPY_MADVISE_HUGEPAGE="0",
               CUDA_VISIBLE_DEVICES=S.GPUS.split(",")[0])  # fmt: skip
    with (it_dir / "isolate.log").open("w") as fh:
        rc = subprocess.run(cmd, cwd=S.REPO, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc:
        LG.ledger(dict(step="attribution", iter=n, error=f"isolate.py exit {rc}"))
        return None
    rows = [LG.jload(p) for p in sorted(out.glob("*.json")) if p.name != "summary.json"]
    motor = [r for r in rows if r.get("verdict") == "motor"]
    record_escalations(inc, n, motor)
    summary = dict(
        classes=classes,
        rounds=[
            dict(
                label=r["label"],
                failure_class=r.get("original", {}).get("failure_class"),
                verdict=r.get("verdict"),
                command=(r.get("call") or {}).get("name"),
                alone=r.get("isolated", {}).get("call_why"),
            )
            for r in rows
        ],  # fmt: skip
        note="motor: the command alone, from the round's start, fails the same way -- the motor layer's or the "
        "controller's limit ([escalate]): route around it (another first move, another stance), do not try to fix the "
        "layer below; skill: alone it does not fail, the skill's earlier steps cause it",
        dir=str(out),
    )
    LG.ledger(dict(step="attribution", iter=n, classes=classes, motor=[r["label"] for r in motor],
                skill=[r["label"] for r in rows if r.get("verdict") == "skill"]))  # fmt: skip
    return summary


def record_escalations(inc: dict, n: int, motor: list[dict]) -> None:
    """TODO 33.20: each `motor` attribution once per incumbent, as an escalation row (loop/escalations.jsonl, for the
    people, [escalate]) and a body-model constraint (body/constraints.jsonl, for the roles)."""
    esc_f, con_f = S.STATE / "escalations.jsonl", S.LOOP.parent / "body" / "constraints.jsonl"
    old = [json.loads(x) for x in esc_f.read_text().splitlines() if x.strip()] if esc_f.exists() else []
    known = {(e.get("incumbent"), e.get("label")) for e in old}
    for r in motor:
        if (inc["hash"], r["label"]) in known:
            continue
        call = r.get("call") or {}
        what = f"{call.get('name')}({', '.join(json.dumps(a) for a in call.get('args', []))})"
        row = dict(t=LG.now(), iter=n, incumbent=inc["hash"], label=r["label"], task=r.get("task"),
                   failure_class=r["original"]["failure_class"], command=what, alone=r["isolated"]["call_why"],
                   rule=20, layer="motor layer / controller")  # fmt: skip
        with esc_f.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        con_f.parent.mkdir(parents=True, exist_ok=True)
        with con_f.open("a") as fh:
            fh.write(json.dumps(dict(row, kind="isolated_failure")) + "\n")


def refresh_diag_pool(inc: dict, base: Path, n: int | None) -> dict | None:
    """TODO 33.23: after the incumbent has run on the diagnostic pool (`base`), re-draw the pool when it fails fewer than
    DIAG_MIN_FAILS starts (diag_pool.refresh: fresh starts from the diagnostic seed range, the incumbent closed loop on
    them, the failing ones swapped in for passing starts until DIAG_TARGET_FAILS fail). Parked starts and the tasks
    parked as a whole stay. The candidate pool's entries then run on the starts they have not seen, so pool_view's
    flips stay comparable. Returns the ledger row, or None when nothing changed."""
    if S.DIAG_MIN_FAILS <= 0 or not S.CAPS["diag_redraw"]:  # CAPS: backend.layout(family)["caps"]; scene: all on
        return None
    parked = LG.jload(S.STATE / "parked.json", {})
    frozen = {x for p in parked.values() for x in p.get("starts") or []}
    tasks = {p["task"] for p in parked.values() if p.get("scope") == "task"}
    scratch = S.LOOP / "diag_redraw" / (f"iter_{n:02d}" if n is not None else LG.now().replace(":", ""))

    def run(pool_dir: Path, out: Path) -> None:
        B.run_jobs([B.Job(Path(inc["dir"]) / "skills", out, inc["set"], None, True)], S.FAMILY, pool_dir, S.PROCS,
                   scratch / "redraw.log", S.GPUS)  # fmt: skip

    row = DP.refresh(F.load(S.FAMILY), S.LOOP / "pool_diag", base, run, scratch, tasks, frozen, S.DIAG_MIN_FAILS,
                     S.DIAG_TARGET_FAILS, S.DIAG_DRAW_MAX, draw=F.draw)  # fmt: skip
    if row is None:
        return None
    LG.ledger(dict(row, iter=n, incumbent=inc["hash"]))
    print(f"diag pool redrawn: {row['fails_before']} -> {row['fails_after']} failing of {row['n']} "
          f"(+{len(row['added'])} -{len(row['retired'])}, {row['drawn']} fresh starts run)")  # fmt: skip
    jobs = [B.Job(Path(e["dir"]) / "skills", S.LOOP / "diag" / f"candidate_{e['hash']}", overrides(Path(e["dir"])), None, True)
            for e in CS.pool_load() if (S.LOOP / "diag" / f"candidate_{e['hash']}").exists() and Path(e["dir"]).exists()]  # fmt: skip
    if jobs and row["added"]:
        B.run_jobs(jobs, S.FAMILY, S.LOOP / "pool_diag", S.PROCS, scratch / "redraw.log", S.GPUS)
    return row


def probe(hyps: list[dict], inc: dict, base: Path, it_dir: Path) -> list[dict]:
    """Every hypothesis on the diagnostic starts (probe_labels). One with `needs_code` is not run: `untested`. One with
    `sweep` (change 1) runs one job per value of its first key, then one per value of its second with the first at its
    best, each on the same starts (subsampled when values x starts exceed its share of caps sweep_rounds); its row
    describes the best set (`best_set`, `sweep`). Any other runs once with its delta."""
    S.SWEEP = DR.SweepBudget(S.CAPS.get("sweep_rounds", 1200))
    skills = Path(inc["dir"]) / "skills"
    scope = {h["id"]: DR.probe_labels(h) for h in hyps}
    before = DR._ok_of(base)
    pool = B.pool_labels(S.LOOP / "pool_diag")
    runnable = [h for h in hyps if not h.get("needs_code")]
    share = S.SWEEP.total // max(1, sum(1 for h in runnable if DR.sweep_specs(h)))
    plan: dict = {}
    for h in runnable:
        specs, labels, sub = DR.sweep_specs(h), scope[h["id"]], False
        if specs:
            starts = sorted(labels if labels is not None else pool)
            n_jobs = sum(len(s["values"]) for s in specs)
            if n_jobs * len(starts) > share:
                starts, sub = DR.subsample_starts(starts, before, max(1, share // n_jobs)), True
                labels = tuple(starts)
            S.SWEEP.reserve(n_jobs * len(starts))
        plan[h["id"]] = dict(labels=labels, specs=specs, sub=sub)
    d1 = lambda h, i: it_dir / "probe" / (f"{h['id']}__s1_{i}" if plan[h["id"]]["specs"] else h["id"])  # noqa: E731
    jobs, failed = [], None
    for h in runnable:
        p, fixed_set = plan[h["id"]], {**inc["set"], **h["delta"].get("set", {})}
        if p["specs"]:
            jobs += [B.Job(skills, d1(h, i), {**fixed_set, p["specs"][0]["key"]: v}, p["labels"])
                     for i, v in enumerate(p["specs"][0]["values"])]  # fmt: skip
        else:
            jobs.append(B.Job(skills, d1(h, 0), fixed_set, p["labels"]))
    try:
        B.run_jobs(jobs, S.FAMILY, S.LOOP / "pool_diag", S.PROCS, it_dir / "prober.log", S.GPUS)
    except Exception as e:
        failed = f"{type(e).__name__}: {e}"
    final: dict = {}  # id -> (rows, best_set, best_dir, second key dropped)
    jobs2, meta2 = [], {}
    for h in runnable:
        p = plan[h["id"]]
        if not p["specs"] or failed:
            continue
        fixed_set = {**inc["set"], **h["delta"].get("set", {})}
        s1 = p["specs"][0]
        rows1 = [DR.sweep_row(s1["key"], v, before, DR._ok_of(d1(h, i))) for i, v in enumerate(s1["values"])]
        live = [i for i, r in enumerate(rows1) if r["n"]]
        if not live:
            continue
        b1 = live[DR.best_row([rows1[i] for i in live])]
        set1 = {**h["delta"].get("set", {}), s1["key"]: s1["values"][b1]}
        final[h["id"]] = (rows1, set1, d1(h, b1), False)
        if len(p["specs"]) == 2:
            s2 = p["specs"][1]
            meta2[h["id"]] = (rows1, b1, set1)
            jobs2 += [B.Job(skills, it_dir / "probe" / f"{h['id']}__s2_{i}", {**fixed_set, **set1, s2["key"]: v}, p["labels"])
                      for i, v in enumerate(s2["values"])]  # fmt: skip
    if jobs2:
        try:
            B.run_jobs(jobs2, S.FAMILY, S.LOOP / "pool_diag", S.PROCS, it_dir / "prober.log", S.GPUS)
        except Exception as e:
            failed = failed or f"{type(e).__name__}: {e}"
        for h in runnable:
            if h["id"] not in meta2:
                continue
            rows1, b1, set1 = meta2[h["id"]]
            s2 = plan[h["id"]]["specs"][1]
            rows2 = [DR.sweep_row(s2["key"], v, before, DR._ok_of(it_dir / "probe" / f"{h['id']}__s2_{i}"))
                     for i, v in enumerate(s2["values"])]  # fmt: skip
            live = [i for i, r in enumerate(rows2) if r["n"]]
            b2 = live[DR.best_row([rows2[i] for i in live])] if live else None
            if b2 is not None and rows2[b2]["net"] >= rows1[b1]["net"]:  # a second key that only hurts is left out
                final[h["id"]] = (rows1 + rows2, {**set1, s2["key"]: s2["values"][b2]},
                                  it_dir / "probe" / f"{h['id']}__s2_{b2}", False)  # fmt: skip
            else:
                final[h["id"]] = (rows1 + rows2, set1, final[h["id"]][2], True)
    res = []
    for h in hyps:
        if h.get("needs_code"):
            res.append(dict(id=h["id"], status="untested", reason="needs code"))
            continue
        p = plan[h["id"]]
        if p["specs"]:
            rows, best_set, rdir, dropped = final.get(h["id"], ([], None, None, False))
        else:
            rows, best_set, rdir, dropped = None, dict(h["delta"].get("set", {})), d1(h, 0), False
        after = DR._ok_of(rdir) if rdir is not None else {}
        if failed or not after:
            res.append(dict(id=h["id"], error=failed or "no rounds ran (an unknown parameter? see prober.log)"))
            continue
        fixed = sorted(k for k in after if after[k] and not before.get(k))
        broken = sorted(k for k in after if before.get(k) and not after[k])
        pf, pb = set(h["predicted_fixed"]), set(h["predicted_broken"])
        res.append(dict(id=h["id"], n=len(after), ok_before=sum(before.values()), ok_after=sum(after.values()),
                        fixed=fixed, broken=broken, predicted_fixed_hit=len(pf & set(fixed)), predicted_fixed_n=len(pf),
                        predicted_broken_hit=len(pb & set(broken)), predicted_broken_n=len(pb),
                        summary=RL.summarize(rdir), best_set=best_set, best_dir=str(rdir)))  # fmt: skip
        if rows is not None:
            res[-1].update(sweep=rows, **({"second_key_dropped": True} if dropped else {}))
        if p["sub"]:
            res[-1].update(subsampled=True, starts=sorted(p["labels"]))
        if scope[h["id"]] is not None or p["sub"]:  # both counts are over the starts it ran, and it says which
            res[-1].update(ok_before=sum(bool(before.get(k)) for k in after))
        if scope[h["id"]] is not None:
            res[-1].update(scope=sorted({k.rsplit("_s", 1)[0] for k in scope[h["id"]]}))
    return res


def probe_now(n: int, hyps: dict, inc: dict, base: Path, it_dir: Path) -> list[dict]:
    """The Prober on the iteration's hypotheses, their statuses, probe.json and the ledger's probe row."""
    pr = probe(hyps["hypotheses"][: S.MAX_HYPOTHESES], inc, base, it_dir)
    set_statuses(hyps, pr)  # TODO 33.17
    records_statuses(hyps, pr, base)  # TODO 33.15.1
    LG.jdump(it_dir / "hypotheses.json", hyps)
    LG.jdump(it_dir / "probe.json", pr)
    LG.ledger(dict(step="probe", iter=n, results=[{k: v for k, v in p.items() if k != "summary"} for p in pr]))
    return pr


def after_review(n, it_dir, inc, base, base_changes, hyps, common, extra, stub) -> str | None:
    """Prober -> Manager -> Engineer -> Gate for a reviewed iteration (also `run --reprobe N`)."""
    pr = probe_now(n, hyps, inc, base, it_dir)
    claims = [c for c in hyps.get("infeasible") or [] if isinstance(c, dict)]  # older hypotheses carry a string
    if not S.CAPS["claims"]:  # a backend without claim checks: an infeasibility claim parks nothing
        claims = []
    cr = verify_claims(claims, inc, it_dir) if claims else []
    LG.jdump(it_dir / "claims.json", cr)
    park(cr, claims, n)
    if S.STOP_AFTER == "probe":
        return "replay: stopped after the probe (--stop-after probe)"
    designs = run_designs(hyps, inc, it_dir, n)  # TODO 33.19
    gaps = gap_candidates()
    LG.jdump(S.STATE / "gap_candidates.json", gaps)
    ctx = it_dir / "manager_context.json"
    es = LG.entries()
    may_reopen = reopen_allowed(es)
    lead = dict(recurring=recurring(es))  # TODO 33.14: first in the context
    if extra.get("attribution"):  # TODO 33.20: which recurring failures are the motor layer's
        lead["attribution"] = extra["attribution"]
    if designs:  # TODO 33.19: how the factors the Reviewer asked about change the outcome (not evidence)
        lead["designs"] = designs
    notes = repair_notes(es, base, n)  # TODO 33.17: after a rejection
    if notes:
        lead["repair_notes"] = notes
    if not may_reopen:
        lead["reopen_blocked"] = (f"the last {MAX_REOPEN_RUN} iterations were reopens ([scene-candidates]): build a "
                                  "hypothesis this iteration")  # fmt: skip
    body = LG.context(inc, dict(extra, hypotheses=hyps, probe=pr, claims=cr,
                             parked=LG.jload(S.STATE / "parked.json", {}),
                             gap_candidates=gaps, candidate_pool=CS.pool_view(inc, base)), compact=True)  # fmt: skip
    LG.jdump(ctx, dict(lead, **body))
    prompt = RL.fill(RL.role_template("manager"), CONTEXT=ctx, **common)
    stub_m = dict(decision="engineer", hypothesis="H1", reason="stub", summary_zh="桩测试")
    ids = [h["id"] for h in hyps["hypotheses"]]
    dec, acct = RL.claude("manager", prompt, manager_schema(may_reopen, ids), it_dir, "Edit,Write,NotebookEdit,Agent,Bash",
                       stub_m if stub else None)  # fmt: skip
    leaks = AU.transcript_leaks(acct["session"])
    if leaks and S.cap("role_rerun") and not stub:  # TODO 48.17: the first leak discards the decision, one more call
        RL.discard_attempt(n, "manager", acct, leaks)
        dec, acct = RL.claude("manager", prompt + RL.rerun_note("Manager", leaks), manager_schema(may_reopen, ids), it_dir,
                           "Edit,Write,NotebookEdit,Agent,Bash")  # fmt: skip
        leaks = AU.transcript_leaks(acct["session"])
    LG.jdump(it_dir / "manager.json", dec)
    tune, assign = dec.get("tune") or [], {}
    if (
        dec["decision"] == "engineer" and S.CANDIDATES > 1
    ):  # change 3: the Engineers' ids as multi_engineer_step takes them
        ids_, dups = DR.assign_ids(dec, hyps["hypotheses"], S.CANDIDATES, tune)
        assign = dict(assigned=ids_, dropped_duplicates=dups)
    elif tune:  # change 1: tuned drafts need the Composer's candidates (decision engineer, K > 1)
        assign = dict(tune_ignored=True)
    LG.ledger(dict(step="manager", iter=n, decision=dec["decision"], hypothesis=dec.get("hypothesis"), tune=tune or None, **assign,
                hypotheses=dec.get("hypotheses") if S.CANDIDATES > 1 else None, reopen_claim=dec.get("reopen_claim"),
                base=dec.get("base"), repeats=dec.get("repeats"), skip=dec.get("skip"),
                reason=dec["reason"], summary_zh=dec["summary_zh"], escalation=dec.get("escalation"), infeasible=dec.get("infeasible"),
                gap_proposals=dec.get("gap_proposals"), leaks=leaks, void=bool(leaks), **acct))  # fmt: skip
    if dec.get("gap_proposals") and not leaks:
        with (S.STATE / "gaps.jsonl").open("a") as fh:
            for g in dec["gap_proposals"]:
                fh.write(json.dumps(dict(t=LG.now(), iter=n, **g), ensure_ascii=False) + "\n")
    if leaks:
        AU.void_iteration(n, "manager", leaks)
        return None
    if dec["decision"] == "reopen" and not may_reopen:  # the schema left it out; an invalid decision ends the iteration
        LG.ledger(dict(step="manager_invalid", iter=n, reason=f"reopen after {MAX_REOPEN_RUN} reopens in a row"))
        return None
    if dec["decision"] == "stop":
        return "manager: stop"
    if dec["decision"] == "review":
        LG.jdump(S.STATE / "focus.json", dict(iter=n, focus=dec.get("focus", ""), reason=dec["reason"]))
        return None
    dec = force_build(n, dec, hyps, it_dir)  # TODO 33.17 ([scene-status])
    return CS.engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub)


def recurring(es: list[dict]) -> list[dict]:
    """TODO 33.14 (D11, user 2026-10-22): mechanisms Reviewers proposed in two or more iterations that no Engineer was
    given. The Manager links a hypothesis to earlier iterations' (`repeats`); linked ids are one mechanism; one given
    to an Engineer (a Manager `engineer` decision, an Engineer row) counts as built. 32.12 proposed the dish soap's
    narrow side in all five iterations and built it in none."""
    groups, built, _ = LG.mechanism_links(es)
    out = []
    for members in groups.values():
        iters = {m.split(":", 1)[0] for m in members}
        if len(iters) < 2 or members & built:
            continue
        latest = max(members)
        it, hid = latest.split(":", 1)
        hs = (LG.jload(S.STATE / f"iter_{it}" / "hypotheses.json", {}) or {}).get("hypotheses") or []
        claim = next((h.get("claim") for h in hs if isinstance(h, dict) and h.get("id") == hid), "")
        out.append(dict(proposed=len(iters), ids=sorted(members), latest=latest, claim=LG._cut(claim, 300)))
    return sorted(out, key=lambda g: (-g["proposed"], g["latest"]))


def set_statuses(hyps: dict, pr: list[dict]) -> None:
    """Every hypothesis's `status` from its probe result (probe_status), on the hypothesis and on its probe row."""
    by = {r.get("id"): r for r in pr}
    for h in hyps.get("hypotheses") or []:
        if isinstance(h, dict) and "id" in h:
            h["status"] = LG.probe_status(h, by.get(h["id"]))
            if h["id"] in by:
                by[h["id"]]["status"] = h["status"]


def records_statuses(hyps: dict, pr: list[dict], base: Path) -> None:
    """TODO 33.15.1 (plan section 3.6): a hypothesis the probe left at `hypothesis` (its test needs code) that names a
    `measure` gets its status from the record library (records.judge on the rounds the roles may see, this machine,
    the incumbent's version by default): verified, refuted, or still a hypothesis when there are too few rounds. The
    verdict and its counts go on the hypothesis and on its probe row (`records`), so statuses() and 33.17 read it."""
    want = [h for h in hyps.get("hypotheses") or []
            if isinstance(h, dict) and isinstance(h.get("measure"), dict)
            and h.get("status") in ("hypothesis", "untested")]  # fmt: skip
    if not want:
        return
    rows, version, host = RC.load(S.REPO, S.SCRATCH_ROOT), RC.version_of(base), RC.machine()
    by = {r.get("id"): r for r in pr}
    for h in want:
        q = RC.judge(rows, h["measure"], version, host)
        if h.get("status") == "untested" and q["verdict"] == "hypothesis":  # too few rounds: still not tested
            continue
        h["status"], h["records"] = q["verdict"], q
        if h["id"] in by:
            by[h["id"]].update(status=q["verdict"], records=q)


def run_designs(hyps: dict, inc: dict, it_dir: Path, n: int) -> list[dict]:
    """TODO 33.19 (plan section 1.3, D16): the hypotheses' design probes (design.py), at most DS.MAX_PROBES an
    iteration and DS.MAX_ROUNDS rounds each, under the incumbent on the diagnostic pool's starts. Not evidence ([evidence]):
    the per-level outcome goes to this iteration's Manager (`lead`), the next Reviewer (`designs_last`), the ledger, and
    this run's measured curves (body/curves_run.jsonl), each tagged with the machine, the controller and its scope --
    `skill` for a parameter of the incumbent, `start` for a start factor (TODO 33.1.7). Not body/curves.jsonl: that is
    the snapshot the sandbox was built with, the one skills read at run time (body.curves, body.best_level), fixed for
    the whole run so a skill behaves the same in every round of it; scripts/rsi/body_curves.py gathers a finished
    run's curves into the origin's."""
    if not S.CAPS["design_probes"]:  # a backend without design probes (design.py varies scene start factors)
        return []
    specs = [(h["id"], h["design"]) for h in hyps.get("hypotheses") or []
             if isinstance(h, dict) and isinstance(h.get("design"), dict)]  # fmt: skip
    if not specs:
        return []
    tasks_all, params, out = list(F.load(S.FAMILY)["tasks"]), RL.parameters_of(inc), []
    for k, (hid, spec) in enumerate(specs):
        why = (
            "over the limit of design probes per iteration" if k >= DS.MAX_PROBES else DS.check(spec, tasks_all, params)
        )
        if why:
            out.append(dict(hypothesis=hid, factor=spec.get("factor"), error=why))
            continue
        res = DS.run(spec, S.FAMILY, S.LOOP / "pool_diag", Path(inc["dir"]) / "skills", inc["set"], it_dir / f"design_{k}",
                     S.PROCS, S.GPUS)  # fmt: skip
        out.append(dict(hypothesis=hid, **res))
    LG.jdump(it_dir / "designs.json", out)
    LG.ledger(dict(step="design", iter=n, results=[{k: v for k, v in r.items() if k != "dir"} for r in out]))
    curves = S.LOOP.parent / "body" / "curves_run.jsonl"
    curves.parent.mkdir(parents=True, exist_ok=True)
    with curves.open("a") as fh:
        for r in out:
            if "levels" in r:
                scope = {"param": "skill", "primitive": "body"}.get(r.get("factor"), "start")  # 33.19.1: body alone
                tag = dict(machine=RC.machine(), scope=scope)
                fh.write(
                    json.dumps(dict(t=LG.now(), iter=n, incumbent=inc["hash"], family=S.FAMILY, **tag, **r)) + "\n"
                )
    return out


def statuses(es: list[dict]) -> dict[str, str]:
    """`NN:id` -> status, from the ledger's probe rows (rows from before TODO 33.17 carry none)."""
    out = {}
    for e in es:
        if e.get("step") == "probe" and isinstance(e.get("iter"), int):
            for r in e.get("results") or []:
                if isinstance(r, dict) and r.get("id") and r.get("status"):
                    out[LG.hyp_key(e["iter"], r["id"])] = r["status"]
    return out


def force_build(n: int, dec: dict, hyps: dict, it_dir: Path) -> dict:
    """TODO 33.17 (plan section 3.4, D14; user 2026-10-22: "把 33.15 到 33.18 先做了吧"): a mechanism Reviewers proposed
    in two or more iterations (recurring, with this Manager's `repeats` in the ledger already), never given to an
    Engineer, with a member among this iteration's hypotheses, and not refuted (its latest probed member's status) takes
    one Engineer slot -- the last one -- unless the Manager's `skip` names one of SKIP_CHECKS for it. The first such
    mechanism only (the most proposed). Returns the decision the Engineers carry out; the Manager's own stays in
    manager.json. 32.12 proposed the dish soap's narrow side five times and built it none; option B built it in its
    eighth iteration, two after the prompt rule (33.14) asked for it."""
    if dec.get("decision") != "engineer":
        return dec
    es = LG.entries()
    st = statuses(es)
    known = [h["id"] for h in hyps.get("hypotheses") or [] if isinstance(h, dict) and "id" in h]
    for g in recurring(es):
        mine = [m for m in g["ids"] if m.startswith(f"{n:02d}:") and m.split(":", 1)[1] in known]
        if not mine:
            continue
        probed = [m for m in sorted(g["ids"]) if st.get(m) and st[m] not in ("hypothesis", "untested")]
        if probed and st[probed[-1]] == "refuted":
            continue
        hid = mine[0].split(":", 1)[1]
        skip = dec.get("skip") or {}
        check = skip.get(hid) or skip.get(mine[0])
        if check in SKIP_CHECKS:
            LG.ledger(dict(step="forced_skip", iter=n, hypothesis=hid, mechanism=g["ids"], check=check))
            return dec
        if S.CANDIDATES > 1:
            ids, _ = DR.assign_ids(
                dec, hyps.get("hypotheses") or [], S.CANDIDATES, [*(dec.get("tune") or []), hid]
            )  # hid keeps its slot
            replaced = None
            if hid not in ids:  # the last slot, or a free one
                replaced = ids[-1] if len(ids) >= S.CANDIDATES else None
                ids = [*ids[: S.CANDIDATES - 1], hid] if len(ids) >= S.CANDIDATES else [*ids, hid]
            out = dict(dec, hypotheses=ids)
        else:
            replaced = dec.get("hypothesis")
            out = dict(dec, hypothesis=hid)
        LG.ledger(dict(step="forced", iter=n, hypothesis=hid, replaced=replaced, mechanism=g["ids"],
                    proposed=g["proposed"], skip_given=check))  # fmt: skip
        LG.jdump(it_dir / "manager_forced.json", out)
        return out
    return dec


def repair_notes(es: list[dict], diag: Path, n: int) -> dict | None:
    """TODO 33.17 (plan sections 3.4, 3.5): when the last iteration before `n` ended with a gate rejection or the
    Composer kept the incumbent, what the Manager reads first after `recurring`: the incumbent's diagnostic failures by
    class (`failure_class`, 33.15), the rejected candidate's tally by class from the gate (wins under the class the
    incumbent failed with, losses under the class the candidate failed with), and the standing suggestion per class
    (CLASS_SUGGEST). DexAgent attaches its diagnosis to every rejection the same way."""
    prev = max((e["iter"] for e in es if isinstance(e.get("iter"), int) and e["iter"] < n), default=None)
    if prev is None:
        return None
    rows = [e for e in es if e.get("iter") == prev]
    gate = next((e for e in reversed(rows) if e.get("step") == "gate" and not e.get("void")), None)
    kept = any(e.get("step") == "compose" and e.get("decision") == "keep" for e in rows)
    if gate and not gate.get("accepted"):
        r = gate.get("result") or gate.get("stage1") or {}
        trigger = f"iteration {prev}: the gate rejected ({LG._cut(gate.get('reason'), 80)})"
    elif kept and not gate:
        r, trigger = {}, f"iteration {prev}: the Composer kept the incumbent"
    else:
        return None
    classes = collections.Counter(
        (x.get("failure_class") or "unknown") for x in B.outcomes(diag).values() if not x.get("ok")
    )  # fmt: skip
    per = r.get("per_class") or {}
    names = sorted(set(classes) | set(per), key=lambda k: (-classes.get(k, 0), k))
    return dict(trigger=trigger, diag_failures_by_class=dict(classes.most_common()), gate_by_class=per,
                suggestions={k: S.CLASS_SUGGEST[k] for k in names if k in S.CLASS_SUGGEST})  # fmt: skip


def reopen_allowed(es: list[dict]) -> bool:
    """TODO 33.14 (D11): no more than MAX_REOPEN_RUN Manager `reopen` decisions in a row (the last valid decision per
    iteration). 32.12's iterations 3-5 were all reopens of one park each."""
    last = {}
    for e in es:
        if e.get("step") == "manager" and not e.get("void") and isinstance(e.get("iter"), int):
            last[e["iter"]] = e.get("decision")
    recent = [last[k] for k in sorted(last)][-MAX_REOPEN_RUN:]
    return not (len(recent) == MAX_REOPEN_RUN and all(d == "reopen" for d in recent))


def manager_schema(allow_reopen: bool, ids: list | None = None) -> dict:
    """`ids`: this iteration's hypotheses, the only ids the Manager can name. multi_engineer_step keeps the ids it knows,
    so one the Manager shortened was dropped without a word: a replay gave its three Engineers one hypothesis."""
    sch = json.loads(json.dumps(MANAGER_SCHEMA))
    props = sch["properties"]
    if not allow_reopen or not S.cap("claims"):
        props["decision"]["enum"] = [d for d in props["decision"]["enum"] if d != "reopen"]
    if not S.cap("claims"):  # TODO 54: no parks, no infeasibility claims or gap proposals without caps claims
        for k in ("reopen_claim", "infeasible", "gap_proposals"):
            props.pop(k, None)
    if not S.cap("prober", True):
        props.pop("tune", None)
    if ids:
        sch["properties"]["hypothesis"]["enum"] = list(ids)
        sch["properties"]["hypotheses"]["items"]["enum"] = list(ids)  # may be empty when `tune` names the drafts
        if "tune" in props:
            props["tune"]["items"]["enum"] = list(ids)
    return sch


_NAMES: dict = {}


def object_names() -> set[str]:
    """Names a skills package may not branch on ([branch], TODO 33.13, D10): every scene object's key, body and ADT
    instance name, and the key's stem (Mug_2 -> Mug) unless the stem is also a semantic name (prototype, category:
    Orange_A -> Orange stays allowed). Semantic fields (category, prototype) are features a flow may use."""
    scene = S.family_spec().get("scene")
    if scene not in _NAMES:
        objs = (LG.jload(S.REPO / scene / "scene_meta.json", {}) or {}).get("objects") or {} if scene else {}
        names, semantic = set(), set()
        for k, o in objs.items():
            sem = o.get("semantic") or {}
            adt = sem.get("adt") or {}
            semantic |= {str(x).lower() for x in (adt.get("prototype_name"), adt.get("category")) if x}
            names |= {k, o.get("body") or "", adt.get("instance_name") or ""}
        stems = {re.sub(r"_(\d+|[A-Z])$", "", k) for k in objs}
        names |= {x for x in stems if len(x) >= 3 and x.lower() not in semantic}
        _NAMES[scene] = {x for x in names if x}
    return _NAMES[scene]


def named_objects(skills: Path, names: set[str] | None = None) -> list[str]:
    """String constants in the skills package (docstrings aside, f-string parts included) equal to an object name
    (object_names): `file:line 'name'`. A hit voids the draft ([branch]): a flow chosen by name cannot be reused for
    a similar object."""
    import ast

    names = object_names() if names is None else names
    hits = []
    for f in sorted(Path(skills).rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        try:
            tree = ast.parse(f.read_text())
        except (SyntaxError, ValueError):
            continue  # the gate's static step reports a package that does not load
        docs = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and body:
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                    docs.add(id(first.value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
                if node.value in names:
                    hits.append(f"{f.relative_to(skills).as_posix()}:{node.lineno} {node.value!r}")
    return hits


def tuned_drafts(n, it_dir, inc, known, pr, tune, src, bentry, number0, taken, jobs) -> list[dict]:
    """Change 1: a candidate the driver builds from a probed hypothesis -- the incumbent's skills (or the Manager's base)
    with the incumbent's overrides plus the probe's best set. Its diagnostic rounds are the probe rounds of the best value,
    copied under the candidate's hash (same skills, same set), and the other starts of change 4's labels run as jobs.
    Needs a probe net of at least 1 and no `needs_code`."""
    by = {p.get("id"): p for p in pr}
    out: list[dict] = []
    pool = B.pool_labels(S.LOOP / "pool_diag")
    ok_inc = DR._ok_of(S.LOOP / "diag" / f"incumbent_{inc['hash']}")
    for hid in tune:
        p, row = by.get(hid) or {}, dict(step="tuned", iter=n, hypothesis=hid)
        net = len(p.get("fixed") or []) - len(p.get("broken") or [])
        why = ("needs_code" if known[hid].get("needs_code") else "no probe result" if "fixed" not in p
               else f"probe net {net} < 1" if net < 1 else None)  # fmt: skip
        if why:
            LG.ledger(dict(row, skipped=why))
            continue
        best_set = dict(p.get("best_set") or {})
        num = number0 + len(out)
        cand = it_dir / f"tuned_{hid}"
        if cand.exists():
            shutil.rmtree(cand)
        shutil.copytree(src / "skills", cand / "skills", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (cand / "candidate.json").write_text(json.dumps({"set": {**overrides(src), **best_set}}, indent=1))
        h = candidate_hash(cand)
        if h == inc["hash"] or h in taken:
            LG.ledger(dict(row, skipped="the incumbent" if h == inc["hash"] else f"same as draft {taken[h]['number']}",
                        best_set=best_set, hash=h))  # fmt: skip
            continue
        declared = list(DR.probe_labels(known[hid]) or ["all"])
        labels, ran = DR.draft_labels(pool, declared, ok_inc, int(S.CAPS.get("draft_sentinels", 3)))
        diag = S.LOOP / "diag" / f"candidate_{h}"
        if bentry is None and p.get("best_dir") and Path(p["best_dir"]).is_dir():
            DR.seed_rounds(Path(p["best_dir"]), diag, labels)
        (it_dir / f"code_{num}.patch").write_text("")
        c = dict(number=num, kind="tuned", hypothesis=hid, mechanism=DR.sweep_mechanism(best_set, list(best_set)),
                 scope=dict(when="the probe's starts", starts=declared), dir=str(cand), patch=str(it_dir / f"code_{num}.patch"),
                 hash=h, diag=str(diag), ran=ran, best_set=best_set)  # fmt: skip
        out.append(c)
        taken[h] = c
        jobs.append(B.Job(cand / "skills", diag, overrides(cand), labels, True))
        LG.ledger(dict(row, candidate=num, best_set=best_set, hash=h, net=net))
    return out


def engineer_again(n: int, src: Path, stub: bool) -> str | None:
    """Iteration n as a replay of another run's iteration `src` from its Manager's decision on (TODO 32.8): that
    iteration's hypotheses, probes, claims and Manager decision are taken as they were, and only the Engineer and the
    gate run again -- so a change to the Engineer's call is compared on the same decision."""
    it_dir = S.STATE / f"iter_{n:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    CS.snapshot(n)
    for f in ("hypotheses.json", "probe.json", "claims.json", "manager.json"):
        if (src / f).exists():
            shutil.copy2(src / f, it_dir / f)
    if (src / "review_notes").is_dir() and not (it_dir / "review_notes").exists():
        shutil.copytree(src / "review_notes", it_dir / "review_notes")
    hyps, pr, dec = (
        LG.jload(it_dir / "hypotheses.json"),
        LG.jload(it_dir / "probe.json", []),
        LG.jload(it_dir / "manager.json"),
    )
    if hyps is None or dec is None:
        raise SystemExit(f"{src} has no hypotheses or Manager decision")
    if dec["decision"] not in ("engineer", "reopen"):
        raise SystemExit(f"{src}: the Manager decided {dec['decision']!r}; nothing for the Engineer")
    claims = [c for c in hyps.get("infeasible") or [] if isinstance(c, dict)]
    if claims and (it_dir / "claims.json").exists():  # that iteration's parks, as its after_review recorded them
        park(LG.jload(it_dir / "claims.json"), claims, n)
    inc = CS.incumbent()
    CS.put_skills(Path(inc["dir"]) / "skills")
    base_changes = CS.changed_files()
    base = CS.diag_incumbent(inc, it_dir)
    LG.ledger(dict(step="manager", iter=n, decision=dec["decision"], hypothesis=dec.get("hypothesis"), reason=dec["reason"],
                summary_zh=dec.get("summary_zh", ""), replayed_from=str(src), cost_usd=0.0, seconds=0))  # fmt: skip
    common = dict(**RL.prompt_vars(), LOOP=S.LOOP, SIM_PY=S.SIM_PY, DIAG_INC=base, SKILLS=S.SKILLS, FAMILY=S.FAMILY,
                  FAMILY_FILE=S.HERE / "families" / f"{S.FAMILY}.json", POOL=S.LOOP / "pool_diag", INC_DIR=inc["dir"],
                  TMP=AU.role_tmp(), REVIEW_NOTES=RL.review_notes_dir(it_dir))  # fmt: skip
    return CS.engineer_step(n, it_dir, inc, hyps, pr, dec, common, base_changes, stub)


def reprobe(n: int, stub: bool) -> str | None:
    """Re-run iteration n's probes on its Reviewer's unchanged hypotheses and continue from the Manager: for probes
    lost to the harness or never run because a driver bug stopped the loop (the hypotheses never saw a probe result,
    so they stand), or after a review the audit voided and a fixed audit passes (`reaudit`). The earlier Manager call
    is voided in the ledger; the human fix that made this necessary goes in interventions.jsonl."""
    it_dir = S.STATE / f"iter_{n:02d}"
    hyps = LG.jload(it_dir / "hypotheses.json")
    if hyps is None:
        raise SystemExit(f"{it_dir} has no hypotheses")
    reaudit(n)
    managed = (it_dir / "manager.json").exists()
    for f in ("probe", "manager.json", "probe.json"):
        p = it_dir / f
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.rename(p.with_name(p.name + ".void"))
    if managed:  # the voided Manager's focus; before a Manager ran, focus.json is the one this Reviewer was given
        (S.STATE / "focus.json").unlink(missing_ok=True)
    LG.ledger(
        dict(step="void", iter=n, reason="re-probing the same hypotheses after a driver fix (interventions.jsonl)")
    )
    inc = CS.incumbent()
    CS.put_skills(Path(inc["dir"]) / "skills")
    base_changes = CS.changed_files()
    base = CS.diag_incumbent(inc, it_dir)
    refresh_diag_pool(inc, base, n)
    unpark(base, n)
    attribution = attribute(inc, base, n, it_dir, stub)  # TODO 33.20
    common = dict(**RL.prompt_vars(), LOOP=S.LOOP, SIM_PY=S.SIM_PY, DIAG_INC=base, SKILLS=S.SKILLS, FAMILY=S.FAMILY,
                  FAMILY_FILE=S.HERE / "families" / f"{S.FAMILY}.json", POOL=S.LOOP / "pool_diag", INC_DIR=inc["dir"],
                  TMP=AU.role_tmp(), REVIEW_NOTES=RL.review_notes_dir(it_dir))  # fmt: skip
    extra = dict(diag_summary=RL.summarize(base), parameters=RL.parameters_of(inc), focus=LG.jload(S.STATE / "focus.json", {}),
                 parked=LG.jload(S.STATE / "parked.json", {}),
                 segments=SG.segment_table(B.outcomes(base).values()))  # fmt: skip  # TODO 48.17.6 ([segments])
    if attribution:
        extra["attribution"] = attribution
    return after_review(n, it_dir, inc, base, base_changes, hyps, common, extra, stub)


def cmd_redraw(a) -> int:
    """TODO 33.23: re-draw the diagnostic pool for the present incumbent outside an iteration (first use: run C's
    incumbent, 30/30 on its pool, before that run continues). The incumbent's rounds and the redraw's log go with the
    last iteration's directory; the ledger row carries that iteration's number. Same --diag-* settings as `run`."""
    meta = LG.jload(S.STATE / "meta.json", {})
    cfg = S.resolve_settings(a, meta, bool(meta and "started" in meta))
    S.DIAG_MIN_FAILS, S.DIAG_TARGET_FAILS, S.DIAG_DRAW_MAX = cfg["diag_min_fails"], cfg["diag_target_fails"], cfg["diag_draw_max"]  # fmt: skip
    AU.unlock_hidden()
    AU.unlock_rounds()
    inc = CS.incumbent()
    n = max([e["iter"] for e in LG.entries() if isinstance(e.get("iter"), int)] or [0])
    it_dir = S.STATE / f"iter_{n:02d}"
    it_dir.mkdir(parents=True, exist_ok=True)
    base = CS.diag_incumbent(inc, it_dir)
    row = refresh_diag_pool(inc, base, n)
    print(json.dumps(row or dict(step="diag_pool", incumbent=inc["hash"], skipped="the pool keeps enough failing starts"),
                     ensure_ascii=False)[:1200])  # fmt: skip
    return 0


# --- the hooks plugin.NULL has no-op versions of (state.H is this module for a scene family) -----------------------

scene = True

# the Reviewer's schema fields only a scene family has (infeasibility claims, the Prober's sweep, record-library
# `measure`, design probes), each switched by its cap in schema_fields
HYPOTHESIS_FIELDS = {
    # change 1: the model names the knobs, the driver sweeps them (sweep_specs, probe)
    "sweep": {
        "type": "array",
        "maxItems": 2,
        "items": {
            "type": "object",
            "required": ["key", "values"],
            "properties": {
                "key": {"type": "string"},
                "values": {"type": "array", "minItems": 2, "maxItems": 5},
            },
        },
    },
    # TODO 33.15.1: a measurable claim the driver checks on the record library when the probe cannot
    "measure": {
        "type": "object",
        "required": ["feature", "op", "value", "expect"],
        "properties": {
            "feature": {"type": "string"},
            "op": {"enum": sorted(RC.OPS)},
            "value": {"type": ["number", "string"]},
            "expect": {"enum": ["fails", "succeeds"]},
            "tasks": {"type": "array", "items": {"type": "string"}},
            "scope": {"enum": ["incumbent", "all"]},
        },
    },
    # TODO 33.19: a design probe -- one factor at a few levels, forced on the same diagnostic starts
    "design": {
        "type": "object",
        "required": ["factor", "levels"],
        "properties": {
            "factor": {"enum": list(DS.FACTORS)},
            "levels": {"type": "array", "items": {"type": "number"}},
            "starts": {"type": "integer"},
            "tasks": {"type": "array", "items": {"type": "string"}},
            "name": {"type": "string"},
        },
    },
}


def before_review(inc: dict, base: Path, n: int, it_dir: Path, stub: bool) -> dict | None:
    """What iterate does before the Reviewer: the diagnostic redraw, the parks a finished start lifts, the attribution."""
    refresh_diag_pool(inc, base, n)
    unpark(base, n)
    return attribute(inc, base, n, it_dir, stub)  # TODO 33.20


def accept_pool() -> Path | None:
    """The redrawn acceptance pool when the parks leave less than UNPARKED_MIN of the family's own."""
    pool, _, _ = _live_pool()
    return pool if pool.exists() else None


def schema_fields(cap) -> dict:
    """The scene line's fields of the Reviewer's schema, each by its cap."""
    top = {"infeasible": {"type": "array", "items": CLAIM_SCHEMA}} if cap("claims") else {}
    switch = {"sweep": cap("prober", True), "measure": cap("measure_claims", True), "design": cap("design_probes")}
    return dict(top=top, hypothesis={k: v for k, v in HYPOTHESIS_FIELDS.items() if switch[k]})


def parameters_of(inc: dict) -> dict:
    """The incumbent's overridable parameters, read by loading its skills package in a process of its own."""
    r = subprocess.run([str(S.SIM_PY), "-c", (
        "import json,sys; sys.path[:0]=[sys.argv[1], 'src']; from scene_round import load_skills; "
        "p=load_skills(__import__('pathlib').Path(sys.argv[2])); p.apply_set(p, json.loads(sys.argv[3])); "
        "print(json.dumps(p.parameters(p), default=str))"), str(S.HERE / "scene"), str(Path(inc["dir"]) / "skills"),
        json.dumps(inc["set"])], cwd=S.REPO, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))  # fmt: skip
    return json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-500:]}


def reviewer_cards(n: int, base: Path, it_dir: Path, log: bool = True) -> Path:
    """The failure cards of the incumbent's diagnostic rounds from cards.py and the scene file."""
    out = it_dir / "cards"
    r = subprocess.run([str(S.SIM_PY), str(S.HERE / "scene" / "cards.py"), str(base), str(S.REPO / S.family_spec()["scene"]), "--out", str(out)],
                       capture_output=True, text=True)  # fmt: skip
    (it_dir / "cards.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise RuntimeError(f"cards.py failed ({r.returncode}): {r.stderr[-800:]}")
    if log:
        LG.ledger(RL.cards_row(n, out))
    return out / "cards.md"


def unknown_keys(skills: Path, set_: dict, items: list[dict]) -> list[str]:
    """The tunable keys the draft's package does not accept, found by loading it in a process of its own."""
    code = ("import json,sys; sys.path[:0]=[sys.argv[1], 'src']; from scene_round import load_skills; "
            "p=load_skills(__import__('pathlib').Path(sys.argv[2])); s=json.loads(sys.argv[3]); bad=[]\n"
            "for it in json.loads(sys.argv[4]):\n"
            "    try: p.apply_set(p, {**s, it['key']: it['values'][0]})\n"
            "    except Exception: bad.append(it['key'])\n"
            "print(json.dumps(bad))")  # fmt: skip
    r = subprocess.run([str(S.SIM_PY), "-c", code, str(S.HERE / "scene"), str(skills), json.dumps(set_), json.dumps(items)], cwd=S.REPO,
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH="src"))  # fmt: skip
    return (
        json.loads(r.stdout.strip().splitlines()[-1])
        if r.returncode == 0 and r.stdout.strip()
        else [it["key"] for it in items]
    )


def stop_reason(es: list[dict]) -> str | None:
    """The scene line's stop rules: a Manager `stop`, and the incumbent winning the acceptance starts outside the parked tasks."""
    m = [e for e in es if e.get("step") == "manager" and not e.get("void")]  # a voided Manager decided nothing
    if m and m[-1].get("decision") == "stop":
        return "manager: stop"
    inc = LG.jload(S.GATE_DIR / "incumbent.json")
    v = unparked_view(Path(inc["rounds"])) if inc else None
    if v and v["n"] >= UNPARKED_MIN * v["all_n"] and v["ok"] >= S.STOP_RATE * v["n"]:
        return (f"the incumbent wins {v['ok']}/{v['n']} >= {S.STOP_RATE:.0%} of the acceptance starts outside the parked "
                f"tasks {v['parked_tasks']} ({v['all_ok']}/{v['all_n']} of all): sealed set")  # fmt: skip
    return None
