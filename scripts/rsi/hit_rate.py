"""TODO 56.4 (user 2026-11-27, from main's TODO 48.19.3): how often each role's proposals hold up, from a loop's own
files.

A smarter model should need fewer iterations; what it can change is how many of its proposals survive each filter.
This module counts that funnel for one loop directory (`outputs/scene_rsi_20261016/<family>/loop`), from the ledger
and the iterations' candidates.json only (nothing of the hidden tree):

    Reviewer   hypotheses the Prober ran on the diagnostic starts: better (more starts fixed than broken), same,
               worse, how many broke at least one start, the probe statuses, and its predicted fixes the probe confirmed
    Engineer   drafts built (void and errored ones counted apart); for the valid drafts (candidates.json) their
               diagnostic flips against the incumbent: better, same, worse, broke at least one start
    Composer   decisions (merge / adopt / keep) and who made them (`by`: the program or the model, TODO 56.2)
    gate       calls that judged (not void, not dry runs, not another family's sync), how many passed stage 1, how
               many were accepted, how many were second shots; the sync calls apart

`compute(loop_dir) -> dict`, `line(dict) -> str` (one line for `loop.py status` and `alternate.py status`). The loop
writes compute()'s result to loop/hit_rate.json after every iteration and at a stop. As a script it prints one line
per loop directory:

    python3 scripts/rsi/hit_rate.py <loop dir> [<loop dir> ...]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

STATIC = (
    "dry run",
    "the candidate does not load",
    "this exact candidate",
    "the candidate is the",
    "acceptance pool has",
)


def _rows(loop_dir: Path) -> list[dict]:
    f = Path(loop_dir) / "ledger.jsonl"
    if not f.exists():
        return []
    out = []
    for x in f.read_text().splitlines():
        if x.strip():
            try:
                out.append(json.loads(x))
            except json.JSONDecodeError:
                continue
    return out


def _sign(fixed: int, broken: int) -> str:
    return "better" if fixed > broken else "worse" if broken > fixed else "same"


def _rate(a: int, b: int) -> float | None:
    return round(a / b, 3) if b else None


def compute(loop_dir: Path) -> dict:
    loop_dir = Path(loop_dir)
    rows = _rows(loop_dir)
    per_iter: dict[int, collections.Counter] = collections.defaultdict(collections.Counter)

    # every key is present even in an empty loop, so hit_rate.json keeps one shape
    rev = collections.Counter(probed=0, better=0, same=0, worse=0, broke_any=0, predicted_fixed_hit=0,
                              predicted_fixed_n=0)  # fmt: skip
    status: collections.Counter = collections.Counter()
    for r in rows:
        if r.get("step") != "probe" or r.get("void"):
            continue
        for h in r.get("results") or []:
            fx, br = len(h.get("fixed") or []), len(h.get("broken") or [])
            status[h.get("status") or "unknown"] += 1
            rev["probed"] += 1
            rev[_sign(fx, br)] += 1
            rev["broke_any"] += br > 0
            rev["predicted_fixed_hit"] += int(h.get("predicted_fixed_hit") or 0)
            rev["predicted_fixed_n"] += int(h.get("predicted_fixed_n") or 0)
            per_iter[r.get("iter")]["probed"] += 1
            per_iter[r.get("iter")][f"probe_{_sign(fx, br)}"] += 1

    eng = collections.Counter(built=0, void=0, error=0, late=0, screened=0, diag_better=0, diag_same=0, diag_worse=0,
                              diag_broke_any=0)  # fmt: skip
    for r in rows:
        if r.get("step") != "engineer":
            continue
        eng["void" if r.get("void") else "error" if r.get("error") else "built"] += 1
        eng["late"] += bool(r.get("late"))
        per_iter[r.get("iter")]["drafts"] += 0 if (r.get("void") or r.get("error")) else 1
    for cj in sorted(loop_dir.glob("iter_*/candidates.json")):
        try:
            cands = json.loads(cj.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for d in cands if isinstance(cands, list) else []:
            if "fixed" not in d:
                continue
            fx, br = len(d.get("fixed") or []), len(d.get("broken") or [])
            eng["screened"] += 1
            eng[f"diag_{_sign(fx, br)}"] += 1
            eng["diag_broke_any"] += br > 0

    comp: collections.Counter = collections.Counter()
    by: collections.Counter = collections.Counter()
    for r in rows:
        if r.get("step") == "compose" and not r.get("void"):
            comp[r.get("decision") or "?"] += 1
            by[r.get("by") or "llm"] += 1

    gate = collections.Counter(static=0, calls=0, stage1_passed=0, accepted=0, second_shots=0, rejected_net_negative=0)
    sync = collections.Counter(calls=0, accepted=0)
    for r in rows:
        if r.get("step") != "gate" or r.get("void") or r.get("dry_run"):
            continue
        if str(r.get("reason") or "").startswith(STATIC):
            gate["static"] += 1
            continue
        if r.get("sync") is not None:  # another family's incumbent merged in (alternate.py), judged non-inferior
            sync["calls"] += 1
            sync["accepted"] += bool(r.get("accepted"))
            continue
        gate["calls"] += 1
        gate["stage1_passed"] += bool(r.get("result")) or bool(r.get("accepted"))
        gate["accepted"] += bool(r.get("accepted"))
        gate["second_shots"] += (r.get("shot") or 1) > 1
        t = r.get("result") or r.get("stage1") or {}
        gate["rejected_net_negative"] += (not r.get("accepted")) and (t.get("b", 0) - t.get("c", 0)) < 0
        per_iter[r.get("iter")]["gate"] += 1
        per_iter[r.get("iter")]["stage1_passed"] += bool(r.get("result")) or bool(r.get("accepted"))
        per_iter[r.get("iter")]["accepted"] += bool(r.get("accepted"))

    return dict(
        reviewer=dict(rev, status=dict(status), better_rate=_rate(rev["better"], rev["probed"]),
                      worse_rate=_rate(rev["worse"], rev["probed"]),
                      predicted_fixed_rate=_rate(rev["predicted_fixed_hit"], rev["predicted_fixed_n"])),
        engineer=dict(eng, diag_better_rate=_rate(eng["diag_better"], eng["screened"])),
        composer=dict(decisions=dict(comp), by=dict(by)),
        gate=dict(gate, stage1_rate=_rate(gate["stage1_passed"], gate["calls"]),
                  accept_rate=_rate(gate["accepted"], gate["calls"])),
        sync=dict(sync, accept_rate=_rate(sync["accepted"], sync["calls"])),
        per_iter={str(k): dict(v) for k, v in sorted(per_iter.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
    )  # fmt: skip


def line(d: dict) -> str:
    r, e, c, g, s = d["reviewer"], d["engineer"], d["composer"], d["gate"], d.get("sync") or {}
    dec = ", ".join(f"{k} {v}" for k, v in sorted(c.get("decisions", {}).items())) or "none"
    return (f"hit rate: Reviewer {r.get('probed', 0)} probed (better {r.get('better', 0)}, same {r.get('same', 0)}, "
            f"worse {r.get('worse', 0)}); Engineer {e.get('built', 0)} drafts built, {e.get('screened', 0)} screened "
            f"(better {e.get('diag_better', 0)}, broke a start {e.get('diag_broke_any', 0)}); Composer {dec} "
            f"(program {c.get('by', {}).get('program', 0)}, model {c.get('by', {}).get('llm', 0)}); gate "
            f"{g.get('accepted', 0)}/{g.get('calls', 0)} accepted ({g.get('stage1_passed', 0)} past stage 1, "
            f"{g.get('second_shots', 0)} second shots)"
            + (f"; sync {s.get('accepted', 0)}/{s.get('calls', 0)} accepted" if s.get("calls") else ""))  # fmt: skip


def main(argv: list[str]) -> int:
    for a in argv:
        print(f"{a}: {line(compute(Path(a)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
