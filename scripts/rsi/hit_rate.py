"""TODO 48.19.3 (user 2026-11-23, "三条都加上"): how often each role's proposals hold up, from a loop's own files.

A smarter model should need fewer iterations; what it can change is how many of its proposals survive each filter.
This module counts that funnel for one run directory (`outputs/scene_rsi_20261016/<family>/loop`), from the ledger and
the iterations' composer contexts only (nothing of the hidden tree):

    Reviewer   hypotheses the Prober ran on the visible diagnostic starts: better (more starts fixed than broken),
               same, worse, and how many broke at least one start; its predicted fixes that the probe confirmed
    Engineer   drafts built (void and errored ones counted apart); for drafts a Composer saw, their visible flips
               against the incumbent: better, same, worse, broke at least one start
    check      candidates run on the hidden half of the diagnostic pool, and how many it sent on ([check-half])
    gate       calls that judged (not void, not dry runs), how many passed stage 1, how many were accepted

`compute(loop_dir) -> dict`, `line(dict) -> str` (one line for `loop.py status`). The loop writes compute()'s result to
loop/hit_rate.json after every iteration and at a stop. As a script it prints one line per run directory:

    python scripts/rsi/hit_rate.py <loop dir> [<loop dir> ...]
"""

from __future__ import annotations

import ast
import collections
import json
import sys
from pathlib import Path


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


def _list(v) -> list:
    """A list field that a context may hold as text (str(list))."""
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.startswith("["):
        try:
            return list(ast.literal_eval(v))
        except (ValueError, SyntaxError):
            return []
    return []


def _sign(fixed: int, broken: int) -> str:
    return "better" if fixed > broken else "worse" if broken > fixed else "same"


def compute(loop_dir: Path) -> dict:
    loop_dir = Path(loop_dir)
    rows = _rows(loop_dir)
    per_iter: dict[int, dict] = collections.defaultdict(collections.Counter)

    rev = collections.Counter()
    status = collections.Counter()
    for r in rows:
        if r.get("step") != "probe" or r.get("void"):
            continue
        for h in r.get("results") or []:
            fx, br = len(h.get("fixed") or []), len(h.get("broken") or [])
            st = h.get("status") or "unknown"
            status[st] += 1
            rev["probed"] += 1
            rev[_sign(fx, br)] += 1
            rev["broke_any"] += br > 0
            rev["predicted_fixed_hit"] += int(h.get("predicted_fixed_hit") or 0)
            rev["predicted_fixed_n"] += int(h.get("predicted_fixed_n") or 0)
            per_iter[r.get("iter")]["probed"] += 1
            per_iter[r.get("iter")][f"probe_{_sign(fx, br)}"] += 1

    eng = collections.Counter()
    for r in rows:
        if r.get("step") != "engineer":
            continue
        eng["void" if r.get("void") else "error" if r.get("error") else "built"] += 1
        per_iter[r.get("iter")]["drafts"] += 0 if (r.get("void") or r.get("error")) else 1
    for cc in sorted(loop_dir.glob("iter_*/composer_context.json")):
        try:
            ctx = json.loads(cc.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for d in ctx.get("candidates") or []:
            fx, br = len(_list(d.get("fixed"))), len(_list(d.get("broken")))
            eng["seen"] += 1
            eng[f"visible_{_sign(fx, br)}"] += 1
            eng["visible_broke_any"] += br > 0

    comp = collections.Counter(
        r.get("decision") or "?" for r in rows if r.get("step") == "compose" and not r.get("void")
    )

    chk = collections.Counter()
    for r in rows:
        if r.get("step") == "check":
            chk["n"] += 1
            chk["sent"] += bool(r.get("sent"))
            per_iter[r.get("iter")]["checked"] += 1
            per_iter[r.get("iter")]["check_sent"] += bool(r.get("sent"))

    gate = collections.Counter()
    for r in rows:
        if r.get("step") != "gate" or r.get("void") or r.get("dry_run"):
            continue
        reason = str(r.get("reason") or "")
        if reason.startswith(
            ("dry run", "the candidate does not load", "this exact candidate", "the candidate is the")
        ):
            gate["static"] += 1
            continue
        gate["calls"] += 1
        gate["stage1_passed"] += bool(r.get("result")) or bool(r.get("accepted"))
        gate["accepted"] += bool(r.get("accepted"))
        if r.get("result") and not r.get("accepted"):
            res = r["result"]
            gate["rejected_net_negative"] += (res.get("b", 0) - res.get("c", 0)) < 0
        if not r.get("result") and not r.get("accepted") and isinstance(r.get("stage1"), dict):
            s1 = r["stage1"]
            gate["rejected_net_negative"] += (s1.get("b", 0) - s1.get("c", 0)) < 0
        per_iter[r.get("iter")]["gate"] += 1
        per_iter[r.get("iter")]["accepted"] += bool(r.get("accepted"))

    ov = collections.Counter()  # TODO 54 phase 4: the gate and the check half beside the next iteration (overlap.py)
    for r in rows:
        if r.get("step") == "overlap" and r.get("kind") == "settled":
            ov["calls"] += 1
            ov["call_s"] += int(r.get("call_s") or 0)
            ov["waited_s"] += int(r.get("waited_s") or 0)
            ov["beside_s"] += int(r.get("beside_s") or 0)
            ov["accepted"] += bool(r.get("accepted"))
        elif r.get("step") == "overlap" and r.get("kind") == "abandoned":
            ov["abandoned"] += 1
        elif r.get("step") == "rebase" and r.get("kind") == "draft":
            ov["rebased_drafts"] += 1
        elif r.get("step") == "drop":
            ov[f"dropped_{r.get('kind')}s"] += 1
            ov[f"dropped_{r.get('reason')}"] += r.get("kind") == "draft"
        elif r.get("step") == "rescreen":
            ov["rescreens"] += 1
        if r.get("step") in ("rebase", "drop", "rescreen") and isinstance(r.get("iter"), int):
            per_iter[r["iter"]][r["step"]] += 1

    def rate(a: int, b: int) -> float | None:
        return round(a / b, 3) if b else None

    return dict(
        reviewer=dict(rev, status=dict(status), better_rate=rate(rev["better"], rev["probed"]),
                      worse_rate=rate(rev["worse"], rev["probed"])),
        engineer=dict(eng, visible_better_rate=rate(eng["visible_better"], eng["seen"])),
        composer=dict(decisions=dict(comp)),
        check=dict(chk, pass_rate=rate(chk["sent"], chk["n"])),
        gate=dict(gate, accept_rate=rate(gate["accepted"], gate["calls"])),
        overlap=dict(ov, rebased_drafts=ov["rebased_drafts"], beside_share=rate(ov["beside_s"], ov["call_s"])),
        per_iter={str(k): dict(v) for k, v in sorted(per_iter.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
    )  # fmt: skip


def line(d: dict) -> str:
    r, e, c, g = d["reviewer"], d["engineer"], d["check"], d["gate"]
    return (f"hit rate: Reviewer {r.get('probed', 0)} probed (better {r.get('better', 0)}, same {r.get('same', 0)}, "
            f"worse {r.get('worse', 0)}); Engineer {e.get('built', 0)} drafts built; the Composer saw {e.get('seen', 0)} "
            f"drafts with the program's tuned ones (visible better {e.get('visible_better', 0)}, broke a start "
            f"{e.get('visible_broke_any', 0)}); "
            f"check {c.get('sent', 0)}/{c.get('n', 0)} sent on; gate {g.get('accepted', 0)}/{g.get('calls', 0)} accepted "
            f"({g.get('rejected_net_negative', 0)} rejected below the incumbent)"
            + (f"; overlap {d['overlap']['calls']} calls, {d['overlap']['beside_s']} of {d['overlap']['call_s']} s beside "
               f"the next iteration, {d['overlap']['rebased_drafts']} drafts rebased" if d.get("overlap", {}).get("calls") else ""))  # fmt: skip


def main(argv: list[str]) -> int:
    for a in argv:
        print(f"{a}: {line(compute(Path(a)))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
