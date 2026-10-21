"""TODO 32.6: the loop's history in one table -- per iteration: every hypothesis (id, the first line of its claim, its
probe before -> after), the infeasibility claims and parks, the Manager's decision, the Engineer's mechanism and the
gate's verdict; then the parks standing now. What 34 of 44 Reviewers dug out of the context and ledger JSON by hand.

    python3 scripts/rsi/tools/history.py <loop dir>     # outputs/scene_rsi_20261016/<family>/loop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("loop", type=Path)
    a = ap.parse_args(argv)
    rows = jl(a.loop / "ledger.jsonl")
    for n in sorted({r["iter"] for r in rows if r.get("iter") is not None}):
        ev = [r for r in rows if r.get("iter") == n]
        print(f"== iteration {n}")
        hyps = (
            json.loads((a.loop / f"iter_{n:02d}" / "hypotheses.json").read_text())
            if (a.loop / f"iter_{n:02d}" / "hypotheses.json").exists()
            else {}
        )
        probe = {p.get("id"): p for r in ev if r.get("step") == "probe" for p in r.get("results", [])}
        for h in hyps.get("hypotheses", []):
            p = probe.get(h["id"], {})
            pr = f"probe {p.get('ok_before')} -> {p.get('ok_after')} / {p.get('n')}" if p else "no probe"
            print(f"  {h['id']}: {h.get('claim', '').splitlines()[0][:150]}  [{pr}]")
        for r in ev:
            s = r.get("step")
            if s == "review" and r.get("void"):
                print(f"  review VOID: {r.get('leaks')}")
            elif s in ("park", "unpark", "claim_refused", "claim_unproven", "reopen_test"):
                print(f"  {s} {r.get('claim')} {r.get('task', '')} {r.get('problems') or r.get('won') or ''}")
            elif s == "manager":
                print(f"  manager: {r.get('decision')} {r.get('hypothesis') or ''} -- {str(r.get('reason', ''))[:200]}")
            elif s == "engineer":
                print(f"  engineer: {str(r.get('mechanism', ''))[:200]}")
            elif s == "gate":
                g = r.get("result") or r.get("stage1") or {}
                print(f"  gate {'ACCEPTED' if r.get('accepted') else 'rejected'} {r.get('hash')}: candidate "
                      f"{g.get('candidate_ok')} vs incumbent {g.get('incumbent_ok')} of {g.get('n')}, b {g.get('b')} "
                      f"c {g.get('c')} p {g.get('p_one_sided')} -- {r.get('reason')}")  # fmt: skip
    parked = json.loads((a.loop / "parked.json").read_text()) if (a.loop / "parked.json").exists() else {}
    print(
        "== parked now: "
        + (", ".join(f"{k} {v['task']} ({len(v['starts'])} starts)" for k, v in parked.items()) or "none")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
