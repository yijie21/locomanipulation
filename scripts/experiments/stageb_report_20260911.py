#!/usr/bin/env python3
"""Stage B acceptance: three arms, the same ten cases, read per round.

The arms are paired by `case_id`, so nothing here compares distributions:

    release          APPR_COARSE 0.30 -> 0.203 m/s   (the calibrated line)
    v1.1 coarse 0.30 the constant carried across     -> 0.044 m/s
    v1.1 coarse 0.40 Stage B's one change            -> 0.187 m/s

Acceptance (ledger bottle.md 21.4) is telemetry, not a success rate on ten
rounds: the approach must stop exiting as `stalled`, its coarse phase must come
back down from ~13 s, and the brake/park distribution must look like release's.

    python scripts/experiments/stageb_report_20260911.py
"""

import json
import re
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AB = ROOT / "outputs/experiments/v11_bottle_ab20_20260910"
NEW = ROOT / "outputs/experiments/stageb_coarse_pilot_20260911"
ARMS = (
    ("release (coarse 0.30, 0.203 m/s)", AB / "release/manifest.jsonl", None),
    ("v1.1 coarse 0.30 (0.044 m/s)", AB / "sonic_v1_1/manifest.jsonl", None),
    ("v1.1 coarse 0.40 (0.187 m/s)", NEW / "manifest.jsonl", NEW / "gym.log"),
)
# `[gym] approach: exit=<reason> braked at x <..> edge <..> -> stopped x <..>`
EXIT_RE = re.compile(r"approach: exit=(?P<why>[a-z ]+?) braked at x\s*(?P<bx>[-\d.nan]+)")


def rows(path, limit=10):
    if not path.is_file():
        return []
    out = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return out[:limit]


def arrival(r):
    m = ((r.get("execution") or {}).get("metrics") or {}).get("arrival_measurements") or []
    return m[0] if m else {}


def _clean(xs):
    # A `stalled` exit records brake_depth_m as NaN, and one NaN makes the whole
    # median NaN -- drop them and say how many were dropped instead.
    return [x for x in xs if x is not None and x == x]


def med(xs):
    xs = _clean(xs)
    return round(st.median(xs), 3) if xs else None


def span(xs):
    n_all = len([x for x in xs if x is not None])
    xs = _clean(xs)
    if not xs:
        return "--"
    tail = f"+{n_all - len(xs)}nan" if n_all > len(xs) else ""
    return f"[{min(xs):.2f},{max(xs):.2f}]{tail}"


def main():
    print(f"{'arm':>34} {'ok':>5} {'brake x':>16} {'end x':>16} {'carry':>16} {'park':>16}")
    table = {}
    for label, manifest, _ in ARMS:
        rs = rows(manifest)
        if not rs:
            print(f"{label:>34}   no manifest yet ({manifest})")
            continue
        br = [arrival(r).get("brake_depth_m") for r in rs]
        en = [arrival(r).get("end_depth_m") for r in rs]
        carry = [
            (a.get("brake_depth_m") - a.get("end_depth_m"))
            if a.get("brake_depth_m") is not None and a.get("end_depth_m") is not None
            else None
            for a in (arrival(r) for r in rs)
        ]
        park = [r.get("park") for r in rs]
        ok = sum(1 for r in rs if r.get("ok"))
        print(
            f"{label:>34} {ok:>3}/{len(rs)} {str(med(br)):>7} {span(br):>8}"
            f" {str(med(en)):>7} {span(en):>8} {str(med(carry)):>7} {span(carry):>8}"
            f" {str(med(park)):>7} {span(park):>8}"
        )
        reasons = {}
        for r in rs:
            reasons[arrival(r).get("reason", "(none)")] = reasons.get(arrival(r).get("reason", "(none)"), 0) + 1
        table[label] = {
            "n": len(rs),
            "ok": ok,
            "brake_median": med(br),
            "end_median": med(en),
            "carry_median": med(carry),
            "park_median": med(park),
            "approach_exit_reasons": reasons,
            "failures": [r.get("why", "") for r in rs if not r.get("ok")],
        }

    print("\n== approach exit reason, per arm (the `stalled` count is the acceptance)")
    for label, v in table.items():
        print(f"{label:>34}  {v['approach_exit_reasons']}")

    print("\n== per-case pairing (case_id: release / v1.1-0.30 / v1.1-0.40)")
    per = {}
    for label, manifest, _ in ARMS:
        for r in rows(manifest):
            per.setdefault(r["case_id"], {})[label] = "ok" if r.get("ok") else (r.get("why") or "fail")[:44]
    for cid, got in per.items():
        cells = " | ".join(f"{got.get(label, '--'):<44}" for label, _, _ in ARMS)
        print(f"{cid:>10}  {cells}")

    print("\n== failure strings")
    for label, v in table.items():
        for w in v["failures"]:
            print(f"{label:>34}  {w[:96]}")

    out = NEW / "stageb_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arms": table, "per_case": per}, indent=1) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
