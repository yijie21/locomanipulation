"""Section 3.2 of docs/handoff/single-arm-curobo-route-plan-20261009.md: compare collect runs of the arm-route A/B.

Rounds are paired by `case_id` (seed:round -- the same sampled scene), and each pair's `init` is checked to match.
Per run: success, failure tags, the table clearance from table_clearance.py (run it first), the time from the stop to
the pre-grasp (first post-stop phase -> settle) and the settle error. Per pair of runs: concordance and an exact
two-sided McNemar p on success, and failure tags present in one run but not the other.

    PYTHONPATH=src python3 scripts/experiments/curobo_route_20261009/compare_runs.py \
        A=outputs/runs/bottle_route_lifted_a_20261009 A2=... B=... B2=... --pairs A:A2 B:B2 A:B A2:B2
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from math import comb
from pathlib import Path

import numpy as np


def tag(why: str) -> str:
    """The failure family: the bracket tag plus the message with numbers dropped."""
    if not why:
        return ""
    return re.sub(
        r"[-+]?\d+(\.\d+)?", "#", why.split(" in ")[0] if " in " in why and why.rstrip().endswith("s") else why
    ).strip()


def load(run: Path) -> dict:
    rows = {}
    for manifest in sorted(glob.glob(str(run / "collect" / "manifest_*.jsonl"))):
        for line in Path(manifest).read_text().splitlines():
            r = json.loads(line)
            s = r["execution"]["metrics"].get("simple", {})
            phases = {p["phase"]: p["t"] for p in s.get("phases", [])}
            first = phases.get("route", phases.get("ready"))
            rows[r["case_id"]] = dict(
                ok=bool(r["ok"]),
                why=r.get("why", ""),
                tag=tag(r.get("why", "")),
                init=r["init"],
                seconds=r.get("seconds"),
                arm_route=s.get("arm_route", "lifted_ready"),
                to_pregrasp_s=(phases["settle"] - first) if first is not None and "settle" in phases else None,
                settle_error_m=(s.get("settle") or {}).get("error_m"),
                route=s.get("route"),
            )
    clearance = run / "table_clearance.json"
    if clearance.exists():
        by_round = {}
        for r in json.loads(clearance.read_text())["rounds"]:
            by_round[(r["manifest"], r["round"])] = r
        # table_clearance keys by manifest + round; case ids are seed:round with the batch's seed
        for manifest in sorted(glob.glob(str(run / "collect" / "manifest_*.jsonl"))):
            for line in Path(manifest).read_text().splitlines():
                r = json.loads(line)
                c = by_round.get((Path(manifest).name, r.get("round")))
                if c is not None:
                    rows[r["case_id"]]["clearance_m"] = c.get("min_distance_m")
                    rows[r["case_id"]]["clearance_body"] = c.get("body")
    return rows


def mcnemar(b: int, c: int) -> float:
    n = b + c
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n) if n else 1.0


def describe(name: str, rows: dict) -> dict:
    vals = list(rows.values())
    ok = sum(r["ok"] for r in vals)
    clear = [r["clearance_m"] for r in vals if r.get("clearance_m") is not None and np.isfinite(r["clearance_m"])]
    times = [r["to_pregrasp_s"] for r in vals if r["to_pregrasp_s"] is not None]
    settle = [r["settle_error_m"] for r in vals if r["settle_error_m"] is not None]
    routes = [r["route"] for r in vals if r.get("route")]
    out = dict(
        run=name,
        rounds=len(vals),
        ok=ok,
        arm_route=sorted({r["arm_route"] for r in vals}),
        failures=dict(Counter(r["tag"] for r in vals if not r["ok"]).most_common()),
        touch=sum(1 for c in clear if c <= 0.0),
        near_1cm=sum(1 for c in clear if c < 0.01),
        clearance_min_cm=100 * min(clear) if clear else None,
        clearance_median_cm=100 * float(np.median(clear)) if clear else None,
        to_pregrasp_median_s=float(np.median(times)) if times else None,
        settle_error_median_cm=100 * float(np.median(settle)) if settle else None,
        route_plan_wall_median_s=float(np.median([r["plan_wall_s"] for r in routes if r.get("plan_wall_s")]))
        if routes
        else None,
        route_failed=sum(1 for r in routes if not r.get("ok")),
        route_rebuilt=sum(1 for r in routes if r.get("rebuilt")),
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="NAME=run_dir")
    ap.add_argument("--pairs", nargs="*", default=[], help="NAME:NAME")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    runs = {k: load(Path(v)) for k, v in (a.split("=", 1) for a in args.runs)}
    report = dict(runs=[describe(k, v) for k, v in runs.items()], pairs=[])
    for k in report["runs"]:
        print(json.dumps(k))
    for pair in args.pairs:
        a, b = pair.split(":")
        common = sorted(set(runs[a]) & set(runs[b]))
        mismatched = [c for c in common if runs[a][c]["init"] != runs[b][c]["init"]]
        both = sum(runs[a][c]["ok"] and runs[b][c]["ok"] for c in common)
        a_only = sum(runs[a][c]["ok"] and not runs[b][c]["ok"] for c in common)
        b_only = sum(runs[b][c]["ok"] and not runs[a][c]["ok"] for c in common)
        tags_a = {runs[a][c]["tag"] for c in common if not runs[a][c]["ok"]}
        tags_b = {runs[b][c]["tag"] for c in common if not runs[b][c]["ok"]}
        res = dict(
            pair=pair,
            paired=len(common),
            init_mismatch=len(mismatched),
            ok=(both + a_only, both + b_only),
            both=both,
            only_first=a_only,
            only_second=b_only,
            mcnemar_p=mcnemar(a_only, b_only),
            tags_only_first=sorted(tags_a - tags_b),
            tags_only_second=sorted(tags_b - tags_a),
        )
        report["pairs"].append(res)
        print(json.dumps(res))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
