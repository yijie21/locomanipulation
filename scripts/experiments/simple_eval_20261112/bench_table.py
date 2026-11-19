"""The table of simple_bench.py's rows (2026-11-19, simple_bench_20261119): per task and method, successes within the
official time limit and within 45 s, the palm's error at the end of the entry and at the pregrasp (median, p90), the
target knocked before the grasp closed, falls; and a paired comparison of two methods on the same episodes.

    python bench_table.py <rows dir> [--pair model direct]
"""

import argparse
import collections
import glob
import json

import numpy as np

TASKS = ("XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP")
ORDER = ("direct", "lookup", "inverse", "model", "servo")


def rows_of(d: str) -> dict:
    by = collections.defaultdict(dict)
    for f in sorted(glob.glob(f"{d}/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            by[(r["task"], r["method"])][r["ep"]] = r
    return by


def stage_err(r: dict, stage: str):
    s = [x for x in r["stages"] if x["stage"] == stage]
    return s[0]["err_cm"] if s else np.nan


def knocked_early(r: dict) -> bool:
    k = r.get("knocked")
    return bool(k) and k["phase"] in ("walk", "clear", "pregrasp", "entry")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rows")
    ap.add_argument("--pair", nargs=2, action="append", default=[])
    a = ap.parse_args()
    by = rows_of(a.rows)
    methods = [m for m in ORDER if any(k[1] == m for k in by)]
    print(f"{'task':22s} {'method':8s} {'n':>3s} {'official':>8s} {'45 s':>5s} {'entry med':>9s} {'p90':>6s} "
          f"{'pre med':>7s} {'knock':>5s} {'fell':>4s}")  # fmt: skip
    for t in TASKS:
        for m in methods:
            rs = list(by.get((t, m), {}).values())
            if not rs:
                continue
            e = np.array([r["palm_cm"] if r["palm_cm"] is not None else np.nan for r in rs], float)
            p = np.array([stage_err(r, "pregrasp") for r in rs], float)
            print(f"{t:22s} {m:8s} {len(rs):3d} {sum(r['success_official'] for r in rs):8d} "
                  f"{sum(r['success_45'] for r in rs):5d} {np.nanmedian(e):9.2f} {np.nanpercentile(e, 90):6.2f} "
                  f"{np.nanmedian(p):7.2f} {sum(knocked_early(r) for r in rs):5d} {sum(r['fell'] is not None for r in rs):4d}")  # fmt: skip
    for m1, m0 in a.pair:
        print(f"\npaired {m1} - {m0} (same episodes):")
        for t in TASKS:
            r1, r0 = by.get((t, m1), {}), by.get((t, m0), {})
            eps = sorted(set(r1) & set(r0))
            if not eps:
                continue
            d = np.array([(r1[i]["palm_cm"] or np.nan) - (r0[i]["palm_cm"] or np.nan) for i in eps], float)
            s1 = np.array([r1[i]["success_45"] for i in eps])
            s0 = np.array([r0[i]["success_45"] for i in eps])
            rng = np.random.default_rng(0)
            bs = [np.nanmedian(d[rng.integers(len(d), size=len(d))]) for _ in range(2000)]
            print(f"  {t:22s} n {len(eps)}: entry error median diff {np.nanmedian(d):+.2f} cm "
                  f"(95 % {np.percentile(bs, 2.5):+.2f}..{np.percentile(bs, 97.5):+.2f}); "
                  f"45 s success {s1.sum()} vs {s0.sum()}, {m1} only {int((s1 & ~s0).sum())}, {m0} only {int((s0 & ~s1).sum())}")  # fmt: skip


if __name__ == "__main__":
    main()
