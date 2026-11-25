"""Raw per-task, per-method summary of ha_bench.py rows (plain python, json only): successes, falls, plan rates, hand
errors at the reach's end, stand errors, where each episode ended, and the failure-analysis diagnostics.

    python3 smoke_summary.py <bench out dir>
"""

import collections
import glob
import json
import statistics
import sys


def main() -> int:
    out = sys.argv[1]
    rows = []
    for f in sorted(glob.glob(f"{out}/*.jsonl")):
        rows += [json.loads(line) for line in open(f)]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["task"], r["method"])].append(r)
    for (task, method), rs in sorted(by.items()):
        rs.sort(key=lambda r: (r["seed"], r["repeat"]))
        reach = [r["reach"] for r in rs if "reach" in r]
        errs = [x["hand_err_cm"] for x in reach]
        plan_ok = sum(1 for x in reach if x.get("plan_ok"))
        fb = sum(1 for x in reach if x.get("fallback"))
        print(f"== {task} {method}: n={len(rs)} success={sum(r['success'] for r in rs)} fell={sum(r['fell'] for r in rs)} "
              f"reached={len(reach)} plan_ok={plan_ok if method == 'J' else '-'} fallback={fb if method == 'J' else '-'}")  # fmt: skip
        if errs:
            print(f"   hand err cm: median {statistics.median(errs):.1f}  min {min(errs):.1f}  max {max(errs):.1f}  "
                  f"<=3cm {sum(e <= 3 for e in errs)}/{len(errs)}  <=5cm {sum(e <= 5 for e in errs)}/{len(errs)}")  # fmt: skip
        ends = collections.Counter(f"{r['end']}|{r.get('sequence')}|{r['phase_at_end']}" for r in rs)
        print("   ends:", dict(ends))
        for r in rs:
            x = r.get("reach", {})
            pc = r.get("perceived", {})
            print(f"   s{r['seed']} r{r['repeat']}: end={r['end']} seq={r.get('sequence')} phase={r['phase_at_end']} "
                  f"steps={r['steps']} succ@{r['success_step']} fell@{r['fell_step']} | goal={pc.get('pregrasp')} "
                  f"stand_err={pc.get('stand_err')} | err={x.get('hand_err_cm')} rot={x.get('hand_rot_err_deg')} "
                  f"plan={x.get('plan_status', '-') if method == 'J' else '-'} t_plan={x.get('t_plan', '-')} "
                  f"dur={x.get('dur_s')} | diag={r.get('diag')}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
