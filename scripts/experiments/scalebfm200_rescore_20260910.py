"""Re-score the 2026-09-10 no-retract closed loop under the 2026-09-09 success rule.

41/100 (sonic_lift6cm_hold2s_v1) and the ledger's 25/100 (section 9) were produced by two
different scoring rules, so the raw difference mixes the dataset change with the rule change.
Commit 2ba1a89 is the rule change: it removed the "bottle knocked before close" abort and the
phase/tilt/velocity gates, and moved success from "rise >= 0.08 m held 1.0 s in the carry phase"
to "rise > 0.06 m held > 2.0 s in any phase".

This script re-scores the SAME saved traces four ways. Velocity is not in object_audit.json, so
the old rule is reproduced without its |v| < 0.06 term -- that term can only make the old count
smaller, so the "old rule" column here is an upper bound on what the old rule would have scored.
"""

import collections
import glob
import json
import sys
from pathlib import Path

DT = 0.02  # object_audit rows are one control tick apart


def held(rows, threshold, hold_s, tilt_max=None):
    start = None
    for r in rows:
        ok = r["rise_m"] > threshold and (tilt_max is None or r["tilt_deg"] <= tilt_max)
        if not ok:
            start = None
            continue
        if start is None:
            start = r["t"]
        if r["t"] - start > hold_s + 1e-9:
            return True
    return False


def main(root):
    root = Path(root)
    rows = []
    for path in sorted(glob.glob(str(root / "seed*/result.json"))):
        case = json.loads(Path(path).read_text())
        audit = json.loads((Path(path).parent / "object_audit.json").read_text())
        fm = case.get("first_move") or {}
        knocked = bool(fm) and bool(fm.get("before_close"))
        rows.append(
            dict(
                seed=case["seed"],
                stratum=case["stratum"],
                new_ok=bool(case["ok"]),
                why=case["why"],
                knocked_before_close=knocked,
                # the 2026-09-09 gate, minus its |v| < 0.06 term (upper bound)
                old_lift=held(audit, 0.08, 1.0, tilt_max=30.0),
                new_lift=held(audit, 0.06, 2.0),
            )
        )
    for r in rows:
        r["old_ok"] = r["old_lift"] and not r["knocked_before_close"]

    def tally(key):
        by = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            s = by[r["stratum"]]
            s[1] += 1
            s[0] += bool(r[key])
        return {k: f"{v[0]}/{v[1]}" for k, v in sorted(by.items())}, sum(bool(r[key]) for r in rows)

    report = dict(
        cases=len(rows),
        new_protocol=dict(total=tally("new_ok")[1], by_stratum=tally("new_ok")[0]),
        old_protocol_upper_bound=dict(total=tally("old_ok")[1], by_stratum=tally("old_ok")[0]),
        old_lift_gate_only=dict(total=tally("old_lift")[1], by_stratum=tally("old_lift")[0]),
        knocked_before_close=dict(
            total=sum(r["knocked_before_close"] for r in rows),
            among_new_successes=sum(r["knocked_before_close"] and r["new_ok"] for r in rows),
        ),
        disagreements=dict(
            new_ok_old_fail=sum(r["new_ok"] and not r["old_ok"] for r in rows),
            old_ok_new_fail=sum(r["old_ok"] and not r["new_ok"] for r in rows),
        ),
        caveat="old_protocol_upper_bound omits the |v| < 0.06 term of the 2026-09-09 gate",
        reference="ledger scalebfm.md section 9: 25/100 (high 10/34, low_near 10/33, low_deep 5/33)",
    )
    (root.parent / "RESCORE.json").write_text(json.dumps(dict(report=report, cases=rows), indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
