"""TODO 23.2.6: the online check of the IK presort -- round 0 of every listed batch (same object yaw): planning
seconds, lift, whether the same cached grasp was chosen as in the first batch. Default batches: _cache and _presort_b
(presort off), _presort and _presort_b (on) -- two of each, so the off/on gap can be read against the off/off and
on/on gaps (the A/A noise floor).

    python3 scripts/experiments/graspnet_bottle_20261008/presort_compare.py [_cache _cache_b _presort _presort_b]
"""

import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
objects = json.loads((REPO / "outputs/graspnet_bottle_20261008/objects.json").read_text())["objects"]
TAGS = sys.argv[1:] or ["_cache", "_cache_b", "_presort", "_presort_b"]
pairs = []
for o in objects:
    recs = []
    for tag in TAGS:
        f = REPO / "outputs/runs" / f"gnb_obj_{o['id']:03d}{tag}" / "grasp_planned/round_00_ik/record.json"
        recs.append(json.loads(f.read_text()) if f.exists() else None)
    if all(recs):
        pairs.append((o["id"], *recs))


def plan_s(r):
    return (r.get("plan") or {}).get("plan_seconds")


def row_of(r):
    return (r.get("grasp") or {}).get("row")


print(f"objects with all {len(TAGS)} rounds: {len(pairs)}")
for i, name in enumerate(TAGS, start=1):
    rs = [p[i] for p in pairs]
    secs = [plan_s(r) for r in rs if plan_s(r) is not None]
    print(
        f"{name}: lifted {sum(1 for r in rs if r.get('ok'))}/{len(rs)}; plan s median {st.median(secs):.1f} "
        f"p90 {sorted(secs)[int(0.9 * len(secs))]:.1f} max {max(secs):.1f} (n {len(secs)}); "
        f"plan ok {sum(1 for r in rs if (r.get('plan') or {}).get('ok'))}; "
        f"rejected median {st.median(len((r.get('plan') or {}).get('rejected') or []) for r in rs)}"
    )
for i in range(len(TAGS)):
    for j in range(i + 1, len(TAGS)):
        a_ok = [bool(p[i + 1].get("ok")) for p in pairs]
        b_ok = [bool(p[j + 1].get("ok")) for p in pairs]
        same = sum(1 for p in pairs if row_of(p[i + 1]) is not None and row_of(p[i + 1]) == row_of(p[j + 1]))
        both = sum(1 for p in pairs if row_of(p[i + 1]) is not None and row_of(p[j + 1]) is not None)
        print(
            f"{TAGS[i]} vs {TAGS[j]}: lifted {sum(a_ok)} vs {sum(b_ok)}; flips {sum(a and not b for a, b in zip(a_ok, b_ok))} -> fail, "
            f"{sum(b and not a for a, b in zip(a_ok, b_ok))} -> lift; same grasp {same}/{both}"
        )
for i, name in enumerate(TAGS, start=1):
    pre = [(p[i].get("plan") or {}).get("presort") for p in pairs]
    pre = [x for x in pre if x]
    if pre:
        print(
            f"{name}: presort seconds median {st.median(x['seconds'] for x in pre):.2f}; feasible median {st.median(x['feasible'] for x in pre)} of {st.median(x['candidates'] for x in pre)}"
        )
