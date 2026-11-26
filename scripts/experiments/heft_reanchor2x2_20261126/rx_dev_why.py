"""dev (heft_reanchor2x2_20261126 results): why each failed trial failed -- every success condition that is false
(cross: feet crossed; gap<0 / gap>5; lat>3: |pelvis lateral error| > 3 cm; unsettled: the last wait timed out; fell;
timeout), per arm x cell, and the |lat| values of the lateral failures."""
import glob, json, sys
from collections import Counter, defaultdict
O = sys.argv[1]
CELLS = ((0.05, 0.0), (0.10, 0.0), (0.20, 0.0), (0.30, 0.0), (0.50, 0.0), (1.00, 0.0), (1.50, 0.0),
         (0.20, 0.20), (0.20, -0.20), (1.00, 0.20), (1.00, -0.20), (2.00, 0.0), (3.00, 0.0))
rows = {}
for fn in sorted(glob.glob(f"{O}/heft/runs/*.jsonl")):
    for line in open(fn):
        r = json.loads(line); rows[(r["arm"], r["cid"], r["rep"])] = r
by = defaultdict(list)
for k, r in rows.items():
    by[(k[0], k[1])].append(r)
print("arm cell: n_fail, reasons (a trial can have several), |lat| cm of the lat>3 failures, lateral err of every trial (median, signed)")
for arm in ("p2", "p2t", "p4", "p4t"):
    for cid in range(13):
        rs = by[(arm, cid)]
        bad = [r for r in rs if not r.get("success")]
        if not bad:
            continue
        why = Counter(); lats = []
        for r in bad:
            if r.get("fell") is not None: why["fell"] += 1; continue
            if r.get("timeout"): why["timeout"] += 1
            if r["cross_m"] > 0: why["cross"] += 1
            if r["gap_m"] < 0: why["gap<0"] += 1
            if r["gap_m"] > 0.05: why["gap>5"] += 1
            if abs(r["err_lat_m"]) > 0.03: why["lat>3"] += 1; lats.append(round(100 * r["err_lat_m"], 1))
            if r["final"]["how"] != "settled": why["unsettled"] += 1
        import numpy as np
        d, lat = CELLS[cid]
        print(f"  {arm:4s} {d:.2f} {lat:+.2f}: fail {len(bad)} {dict(why)} lat {sorted(lats)} | all lat median {np.median([100 * r['err_lat_m'] for r in rs]):+.1f}")
# lateral-only failures: success except |lat|
print("\nfailures whose ONLY false condition is |lat| > 3 cm, per arm (cells 0-10 / L cells 5,6,9,10):")
for arm in ("p2", "p2t", "p4", "p4t"):
    lo = [k for k, r in rows.items() if k[0] == arm and not r.get("success") and r.get("fell") is None and not r.get("timeout") and r["cross_m"] <= 0
          and 0 <= r["gap_m"] <= 0.05 and r["final"]["how"] == "settled" and abs(r["err_lat_m"]) > 0.03]
    print(f"  {arm}: {sum(k[1] <= 10 for k in lo)} / {sum(k[1] in (5, 6, 9, 10) for k in lo)}")
