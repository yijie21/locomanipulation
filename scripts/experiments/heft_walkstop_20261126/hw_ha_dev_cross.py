"""dev (heft_walkstop HA part, after the HEFT segment): the crossings' size and timing and the toe reach in HA, per arm x
cell: crossing sizes (cm), how many end behind the line (final gap < 0) vs transient, success with a 0.5 cm crossing
tolerance, the truth toe reach change at the read (final front - pelvis - start reach) and the own-FK one after the last
walk, the signed lateral after the first walk (truth), odometry error (cm)."""
import glob, json, sys
from collections import defaultdict
import numpy as np
O = sys.argv[1]
rows = {}
for fn in glob.glob(O + "/ha_*/runs/*.jsonl"):
    for l in open(fn):
        r = json.loads(l); rows[(r["arm"], r["cid"], r["rep"])] = r
D = {0: .05, 2: .20, 4: .50, 5: 1.00}
def f1(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return f"{np.median(v):+.1f} [{min(v):+.1f}, {max(v):+.1f}]" if v else "-"
print("arm  d     n cross behind  cross sizes cm               s.5  reach truth          own rN               lat after walk 1    odo")
for arm in ("p0", "p2", "p4", "q0", "q2"):
    for cid in (0, 2, 4, 5):
        rs = [r for (a, c, _), r in rows.items() if a == arm and c == cid and "gap_m" in r]
        if not rs:
            continue
        cr = [r for r in rs if r["cross_m"] > 0]
        s5 = sum(1 for r in rs if r["cross_m"] <= 0.005 and 0 <= r["gap_m"] <= 0.05 and abs(r["err_lat_m"]) <= 0.03 and r["final"]["how"] == "settled")
        print(f"{arm:4s} {D[cid]:4.2f} {len(rs):2d} {len(cr):5d} {sum(r['gap_m'] < 0 for r in cr):6d}  {str(sorted(round(100 * r['cross_m'], 1) for r in cr)):28s} {s5:3d}  "
              f"{f1([100 * ((r['final']['front'] - r['final']['fwd']) - r['reach_true']) for r in rs]):20s} "
              f"{f1([100 * (r['walks'][-1].get('reach_own_end', np.nan) - r['reach_own']) for r in rs]):20s} "
              f"{f1([100 * r['walks'][0].get('moved_lat', np.nan) for r in rs]):18s} {f1([r['final']['odo_err_cm'] for r in rs])}")
