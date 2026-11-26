"""dev (heft_reanchor2x2_20261126, after the independent review): re-issue walks asking 1.5-5 cm lateral, by direction (+ =
left, n = left of the start heading) and by the plan's step kinds: how many moved (|moved_lat| >= half the ask, same
sign), median moved / asked; and the sign of every lateral failure."""
import glob, json, sys
from collections import defaultdict
import numpy as np
O = sys.argv[1]
rows = [json.loads(l) for fn in glob.glob(O + "/heft/runs/*.jsonl") for l in open(fn)]
g = defaultdict(list)
for r in rows:
    for k, w in enumerate(r["walks"]):
        if k == 0 or "asked_lat" not in w or w.get("moved_lat") is None:
            continue
        a, m = 100 * w["asked_lat"], 100 * w["moved_lat"]
        if not 1.5 <= abs(a) <= 5:
            continue
        kinds = tuple(sorted(set((w.get("plan") or {}).get("kinds") or [])))
        g[("left" if a > 0 else "right", kinds)].append((a, m))
print("direction, plan kinds: n, moved (>= half the ask, same sign), median moved/asked")
for key in sorted(g):
    v = g[key]
    mv = sum(1 for a, m in v if np.sign(m) == np.sign(a) and abs(m) >= 0.5 * abs(a))
    print(f"  {key[0]:5s} {str(key[1]):40s} n {len(v):3d} moved {mv:3d} ratio median {np.median([m / a for a, m in v]):+.2f}")
lf = [100 * r["err_lat_m"] for r in rows if not r.get("success") and "err_lat_m" in r and abs(r["err_lat_m"]) > 0.03]
print(f"lateral failures (|lat| > 3 cm): {len(lf)}, left (+) {sum(x > 0 for x in lf)}, right (-) {sum(x < 0 for x in lf)}")
