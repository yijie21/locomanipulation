"""selfmodel_via_20261120, fourth step, part two: gen_reach.py's REACH_VIA_SLOTS for a gap_routes.py output.

    python3 -I gap_slots.py <gap routes dir> [--margins 1,2,...] [--levels 3,5,...]

One slot per (base scene, level, margin), base scene outermost, then level, then margin; the path's name where the planner
made one, null where it failed; the walk key is the base scene, so every level and margin of a base scene starts each pass
from the same state. A run of 4 passes is 4 x len(names) resets.
-> <gap routes dir>/slots.json (names, walk, and the (s, f, m) of each slot), printed: slots, planned, resets for 4 passes.
"""

import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
ARGS = sys.argv[2:]
C = [json.load(open(p)) for p in sorted(D.glob("cands_*of*.json"))]
assert C and len({c["n"] for c in C}) == 1 and sum(1 for _ in C) == C[0]["part"][1], "every part's cands file"
assert all(c["base"] == C[0]["base"] for c in C), "the parts drew different base scenes"
plans = {(r["s"], r["f"], r["m"]): r for c in C for r in c["plans"]}
levels = [int(x) for x in ARGS[ARGS.index("--levels") + 1].split(",")] if "--levels" in ARGS else C[0]["levels"]
margins = [int(x) for x in ARGS[ARGS.index("--margins") + 1].split(",")] if "--margins" in ARGS else C[0]["margins"]
names, walk, key = [], [], []
for s in range(C[0]["n"]):
    for f in levels:
        for m in margins:
            r = plans[(s, f, m)]
            names.append(r["name"] if r["ok"] and (D / "via" / f"via_{r['name']}.npz").exists() else None)
            walk.append(s)
            key.append([s, f, m])
json.dump(dict(names=names, walk=walk, key=key, levels=levels, margins=margins), open(D / "slots.json", "w"))
ok = sum(n is not None for n in names)
print(f"{len(names)} slots, {ok} planned; 4 passes: {4 * len(names)} resets, {4 * ok} reaches")
