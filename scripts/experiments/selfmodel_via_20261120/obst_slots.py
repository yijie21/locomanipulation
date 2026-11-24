"""selfmodel_via_20261120, sixth step: gen_reach.py's REACH_VIA_SLOTS for an obst_routes.py output.

    python3 -I obst_slots.py <obst routes dir>

One slot per (base scene, margin), base scene outermost: the path's name where the planner made one, null where it failed.
Only base scenes (planned with their real boxes) get slots. The walk key is the base scene's rank, so every margin of a base
scene starts each pass from the same state. A run of 4 passes is 4 x len(names) resets.
-> <obst routes dir>/slots.json (names, walk, and the (s, 0, m) of each slot), printed: slots, planned, resets for 4 passes.
"""

import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
C = [json.load(open(p)) for p in sorted(D.glob("cands_*of*.json"))]
assert C and len(C) == C[0]["part"][1] and all(c["base"] == C[0]["base"] for c in C), (
    "every part's cands file, same draws"
)
plans = {(r["s"], r["m"]): r for c in C for r in c["plans"]}
margins = C[0]["margins"]
base = [s for s in range(C[0]["n"]) if plans[(s, 0)]["ok"]]
names, walk, key = [], [], []
for k, s in enumerate(base):
    for m in margins:
        r = plans[(s, m)]
        names.append(r["name"] if r["ok"] and (D / "via" / f"via_{r['name']}.npz").exists() else None)
        walk.append(k)
        key.append([s, 0, m])
json.dump(dict(names=names, walk=walk, key=key, levels=[0], margins=margins), open(D / "slots.json", "w"))
ok = sum(n is not None for n in names)
print(f"{len(base)} base scenes, {len(names)} slots, {ok} planned; 4 passes: {4 * len(names)} resets, {4 * ok} reaches")
