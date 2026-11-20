"""TabletopGrasp episodes whose hand got over the table's edge: where the grasp sat on the can, old look against new.

python3 -I near_grasp.py <simple_bench_20261119 dir>
"""

import glob
import json
import statistics as st
import sys

B = sys.argv[1]


def rows(d):
    out = {}
    for f in glob.glob(f"{B}/{d}/direct_TabletopGraspMP_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[r["ep"]] = r
    return out


def stage(r, n):
    s = [x for x in r.get("stages") or [] if x["stage"] == n]
    return s[0] if s else None


for d in ("rows_perc", "rows_eyes"):
    R = rows(d)
    print(f"== {d}: episodes whose hand got over the edge (pregrasp <= 20 cm)")
    print(" ep ok  obj_from_grasp cm (x ahead, y left, z up)  seen_cm  entry_err  entry miss world cm   knocked")
    zs, ok_n = [], 0
    for e in range(30):
        r = R[e]
        p = stage(r, "pregrasp")
        if p is None or p["err_cm"] > 20:
            continue
        en = stage(r, "entry")
        k = r.get("knocked") or {}
        zs.append(r["obj_from_grasp"][2])
        ok_n += r["success_45"]
        print(f"{e:3d} {int(r['success_45'])}   {str(r['obj_from_grasp']):30s}  {str(r.get('seen_cm')):6s}  "
              f"{str(en['err_cm'] if en else None):8s}   {str(en['miss'] if en else ''):22s} {k.get('phase')} {k.get('touch')}")  # fmt: skip
    print(f"   n {len(zs)}, success {ok_n}, obj_from_grasp z median {st.median(zs):.1f} cm\n")
