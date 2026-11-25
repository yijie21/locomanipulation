"""Summarise look_compare.py output: image / depth / SAM 3 differences of the lazy one-shot render (and of warm-up renders) against
the continuous render, with the renderer's own render-to-render noise as the baseline."""

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
ks = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "0,1,4,8,16,32,48")]
rows = []
for f in sorted(glob.glob(str(root / "*" / "*_cont_0.npz"))):
    g = f.replace("_cont_0", "_lazy_0")
    if not Path(g).exists():
        continue
    try:
        c, l = np.load(f), np.load(g)
        rc, rl = json.loads(str(c["rec"])), json.loads(str(l["rec"]))
    except Exception:
        continue
    task = Path(f).parent.name
    cl, ll = c["left"].astype(float), l["left"].astype(float)
    cd, ld = c["depth"], l["depth"]
    row = dict(
        task=task,
        id=rc["id"],
        noise=float(np.abs(cl[3] - cl[4]).mean()),
        n_obj_c=rc["n_obj"],
        n_obj_l=rl["n_obj"],
        parts_c=len(rc["parts"]),
        parts_l=len(rl["parts"]),
    )
    for k in ks:
        if k < len(ll):
            row[f"mad{k}"] = float(np.abs(ll[k] - cl[0]).mean())
            row[f"dep{k}"] = float(np.nanmedian(np.abs(ld[k] - cd[0])))
    # first art() is the lazy one-shot (warm 0): parts matched by nearest centre
    if rc["parts"] and rl["parts"]:
        pc = np.array([p["p"] for p in rc["parts"]])
        pl = np.array([p["p"] for p in rl["parts"]])
        row["part_cm"] = float(100 * np.median([np.linalg.norm(pc - p, axis=1).min() for p in pl]))
    rows.append(row)
print(f"{'task':16} {'id':20} noise " + " ".join(f"mad{k:<3}" for k in ks) + " | n_obj c/l  parts c/l  part_cm")
for r in rows:
    print(
        f"{r['task']:16} {r['id']:20} {r['noise']:.2f} "
        + " ".join(f"{r.get(f'mad{k}', float('nan')):6.2f}" for k in ks)
        + f" | {r['n_obj_c']:6d}/{r['n_obj_l']:<6d} {r['parts_c']}/{r['parts_l']}  {r.get('part_cm', float('nan')):.1f}"
    )
by = defaultdict(list)
for r in rows:
    by[r["task"]].append(r)
print("\nmean over ids (task):")
for t, rs in by.items():
    print(
        t,
        "noise %.2f" % np.mean([r["noise"] for r in rs]),
        " ".join(f"mad{k}={np.mean([r[f'mad{k}'] for r in rs if f'mad{k}' in r]):.2f}" for k in ks),
        " depth_med_abs(k=0,48): %.4f %.4f"
        % (np.mean([r["dep0"] for r in rs]), np.mean([r.get("dep48", np.nan) for r in rs])),
    )
