"""The --trace rows of diag_tabletop_eyes: where the hand went in the pregrasp reach and what of the robot touched the table.

PYTHONPATH=src $PY -I trace_table.py <diag dir>     (from the repo root)
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.cwd() / "scripts/experiments/simple_eval_20261112"))
import episodes as EP  # noqa: E402

eps = EP.episodes("TabletopGraspMP")
for f in sorted(glob.glob(f"{sys.argv[1]}/direct_*.jsonl")):
    r = json.loads(open(f).readline())
    ep = eps[r["ep"]]
    cfg = ep["cfg"]
    tabs = []
    for k, a in cfg["layout"]["actors"].items():
        if a.get("uid") == "box":
            p, s = a["pose"]["position"], a["size"]
            tabs.append((k, round(p[2] + 0.5 * s[2], 3), round(p[0] - 0.5 * s[0], 3)))
    tr = [t for t in r["trace"] if t[1] == "pregrasp"]
    pre = [s for s in r["stages"] if s["stage"] == "pregrasp"][0]
    print(f"\nep {r['ep']}: pregrasp miss {pre['err_cm']} cm, tables (name, top z, front edge x) {tabs}")
    first = next((t for t in tr if t[5]), None)
    if first:
        print(f"  first robot-table contact t={first[0]} s, links {first[5]}, jaw mm {first[2]}")
    links = sorted({n for t in tr for n in t[5]})
    print(f"  links that touched the table during the pregrasp: {links}")
    n = len(tr)
    for i in (0, n // 8, n // 4, 3 * n // 8, n // 2, 3 * n // 4, n - 1):
        t = tr[i]
        print(f"  t={t[0]:6.2f}  jaw mm {t[2]}  on table {t[5]}")
    print(f"  wanted pregrasp (jaw) z ~ {round(1000 * (r['grasp_h'][2] + 0.08))} mm; grasp_h {r['grasp_h']}")
