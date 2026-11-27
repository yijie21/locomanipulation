"""cmp_v3a.py <v3a runs> <rerun dir>: the final version's bit check (README section 2). Read only. notebook: skip figure
render, not an experiment.

Per placement and arm, the rerun against v3a's record: the scene XML's MD5; every snapshot array v3a has (pregrasp_end,
entry_end, close_end: the whole state; the goals) equal to the bit; every film array equal; the row equal except the
elapsed times (wall_s, think_s) and the rerun's added snapshot names (fig.snaps); the close distance (|palm - goal| at
the close command) and the four classes from both. Prints every difference; exit 1 when there is one.
"""

import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OLD, NEW = Path(sys.argv[1]), Path(sys.argv[2])
IGNORE = {"wall_s", "think_s"}
diffs = []


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def rows(d, arm):
    out = {}
    for f in sorted(glob.glob(str(d / "rows" / f"{arm}_*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            out[int(r["index"])] = r
    return out


def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in IGNORE or (path.endswith(".fig") and k == "snaps"):
                continue
            if k not in a or k not in b:
                diffs.append((path + "." + k, "missing in " + ("old" if k not in a else "new")))
                continue
            walk(a[k], b[k], path + "." + k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append((path, f"length {len(a)} vs {len(b)}"))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif a != b:
        diffs.append((path, f"{a!r} vs {b!r}"))


def cls(a, r, thr=2.0):
    return "A miss R hit" if a > thr and r <= thr else ("both hit" if a <= thr and r <= thr else ("both miss" if a > thr else "A hit R miss"))


n_arr = n_film = n_rows = 0
dist = {"old": {}, "new": {}}
for arm in ("A", "R"):
    ro, rn = rows(OLD, arm), rows(NEW, arm)
    if sorted(ro) != sorted(rn):
        diffs.append((arm, f"placements {sorted(ro)} vs {sorted(rn)}"))
    for i in sorted(ro):
        tag = f"{arm} p{i:02d}"
        so, sn = OLD / "scene" / f"{arm}_p{i:02d}" / "scene_locomani.xml", NEW / "scene" / f"{arm}_p{i:02d}" / "scene_locomani.xml"
        if md5(so) != md5(sn):
            diffs.append((tag, "scene md5"))
        zo, zn = np.load(OLD / "snap" / f"{arm}_p{i:02d}.npz"), np.load(NEW / "snap" / f"{arm}_p{i:02d}.npz")
        for k in zo.files:
            n_arr += 1
            if k not in zn.files or zo[k].shape != zn[k].shape or not np.array_equal(zo[k], zn[k]):
                diffs.append((tag, f"snap {k}"))
        fo = OLD / "film" / arm / f"ShelfReach_L0_e{i}_truth_direct.npz"
        fn = NEW / "film" / arm / f"ShelfReach_L0_e{i}_truth_direct.npz"
        a, b = np.load(fo), np.load(fn)
        if sorted(a.files) != sorted(b.files):
            diffs.append((tag, "film keys"))
        for k in a.files:
            n_film += 1
            if k not in b.files or a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
                diffs.append((tag, f"film {k}"))
        walk(ro[i], rn[i], tag)
        n_rows += 1
        for lab, z in (("old", zo), ("new", zn)):
            dist[lab].setdefault(i, {})[arm] = 100 * float(np.linalg.norm(z["entry_end__palm"] - z["goal__entry"]))
print(f"compared: {n_arr} snapshot arrays, {n_film} film arrays, {n_rows} rows (40 = 20 placements x 2 arms)")
print(f"{'p':>3} {'A old':>9} {'A new':>9} {'R old':>9} {'R new':>9}  class old | new")
same_cls = True
for i in sorted(dist["old"]):
    o, n = dist["old"][i], dist["new"][i]
    co, cn = cls(o["A"], o["R"]), cls(n["A"], n["R"])
    same_cls &= co == cn
    print(f"{i:3d} {o['A']:9.5f} {n['A']:9.5f} {o['R']:9.5f} {n['R']:9.5f}  {co} | {cn}" + ("" if (o == n and co == cn) else "   <-- DIFFERS"))
print("close distances bit-equal:", all(dist["old"][i] == dist["new"][i] for i in dist["old"]), "| four classes equal:", same_cls)
print("differences:", len(diffs))
for d in diffs[:50]:
    print("  ", d)
sys.exit(1 if diffs or not same_cls else 0)
