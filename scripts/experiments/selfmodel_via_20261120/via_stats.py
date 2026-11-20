"""The shapes of the cuRobo paths the bench ran (via2.py's via_<NAME>.npz + prep2.py's prep_<NAME>.npz), in the reach
start's heading frame (x ahead, y left, z up), so a generator of random paths can be checked against them.
    python -I via_stats.py <via dir> <prep dir> [<via dir> <prep dir> ...]"""

import glob
import sys
from pathlib import Path

import numpy as np


def ang(Ra, Rb):
    return np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


rows = []
args = sys.argv[1:]
for vd, pd in zip(args[::2], args[1::2]):
    for f in sorted(glob.glob(str(Path(vd) / "via_*.npz"))):
        name = Path(f).stem[4:]
        pf = Path(pd) / f"prep_{name}.npz"
        if not pf.exists():
            continue
        v, p = np.load(f), np.load(pf, allow_pickle=True)
        Rp = p["pelvis_R"]
        yaw = np.arctan2(Rp[1, 0], Rp[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        H = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])  # world -> heading frame
        o = p["pelvis_p"]
        a, b = H @ (p["jaw_p"] - o), H @ (v["goal"] - o)
        pts = np.vstack([a, (H @ (v["p"] - o).T).T, b])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        ab = b - a
        u = np.clip(((pts - a) @ ab) / (ab @ ab), 0, 1)
        off = pts - (a + u[:, None] * ab)
        i = int(np.argmax(np.linalg.norm(off, axis=1)))
        Rs = [p["wrist_R"]] + list(v["R"])
        turn = sum(ang(Rs[k], Rs[k + 1]) for k in range(len(Rs) - 1))
        rows.append(
            dict(
                name=name,
                n_via=len(v["p"]),
                straight=np.linalg.norm(ab),
                length=seg.sum(),
                bulge=np.linalg.norm(off[i]),
                up=off[i, 2],
                side=off[i, 1],
                back=off[i, 0],
                z_max_over_goal=pts[:, 2].max() - b[2],
                start=a,
                goal=b,
                wrist_turn=turn,
                first_up=pts[1, 2] - a[2],
            )
        )
print("paths", len(rows))
for k in ("n_via", "straight", "length", "bulge", "up", "side", "back", "z_max_over_goal", "first_up", "wrist_turn"):
    x = np.array([r[k] for r in rows], float)
    sc = 1 if k in ("n_via", "wrist_turn") else 100
    unit = "" if k == "n_via" else " deg" if k == "wrist_turn" else " cm"
    print(
        f"{k:16s} min {sc * x.min():7.1f}  p10 {sc * np.percentile(x, 10):7.1f}  median {sc * np.median(x):7.1f}  "
        f"p90 {sc * np.percentile(x, 90):7.1f}  max {sc * x.max():7.1f}{unit}"
    )
S, G = np.array([r["start"] for r in rows]), np.array([r["goal"] for r in rows])
print("start (pelvis heading frame) cm min", np.round(100 * S.min(0)), "max", np.round(100 * S.max(0)))
print("goal  (pelvis heading frame) cm min", np.round(100 * G.min(0)), "max", np.round(100 * G.max(0)))
by = {}
for r in rows:
    by.setdefault(r["name"].split("_")[0], []).append(r)
for t, rs in by.items():
    print(
        f"{t:22s} n {len(rs):3d}  bulge median {100 * np.median([r['bulge'] for r in rs]):5.1f} cm  "
        f"up {100 * np.median([r['up'] for r in rs]):5.1f}  side {100 * np.median([r['side'] for r in rs]):5.1f}  "
        f"length/straight {np.median([r['length'] / r['straight'] for r in rs]):4.2f}"
    )
