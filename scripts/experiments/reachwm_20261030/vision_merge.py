"""Two looks at one object, one cloud: the second look's points join the first's when they are the same object.

Both clouds are in the world frame the robot believes in (vision_seg.py, from the camera poses it estimated). The second look
names its object on its own, so it can name another one: its points count only if their centre lies inside the first cloud's
box grown by --grow. The joined cloud is thinned to one point per --voxel cell.

    python vision_merge.py <set>/vision --a wb2 --b wb2v --tag wb2m
-> see_<tag>_L<k>_s<j>.npz (pts, p_o, R_o), merge_<tag>.json
"""

import argparse
import glob
import json
import os
import re

import numpy as np


def thin(p: np.ndarray, voxel: float) -> np.ndarray:
    _, i = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_index=True)
    return p[np.sort(i)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--grow", type=float, default=0.04)
    ap.add_argument("--voxel", type=float, default=0.003)
    ap.add_argument("--min-points", type=int, default=30)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, f"see_{a.a}_L*_s*.npz"))):
        k, j = map(int, re.search(r"_L(\d+)_s(\d+)\.npz$", f).groups())
        z = np.load(f)
        A = z["pts"].astype(np.float64)
        row = dict(layout=k, slot=j, n_a=len(A), n_b=0, used=False)
        fb = os.path.join(a.dir, f"see_{a.b}_L{k}_s{j}.npz")
        pts = A
        if os.path.exists(fb):
            B = np.load(fb)["pts"].astype(np.float64)
            row["n_b"] = len(B)
            if len(B) >= a.min_points and len(A) >= a.min_points:
                lo, hi = np.percentile(A, 2, axis=0) - a.grow, np.percentile(A, 98, axis=0) + a.grow
                c = np.median(B, axis=0)
                row["centre_off_cm"] = float(100 * np.linalg.norm(c - np.median(A, axis=0)))
                if np.all(c >= lo) and np.all(c <= hi):
                    keep = np.all((B >= lo - a.grow) & (B <= hi + a.grow), axis=1)
                    pts, row["used"] = np.vstack([A, B[keep]]), True
            elif len(B) >= a.min_points:  # the first look found nothing: the second alone
                pts, row["used"], row["only_b"] = B, True, True
        pts = thin(pts, a.voxel) if len(pts) else pts
        row["n"] = len(pts)
        np.savez_compressed(os.path.join(a.dir, f"see_{a.tag}_L{k}_s{j}.npz"), pts=pts.astype(np.float32), p_o=z["p_o"],
                            R_o=z["R_o"])  # fmt: skip
        rows.append(row)
    json.dump(rows, open(os.path.join(a.dir, f"merge_{a.tag}.json"), "w"))
    print(len(rows), "clouds, second look used in", sum(r["used"] for r in rows))


if __name__ == "__main__":
    main()
