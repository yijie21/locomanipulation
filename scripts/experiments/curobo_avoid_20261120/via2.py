"""curobo_avoid_20261120 step 3a: a cuRobo plan (plan2.py) -> the bench's via points (simple_bench.py --avoid).

    python3 via2.py <out dir> <tol m> <plan dir> [<plan dir> ...]

Per plan_<NAME>.npz with a path (the first plan dir that has one: the pre-registered setting first, then the fallbacks): the
jaw path (world) simplified by Douglas-Peucker to within <tol>; its inner points (not the start, not the goal) with the
planned wrist rotation there (world: the plan's pelvis-frame quaternion turned by the start pelvis, prep_<NAME>.npz) ->
<out>/via_<NAME>.npz: p (k, 3), R (k, 3, 3), goal (the plan's goal jaw, for the bench's check against its own pregrasp),
src (the plan dir), max_dev (m, the simplified path's largest distance from the planned one).
"""

import sys
from pathlib import Path

import numpy as np

OUT, TOL, DIRS = Path(sys.argv[1]), float(sys.argv[2]), [Path(x) for x in sys.argv[3:]]
OUT.mkdir(parents=True, exist_ok=True)


def mat_of(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def seg_dist(X, a, b):
    ab = b - a
    u = np.clip(((X - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(X - (a + u[:, None] * ab), axis=1)


def rdp(X, tol):
    keep = [0, len(X) - 1]
    todo = [(0, len(X) - 1)]
    while todo:
        i, j = todo.pop()
        if j <= i + 1:
            continue
        d = seg_dist(X[i + 1 : j], X[i], X[j])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep.append(i + 1 + k)
            todo += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(set(keep))


def poly_dev(X, idx):  # the planned points' largest distance from the simplified polyline
    return max(float(seg_dist(X[a : b + 1], X[a], X[b]).max()) for a, b in zip(idx[:-1], idx[1:]))


names = sorted({f.name[5:-4] for d in DIRS for f in d.glob("plan_*.npz")})
for name in names:
    for d in DIRS:
        f = d / f"plan_{name}.npz"
        if f.exists() and "jaw" in np.load(f).files:
            break
    else:
        print(name, "no path")
        continue
    P = np.load(f)
    pr = np.load(DIRS[0] / f"prep_{name}.npz")
    J = P["jaw"]
    idx = rdp(J, TOL)
    inner = idx[1:-1]
    Rw = np.array([pr["pelvis_R"] @ mat_of(P["wrist_q"][k]) for k in inner]).reshape(-1, 3, 3)
    np.savez(
        OUT / f"via_{name}.npz", p=J[inner], R=Rw, goal=J[-1], src=str(d), max_dev=poly_dev(J, idx), k=np.array(inner)
    )
    print(name, "src", d.name, "points", len(J), "via", len(inner), "max_dev_cm", round(100 * poly_dev(J, idx), 2),
          "goal vs prep mm", round(1000 * float(np.linalg.norm(J[-1] - pr["goal_jaw"])), 2),
          "path cm", round(100 * float(np.linalg.norm(np.diff(J, axis=0), axis=1).sum()), 1))  # fmt: skip
