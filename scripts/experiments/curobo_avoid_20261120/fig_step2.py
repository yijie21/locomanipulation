"""curobo_avoid_20261120 step 2, the figure: per episode a side view (x ahead, z up) of the obstacle voxels within 12 cm of
the right hand's start y (grey; the dropped ones, start pose and target, light red), the true table slab (red), the planned
jaw path (blue; the direct run's jaw path from the trace, orange), the start's and the goal's jaw (dots).

    $SIMPLE/.venv/bin/python -I fig_step2.py <step1 dir> <step2 dir> <out png> EP [EP ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

S1, S2, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
eps = [int(x) for x in sys.argv[4:]]
fig, axes = plt.subplots(1, len(eps), figsize=(5.2 * len(eps), 5.2))
axes = np.atleast_1d(axes)
for ax, e in zip(axes, eps):
    name = f"TabletopGraspMP_L{e // 10}_e{e % 10}"
    O, pr = np.load(S1 / f"obst_{name}.npz"), np.load(S2 / f"prep_{name}.npz")
    pl = S2 / f"plan_{name}.npz"
    P = np.load(pl) if pl.exists() else None
    V, vs = O["vox"], float(O["vox_size"])
    y0 = float(pr["jaw_p"][1])
    sel = np.abs(V[:, 1] - y0) < 0.12
    ax.scatter(V[sel, 0], V[sel, 2], s=14, marker="s", c="0.55", alpha=0.35, lw=0)
    c, h = pr["table_pos"], 0.5 * pr["table_size"]
    ax.add_patch(plt.Rectangle((c[0] - h[0], c[2] - h[2]), 2 * h[0], 2 * h[2], fill=False, ec="r", lw=1.2))
    if len(pr["path"]):
        ax.plot(pr["path"][:, 0], pr["path"][:, 2], "-", c="orange", lw=1.2, label="direct jaw (trace)")
    if P is not None and "jaw" in P.files:
        ax.plot(P["jaw"][:, 0], P["jaw"][:, 2], "-", c="b", lw=1.5, label="cuRobo jaw")
    ax.plot(*pr["jaw_p"][[0, 2]], "ko", ms=5, label="start jaw")
    ax.plot(*pr["goal_jaw"][[0, 2]], "g*", ms=10, label="pregrasp goal")
    ok = bool(P["ok"]) if P is not None and "ok" in P.files else None
    ax.set_title(f"ep {e}: plan {'ok' if ok else 'FAIL'}; voxels |y - hand y| < 12 cm", fontsize=9)
    ax.set_xlim(c[0] - h[0] - 0.25, c[0] - h[0] + 0.45)
    ax.set_ylim(0.45, 1.15)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, ahead)")
    ax.set_ylabel("z (m)")
    ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(OUT, dpi=90)
print("wrote", OUT)
