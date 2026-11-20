"""curobo_avoid_20261120 step 1, the figure: per episode the look's left image (the robot's own pixels blue, the target's green,
obstacle points within 1.2 cm of the robot's table level yellow, the true near edge red) and a side view of the obstacle
points within 0.3 m of the target's y (x ahead, z up; the true table box red).

    $SIMPLE/.venv/bin/python -I fig_step1.py <look dumps> <step1 dir> <out png> EP [EP ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LOOK, S1, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
eps = [int(x) for x in sys.argv[4:]]
fig, axes = plt.subplots(len(eps), 2, figsize=(15, 3.6 * len(eps)), gridspec_kw=dict(width_ratios=[1.25, 1]))
axes = np.atleast_2d(axes)
for k, e in enumerate(eps):
    name = f"TabletopGraspMP_L{e // 10}_e{e % 10}"
    L, O = np.load(LOOK / f"{name}.npz"), np.load(S1 / f"obst_{name}.npz")
    img = L["left"].astype(float) / 255
    over = img.copy()
    over[O["robot_px"]] = 0.5 * over[O["robot_px"]] + 0.5 * np.array([0.1, 0.3, 1.0])
    over[L["mask"]] = 0.5 * over[L["mask"]] + 0.5 * np.array([0.1, 0.9, 0.2])
    pts, keep, level = L["pts"], O["keep"], float(O["level"])
    y_t = float(np.median(L["cloud"][:, 1]))
    tl = keep & (np.abs(pts[..., 2] - level) < 0.012) & (np.abs(pts[..., 1] - y_t) < 0.3)
    over[tl] = 0.4 * over[tl] + 0.6 * np.array([1.0, 0.85, 0.0])
    near = (
        keep
        & (np.abs(pts[..., 0] - O["edge_true"]) < 0.015)
        & (np.abs(pts[..., 1] - y_t) < 0.3)
        & (pts[..., 2] > O["top_true"] - 0.3)
    )
    ax = axes[k, 0]
    ax.imshow(over)
    ax.set_title(
        f"ep {e} {name}: blue robot, green target, yellow table level, (cyan: points 1.5 cm about the true edge's x)",
        fontsize=8,
    )
    ax.contour(near, levels=[0.5], colors="cyan", linewidths=0.6)
    ax.axis("off")
    ax = axes[k, 1]
    P = pts[keep & L["ok"]]
    P = P[
        (np.abs(P[:, 1] - y_t) < 0.3)
        & (P[:, 0] < O["edge_true"] + 0.6)
        & (P[:, 0] > O["edge_true"] - 0.4)
        & (P[:, 2] > 0.3)
    ]
    ax.scatter(P[:, 0], P[:, 2], s=0.3, c="k", alpha=0.3)
    et, tt = float(O["edge_true"]), float(O["top_true"])
    ax.plot([et, et + 0.6, et + 0.6, et, et], [tt, tt, tt - 0.1, tt - 0.1, tt], "r-", lw=1)
    ax.axhline(level, color="orange", lw=0.6, ls="--")
    ax.set_xlim(et - 0.4, et + 0.6)
    ax.set_ylim(0.3, 1.4)
    ax.set_aspect("equal")
    ax.set_title(
        "side view, |y - target y| < 0.3 m: points (black), true table (red), robot's level (orange)", fontsize=8
    )
    ax.set_xlabel("x (m, ahead)")
    ax.set_ylabel("z (m)")
fig.tight_layout()
fig.savefig(OUT, dpi=90)
print("wrote", OUT)
