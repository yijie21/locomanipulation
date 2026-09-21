"""(h × d) panels of the measured x boundary over y × yaw, from a search run's boundary_table.json.

python -m locomani.tasks.crate.reach_envelope.plot_boundary <run_dir>
"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
RUN = Path(sys.argv[1]).resolve()
rows = json.load(open(RUN / "boundary_table.json"))
grid = json.load(open(RUN / "search.json"))["grid"]
H, D, Y, YAW = tuple(grid["h"]), tuple(grid["d"]), tuple(grid["y"]), tuple(grid["yaw"])
n_trials = sum(r["trials"] for r in rows)


def xb(r):
    if r["x_pass"] is None:
        return np.nan
    return r["x_pass"] + 0.025 if r["x_fail"] is not None else r["x_pass"]


cmap = LinearSegmentedColormap.from_list("blue", ["#dbe9f6", "#1f4e8c"])
fig, axes = plt.subplots(len(H), len(D), figsize=(4.2 * len(D), 3.5 * len(H)), sharex=True, sharey=True, squeeze=False)
for i, h in enumerate(H):
    for j, d in enumerate(D):
        ax = axes[i, j]
        M = np.full((len(Y), len(YAW)), np.nan)
        for a, y in enumerate(Y):
            for b, yaw in enumerate(YAW):
                r = next(r for r in rows if (r["y"], r["h"], r["d"], r["yaw"]) == (y, h, d, yaw))
                M[a, b] = xb(r)
        ax.imshow(M, cmap=cmap, vmin=0.20, vmax=0.65, aspect="auto", origin="lower")
        for a in range(len(Y)):
            for b in range(len(YAW)):
                v = M[a, b]
                if np.isnan(v):
                    ax.add_patch(
                        plt.Rectangle(
                            (b - 0.5, a - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#8a8f98", linewidth=0
                        )
                    )
                    ax.text(b, a, "×", ha="center", va="center", color="#6b7280", fontsize=12)
                else:
                    ax.text(
                        b,
                        a,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        color="white" if v > 0.47 else "#1f2937",
                        fontsize=10.5,
                    )
        ax.set_title(f"h = {h:.2f} m    d = {d:.2f} m", fontsize=11, pad=6)
        ax.set_xticks(range(len(YAW)))
        ax.set_xticklabels([f"{v:+.0f}°" for v in YAW])
        ax.set_yticks(range(len(Y)))
        ax.set_yticklabels([f"{v:+.2f}" for v in Y])
        ax.tick_params(length=0, labelsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
for ax in axes[-1]:
    ax.set_xlabel("连线偏角 yaw", fontsize=10)
for ax in axes[:, 0]:
    ax.set_ylabel("横向偏移 y (m)", fontsize=10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0.20, 0.65))
sm.set_array([])
cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02)
cb.set_label("最远可达 x（双手位置误差 ≤ 2 cm 的边界，m）", fontsize=10)
cb.outline.set_visible(False)
fig.suptitle(
    f"G1 双臂前伸边界 · {len(rows)} 条线 / {n_trials} 次实测 · 双脚不动 · 姿态误差只记录不筛选（× = 0.20 m 处仍失败）",
    fontsize=12,
    y=0.995,
)
fig.savefig(RUN / "boundary_slices.png", dpi=130, bbox_inches="tight")
print(RUN / "boundary_slices.png")
