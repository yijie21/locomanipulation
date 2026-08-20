"""Visualize plan_core_stand's squat-depth scan as an animation.

Left panel (world frame, side view): the sweet-core silhouette rides down
with the pelvis as the squat depth increases; the fixed horizontal line is
the can's grasp height. Right panel (pelvis frame, top view): the core
slice that currently sits at grasp height, with the three constraints
overlaid — scatter margin (darker green), the 33-deg ego cone, and the
0.28 m servo depth cap — and the far-band aim cells starred. The scan
keeps the layer whose usable cells reach farthest; the closing frames
freeze on the winner.

Usage: python src/teleop_free/diag_plan_scan.py   (offline, no stack)
Output: outputs/videos/plan_scan.mp4
"""

from __future__ import annotations

import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_map as rmm  # noqa: E402

OUT = "/workspace/code/locomanipulation/outputs/videos/plan_scan.mp4"
TMP = "/tmp/claude-1002/plan_scan_frames"

GRIP_Z = 0.707          # world grasp height of the can
PELVIS = 0.793          # standing pelvis height
AIM_ANG = np.deg2rad(33.0)
R_CAP = 0.28
MIN_MARGIN = 0.05

# Nested hierarchy -> one sequential green ramp + a labelled gold accent.
C_ENV = "#DDE9DD"       # envelope slice (context)
C_CORE = "#9CC7A4"      # sweet-core cells
C_MARG = "#3E8D57"      # margin-passing core cells
C_AIM = "#D9922E"       # far-band aim cells (starred + labelled)
INK, INK2 = "#1F2933", "#5B6570"


def main() -> None:
    os.makedirs(TMP, exist_ok=True)
    rm = rmm.ReachMap()
    core = rm._grid("right", core=True)
    env = rmm._close6(rm._grid("right", core=False))
    vox, org = rm.vox, rm.origin
    obj_z_pf = GRIP_Z - PELVIS

    # Side-view silhouette of the core (x extent per z layer, y-projected).
    sil = []
    for k in range(core.shape[2]):
        occ = np.argwhere(core[:, :, k])
        if len(occ):
            xs = org[0] + (occ[:, 0] + 0.5) * vox
            sil.append((org[2] + (k + 0.5) * vox, xs.min(), xs.max()))
    sil = np.asarray(sil)

    def layer(dz):
        k = int((obj_z_pf + dz - org[2]) / vox)
        if not (0 <= k < core.shape[2]):
            return None
        cocc = np.argwhere(core[:, :, k])
        eocc = np.argwhere(env[:, :, k])
        if len(cocc) == 0:
            return None
        eset = {(int(i), int(j)) for i, j in eocc}

        def marg(i, j):
            for r in range(1, 7):
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        if (max(abs(di), abs(dj)) == r
                                and (i + di, j + dj) not in eset):
                            return (r - 1) * vox
            return 6 * vox
        cpts = cocc * vox + org[:2] + vox / 2.0
        epts = eocc * vox + org[:2] + vox / 2.0
        margs = np.array([marg(int(i), int(j)) for i, j in cocc])
        ang = np.abs(np.arctan2(cpts[:, 1], cpts[:, 0]))
        r = np.hypot(cpts[:, 0], cpts[:, 1])
        use = (margs >= MIN_MARGIN) & (ang <= AIM_ANG) & (r <= R_CAP)
        return epts, cpts, margs >= MIN_MARGIN, use, r

    dzs = np.arange(0.0, 0.195 + 1e-9, vox)
    best = (None, -1.0)      # (dz, r_max)
    rows = []
    for dz in dzs:
        L = layer(dz)
        r_max = float(L[4][L[3]].max()) if (L is not None and L[3].any()) else 0.0
        if r_max > best[1]:
            best = (dz, r_max)
        rows.append((dz, L, r_max))

    frames = []
    run_best = 0.0
    for idx, (dz, L, r_max) in enumerate(rows + [rows[
            int(round(best[0] / vox))]] * 3):
        chosen = idx >= len(rows)
        run_best = max(run_best, r_max)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12.8, 6.2), dpi=100,
            gridspec_kw={"width_ratios": [1.0, 1.25]})
        fig.patch.set_facecolor("white")

        # -- side view (world) ------------------------------------------
        z_w = sil[:, 0] + (PELVIS - dz)
        ax1.fill_betweenx(z_w, sil[:, 1], sil[:, 2], color=C_CORE,
                          alpha=0.85, lw=0)
        ax1.axhline(GRIP_Z, color=C_AIM, lw=2)
        ax1.text(0.62, GRIP_Z + 0.012, "grasp height 0.707 m",
                 color=INK, fontsize=10)
        ax1.plot([0], [PELVIS - dz], "o", color=INK, ms=9)
        ax1.text(-0.055, PELVIS - dz, "pelvis", color=INK, fontsize=10,
                 ha="right", va="center")
        ax1.set_xlim(-0.12, 0.95)
        ax1.set_ylim(0.55, 1.35)
        ax1.set_xlabel("forward x (m)", color=INK2)
        ax1.set_ylabel("world z (m)", color=INK2)
        ax1.set_title("core rides down with the squat", color=INK,
                      fontsize=12)
        ax1.grid(color="#EEF1F3", lw=0.8)
        for s in ax1.spines.values():
            s.set_color("#D5DADF")

        # -- top view (pelvis frame, slice at grasp height) --------------
        ax2.add_patch(Arc((0, 0), 2 * R_CAP, 2 * R_CAP, color=INK2,
                          lw=1.4, ls="--"))
        ax2.text(R_CAP * 0.72, R_CAP * 0.74, "servo cap 0.28 m",
                 color=INK2, fontsize=9)
        for sgn in (1, -1):
            ax2.plot([0, 0.5 * np.cos(sgn * AIM_ANG)],
                     [0, 0.5 * np.sin(sgn * AIM_ANG)], color=INK2,
                     lw=1.2, ls=":")
        ax2.text(0.42, 0.30, "ego cone ±33°", color=INK2, fontsize=9)
        ax2.annotate("", xy=(0.10, 0), xytext=(0, 0),
                     arrowprops=dict(arrowstyle="-|>", color=INK, lw=2))
        ax2.plot(0, 0, "s", color=INK, ms=10)
        if L is not None:
            epts, cpts, m_ok, use, r = L
            ax2.plot(epts[:, 0], epts[:, 1], "s", color=C_ENV, ms=7,
                     mec="none")
            ax2.plot(cpts[:, 0], cpts[:, 1], "s", color=C_CORE, ms=7,
                     mec="none")
            ax2.plot(cpts[m_ok, 0], cpts[m_ok, 1], "s", color=C_MARG,
                     ms=7, mec="none")
            if use.any():
                far = use & (r >= r[use].max() - 0.03)
                ax2.plot(cpts[far, 0], cpts[far, 1], "*", color=C_AIM,
                         ms=17, mec="white", mew=0.8)
                fx, fy = cpts[far][np.argmax(r[far])]
                ax2.annotate(f"aim r={r[use].max():.2f} m",
                             xy=(fx, fy), xytext=(fx + 0.06, fy - 0.14),
                             color=INK, fontsize=10,
                             arrowprops=dict(arrowstyle="->", color=INK2))
        ax2.set_xlim(-0.12, 0.62)
        ax2.set_ylim(-0.55, 0.42)
        ax2.set_aspect("equal")
        ax2.set_xlabel("pelvis-frame x (m)", color=INK2)
        ax2.set_ylabel("pelvis-frame y (m)", color=INK2)
        ax2.set_title("slice now at grasp height", color=INK, fontsize=12)
        ax2.grid(color="#EEF1F3", lw=0.8)
        for s in ax2.spines.values():
            s.set_color("#D5DADF")
        leg = [plt.Line2D([], [], marker="s", ls="", color=C_ENV,
                          label="envelope (possible)"),
               plt.Line2D([], [], marker="s", ls="", color=C_CORE,
                          label="sweet core"),
               plt.Line2D([], [], marker="s", ls="", color=C_MARG,
                          label="margin ≥ 5 cm"),
               plt.Line2D([], [], marker="*", ls="", color=C_AIM, ms=13,
                          label="aim band (farthest)")]
        ax2.legend(handles=leg, loc="lower left", fontsize=9,
                   framealpha=0.9, edgecolor="#D5DADF")

        head = (f"CHOSEN: squat {best[0]*100:.1f} cm — farthest overlap "
                f"r = {best[1]:.2f} m" if chosen else
                f"scanning squat depth Δz = {dz*100:4.1f} cm    "
                f"usable reach r = {r_max:.2f} m    "
                f"best so far = {run_best:.2f} m")
        fig.suptitle(head, color=(C_AIM if chosen else INK), fontsize=14,
                     y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fp = f"{TMP}/f_{idx:03d}.png"
        fig.savefig(fp)
        plt.close(fig)
        frames.append(fp)

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate",
                    "0.8", "-i", f"{TMP}/f_%03d.png", "-vf",
                    "fps=25,format=yuv420p", "-c:v", "libx264", OUT],
                   check=True)
    print("wrote", OUT, os.path.getsize(OUT) // 1024, "KB,",
          len(frames), "frames")


if __name__ == "__main__":
    main()
