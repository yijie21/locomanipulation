"""Visualize the JOINT squat-depth / stand-distance optimization.

The table and the can are FIXED; the robot is what moves. Each squat depth
enlarges the arm's reach at the can's height, so the robot can stand
farther back while the sweet core still covers the can — the squat is
traded for table clearance. Left panel (world side view): table + can
fixed, the robot retreats as it squats, the core silhouette rides down
and out with it; the pelvis-to-edge clearance is annotated. Right panel
(pelvis frame, top view): the core slice at can height with the margin
and ego-cone constraints; the star is the farthest usable aim. The old
servo depth cap (0.28 m, shallow-squat era) is drawn as a reference line
only — it no longer truncates the choice. The scan keeps the depth whose
aim reaches farthest; closing frames freeze on the winner.

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
from matplotlib.patches import Arc, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reach_map as rmm  # noqa: E402
import paths

OUT = f"{paths.REPO}/outputs/videos/plan_scan.mp4"
TMP = "/tmp/claude-1002/plan_scan_frames"

GRIP_Z = 0.707          # world grasp height of the can
CAN_TOP = 0.727
TABLE_TOP = 0.62
PELVIS = 0.793          # standing pelvis height
CAN_INBOARD = 0.06      # can centre sits this far in from the table edge
AIM_ANG = np.deg2rad(33.0)
MIN_MARGIN = 0.05
OLD_CAP = 0.28          # shallow-squat-era servo cap: reference line only
R_CAP = 0.34            # execution wall (pregrasp staging ~0.36): the map
                        # now reaches past it, so THIS is what binds
SIDE_Y = -0.06          # right-shoulder aim bias, same as plan_core_stand

# Nested hierarchy -> one sequential green ramp + a labelled gold accent.
C_ENV = "#DDE9DD"
C_CORE = "#9CC7A4"
C_MARG = "#3E8D57"
C_AIM = "#D9922E"
C_TBL = "#B07A4F"
INK, INK2 = "#1F2933", "#5B6570"


def main() -> None:
    os.makedirs(TMP, exist_ok=True)
    rm = rmm.ReachMap()
    core = rm._grid("right", core=True)
    env = rmm._close6(rm._grid("right", core=False))
    vox, org = rm.vox, rm.origin
    obj_z_pf = GRIP_Z - PELVIS

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
        # Same rule as plan_core_stand: margin + ego cone + the execution
        # wall R_CAP, preferring right-shoulder-side cells (untwisted arm).
        use = (margs >= MIN_MARGIN) & (ang <= AIM_ANG) & (r <= R_CAP)
        side = cpts[:, 1] <= SIDE_Y
        if (use & side).any():
            use = use & side
        return epts, cpts, margs >= MIN_MARGIN, use, r

    dzs = np.arange(0.0, 0.195 + 1e-9, vox)
    rows, best = [], (None, -1.0)
    for dz in dzs:
        L = layer(dz)
        r_aim = float(L[4][L[3]].max()) if (L is not None and L[3].any()) else 0.0
        if r_aim > best[1]:
            best = (dz, r_aim)
        rows.append((dz, L, r_aim))

    frames, run_best = [], 0.0
    seq = rows + [rows[int(round(best[0] / vox))]] * 3
    for idx, (dz, L, r_aim) in enumerate(seq):
        chosen = idx >= len(rows)
        run_best = max(run_best, r_aim)
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(12.8, 6.4), dpi=100,
            gridspec_kw={"width_ratios": [1.15, 1.0]})
        fig.patch.set_facecolor("white")

        # -- side view: table+can fixed, ROBOT retreats as it squats ------
        # World x: can at 0, table extends to the right, edge at -0.06.
        ax1.add_patch(Rectangle((-CAN_INBOARD, TABLE_TOP - 0.03), 0.66,
                                0.03, color=C_TBL))
        ax1.add_patch(Rectangle((-CAN_INBOARD + 0.02, 0.30), 0.03,
                                TABLE_TOP - 0.33, color=C_TBL))
        ax1.add_patch(Rectangle((0.50, 0.30), 0.03, TABLE_TOP - 0.33,
                                color=C_TBL))
        ax1.add_patch(Rectangle((-0.018, TABLE_TOP), 0.036,
                                CAN_TOP - TABLE_TOP, color="#C0504D"))
        ax1.axhline(GRIP_Z, color=C_AIM, lw=1.2, ls="--", alpha=0.7)
        have = r_aim > 0
        px = -r_aim if have else -0.22
        pz = PELVIS - dz
        z_w = sil[:, 0] + pz
        ax1.fill_betweenx(z_w, px + sil[:, 1], px + sil[:, 2],
                          color=C_CORE, alpha=0.55 if have else 0.25,
                          lw=0)
        ax1.plot([px], [pz], "o", color=INK, ms=10)
        ax1.plot([px, px], [0.30, pz], color=INK, lw=2.5)
        if have:
            clr = r_aim - CAN_INBOARD
            ax1.annotate("", xy=(-CAN_INBOARD, 0.34), xytext=(px, 0.34),
                         arrowprops=dict(arrowstyle="<->", color=INK,
                                         lw=1.6))
            ax1.text((px - CAN_INBOARD) / 2, 0.355,
                     f"edge clearance {clr:.2f} m", color=INK,
                     fontsize=11, ha="center")
        else:
            ax1.text(-0.35, 0.40, "no usable slice yet", color=INK2,
                     fontsize=11)
        ax1.set_xlim(-0.62, 0.58)
        ax1.set_ylim(0.25, 1.30)
        ax1.set_xlabel("world x (m) — can at 0, table edge at -0.06",
                       color=INK2)
        ax1.set_ylabel("world z (m)", color=INK2)
        ax1.set_title("deeper squat -> farther reach -> the ROBOT "
                      "retreats", color=INK, fontsize=12)
        ax1.grid(color="#EEF1F3", lw=0.8)
        for s in ax1.spines.values():
            s.set_color("#D5DADF")

        # -- top view: slice at can height, no distance cap ---------------
        ax2.add_patch(Arc((0, 0), 2 * OLD_CAP, 2 * OLD_CAP, color="#B0B7BE",
                          lw=1.2, ls="--"))
        ax2.text(0.02, OLD_CAP + 0.015, "old servo cap 0.28 (to lift)",
                 color="#8A9199", fontsize=8.5)
        ax2.add_patch(Arc((0, 0), 2 * R_CAP, 2 * R_CAP, color=C_AIM,
                          lw=1.4, ls="--", alpha=0.8))
        ax2.text(0.02, R_CAP + 0.015, "R_CAP 0.34 (execution wall)",
                 color=C_AIM, fontsize=8.5)
        for sgn in (1, -1):
            ax2.plot([0, 0.55 * np.cos(sgn * AIM_ANG)],
                     [0, 0.55 * np.sin(sgn * AIM_ANG)], color=INK2,
                     lw=1.2, ls=":")
        ax2.text(0.44, 0.31, "ego cone ±33°", color=INK2, fontsize=9)
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
                ax2.annotate(f"aim r={r[use].max():.2f} m", xy=(fx, fy),
                             xytext=(fx + 0.05, fy - 0.15), color=INK,
                             fontsize=10,
                             arrowprops=dict(arrowstyle="->", color=INK2))
        ax2.set_xlim(-0.12, 0.62)
        ax2.set_ylim(-0.55, 0.42)
        ax2.set_aspect("equal")
        ax2.set_xlabel("pelvis-frame x (m)", color=INK2)
        ax2.set_ylabel("pelvis-frame y (m)", color=INK2)
        ax2.set_title("slice now at can height", color=INK, fontsize=12)
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
                          label="aim (farthest usable)")]
        ax2.legend(handles=leg, loc="lower left", fontsize=9,
                   framealpha=0.9, edgecolor="#D5DADF")

        head = (f"CHOSEN: squat {best[0]*100:.1f} cm -> stand r = "
                f"{best[1]:.2f} m, edge clearance "
                f"{best[1]-CAN_INBOARD:.2f} m" if chosen else
                f"squat Δz = {dz*100:4.1f} cm    stand distance r = "
                f"{r_aim:.2f} m    best so far = {run_best:.2f} m")
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
