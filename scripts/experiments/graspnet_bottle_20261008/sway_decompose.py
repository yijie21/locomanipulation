"""TODO 23.4.6: where the entry-phase hand error comes from, tick by tick, from the 46-column expert trace (the
pinned-arm runs from 2026-10-10 10:34 on: `_pinC`).

Per tick k of the entry the trace holds the world target T_k, the measured tool A_k, the commanded joints C_k (right
arm, waist), the measured joints M_k, the whole-body IK's own hand error, and the measured pelvis pose P_k the IK
solved against. The hand error A_{k+1} - T_k splits into
  feedback= the tracking correction SupportedMotion adds to T_k (49-column traces; before that it sits in ik)
  ik      = FK(C_k, P_k) - T_k - feedback          the IK did not reach what it aimed at
  joint   = FK(M_{k+1}, P_k) - FK(C_k, P_k)        the motors did not reach the command
  pelvis  = FK(M_{k+1}, P_{k+1}) - FK(M_{k+1}, P_k)  the pelvis moved between the solve and the measurement
  rest    = A_{k+1} - FK(M_{k+1}, P_{k+1})          FK vs measured tool (should be ~0; a check of the frames)
all expressed in the grasp's palm frame (x approach, y jaw, z pad height). Also per round: the pelvis's lateral
displacement, yaw change and sway speed over the entry, against the round's outcome. FK by ArmIK on the stock
scene (the robot is the same in every object scene). Prints the aggregate; writes <root>/cache/sway/summary.md.

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/graspnet_bottle_20261008/sway_decompose.py [--tag _pinC]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "outputs" / "graspnet_bottle_20261008"
OFFSET = np.array([0.1515, 0.0, 0.0])
PH_ENTRY = 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="_pinC")
    ap.add_argument("--window-s", type=float, default=1.5, help="entry ticks analysed from the entry start")
    args = ap.parse_args()
    from locomani.primitives.arm_ik import ArmIK

    ik = ArmIK(str(REPO / "outputs" / "scene" / "sim_lowcom" / "scene_locomani.xml"))

    def fk_world(q29, pel_pos, pel_quat_wxyz):
        p, R = ik.fk(q29, "right")
        Rp = Rotation.from_quat(np.asarray(pel_quat_wxyz)[[1, 2, 3, 0]]).as_matrix()
        return Rp @ (p + R @ OFFSET) + np.asarray(pel_pos)

    objects = json.loads((ROOT / "objects.json").read_text())["objects"]
    rounds = []
    for o in objects:
        d = REPO / "outputs" / "runs" / f"gnb_obj_{o['id']:03d}{args.tag}" / "grasp_planned" / "round_00_ik"
        if not (d / "record.json").exists() or not (d / "expert_trace.npy").exists():
            continue
        rec = json.loads((d / "record.json").read_text())
        tr = np.load(d / "expert_trace.npy")
        if tr.shape[1] < 46 or "grasp" not in rec:
            continue
        Rg = np.array(rec["grasp"]["R"])
        t, ph = tr[:, 0], tr[:, 1].astype(int)
        e = np.nonzero(ph == PH_ENTRY)[0]
        if len(e) < 3:
            continue
        e = e[(t[e] - t[e[0]]) <= args.window_s]
        q_meas = np.zeros((len(tr), 29))
        q_meas[:, 22:29] = tr[:, 16:23]
        q_meas[:, 12:15] = tr[:, 30:33]
        q_cmd = q_meas.copy()
        q_cmd[:, 22:29] = tr[:, 23:30]
        q_cmd[:, 12:15] = tr[:, 33:36]
        P = tr[:, 39:42]
        Q = tr[:, 42:46]
        has_fb = tr.shape[1] >= 49
        parts = dict(ik=[], feedback=[], joint=[], pelvis=[], rest=[], total=[])
        for k in e[:-1]:
            T, A1 = tr[k, 2:5], tr[k + 1, 5:8]
            fb = tr[k, 46:49] if has_fb else np.zeros(3)
            f_cmd = fk_world(q_cmd[k], P[k], Q[k])
            f_meas_old = fk_world(q_meas[k + 1], P[k], Q[k])
            f_meas_new = fk_world(q_meas[k + 1], P[k + 1], Q[k + 1])
            # the solver aims at T + feedback (SupportedMotion's integrated tracking correction): split it off
            parts["feedback"].append(fb @ Rg)
            parts["ik"].append((f_cmd - T - fb) @ Rg)
            parts["joint"].append((f_meas_old - f_cmd) @ Rg)
            parts["pelvis"].append((f_meas_new - f_meas_old) @ Rg)
            parts["rest"].append((A1 - f_meas_new) @ Rg)
            parts["total"].append((A1 - T) @ Rg)
        parts = {k: np.array(v) for k, v in parts.items()}
        # pelvis motion over the window: lateral (pelvis-frame y at the window start), yaw, speed
        yaw0 = Rotation.from_quat(Q[e[0]][[1, 2, 3, 0]]).as_euler("xyz")[2]
        lat_axis = np.array([-np.sin(yaw0), np.cos(yaw0), 0.0])
        disp = P[e] - P[e[0]]
        yaw = np.array([Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler("xyz")[2] for q in Q[e]])
        dt = np.diff(t[e])
        speed = np.linalg.norm(np.diff(P[e, :2], axis=0), axis=1) / np.maximum(dt, 1e-3)
        why = rec.get("why", "")
        rounds.append(
            dict(
                id=o["id"], ok=bool(rec.get("ok")), knocked="knocked" in why,
                h=rec["grasp"]["tool"][2] - rec["table_top"],
                lat_max={k: float(np.abs(v[:, 1]).max()) for k, v in parts.items()},
                z_max={k: float(np.abs(v[:, 2]).max()) for k, v in parts.items()},
                lat_signed_med={k: float(np.median(v[:, 1])) for k, v in parts.items()},
                pelvis_lat_disp=float(np.abs(disp @ lat_axis).max()),
                pelvis_yaw_deg=float(np.degrees(np.abs(np.unwrap(yaw) - yaw[0]).max())),
                pelvis_speed_max=float(speed.max()) if len(speed) else 0.0,
                pelvis_z_disp=float(np.abs(disp[:, 2]).max()),
                ticks=len(e),
            )
        )  # fmt: skip
    if not rounds:
        print("no 46-column traces found")
        return 1

    def med(key, sub=None, rows=rounds):
        v = [r[key][sub] if sub else r[key] for r in rows]
        return 100 * st.median(v)

    lines = [
        f"# Entry-phase hand error decomposition ({args.tag}, {len(rounds)} rounds, first {args.window_s} s of the entry)",
        "",
    ]
    lines.append("| part | max |lateral| median (cm) | p75 | max |height| median | signed lateral median |")
    lines.append("|---|---|---|---|---|")
    for part in ("total", "feedback", "ik", "joint", "pelvis", "rest"):
        v = [r["lat_max"][part] for r in rounds]
        lines.append(
            f"| {part} | {100 * st.median(v):.2f} | {100 * np.percentile(v, 75):.2f} | {med('z_max', part):.2f} | "
            f"{100 * st.median(r['lat_signed_med'][part] for r in rounds):+.2f} |"
        )
    lines.append("")
    lines.append(
        f"- pelvis over the window: lateral displacement median {med('pelvis_lat_disp'):.2f} cm (p75 "
        f"{100 * np.percentile([r['pelvis_lat_disp'] for r in rounds], 75):.2f}), yaw {st.median(r['pelvis_yaw_deg'] for r in rounds):.2f} deg, "
        f"height {med('pelvis_z_disp'):.2f} cm, speed max median {med('pelvis_speed_max'):.1f} cm/s"
    )
    for name, sub in (
        ("lifted", [r for r in rounds if r["ok"]]),
        ("knocked", [r for r in rounds if r["knocked"]]),
        ("other fail", [r for r in rounds if not r["ok"] and not r["knocked"]]),
    ):
        if sub:
            lines.append(
                f"- {name} n {len(sub)}: total lat {med('lat_max', 'total', sub):.2f}, ik {med('lat_max', 'ik', sub):.2f}, joint {med('lat_max', 'joint', sub):.2f}, "
                f"pelvis {med('lat_max', 'pelvis', sub):.2f} cm; pelvis lat disp {med('pelvis_lat_disp', rows=sub):.2f} cm, yaw {st.median(r['pelvis_yaw_deg'] for r in sub):.2f} deg, speed {med('pelvis_speed_max', rows=sub):.1f} cm/s"
            )
    for lo, hi in ((0, 0.06), (0.06, 0.1), (0.1, 1)):
        sub = [r for r in rounds if lo <= r["h"] < hi]
        if sub:
            lines.append(
                f"- grasp {100 * lo:.0f}-{100 * hi:.0f} cm above the table n {len(sub)}: total lat {med('lat_max', 'total', sub):.2f}, ik {med('lat_max', 'ik', sub):.2f}, "
                f"joint {med('lat_max', 'joint', sub):.2f}, pelvis {med('lat_max', 'pelvis', sub):.2f} cm; lifted {sum(r['ok'] for r in sub)}/{len(sub)}"
            )
    out = ROOT / "cache" / "sway"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    (out / "rounds.json").write_text(json.dumps(rounds, indent=1))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
