"""selfmodel_faildiag_20261126 revision 6 smokes (not in the statistics): the tucked walks' arm tracking and the
no-turn correction walks (N), on dev episodes outside question 1's 139.

Tracking: from the tuck's start (the first row whose pushed arm reference leaves the episode's start reference) + 90 rows
(the 0.8 s blend and the 1.0 s hold) to the first reach command after it: every 5th row, both hand points by FK of the
measured joints and of the pushed reference (fd_tuckpose.Body, pelvis at the origin), the larger of the two errors.
N: the first row where the run leaves the same episode's rerun (step log), the correction walk's row (its wait's fire),
the correction walk's plan (turn in place or not, from the pushed pelvis yaw), the pelvis at the first reach command
against the plan's stance, falls / lost / outcome.

    python fd_smoke6.py --tuck <smoke/tuckc> --noturn <smoke/noturn>      (locomanilab_py311, from code_tuck/)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import fd_tuckpose as P

SO = ("left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
      "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
      "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
      "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
      "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint", "right_elbow_joint",
      "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint", "right_wrist_pitch_joint",
      "left_wrist_yaw_joint", "right_wrist_yaw_joint")  # fmt: skip  (the step log's q columns: ScaleBFM's order)
T = {"wcan": "IThorLivingRoomApproachTableAndLiftWateringCan", "dumbbell": "IThorLivingRoomApproachTableAndLiftDumbbell"}


def load(base: Path, tn: str, method: str, ep: int):
    r = json.load(open(base / "episodes" / T[tn] / method / f"ep{ep:+05d}.json"))
    z = np.load(base / "steplog" / T[tn] / method / f"ep{ep:+05d}.npz")
    return r, z["rows"], {str(c): i for i, c in enumerate(z["cols"])}


def track(B, so2mj, r, rows, C):
    ref = rows[:, [C[f"push{7 + j}"] for j in range(15, 29)]]
    ch = np.nonzero(np.abs(ref - ref[60]).max(1) > 1e-4)[0]
    if not len(ch):
        return None
    f0 = int(ch[0])
    cmds = [int(w["fire"]) for w in r["waits"] if w.get("then") != "correction walk" and w.get("fire") is not None and int(w["fire"]) > f0]
    f1 = min(cmds) if cmds else len(rows)
    E, J = [], []
    for f in range(f0 + 90, f1, 5):
        qa = rows[f, [C[f"q{i}"] for i in range(29)]][so2mj]
        qr = rows[f, [C[f"push{7 + j}"] for j in range(29)]]
        B.set(qa)
        ha = B.hands()
        B.set(qr)
        hr = B.hands()
        E.append(max(np.linalg.norm(ha[s]["hp"] - hr[s]["hp"]) for s in ("left", "right")))
        J.append(np.abs(qa[15:29] - qr[15:29]).max())
    if not E:
        return dict(f0=f0, f1=f1, n=0)
    e = np.array(E) * 100
    return dict(f0=f0, f1=f1, n=len(E), p50=float(np.median(e)), p90=float(np.percentile(e, 90)), max=float(e.max()),
                joint_p50_deg=float(np.degrees(np.median(J))))  # fmt: skip


def yaw(rows, C, pre="push", f=None):
    q = rows[:, [C[f"{pre}{k}"] for k in (3, 4, 5, 6)]] if pre == "push" else rows[:, [C[c] for c in ("pel_qw", "pel_qx", "pel_qy", "pel_qz")]]
    return np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))


def walk_info(r, rows, C):
    """the correction walk: its row, the pushed pelvis yaw swing over it (turn in place or not), the pelvis at the
    first reach command against the stance, the true pelvis's travel over it."""
    fx = [w for w in r.get("waits", []) if w.get("then") == "correction walk" and w.get("fire") is not None]
    if not fx:
        return dict(f_fix=None)
    a = int(fx[0]["fire"])
    cm = [int(w["fire"]) for w in r["waits"] if w.get("then") != "correction walk" and w.get("fire") is not None and int(w["fire"]) > a]
    b = min(cm) if cm else len(rows) - 1
    yr = np.unwrap(yaw(rows[a:b + 1], C))
    st = (r.get("plan") or {}).get("stance") or {}
    pel = rows[:, [C["pel_px"], C["pel_py"]]]
    out = dict(f_fix=a, f_cmd=b, ref_yaw_swing_deg=float(np.degrees(yr.max() - yr.min())),
               pel_travel_cm=float(100 * np.linalg.norm(pel[b] - pel[a])), walk_left_cm=(r.get("walk_left_cm") or [None])[0])  # fmt: skip
    if st.get("xy") is not None:
        out["stance_err_cmd_cm"] = float(100 * np.linalg.norm(pel[b] - np.asarray(st["xy"], float)))
    return out


def first_diff(ra, rb):
    k = min(len(ra), len(rb))
    d = np.nonzero(np.any(ra[:k] != rb[:k], axis=1))[0]
    return int(d[0]) if len(d) else (None if len(ra) == len(rb) else k)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuck", required=True)
    ap.add_argument("--noturn", required=True)
    a = ap.parse_args()
    so2mj = [SO.index(n) for n in P.MJ_NAMES]
    B = P.Body()
    print("== new pose (T1b / T2b), dev episodes (alone); thresholds fixed beforehand: tracking p50 <= 3 cm, no 'lost' while walking")
    for g in ("tuck1b", "tuck2b"):
        for tn, ep in (("wcan", 35), ("wcan", 39), ("dumbbell", 39)):
            r, rows, C = load(Path(a.tuck) / g / "sonic", tn, "alone", ep)
            t = track(B, so2mj, r, rows, C)
            print(f"  {g} {tn} {ep}: reason {r.get('reason')} why {r.get('why')!r} fell {r.get('fell')} | tracking rows {t['f0'] + 90}..{t['f1']} "
                  f"hand point error cm p50 {t.get('p50', float('nan')):.1f} p90 {t.get('p90', float('nan')):.1f} max {t.get('max', float('nan')):.1f}, "
                  f"largest arm joint error p50 {t.get('joint_p50_deg', float('nan')):.0f} deg | arrive {(r.get('arrive') or {}).get('pos_err_cm')}")  # fmt: skip
    print("\n== N (HG_SHORT_M=0.40) and N + tuck, dev episodes (reachable, outside the 139; 15-40 cm correction walks)")
    base = Path(a.noturn)
    for tn, m, ep in (("dumbbell", "reachable", 30), ("wcan", "reachable", 30), ("wcan", "reachable", 31)):
        r0, rows0, C0 = load(base / "rerun" / "sonic", tn, m, ep)
        w0 = walk_info(r0, rows0, C0)
        print(f"  {tn} {m} {ep}: rerun: reason {r0.get('reason')} fell {r0.get('fell')} | correction walk row {w0['f_fix']}, asked {w0['walk_left_cm']:.1f} cm, "
              f"reference yaw swing {w0['ref_yaw_swing_deg']:.0f} deg, pelvis travel {w0['pel_travel_cm']:.1f} cm, stance error at the next command {w0.get('stance_err_cmd_cm', float('nan')):.1f} cm, arrive {(r0.get('arrive') or {}).get('pos_err_cm')}")  # fmt: skip
        for g in ("noturn", "noturn_tuck1", "noturn_tuck1b"):
            r, rows, C = load(base / g / "sonic", tn, m, ep)
            w = walk_info(r, rows, C)
            fd = first_diff(rows, rows0)
            t = track(B, so2mj, r, rows, C) if g != "noturn" else None
            ts = f" | tracking p50 {t['p50']:.1f} p90 {t['p90']:.1f} cm" if t and t.get("n") else ""
            print(f"    {g}: first row off the rerun {fd} (correction walk row {w0['f_fix']}: identical through row {w0['f_fix']} -> {fd is not None and fd > w0['f_fix']}) | "
                  f"reason {r.get('reason')} why {r.get('why')!r} fell {r.get('fell')} success {r.get('success')} | reference yaw swing {w.get('ref_yaw_swing_deg', float('nan')):.0f} deg, "
                  f"pelvis travel {w.get('pel_travel_cm', float('nan')):.1f} cm, stance error at the next command {w.get('stance_err_cmd_cm', float('nan')):.1f} cm, arrive {(r.get('arrive') or {}).get('pos_err_cm')}, "
                  f"n_walks {r.get('n_walks')}, fd_noturn {r.get('fd_noturn')}{ts}")  # fmt: skip
    # N on an alone episode whose correction walk is < 15 cm (already a shuffle at SHORT_M 0): the whole episode should
    # be lmlcmp3's (the record, the fd_run.py check way, is compared outside this script)
    r, rows, C = load(base / "noturn" / "sonic", "wcan", "alone", 35)
    print(f"\n  noturn wcan alone 35 (correction walk {(r.get('walk_left_cm') or [None])[0]} cm < 15): reason {r.get('reason')} steps {len(rows)}")


if __name__ == "__main__":
    main()
