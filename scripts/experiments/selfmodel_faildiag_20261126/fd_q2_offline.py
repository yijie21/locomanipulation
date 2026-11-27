"""selfmodel_faildiag_20261126 question 2, the offline quantity: how far the grasp the look's plan picked is from the
true object's grasps (measurement only).

Per (controller, task, episode), lmlcmp3's A plan (the look's grasp; R picks the same one):
  pos_cm        its hand point to the nearest true-object grasp's hand point
  rot_deg       the wrist rotation between them (that nearest grasp)
  best_*        the true grasp nearest by cm + deg / 5 (the servo's metric): its cm and deg
  surf_cm       its hand point to the true object's surface (fd_gt's 2048 surface points)
The true grasps (fd_gt.py, object frame) are moved to the object's simulator pose at the plan's load (the servo group's
step log at row 35 + the delay shift, where fd_run placed them; every group equals A up to the first pre-grasp command). Self-test: the gtgrasp group's own pick must sit
on a true grasp (pos and rot ~ 0).
Also the gtgrasp group's pick against A's: approach yaw from the stance heading, pitch, height, and the outcomes.

    python fd_q2_offline.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import fd_q2_an as Q

sys.path.insert(0, "/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126/code/scripts/experiments/hardgrasp_20261125")
import hg_look as HL  # noqa: E402

GT_DIR = Q.FD / "gt_looks"


def quat_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def rot_of(plan: dict) -> np.ndarray:
    x = np.asarray(plan["ad"], float)
    y = np.asarray(plan["n_pal"], float)
    y = y - x * (x @ y)
    x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def ang(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Rotation angle (deg) between Ra and each of Rb (n,3,3)."""
    tr = np.einsum("ij,nij->n", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def look_T(be: str, task: str, ep: int) -> np.ndarray | None:
    pj, pz = Q.paths("servo", be, task, ep)
    if pz is None or not pz.exists():
        return None
    r = json.loads(pj.read_text())
    z = np.load(pz)
    C = {str(c): i for i, c in enumerate(z["cols"])}
    # the row at which fd_run's patched load placed the true grasps (found from the gtgrasp picks: 35 + the delay shift;
    # the candle still rolls a few mm between there and fd_q1_moved's f_look = 55 + shift)
    f = 35 + max(int(r.get("shift_f") or 0), 0)
    row = z["rows"][f]
    T = np.eye(4)
    T[:3, :3] = quat_mat([row[C[k]] for k in ("obj_qw", "obj_qx", "obj_qy", "obj_qz")])
    T[:3, 3] = [row[C[k]] for k in ("obj_px", "obj_py", "obj_pz")]
    return T


def compare(plan: dict, Tl: np.ndarray, g: dict) -> dict:
    Tw = Tl @ g["grasp_G_obj"] @ HL.T_G_WRIST
    hand = Tw[:, :3, 3] + Tw[:, :3, :3] @ HL.HAND_OFF
    goal = np.asarray(plan["goal"], float)
    d = 100 * np.linalg.norm(hand - goal, axis=1)
    a = ang(rot_of(plan), Tw[:, :3, :3])
    j = int(np.argmin(d))
    k = int(np.argmin(d + a / 5))
    cl = (Tl @ np.linalg.inv(g["obj_T_reset"]) @ np.c_[g["cloud_w"], np.ones(len(g["cloud_w"]))].T).T[:, :3]
    return dict(pos_cm=float(d[j]), rot_deg=float(a[j]), best_cm=float(d[k]), best_deg=float(a[k]), surf_cm=float(100 * np.min(np.linalg.norm(cl - goal, axis=1))))


def geom(plan: dict) -> dict:
    ad = np.asarray(plan["ad"], float)
    st = plan["stance"]
    return dict(dyaw_deg=math.degrees(Q_wrap(math.atan2(ad[1], ad[0]) - float(st["yaw"]))), pitch_deg=math.degrees(math.asin(max(-1.0, min(1.0, -ad[2])))),
                z=float(plan["goal"][2]), score=float(plan["score"]))  # fmt: skip


def Q_wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    per = json.load(open(Path(a.out) / "q2_episodes.json"))
    rows = []
    for x in per:
        be, tn, ep = x["backend"], x["task"], x["ep"]
        task = Q.TN[tn]
        gf = GT_DIR / task / f"ep{ep:+05d}.npz"
        Tl = look_T(be, task, ep)
        if Tl is None or not gf.exists():
            continue
        g = {k: v for k, v in np.load(gf, allow_pickle=True).items()}
        rA = json.loads(Q.paths("A", be, task, ep)[0].read_text())
        row = dict(backend=be, task=tn, ep=ep, R_success=x.get("R", {}).get("success"), R_reason=x.get("R", {}).get("reason"),
                   obj_center_err_cm=(rA.get("look") or {}).get("obj_center_err_cm"))  # fmt: skip
        if rA.get("plan") and rA["plan"].get("fail") is None:
            row.update(compare(rA["plan"], Tl, g))
            row["A_geom"] = geom(rA["plan"])
        pg = Q.paths("gtgrasp", be, task, ep)[0]
        if pg is not None and pg.exists():
            rg = json.loads(pg.read_text())
            if rg.get("plan") and rg["plan"].get("fail") is None:
                row["self_test"] = compare(rg["plan"], Tl, g)
                row["G_geom"] = geom(rg["plan"])
                row.update(G_success=bool(rg.get("success")), G_reason=rg.get("reason"))
        rows.append(row)
    Path(a.out, "q2_offline.json").write_text(json.dumps(rows, indent=0, default=float))
    p = lambda v: np.percentile(v, [25, 50, 75]).round(1).tolist() if len(v) else None  # noqa: E731
    st = [r["self_test"] for r in rows if "self_test" in r]
    print(f"self-test (the gtgrasp pick against the true grasps): n {len(st)}, pos max {max([s['pos_cm'] for s in st], default=float('nan')):.2e} cm, rot max {max([s['rot_deg'] for s in st], default=float('nan')):.2e} deg")
    for be in Q.BACKENDS:
        for tn in ("dumbbell", "wcan", "candle", "apple"):
            S = [r for r in rows if r["backend"] == be and r["task"] == tn and "pos_cm" in r]
            if not S:
                continue
            print(f"\n{be} {tn}: {len(S)} episodes with A's plan")
            for k, lab in (("pos_cm", "look's pick -> nearest true grasp, cm"), ("rot_deg", "  its wrist rotation, deg"), ("best_cm", "nearest by cm + deg/5: cm"),
                           ("best_deg", "  deg"), ("surf_cm", "look's pick hand point -> true surface, cm"), ("obj_center_err_cm", "the look's object centre error, cm")):  # fmt: skip
                v = [r[k] for r in S if r.get(k) is not None]
                ok = [r[k] for r in S if r.get(k) is not None and r["R_success"]]
                bad = [r[k] for r in S if r.get(k) is not None and r["R_success"] is False]
                print(f"  {lab:44s} p25/50/75 {p(v)}; R success {p(ok)} (n {len(ok)}), R failed {p(bad)} (n {len(bad)})")
            G = [r for r in S if "G_geom" in r]
            if G:
                for k in ("dyaw_deg", "pitch_deg", "z", "score"):
                    print(f"  pick geometry {k:10s} A (look) p50 {np.median([r['A_geom'][k] for r in G]):.2f}, gtgrasp p50 {np.median([r['G_geom'][k] for r in G]):.2f}; |dyaw| > 30 deg: A {sum(abs(r['A_geom']['dyaw_deg']) > 30 for r in G) if k == 'dyaw_deg' else ''} G {sum(abs(r['G_geom']['dyaw_deg']) > 30 for r in G) if k == 'dyaw_deg' else ''}")
                print(f"  outcome R -> gtgrasp: {dict(Counter((r['R_reason'], r['G_reason']) for r in G).most_common(8))}")


if __name__ == "__main__":
    main()
