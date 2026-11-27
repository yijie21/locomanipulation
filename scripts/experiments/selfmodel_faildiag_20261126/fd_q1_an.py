"""selfmodel_faildiag_20261126 question 1: why SONIC misses (dumbbell, watering can) -- from the rerun group's records,
step logs and lmlcmp3's looks. Run with the simulation env (needs lml_selfmodel's Planner for the reference's FK):

    cd $G; PYTHONPATH=$G:$C0/src python fd_q1_an.py --out <dir>

Per episode with a close (the close command's row fc, the lift command's row fl):
  close_dist_cm         hand point (right wrist_yaw_link + R (0.13, 0, 0)) to the plan's grasp hand point, at fc
                        (checked against the record's close_dist_cm)
  rot_err_deg           the wrist frame at fc vs the planned grasp's wrist frame (the look's grasp_T[idx], roll included)
  express_deg           the planned grasp frame vs the entry command's rotation rot_z(yaw_cmd + dyaw) rot_y(pitch)
  ik_deg                the command's rotation vs the reference's wrist at the entry's end (FK of the pushed frame at
                        the entry's f_end; world frames)
  track_deg             the reference's wrist at the entry's end vs the wrist at fc
  v_hand_cms, w_hand_dps the hand point's speed / the wrist's angular speed at fc (0.1 s difference)
  close_move_cm         the hand point's largest displacement from fc over fc .. fl, and its peak speed
  fingers               the largest normal force of thumb / index / middle / palm on the object over fc .. fl;
                        fingers in contact (> 0.5 N) at fl
  obj_push_cm           the object's horizontal displacement fc -> fl
  rise                  the object's and the hand point's rise over fl .. fl + 3 s
Clean closes: reason ok / missed / dropped, the pre-grasp segment without object contact (lc3_eval's
pregrasp_touch), past the split. Missed mechanism (first that applies): never touched (no hand-object contact row in
fc .. fl), pushed away (touched, object moved > 2 cm), slipped (a finger in contact at fl, rise < 2 cm), other.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FD = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126")
LC3 = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_lmlcmp3_20261126")
CODE = LC3 / "code" / "scripts" / "experiments"
sys.path[:0] = [str(CODE / "locomanilab_20261117"), str(CODE / "hardgrasp_20261125")]
import lml_selfmodel as L  # noqa: E402

TASKS = {"IThorLivingRoomApproachTableAndLiftDumbbell": "dumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan": "wcan"}
GROUPS = (("sonic", "alone"), ("sonic", "selfmodel"), ("sonic", "reachable"), ("scalebfm", "alone"), ("scalebfm", "reachable"))
HAND_OFF = np.array([0.13, 0.0, 0.0])


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def ang(A, B) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def episode(pl, r: dict, z) -> dict | None:
    w = {x["key"]: x for x in r.get("waits", [])}
    if "close" not in w or "lift" not in w:
        return None
    rows, cols = z["rows"], [str(c) for c in z["cols"]]
    C = {c: i for i, c in enumerate(cols)}
    g = lambda *ks: rows[:, [C[k] for k in ks]]  # noqa: E731
    wp, wq = g("wr_px", "wr_py", "wr_pz"), g("wr_qw", "wr_qx", "wr_qy", "wr_qz")
    Rw = np.array([qmat(q) for q in wq])
    H = wp + Rw @ HAND_OFF
    obj = g("obj_px", "obj_py", "obj_pz")
    goal = np.asarray(r["plan"]["goal"], float)
    fc, fl = int(w["close"]["fire"]), int(w["lift"]["fire"])
    # the row of the close command: the record's close distance
    cand = {k: 100 * float(np.linalg.norm(H[fc + k] - goal)) for k in (-1, 0, 1) if 0 <= fc + k < len(H)}
    k0 = min(cand, key=lambda k: abs(cand[k] - r["close_dist_cm"]))
    fc, fl = fc + k0, fl + k0
    lk = np.load(LC3 / "hg" / "looks" / r["task"] / f"ep{r['ep']:+05d}.npz")
    Rg = np.asarray(lk["grasp_T"][r["plan"]["idx"]], float)[:3, :3]
    ent = [x for x in r["reaches"] if x["reach_no"] == 2 and x["f_cmd"] < fc]
    out = dict(row_offset=k0, close_dist_cm=cand[k0], close_dist_rec=r["close_dist_cm"], rot_err_deg=ang(Rg, Rw[fc]))
    # where the object really is: the plan's grasp point vs the object (body origin) at the close; the object in the
    # hand's frame (wrist axes, from the hand point) at the close and in the planned grasp's frame (from the goal)
    out["goal_obj_cm"] = 100 * float(np.linalg.norm(goal - obj[fc]))
    out["hand_obj_cm"] = 100 * float(np.linalg.norm(H[fc] - obj[fc]))
    out["obj_in_hand_cm"] = (100 * (Rw[fc].T @ (obj[fc] - H[fc]))).tolist()
    out["obj_in_plan_cm"] = (100 * (Rg.T @ (obj[fc] - goal))).tolist()
    out["obj_look_err_cm"] = 100 * float(np.linalg.norm(np.asarray(lk["obj_w"], float).mean(0) - obj[fc])) if "obj_w" in lk.files else float("nan")
    if ent:
        e = ent[-1]
        rot_w = rz(e["yaw_cmd"] + math.radians(e["dyaw_deg"])) @ ry(math.radians(e["pitch_deg"]))
        fe = min(int(e["f_end"]), len(rows) - 1)
        push = rows[fe, C["push0"] : C["push0"] + 36].astype(float)
        fk = pl.fk_world(push[:3], push[3:7], push[7:])
        Rref = fk["right"][1]
        out.update(express_deg=ang(Rg, rot_w), ik_deg=ang(rot_w, Rref), track_deg=ang(Rref, Rw[fc]), ref_hand_err_cm=100 * float(np.linalg.norm(fk["right"][0] - np.asarray(e["goal_w"]))),
                   entry_ik_cm=e.get("ik_err_cm"), entry_dyaw=e["dyaw_deg"], entry_pitch=e["pitch_deg"])  # fmt: skip
    d5 = 5
    out["v_hand_cms"] = 100 * float(np.linalg.norm(H[fc] - H[fc - d5])) / (d5 / L.FPS)
    out["w_hand_dps"] = ang(Rw[fc - d5], Rw[fc]) / (d5 / L.FPS)
    win = slice(fc, fl + 1)
    dH = np.linalg.norm(H[win] - H[fc], axis=1)
    out["close_move_cm"] = 100 * float(dH.max())
    vH = np.linalg.norm(np.diff(H[win], axis=0), axis=1) * L.FPS
    out["close_vmax_cms"] = 100 * float(vH.max()) if len(vH) else float("nan")
    F = g("f_thumb", "f_index", "f_middle", "f_palm")
    out["fmax"] = F[win].max(axis=0).tolist()
    out["fingers_at_lift"] = int((F[fl] > 0.5).sum())
    out["touched"] = bool((rows[win, C["hand_obj_n"]] > 0).any())
    out["obj_push_cm"] = 100 * float(np.linalg.norm(obj[fl, :2] - obj[fc, :2]))
    l3 = slice(fl, min(fl + 3 * L.FPS, len(rows)))
    out["obj_rise_cm"] = 100 * float(obj[l3, 2].max() - obj[fl, 2])
    out["hand_rise_cm"] = 100 * float(H[l3, 2].max() - H[fl, 2])
    return out


def mech(e: dict) -> str:
    if not e["touched"]:
        return "never touched"
    if e["obj_push_cm"] > 2.0:
        return "pushed away"
    if e["fingers_at_lift"] > 0 and e["obj_rise_cm"] < 2.0:
        return "slipped"
    return "other"


def q(v, p):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return "nan" if not v else f"{np.percentile(v, p):.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pl = L.Planner()
    per = []
    for be, m in GROUPS:
        for task, tn in TASKS.items():
            for f in sorted(glob.glob(str(FD / "rerun" / be / "episodes" / task / m / "ep*.json"))):
                r = json.loads(Path(f).read_text())
                sl = FD / "rerun" / be / "steplog" / task / m / Path(f).with_suffix(".npz").name
                d1 = next((d for d in r.get("d3", []) if d["reach_no"] == 1), None)
                pre_touch = d1 is not None and d1["obj_rows"] > 0
                row = dict(backend=be, method=m, task=tn, ep=r["ep"], reason=r["reason"], success=bool(r["success"]), pre_touch=pre_touch,
                           idx=(r.get("plan") or {}).get("idx"), clean=r["reason"] in ("ok", "missed", "dropped") and not pre_touch)  # fmt: skip
                if sl.exists() and r.get("plan") and r["plan"].get("fail") is None:
                    e = episode(pl, r, np.load(sl))
                    if e:
                        row.update(e)
                        row["mech"] = mech(e) if r["reason"] == "missed" else None
                per.append(row)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "q1_episodes.json").write_text(json.dumps(per, indent=0, default=float))
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    chk = [e for e in per if "close_dist_cm" in e]
    log(f"fd_q1_an: episodes {len(per)}, with a close {len(chk)}; close distance recomputed vs the record: max |diff| "
        f"{max(abs(e['close_dist_cm'] - e['close_dist_rec']) for e in chk):.2e} cm; row offsets {dict(Counter(e['row_offset'] for e in chk))}")  # fmt: skip
    keys = ("close_dist_cm", "goal_obj_cm", "hand_obj_cm", "obj_look_err_cm", "rot_err_deg", "express_deg", "ik_deg", "track_deg", "v_hand_cms", "w_hand_dps", "close_move_cm", "close_vmax_cms", "obj_push_cm")
    for be, m in GROUPS:
        for tn in ("dumbbell", "wcan", "both"):
            E = [e for e in per if e["backend"] == be and e["method"] == m and (tn == "both" or e["task"] == tn)]
            cl = [e for e in E if e["clean"] and "close_dist_cm" in e]
            ok = [e for e in cl if e["reason"] == "ok"]
            ms = [e for e in cl if e["reason"] == "missed"]
            log(f"== {be} {m} {tn}: episodes {len(E)}, reasons {dict(Counter(e['reason'] for e in E))}, pre-grasp touch {sum(e['pre_touch'] for e in E)}; clean closes {len(cl)} (ok {len(ok)}, missed {len(ms)}, dropped {sum(e['reason'] == 'dropped' for e in cl)})")
            for k in keys:
                log(f"   {k:15s} ok p50 {q([e.get(k) for e in ok], 50):>6s} (p10 {q([e.get(k) for e in ok], 10)}, p90 {q([e.get(k) for e in ok], 90)}) | "
                    f"missed p50 {q([e.get(k) for e in ms], 50):>6s} (p10 {q([e.get(k) for e in ms], 10)}, p90 {q([e.get(k) for e in ms], 90)}) | AUC(missed > ok) {auc([e.get(k, np.nan) for e in ms], [e.get(k, np.nan) for e in ok]):.2f}")  # fmt: skip
            for nm, S in (("ok", ok), ("missed", ms)):
                if S:
                    log(f"   object in the planned grasp frame (from the goal; x = wrist x, y = palm normal) cm p50 {np.round(np.median([e['obj_in_plan_cm'] for e in S], axis=0), 1).tolist()}, "
                        f"in the hand frame at the close p50 {np.round(np.median([e['obj_in_hand_cm'] for e in S], axis=0), 1).tolist()} ({nm})")  # fmt: skip
            if ms:
                log(f"   hand speed at the close > 3 cm/s: missed {sum(e['v_hand_cms'] > 3 for e in ms)}/{len(ms)}, ok {sum(e['v_hand_cms'] > 3 for e in ok)}/{len(ok)}")
                log(f"   missed mechanism: {dict(Counter(e['mech'] for e in ms))}; fingers in contact at the lift (missed) {dict(Counter(e['fingers_at_lift'] for e in ms))}")
                log(f"   finger max force on the object N (thumb/index/middle/palm) missed p50 {np.round(np.median([e['fmax'] for e in ms], axis=0), 1).tolist()}, ok p50 "
                    f"{np.round(np.median([e['fmax'] for e in ok], axis=0), 1).tolist() if ok else '-'}; object rise cm missed p50 {q([e['obj_rise_cm'] for e in ms], 50)}, hand rise {q([e['hand_rise_cm'] for e in ms], 50)}")  # fmt: skip
    # same episode, same grasp, ScaleBFM vs SONIC (alone, reachable)
    for m in ("alone", "reachable"):
        S = {(e["task"], e["ep"]): e for e in per if e["backend"] == "sonic" and e["method"] == m}
        B = {(e["task"], e["ep"]): e for e in per if e["backend"] == "scalebfm" and e["method"] == m}
        both = [k for k in S if k in B and S[k]["idx"] is not None and B[k]["idx"] is not None]
        same = [k for k in both if S[k]["idx"] == B[k]["idx"]]
        tab = Counter((B[k]["success"], S[k]["success"]) for k in same)
        log(f"== same grasp, {m}: episodes with both {len(both)}, same grasp chosen {len(same)}; on the same grasp (ScaleBFM ok, SONIC ok): {dict(tab)}")
    (Path(a.out) / "q1.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
