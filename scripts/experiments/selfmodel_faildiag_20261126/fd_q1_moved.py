"""selfmodel_faildiag_20261126 question 1 (added after the servo smoke): episodes whose object left its place before the
hand got there, per controller -- from the rerun group's step logs (measurement only).

Per episode with a plan:
  f_look        the look's row (lc3_run: max(START_HOLD_S, 5 + LEAN_S + LOOK_AFTER_S) * FPS = 55, + the delay shift)
  f_arr         the hand's arrival: the close command's row, or the first right-hand / object contact from the entry
                command (cmd2) on if earlier; the episode's last row if neither
  moved         the object (body origin) more than MOVE_CM from its pose at f_look, or turned more than MOVE_DEG,
                at some row in f_look .. f_arr - 1
  onset         the first row of that disturbance: the last row before the first MOVE_CM / MOVE_DEG exceedance at
                which the object was still within 0.5 cm and 3 deg of its look pose, + 1
  phase         at the onset: first walk (f_look .. the first wait that turned into a correction walk), correction
                walk (.. the reach's cmd0 / cmd1), reach1 (.. cmd2), reach2 (.. close), other
  near          at the onset: the closest of left wrist / right wrist / pelvis to the object (body origins, cm)
  at_close_cm   the object's displacement from its look pose at f_arr (how stale the plan's grasp is)
With --hits (step logs from fd_run.py with FD_HITLOG=1): the robot bodies in contact with the object at the onset
+- 0.2 s and their summed normal force (the hitter), and whether any robot body touched it before f_arr.

    python fd_q1_moved.py --out <dir> [--root <fd out with rerun/>] [--hits]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FD = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs/selfmodel_faildiag_20261126")
TASKS = {"IThorLivingRoomApproachTableAndLiftDumbbell": "dumbbell", "IThorLivingRoomApproachTableAndLiftWateringCan": "wcan"}
FPS = 50
F_LOOK0 = 55
MOVE_CM, MOVE_DEG = 2.0, 15.0


def qang(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def episode(r: dict, z, hits: bool) -> dict:
    rows = z["rows"]
    C = {str(c): i for i, c in enumerate(z["cols"])}
    g = lambda *ks: rows[:, [C[k] for k in ks]]  # noqa: E731
    obj, oq = g("obj_px", "obj_py", "obj_pz"), g("obj_qw", "obj_qx", "obj_qy", "obj_qz")
    n = len(rows)
    f_look = F_LOOK0 + max(int(r.get("shift_f") or 0), 0)
    w = {x["key"]: int(x["fire"]) for x in r.get("waits", []) if x.get("fire") is not None}
    hand = np.nonzero(rows[:, C["hand_obj_n"]] > 0)[0]
    # the grasp's arrival: a right-hand contact counts only from the entry command (cmd2) on; earlier ones (a brush
    # while walking, the pre-grasp reach) are disturbances like any other body's
    hand = hand[hand >= w["cmd2"]] if "cmd2" in w else hand[:0]
    cands = [v for v in (w.get("close"), int(hand[0]) if len(hand) else None) if v is not None]
    f_arr = min(cands) if cands else n - 1
    f_arr = min(f_arr, n - 1)
    dp = 100 * np.linalg.norm(obj - obj[f_look], axis=1)
    da = np.array([qang(oq[f_look], q) for q in oq])
    out = dict(f_look=f_look, f_arr=f_arr, arr_by="close" if w.get("close") == f_arr else "hand contact" if len(hand) and int(hand[0]) == f_arr else "end",
               at_arr_cm=float(dp[f_arr]), at_arr_deg=float(da[f_arr]), max_before_cm=float(dp[f_look:f_arr].max()) if f_arr > f_look else 0.0)  # fmt: skip
    bad = np.nonzero((dp[f_look:f_arr] > MOVE_CM) | (da[f_look:f_arr] > MOVE_DEG))[0]
    out["moved"] = bool(len(bad))
    if not len(bad):
        return out
    f1 = f_look + int(bad[0])
    still = np.nonzero((dp[f_look:f1] <= 0.5) & (da[f_look:f1] <= 3.0))[0]
    on = f_look + int(still[-1]) + 1 if len(still) else f_look
    # phases from the record: walks (walk_rec / the first command's schedule), the commands
    c_first = min([w[k] for k in ("cmd0", "cmd1") if k in w], default=None)
    fixes = [int(x["fire"]) for x in r.get("waits", []) if x.get("then") == "correction walk" and x.get("fire") is not None]
    if c_first is None or on < c_first:
        ph = "first walk" if not fixes or on < fixes[0] else "correction walk"
    elif "cmd2" not in w or on < w["cmd2"]:
        ph = "reach1"
    elif "close" not in w or on < w["close"]:
        ph = "reach2"
    else:
        ph = "other"
    P = {"left wrist": g("wl_px", "wl_py", "wl_pz"), "right wrist": g("wr_px", "wr_py", "wr_pz"), "pelvis": g("pel_px", "pel_py", "pel_pz")}
    near = {k: 100 * float(np.linalg.norm(v[on] - obj[on])) for k, v in P.items()}
    rh = np.nonzero(rows[f_look:f_arr, C["hand_obj_n"]] > 0)[0]
    out.update(right_hand_touch_before=bool(len(rh)), right_hand_first=f_look + int(rh[0]) if len(rh) else None)
    out.update(onset=on, onset_s=(on - f_look) / FPS, phase=ph, walk1_end=fixes[0] if fixes else None, near=near, nearest=min(near, key=near.get), f_cmd1=c_first, arm_furn_onset=bool(rows[max(on - 10, 0) : on + 10, C["arm_furn"]].any()))  # fmt: skip
    if hits and "obj_hits" in z.files:
        H = np.asarray(z["obj_hits"], float)
        names = json.loads(str(z["names"]))["bodies"]
        robot = [i for i, nm in enumerate(names) if nm.startswith("robot0_")]
        Hr = H[np.isin(H[:, 1].astype(int), robot)]
        win = Hr[(Hr[:, 0] >= on - 10) & (Hr[:, 0] <= on + 10)]
        agg = defaultdict(float)
        for fr, b, fn in win:
            agg[names[int(b)]] += fn
        w1 = Hr[(Hr[:, 0] >= on - 50) & (Hr[:, 0] <= on + 10)]  # from 1 s before the onset (the onset rule lags a light first touch)
        agg1 = defaultdict(float)
        for fr, b, fn in w1:
            agg1[names[int(b)]] += fn
        out["hit_bodies_1s"] = {k: round(v, 1) for k, v in sorted(agg1.items(), key=lambda kv: -kv[1])}
        pre = Hr[(Hr[:, 0] >= f_look) & (Hr[:, 0] < f_arr)]
        first = pre[0] if len(pre) else None
        out.update(hit_bodies={k: round(v, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])},
                   first_robot_touch=None if first is None else dict(row=int(first[0]), body=names[int(first[1])], s_from_onset=(first[0] - on) / FPS),
                   robot_touch_before_arr=sorted({names[int(b)] for b in pre[:, 1]}))  # fmt: skip
        oth = H[~np.isin(H[:, 1].astype(int), robot) & (H[:, 0] >= on - 10) & (H[:, 0] <= on + 10)]
        out["other_bodies_onset"] = sorted({names[int(b)] for b in oth[:, 1]})
    if hits and "sup_hits" in z.files and len(z["sup_roots"]):
        # FD_HITLOG=2: the object's supports (furniture it rests on) -- did they move first, and what touched them
        roots = [int(v) for v in z["sup_roots"]]
        sp = np.asarray(z["sup_pos"], float).reshape(len(rows), len(roots), 3)
        dsp = np.nan_to_num(100 * np.linalg.norm(sp - sp[f_look][None], axis=2), nan=0.0).max(axis=1)
        mv = np.nonzero(dsp[f_look:] > 0.5)[0]
        SH = np.asarray(z["sup_hits"], float).reshape(-1, 4)
        names = json.loads(str(z["names"]))["bodies"]
        win = SH[(SH[:, 0] >= on - 25) & (SH[:, 0] <= on + 10)]
        agg = defaultdict(float)
        for fr, rb, ob, fn in win:
            agg[names[int(rb)]] += fn
        out.update(support=[names[r] for r in roots], support_moved_cm_at_onset=float(dsp[on]), support_moved_cm_at_arr=float(dsp[f_arr]),
                   support_first_move_s_from_onset=None if not len(mv) else (f_look + int(mv[0]) - on) / FPS,
                   support_hitters={k: round(v, 1) for k, v in sorted(agg.items(), key=lambda kv: -kv[1])})  # fmt: skip
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(FD))
    ap.add_argument("--hits", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    per = []
    for f in sorted(glob.glob(str(root / "rerun" / "*" / "episodes" / "*" / "*" / "ep*.json"))):
        p = Path(f)
        be, task, m = p.parts[-5], p.parts[-3], p.parts[-2]
        if task not in TASKS:
            continue
        r = json.loads(p.read_text())
        sl = root / "rerun" / be / "steplog" / task / m / p.with_suffix(".npz").name
        row = dict(backend=be, method=m, task=TASKS[task], ep=r["ep"], reason=r.get("reason"), success=bool(r.get("success")), plan=bool(r.get("plan") and r["plan"].get("fail") is None))
        if row["plan"] and sl.exists():
            z = np.load(sl)
            row.update(episode(r, z, a.hits))
            if a.hits:  # the hit-logged rerun against question 1's rerun: the same rows, bit for bit
                ref = FD / "rerun" / be / "steplog" / task / m / sl.name
                row["same_rows"] = bool(ref.exists() and np.array_equal(np.load(ref)["rows"], z["rows"]))
        per.append(row)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("q1_moved_hits.json" if a.hits else "q1_moved.json")).write_text(json.dumps(per, indent=0, default=float))
    print(f"episodes {len(per)}; with a plan and a step log {sum('moved' in x for x in per)}; object moved > {MOVE_CM} cm or > {MOVE_DEG} deg from its look pose before the hand arrived:")
    for be in sorted({x["backend"] for x in per}):
        for tn in TASKS.values():
            S = [x for x in per if x["backend"] == be and x["task"] == tn and "moved" in x]
            if not S:
                continue
            M = [x for x in S if x["moved"]]
            print(f"  {be:8s} {tn:8s}: {len(M)}/{len(S)} moved; by method {dict(Counter(x['method'] for x in M))} of {dict(Counter(x['method'] for x in S))}; "
                  f"outcome of the moved {dict(Counter(x['reason'] for x in M))}; phase at the onset {dict(Counter(x['phase'] for x in M))}; "
                  f"nearest at the onset {dict(Counter(x['nearest'] for x in M))}")  # fmt: skip
            if M:
                print(f"      moved: displacement at the hand's arrival p50 {np.median([x['at_arr_cm'] for x in M]):.1f} cm / {np.median([x['at_arr_deg'] for x in M]):.0f} deg; "
                      f"onset s after the look p50 {np.median([x['onset_s'] for x in M]):.1f}; outcome of the not moved {dict(Counter(x['reason'] for x in S if not x['moved']))}")  # fmt: skip
                if a.hits:
                    H = [x for x in M if "hit_bodies" in x and x.get("same_rows")]
                    print(f"      bit-identical to question 1's rerun: {sum(bool(x.get('same_rows')) for x in S)}/{len(S)}; read below: {len(H)} moved")
                    print(f"      hitter at the onset +- 0.2 s (robot body with the largest summed normal force): {dict(Counter(next(iter(x['hit_bodies']), 'no robot contact') for x in H).most_common())}")
                    print(f"      hitter from 1 s before to 0.2 s after the onset: {dict(Counter(next(iter(x['hit_bodies_1s']), 'no robot contact') for x in H).most_common())}")
                    grp = lambda b: "left hand" if b.startswith("robot0_left_hand") else "right hand" if b.startswith("robot0_right_hand") else "left arm" if b.startswith("robot0_left") and any(t in b for t in ("wrist", "elbow", "shoulder")) else "right arm" if b.startswith("robot0_right") and any(t in b for t in ("wrist", "elbow", "shoulder")) else "legs" if any(t in b for t in ("hip", "knee", "ankle")) else "torso / pelvis" if b.startswith("robot0_") else b  # noqa: E731
                    print(f"      the same by body part: {dict(Counter(grp(next(iter(x['hit_bodies_1s']), 'no robot contact')) for x in H).most_common())}; by phase x part {dict(Counter((x['phase'], grp(next(iter(x['hit_bodies_1s']), 'no robot contact'))) for x in H).most_common())}")  # fmt: skip
                    print(f"      first robot body to touch the object after the look: {dict(Counter((x['first_robot_touch'] or {}).get('body', 'none') for x in H).most_common())}; "
                          f"its time from the onset s p50 {np.median([x['first_robot_touch']['s_from_onset'] for x in H if x['first_robot_touch']]) if any(x['first_robot_touch'] for x in H) else float('nan'):.2f}")  # fmt: skip
                    print(f"      no robot contact at the onset: non-robot bodies then {dict(Counter(b for x in H if not x['hit_bodies'] for b in x['other_bodies_onset']).most_common(6))}")
                    V = [x for x in M if "support_hitters" in x and x.get("same_rows")]
                    if V:
                        print(f"      support logged (FD_HITLOG=2): {len(V)}; support moved > 0.5 cm by the onset {sum(x['support_moved_cm_at_onset'] > 0.5 for x in V)}; "
                              f"support's first move s from the onset p50 {np.median([x['support_first_move_s_from_onset'] for x in V if x['support_first_move_s_from_onset'] is not None]) if any(x['support_first_move_s_from_onset'] is not None for x in V) else float('nan'):.2f}; "
                              f"robot body on the support (0.5 s before .. 0.2 s after the onset, largest force) {dict(Counter(next(iter(x['support_hitters']), 'none') for x in V).most_common())}")  # fmt: skip


if __name__ == "__main__":
    main()
