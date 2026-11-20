"""The Isaac-eyes look against the MuJoCo-eyes one, per task: what SAM 3 found, the depth, the chosen grasp (2026-11-19).
python -I look_stats.py <outputs/simple_eval_20261112> <logs dir>"""

import glob
import json
import re
import sys

import numpy as np

O, L = sys.argv[1], sys.argv[2]
AHEAD, ROLL, CLEAR = 0.7, (-30.0, 110.0), 0.06


def choose(d, name, yaw):
    g, cl = np.load(f"{O}/{d}/grasp_{name}.npz"), np.load(f"{O}/{d}/{name}.npz")
    T = g["T_op"].copy()
    T[:, :3, 3] += cl["p_o"]
    a, l = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    Rw = T[:, :3, :3]
    roll = np.degrees(np.arctan2(-Rw[:, 2, 1], Rw[:, :, 1] @ l))
    can = (
        (Rw[:, :, 0] @ a >= AHEAD)
        & (roll >= ROLL[0])
        & (roll <= ROLL[1])
        & (T[:, 2, 3] >= np.percentile(cl["pts"][:, 2], 2) + CLEAR)
    )
    if not can.any():
        return None
    k = int(np.argmax(can))
    return T[k], k, float(g["score"][k]), int(can.sum()), float(cl["iou"]), float(cl["score"])


out = {}
for task in ["XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP"]:
    rows = []
    dep = [float(m) for m in re.findall(r"depth cm: fused (-?[\d.]+)", open(f"{L}/full_look_{task}.log").read())]
    for f in sorted(glob.glob(f"{O}/look_dump_eyes/{task}_L*_e*.npz")):
        name = f.split("/")[-1][:-4]
        yaw = float(np.load(f)["yaw"])
        e, b = choose("cloud_eyes", name, yaw), choose("cloud_blind", name, yaw)
        rows.append((name, e, b))
    ok = [(n, e, b) for n, e, b in rows if e is not None and b is not None]
    dpos = np.array([100 * np.linalg.norm(e[0][:3, 3] - b[0][:3, 3]) for _, e, b in ok])
    dang = np.array(
        [np.degrees(np.arccos(np.clip((np.trace(e[0][:3, :3].T @ b[0][:3, :3]) - 1) / 2, -1, 1))) for _, e, b in ok]
    )
    out[task] = dict(
        n=len(rows),
        no_grasp_eyes=[n for n, e, b in rows if e is None],
        no_grasp_blind=[n for n, e, b in rows if b is None],
        iou_eyes_med=round(float(np.median([e[4] for _, e, b in ok])), 3),
        iou_eyes_min=round(float(min(e[4] for _, e, b in ok)), 3),
        iou_blind_med=round(float(np.median([b[4] for _, e, b in ok])), 3),
        sam_eyes_med=round(float(np.median([e[5] for _, e, b in ok])), 3),
        depth_err_cm_med=round(float(np.median(np.abs(dep))), 2),
        depth_err_cm_max=round(float(np.max(np.abs(dep))), 2),
        n_depth=len(dep),
        grasp_shift_cm_med=round(float(np.median(dpos)), 2),
        grasp_shift_cm_p90=round(float(np.percentile(dpos, 90)), 2),
        grasp_shift_cm_max=round(float(dpos.max()), 2),
        grasp_turn_deg_med=round(float(np.median(dang)), 1),
        shift_over_2cm=int((dpos > 2).sum()),
        shift_over_5cm=int((dpos > 5).sum()),
        can_eyes_med=int(np.median([e[3] for _, e, b in ok])),
        can_blind_med=int(np.median([b[3] for _, e, b in ok])),
    )
    print(task, json.dumps(out[task]))
json.dump(out, open(f"{O}/look_stats_eyes_vs_blind.json", "w"), indent=1)
