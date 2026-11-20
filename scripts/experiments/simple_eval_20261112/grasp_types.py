"""The chosen grasp's kind per task and look: the palm's facing (roll 0 = sideways, 90 = down) and height (2026-11-20).
python -I grasp_types.py <outputs/simple_eval_20261112> <cloud dir> ..."""

import glob
import json
import os
import sys

import numpy as np

O = sys.argv[1]
AHEAD, ROLL, CLEAR = 0.7, (-30.0, 110.0), 0.06


def choose(d, name, yaw):
    if not os.path.exists(f"{O}/{d}/grasp_{name}.npz"):
        return None
    g, cl = np.load(f"{O}/{d}/grasp_{name}.npz"), np.load(f"{O}/{d}/{name}.npz")
    T = g["T_op"].copy()
    T[:, :3, 3] += cl["p_o"]
    a, l = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    Rw = T[:, :3, :3]
    roll = np.degrees(np.arctan2(-Rw[:, 2, 1], Rw[:, :, 1] @ l))
    foot = np.percentile(cl["pts"][:, 2], 2)
    can = (Rw[:, :, 0] @ a >= AHEAD) & (roll >= ROLL[0]) & (roll <= ROLL[1]) & (T[:, 2, 3] >= foot + CLEAR)
    if not can.any():
        return None
    k = int(np.argmax(can))
    return float(roll[k]), float(T[k, 2, 3]), float(T[k, 2, 3] - foot), float(-Rw[k, 2, 1])


res = {}
for task in ["XMovePickTeleop", "XMoveBendPickTeleop", "BendPickMP", "TabletopGraspMP"]:
    for d in sys.argv[2:]:
        rows = []
        for f in sorted(glob.glob(f"{O}/look_dump_eyes/{task}_L*_e*.npz")):
            name = f.split("/")[-1][:-4]
            c = choose(d, name, float(np.load(f)["yaw"]))
            if c:
                rows.append((name,) + c)
        r = np.array([x[1] for x in rows])
        z = np.array([x[2] for x in rows])
        up = np.array([x[3] for x in rows])
        res[f"{task}/{d}"] = dict(
            n=len(rows),
            down=int((r > 60).sum()),
            slant=int(((r > 25) & (r <= 60)).sum()),
            side=int((r <= 25).sum()),
            z_med=round(float(np.median(z)), 3),
            z_down_med=round(float(np.median(z[r > 60])), 3) if (r > 60).any() else None,
            above_foot_med_cm=round(100 * float(np.median(up)), 1),
        )
        print(task, d, json.dumps(res[f"{task}/{d}"]))
        if task == "TabletopGraspMP":
            print("   ", [(n.split("MP_")[1], round(a), round(b, 2)) for n, a, b, c, e in rows][:30])
json.dump(res, open(f"{O}/grasp_types_by_look.json", "w"), indent=1)
