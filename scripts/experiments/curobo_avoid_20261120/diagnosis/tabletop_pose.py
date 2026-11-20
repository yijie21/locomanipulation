"""Why TabletopGrasp's pregrasp fails: the chosen grasp's pose in the robot's heading frame against the outcome.

Run from the repo root on the 4090:
    PYTHONPATH=src $PY -I tabletop_pose.py <simple_bench_20261119 dir> [task]
The grasp is re-chosen offline as simple_bench does (pick.graspgen_grasp with the start yaw from the episode's config);
the row's recorded score checks that the same grasp came out.
"""

import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts/experiments/simple_eval_20261112"))
sys.path.insert(0, str(ROOT / "scripts/experiments/reachwm_20261030"))
import episodes as EP  # noqa: E402
import pick as P  # noqa: E402

B, TASK = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "TabletopGraspMP")


def rows(d, m):
    out = {}
    for f in glob.glob(f"{B}/{d}/{m}_{TASK}_*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            out[r["ep"]] = r
    return out


def stage(r, name):
    s = [x for x in r.get("stages") or [] if x["stage"] == name]
    return s[0] if s else None


def start_yaw(ep):
    cfg = ep["cfg"]
    for key, a in cfg["layout"]["actors"].items():
        if key == cfg["robot_cfg"]["uid"] or "robot" in key:
            q = a["pose"]["quaternion"]
            return float(2 * np.arctan2(q[3], q[0]))
    q = cfg["dr_state_dict"]["spatial"][cfg["robot_cfg"]["uid"]]["quaternion"]
    return float(2 * np.arctan2(q[3], q[0]))


eps = EP.episodes(TASK)
print(
    f"{TASK}: fyaw = the fingers' heading from straight ahead (+ left), fpitch (- down), roll as pick.py (90 palm down)"
)
print("miss = the palm at the pregrasp minus the wanted, cm, heading frame (x ahead, y left, z up)")
summary = []
for d, cloud in (("rows_perc", "cloud_blind"), ("rows_eyes", "cloud_eyes_final")):
    D, M = rows(d, "direct"), rows(d, "model")
    print(f"\n== {cloud}")
    print(" ep  target          fyaw fpitch  roll  score-ok | direct pre cm  miss(x,y,z)        ok | model pre cm ok")
    for e in range(30):
        ep = eps[e]
        name = f"{TASK}_L{ep['level']}_e{ep['index']}"
        yaw = start_yaw(ep)
        tgt = (
            ep["cfg"]["layout"]["actors"]["target"]["asset"]["label"]
            if "target" in ep["cfg"]["layout"]["actors"]
            else "?"
        )
        if not (EP.OUT / cloud / f"grasp_{name}.npz").exists():
            print(f"{e:3d}  {tgt:14s}  no grasp")
            continue
        T_wo = np.eye(4)
        T_wo[:3, 3] = np.load(EP.OUT / cloud / f"{name}.npz")["p_o"]
        found = P.graspgen_grasp(name, T_wo, yaw, cloud=cloud)
        if found is None:
            print(f"{e:3d}  {tgt:14s}  none chosen offline")
            continue
        T_og, _, _, _, score = found
        Rw = (T_wo @ T_og)[:3, :3]
        c, s_ = math.cos(-yaw), math.sin(-yaw)
        Rh = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ Rw
        fx = Rh[:, 0]
        fyaw = math.degrees(math.atan2(fx[1], fx[0]))
        fpitch = math.degrees(math.asin(max(-1, min(1, fx[2]))))
        roll = math.degrees(math.atan2(-Rh[2, 1], Rh[1, 1]))
        rd, rm = D[e], M[e]
        same = abs(score - (rd.get("score") or -1)) < 1e-6
        pd, pm = stage(rd, "pregrasp"), stage(rm, "pregrasp")
        miss = np.zeros(3)
        if pd:
            mw = np.asarray(pd["miss"], float)
            miss = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ mw
        print(f"{e:3d}  {tgt:14s} {fyaw:6.1f} {fpitch:6.1f} {roll:6.1f}  {str(same):5s} | "
              f"{pd['err_cm'] if pd else float('nan'):7.2f}  {np.round(miss, 1)!s:22s} {int(rd['success_45'])} | "
              f"{pm['err_cm'] if pm else float('nan'):7.2f} {int(rm['success_45'])}")  # fmt: skip
        summary.append((cloud, e, fyaw, fpitch, roll, pd["err_cm"] if pd else np.nan))

print("\nfar (direct pregrasp > 20 cm) against near, both looks pooled:")
S = np.array([x[2:] for x in summary], float)
far, near = S[:, 3] > 20, S[:, 3] <= 20
for i, k in enumerate(("fyaw", "fpitch", "roll")):
    print(f"  {k:6s} far  {np.round(np.sort(S[far, i]), 0).astype(int).tolist()}")
    print(f"  {k:6s} near {np.round(np.sort(S[near, i]), 0).astype(int).tolist()}")
