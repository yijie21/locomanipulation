"""Development analysis (SIMPLE's venv; numpy): split each episode's reach error into the reference's own error (FK of
the joints sent, vs the perceived pregrasp: what the method asked for) and the tracking error (FK of the measured joints
vs FK of the joints sent), both in the pelvis frame, at the settle's end; and the arm's mean |sent - measured| (rad).

    .venv/bin/python reach_split.py <bench out dir>
"""

import glob
import json
import sys

import g1arm
import numpy as np

ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "waist_roll_joint", "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint", "left_knee_joint",
    "right_knee_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint", "left_ankle_pitch_joint",
    "right_ankle_pitch_joint", "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
    "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint", "left_elbow_joint",
    "right_elbow_joint", "left_wrist_roll_joint", "right_wrist_roll_joint", "left_wrist_pitch_joint",
    "right_wrist_pitch_joint", "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]  # fmt: skip
OFF = {"ppbox": (0.06, 0.06, 0.0), "doubledesk": (0.07, 0.04, 0.015), "opendoor": (0.07, 0.04, 0.015)}


def main() -> int:
    out = sys.argv[1]
    ch = g1arm.Chain(tip="right_hand_palm_link")
    ai = [ORDER.index(n) for n in g1arm.ARM["right"]]
    for f in sorted(glob.glob(f"{out}/*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            x = r.get("reach")
            if not x:
                continue
            z = np.load(f"{out}/film/{r['task']}_{r['method']}_s{r['seed']}_r{r['repeat']}.npz")
            k = x["settle_k"] - 1
            off = OFF[r["task"]]
            pr = ch.point(dict(zip(ORDER, z["ref"][k].astype(float))), off)[0]
            pm = ch.point(dict(zip(ORDER, z["q"][k].astype(float))), off)[0]
            tgt = np.array(r["perceived"]["pregrasp"])
            arm = np.abs(z["ref"][x["reach_k1"] : x["settle_k"], ai] - z["q"][x["reach_k1"] : x["settle_k"], ai]).mean()
            print(f"{r['task']:>10} {r['method']} s{r['seed']} r{r['repeat']}: total {x['hand_err_cm']:5.1f} cm | "
                  f"reference {100 * np.linalg.norm(pr - tgt):5.1f} | tracking {100 * np.linalg.norm(pm - pr):5.1f} | "
                  f"arm |sent-meas| {arm:.3f} rad | plan {x.get('plan_status', '-')}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
