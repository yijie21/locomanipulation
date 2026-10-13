"""SONIC's bimanual grasp posture, read from a warehouse run's traces: at the first tick of each phase
(descend / close / stand) of every successful episode, pelvis height and tilt, torso pitch, and both
wrists' pose in the pelvis-yaw frame. Writes all three phases; w0_admission.py takes `close` as the bend target.

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/experiments/controller_compare_20261013/sonic_grasp_posture.py \
        outputs/runs/warehouse_mars_guard_off_20261008 warehouse_box_45x32x33_mars outputs/controller_compare_20261013/w0/posture_33_mars.json
"""

import glob
import json
import sys

import mujoco
import numpy as np

run, scene, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
m = mujoco.MjModel.from_xml_path(f"outputs/scene/{scene}/scene_locomani.xml")
d = mujoco.MjData(m)
B = {
    n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    for n in ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link", "nor_cal_box_object")
}


def yaw(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mean_rot(Rs):
    u, _, vt = np.linalg.svd(np.sum(Rs, 0))
    return u @ vt


rows = []
for od in sorted(glob.glob(f"{run}/observations/*/")):
    lab = od.rstrip("/").split("/")[-1]
    if not json.load(open(od + "result.json"))["ok"]:
        continue
    cmd = [json.loads(l) for l in open(od + "commands.jsonl")]
    seq = {r["phase"]: r["state_seq"] for r in reversed(cmd)}
    tr = np.load(f"{run}/trace/{lab}_state_trace.npz")
    for ph in ("descend", "close", "stand"):
        if ph not in seq:
            continue
        i = int(np.argmin(abs(tr["seq"] - seq[ph])))
        d.qpos[:] = tr["qpos"][i]
        mujoco.mj_forward(m, d)
        pp = d.xpos[B["pelvis"]].copy()
        Rp = rz(yaw(d.xquat[B["pelvis"]]))
        rel = lambda b: Rp.T @ (d.xpos[b] - pp)
        relR = lambda b: Rp.T @ d.xmat[b].reshape(3, 3)
        tor = d.xmat[B["torso_link"]].reshape(3, 3)
        rows.append(
            dict(
                ph=ph,
                pel_z=pp[2],
                pel_R=relR(B["pelvis"]),
                torso_pitch=np.degrees(np.arcsin(-tor[2, 0])),
                lw=rel(B["left_wrist_yaw_link"]),
                rw=rel(B["right_wrist_yaw_link"]),
                lR=relR(B["left_wrist_yaw_link"]),
                rR=relR(B["right_wrist_yaw_link"]),
                box=rel(B["nor_cal_box_object"]),
                box_top_z=d.xpos[B["nor_cal_box_object"]][2],
            )
        )
res = {}
for ph in ("descend", "close", "stand"):
    rs = [r for r in rows if r["ph"] == ph]
    A = lambda k: np.array([r[k] for r in rs])
    res[ph] = dict(
        n=len(rs),
        pelvis_z=float(A("pel_z").mean()),
        pelvis_z_sd=float(A("pel_z").std()),
        pelvis_R=mean_rot(A("pel_R")).tolist(),
        torso_pitch_deg=float(A("torso_pitch").mean()),
        left_wrist=A("lw").mean(0).tolist(),
        right_wrist=A("rw").mean(0).tolist(),
        left_wrist_R=mean_rot(A("lR")).tolist(),
        right_wrist_R=mean_rot(A("rR")).tolist(),
        box=A("box").mean(0).tolist(),
        box_origin_z=float(A("box_top_z").mean()),
    )
    print(
        ph,
        rs
        and "n %d pel_z %.3f torso %.1f° lw %s rw %s box %s"
        % (
            len(rs),
            res[ph]["pelvis_z"],
            res[ph]["torso_pitch_deg"],
            np.round(res[ph]["left_wrist"], 3),
            np.round(res[ph]["right_wrist"], 3),
            np.round(res[ph]["box"], 3),
        ),
    )
json.dump(dict(run=run, scene=scene, phases=res), open(out_json, "w"), indent=1)
