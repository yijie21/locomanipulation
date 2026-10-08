"""CuRobo warehouse study, stage 2c: planning requests, the same for CuRobo and for our IK.

Per case, per base and per grasp variant, everything in the base's pelvis frame (the pelvis stays fixed; only
waist + both arms move). Bases: `reach_start` = legs and pelvis as measured at the first reach frame (where the
arms start); `descend_end` = legs and pelvis as measured at the end of descend, i.e. the body already placed the
way the expert's whole-body reach placed it, with the waist and arms still at their reach-start values.
Each request holds:
- start: the 17 measured waist/arm joints (and the full 29 + jaws for the MuJoCo check);
- obstacles: the carton's five walls and the platform as boxes (centre, rotation, half extents);
- goal: both palm (wrist_yaw_link) poses for the variant's grasp (grip point and attitude in the box frame).

Variants: `nominal` is the expert's own grasp (mean over the run's completed rounds, hand_pose.json); the
others are deeper grasps tilted about the jaw axis, from the clear region of the hand-pose sweep.

Joint limits are the expert's: physical ranges, waist pitch <= carry.waist_pitch_max, shoulder yaw and wrist
roll within +-50 deg (BimanualIK), elbow >= visual.elbow_min_deg.

    PYTHONPATH=src:scripts/experiments/curobo_warehouse_20261008 "$VENV_SIM_PY" \
        scripts/experiments/curobo_warehouse_20261008/requests.py outputs/curobo_warehouse_20261008
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np
import yaml
from cases import box_extent, quat_wxyz_to_matrix
from scipy.spatial.transform import Rotation

from locomani.primitives.g1_joints import ARM_MJ
from locomani.tasks.crate.bimanual_carry import OFFSET

RIM_Z = 0.33
VARIANTS = {  # name: (grip depth below the rim or None for the expert's own point, tilt about the jaw axis, deg)
    "nominal": (None, 0.0),
    "d20_t-20": (0.020, -20.0),
    "d30_t-30": (0.030, -30.0),
}
BASES = ("reach_start", "descend_end")
WAIST = [12, 13, 14]
ACTIVE = WAIST + list(ARM_MJ["left"]) + list(ARM_MJ["right"])  # indices into the 29 body joints


def joint_limits(m, hinge_ids, run_config):
    carry, visual = run_config["task"]["carry"], run_config["task"]["visual"]
    lo = np.array([m.jnt_range[j][0] for j in hinge_ids])
    hi = np.array([m.jnt_range[j][1] for j in hinge_ids])
    hi[14] = min(hi[14], float(carry["waist_pitch_max"]))
    for side in ("left", "right"):
        arm = ARM_MJ[side]
        lo[arm[3]] = max(lo[arm[3]], np.radians(float(visual["elbow_min_deg"])))
        for index in (arm[2], arm[4]):
            lo[index], hi[index] = max(lo[index], -np.radians(50)), min(hi[index], np.radians(50))
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir", help="study directory with cases.json and hand_pose.json")
    args = ap.parse_args()

    data = json.load(open(f"{args.dir}/cases.json"))
    meta, cases = data["meta"], [c for c in data["cases"] if "start" in c]
    nominal = json.load(open(f"{args.dir}/hand_pose.json"))["meta"]["nominal"]
    run_config = yaml.safe_load(open(f"{meta['run']}/config.yaml"))
    m = mujoco.MjModel.from_xml_path(meta["scene"])
    d = mujoco.MjData(m)
    hinge_ids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    hinge_qadr = np.array([m.jnt_qposadr[j] for j in hinge_ids])
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in hinge_ids]
    lo, hi = joint_limits(m, hinge_ids, run_config)
    box_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "nor_cal_box_free")]
    obstacles = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) for n in meta["obstacles"]]

    grip_box = np.asarray(nominal["grip_box"])
    R_box = [quat_wxyz_to_matrix(q) for q in nominal["quat_box"]]
    requests = []
    for c, base in [(c, b) for c in cases for b in BASES]:
        if base == "descend_end" and "descend_end" not in c:
            continue
        q = np.asarray(c["start"]["qpos"]).copy()
        if base == "descend_end":
            body = np.asarray(c["descend_end"]["qpos"])
            legs = np.r_[np.arange(7), hinge_qadr[:12]]  # free base + 12 leg joints
            q[legs] = body[legs]
            q[box_qadr : box_qadr + 7] = body[box_qadr : box_qadr + 7]
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        p0, R0 = q[:3], quat_wxyz_to_matrix(q[3:7])
        to_pelvis = lambda x: R0.T @ (np.asarray(x) - p0)  # noqa: E731
        boxes = []
        for g in obstacles:
            offset, half = box_extent(m, g)
            Rg = d.geom_xmat[g].reshape(3, 3)
            boxes.append(
                dict(
                    name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                    center=to_pelvis(d.geom_xpos[g] + Rg @ offset).tolist(),
                    rotation=(R0.T @ Rg).tolist(),
                    half=half.tolist(),
                )
            )
        bq = q[box_qadr : box_qadr + 7]
        Rb, pb = quat_wxyz_to_matrix(bq[3:]), bq[:3]
        goals = {}
        for name, (depth, tilt) in VARIANTS.items():
            palms = []
            for h in range(2):
                p_box = grip_box[h].copy()
                if depth is not None:
                    p_box[2] = RIM_Z - depth
                R = Rb @ Rotation.from_euler("x", np.radians(tilt)).as_matrix() @ R_box[h]
                grip = pb + Rb @ p_box
                palms.append(dict(pos=to_pelvis(grip - R @ OFFSET).tolist(), rot=(R0.T @ R).tolist()))
            goals[name] = palms
        joints = q[hinge_qadr]
        requests.append(
            dict(
                label=c["label"],
                base=base,
                ok=c["ok"],
                why=c["why"],
                qpos=q.tolist(),
                start=[float(joints[i]) for i in ACTIVE],
                obstacles=boxes,
                goals=goals,
            )
        )
    out = dict(
        scene=meta["scene"],
        offset=OFFSET.tolist(),
        active_joints=[names[i] for i in ACTIVE],
        active_index=ACTIVE,
        lower=[float(lo[i]) for i in ACTIVE],
        upper=[float(hi[i]) for i in ACTIVE],
        variants={k: dict(depth=v[0], tilt_deg=v[1]) for k, v in VARIANTS.items()},
        requests=requests,
    )
    with open(f"{args.dir}/requests.json", "w") as f:
        json.dump(out, f)
    print(f"wrote {len(requests)} requests (cases x bases) x {len(VARIANTS)} variants to {args.dir}/requests.json")


if __name__ == "__main__":
    main()
