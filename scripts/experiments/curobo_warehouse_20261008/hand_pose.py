"""CuRobo warehouse study, stage 1b: is the expert's grasp pose itself free of the carton?

The arm does not matter for this question: the hand (wrist_yaw_link with the Dex1 base and the D405 mount,
and both fingers at the opening recorded at the end of descend) is placed directly at a grasp pose and its collision geoms are
measured against the carton walls and the platform (sampled surface-to-box distance, cases.Surfaces). Each case uses its box pose at the first reach frame.

Poses: the expert's nominal grasp (grip point and palm attitude in the box frame, the mean over the run's
completed rounds, see cases.json) and a sweep of grip depth below the rim x finger tilt about the across-wall
axis (the jaw axis), which keeps the jaws straddling the wall. Signed distances, negative = penetration.

    PYTHONPATH=src "$VENV_SIM_PY" scripts/experiments/curobo_warehouse_20261008/hand_pose.py \
        outputs/curobo_warehouse_20261008/cases.json outputs/curobo_warehouse_20261008/hand_pose.json
"""

from __future__ import annotations

import argparse
import json
import os

import mujoco
import numpy as np
from cases import DIST_MAX, Surfaces, quat_wxyz_to_matrix
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.bimanual_carry import OFFSET

SIDES = ("left", "right")
RIM_Z = 0.33  # carton height in its frame (warehouse_box_45x32x33)
DEPTHS = (0.005, 0.01, 0.02, 0.03, 0.04)  # grip point below the rim, m
TILTS = tuple(range(-60, 41, 10))  # deg about the across-wall axis (box x): finger angle within the wall plane


class Hand:
    """Collision geoms of one hand, posed rigidly from a palm pose and the two finger slides."""

    def __init__(self, m, side):
        self.m = m
        palm = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link")
        fingers = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{side}_dex1_finger_{k}_link") for k in (1, 2)]
        self.jaw_qadr = [m.jnt_qposadr[m.body_jntadr[b]] for b in fingers]
        self.parts = []  # (geom, name, finger index or None, T_palm_body at zero slide, slide axis in palm, T_body_geom)
        for k, b in enumerate([palm] + fingers):
            T_pb, axis = np.eye(4), None
            if b != palm:
                T_pb[:3, :3] = quat_wxyz_to_matrix(m.body_quat[b])
                T_pb[:3, 3] = m.body_pos[b]
                axis = T_pb[:3, :3] @ m.jnt_axis[m.body_jntadr[b]]
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                    continue
                T_bg = np.eye(4)
                T_bg[:3, :3] = quat_wxyz_to_matrix(m.geom_quat[g])
                T_bg[:3, 3] = m.geom_pos[g]
                part = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b).replace(f"{side}_", "")
                self.parts.append((g, f"{part}#{g}", k - 1 if axis is not None else None, T_pb, axis, T_bg))

    def jaws(self, qpos):
        return [float(qpos[a]) for a in self.jaw_qadr]

    def place(self, d, grip, R, jaws):
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = grip - R @ OFFSET
        for g, _, finger, T_pb, axis, T_bg in self.parts:
            T_slide = T_pb.copy()
            if finger is not None:
                T_slide[:3, 3] = T_pb[:3, 3] + axis * jaws[finger]
            Tg = T @ T_slide @ T_bg
            d.geom_xpos[g] = Tg[:3, 3]
            d.geom_xmat[g] = Tg[:3, :3].ravel()


def clearance(surfaces, d, hand, obstacles, grip, R, jaws):
    hand.place(d, grip, R, jaws)
    worst = (DIST_MAX, None, None)
    per_part = {}
    for g, part, *_ in hand.parts:
        for g_obs, obs_name in obstacles:
            dist = surfaces.to_box(d, g, g_obs)
            per_part[part] = min(per_part.get(part, DIST_MAX), dist)
            if dist < worst[0]:
                worst = (dist, part, obs_name)
    return dict(dist=worst[0], part=worst[1], obstacle=worst[2], parts=per_part)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cases")
    ap.add_argument("out")
    args = ap.parse_args()

    data = json.load(open(args.cases))
    meta, cases = data["meta"], [c for c in data["cases"] if "start" in c]
    m = mujoco.MjModel.from_xml_path(meta["scene"])
    d = mujoco.MjData(m)
    box_qadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "nor_cal_box_free")]
    obstacle_names = set(meta["obstacles"])
    obstacles = [
        (g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        for g in range(m.ngeom)
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) in obstacle_names
    ]

    done = [c for c in cases if c["ok"] and c.get("descend_end_target")]
    grip_box = np.mean([[h["grip_box"] for h in c["descend_end_target"]["hands"]] for c in done], axis=0)
    quat_box = np.mean([[h["quat_box"] for h in c["descend_end_target"]["hands"]] for c in done], axis=0)
    quat_box /= np.linalg.norm(quat_box, axis=1, keepdims=True)
    nominal = dict(grip_box=grip_box.tolist(), quat_box=quat_box.tolist(), from_rounds=len(done))
    print("nominal grasp (box frame)", np.round(grip_box, 3).tolist(), np.round(quat_box, 3).tolist())

    hands = [Hand(m, s) for s in SIDES]
    surfaces = Surfaces(m, [g for hand in hands for g, *_ in hand.parts])
    rows = []
    for c in cases:
        q = np.asarray(c["start"]["qpos"])
        grasp_q = np.asarray(c.get("descend_end", c["start"])["qpos"])  # the jaws as they arrived at the wall
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        bq = q[box_qadr : box_qadr + 7]
        Rb, pb = quat_wxyz_to_matrix(bq[3:]), bq[:3]
        row = dict(label=c["label"], ok=c["ok"], why=c["why"], jaws=[hand.jaws(grasp_q) for hand in hands])
        row.update(nominal=[], sweep=[])
        for h, hand in enumerate(hands):
            R_box = quat_wxyz_to_matrix(quat_box[h])
            jaws = row["jaws"][h]
            row["nominal"].append(clearance(surfaces, d, hand, obstacles, pb + Rb @ grip_box[h], Rb @ R_box, jaws))
            for depth in DEPTHS:
                for tilt in TILTS:
                    p_box = grip_box[h].copy()
                    p_box[2] = RIM_Z - depth
                    R_sweep = Rotation.from_euler("x", np.radians(tilt)).as_matrix() @ R_box
                    r = clearance(surfaces, d, hand, obstacles, pb + Rb @ p_box, Rb @ R_sweep, jaws)
                    row["sweep"].append(dict(hand=SIDES[h], depth=depth, tilt=tilt, dist=r["dist"], part=r["part"]))
        rows.append(row)
        nom = min(r["dist"] for r in row["nominal"])
        print(f"{c['label']} ok={c['ok']} nominal worst {nom:+.4f}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(meta=dict(nominal=nominal, depths=DEPTHS, tilts=TILTS, rim_z=RIM_Z), rows=rows), f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
