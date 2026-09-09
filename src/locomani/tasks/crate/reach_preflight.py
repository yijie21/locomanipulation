"""Pre-flight for ONE continuous Cartesian reach: pad centre from the side pose
along a cubic Bezier (up along the body, then over the rim, into the hover),
IK warm-started sample to sample. Reports contacts and finger clearance over
the bar at four stand-offs, on the 0.62 m scene."""

import argparse
import json
import os as _os
import sys

import mujoco
import numpy as np

_os.chdir(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))))
)  # repo root: the scene/output paths below are repo-relative
from scipy.spatial.transform import Rotation as Rot  # noqa: E402
from scipy.spatial.transform import Slerp  # noqa: E402

from locomani.backends.sonic.primitives_const import GRASP_CENTRE_OFFSET_CONST as OFF  # noqa: E402
from locomani.primitives.arm_ik import ArmIK  # noqa: E402
from locomani.primitives.g1_joints import ARM_MJ  # noqa: E402

# Argparse flags, same names/defaults as the old env vars (decision 8: a
# diagnostic tool's knobs are CLI flags, not config-dataclass fields).
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--sag", type=float, default=0.06)
_ap.add_argument("--pw", type=float, default=0.3)
_ap.add_argument("--up", type=float, default=0.25)
_ap.add_argument("--over", type=float, default=0.10)
_ap.add_argument("--out-dist", type=float, default=0.0, dest="out_dist", help="was OUT (lateral bezier control offset)")
_ap.add_argument("--warm", default="prev")
_ap.add_argument("--rot", default="pitch")
_ap.add_argument("--pitch0", type=float, default=90.0)
_ap.add_argument("--save", default=None, help="dump the joint trajectory to <SAVE>_so<stand-off>.npy")
_args = _ap.parse_args()

out = "outputs/scene/crate_bimanual"
meta = json.load(open(f"{out}/scene_meta.json"))
xml = f"{out}/scene_locomani.xml"
ik = ArmIK(xml)
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
ik.lo[12] = ik.hi[12] = 0.0
ik.lo[13] = ik.hi[13] = 0.0
ik.lo[14] = ik.hi[14] = 0.0
for sd in ("left", "right"):
    ik.lo[ARM_MJ[sd][0]] = np.radians(-80)
top = meta["surfaces"]["A"]["top"]
edge_x = meta["surfaces"]["A"]["xy"][0] - meta["surfaces"]["A"]["flat_half_extent"][0]
crate0 = np.asarray(meta["objects"]["blue_crate"]["init_pos"])
cq = meta["objects"]["blue_crate"]["init_quat"]
NEAR_X = 0.1347
WALL_Y = 0.1267
TOP_Z = 0.085
BIAS = 0.012
near_face = crate0[0] - NEAR_X
bar_top = crate0[2] + TOP_Z
PZ = 0.79
SAG = _args.sag
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, meta["objects"]["blue_crate"]["body"])
cadr = int(m.jnt_qposadr[m.body_jntadr[cid]])
pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
rb = {b for b in range(m.nbody) if m.body_rootid[b] == m.body_rootid[pel]}
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
finger_g = [g for g in range(m.ngeom) if "dex1_finger" in bname(m.geom_bodyid[g])]
grip_q = [
    m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in (
        "left_hand_gripper_1_joint",
        "left_hand_gripper_2_joint",
        "right_hand_gripper_1_joint",
        "right_hand_gripper_2_joint",
    )
]
PW = _args.pw
UP = _args.up
OVER = _args.over
PITCH = 60.0
OUT = _args.out_dist
WARM = _args.warm


def mirror29(left_deg):
    q = np.zeros(29)
    q[ARM_MJ["left"]] = np.radians(left_deg)
    q[ARM_MJ["right"]] = np.radians(left_deg) * np.array([1, -1, -1, 1, -1, 1, -1])
    return q


q_side = mirror29((10, 10, 0, 60, 0, 30, 0))


def place(q29, so):
    mujoco.mj_resetData(m, d)
    d.qpos[ik.base_qadr : ik.base_qadr + 3] = [near_face - so, 0, PZ]
    d.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = [1, 0, 0, 0]
    d.qpos[ik.qadr] = q29
    d.qpos[grip_q] = 0.0245
    d.qpos[cadr : cadr + 3] = crate0
    d.qpos[cadr + 3 : cadr + 7] = cq
    mujoco.mj_forward(m, d)


def contacts():
    bad = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in rb) != (b2 in rb) and c.dist < 0.0:
            mine, other = (b1, b2) if b1 in rb else (b2, b1)
            if bname(other) != "world":
                bad.add((bname(mine).replace("_link", ""), bname(other).split("_")[0]))
    return sorted(bad)


def Rpitch(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def pad(q, sd):
    pos, rot = ik.fk(q, sd)
    return pos + rot @ np.asarray(OFF), rot


def bezier(p0, p1, p2, p3, u):
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u * u * p2 + u**3 * p3


print(f"scene: table {top:.2f}, bar top {bar_top:.3f}, pelvis {PZ}; UP {UP} OVER {OVER} sag budget {SAG}")
for so in (0.13, 0.18, 0.24, 0.31):
    depth = float(np.clip(0.38 - so, 0.06, 0.11))
    px = near_face - so
    hov = {
        sd: np.array([near_face + depth, (WALL_Y - BIAS) * (1 if sd == "left" else -1), bar_top + 0.06])
        - np.array([px, 0, PZ])
        for sd in ("left", "right")
    }
    q_h = np.zeros(29)  # like the expert's solve_for: ONE pass, 200 iters, from zeros, waist bounded to 0
    for sd in ("left", "right"):
        q_h, _, _ = ik.solve(
            sd, hov[sd], Rpitch(PITCH), q_h, iters=200, use_waist=True, waist_cost=3.0, point_offset=OFF
        )
    q = q_side.copy()
    p0 = {sd: pad(q, sd)[0] for sd in ("left", "right")}
    r0 = {sd: pad(q, sd)[1] for sd in ("left", "right")}
    traj = []
    worst = None
    hits = set()
    ikmax = 0.0
    L = 0.0
    jumps = []
    for sd in ("left", "right"):
        L = max(
            L,
            np.linalg.norm(p0[sd] + [0, 0, UP] - p0[sd])
            + np.linalg.norm(hov[sd] + [0, 0, OVER] - (p0[sd] + [0, 0, UP]))
            + OVER,
        )
    N = 60
    for k in range(1, N + 1):
        u = k / N
        for sd in ("left", "right"):
            sgn = 1 if sd == "left" else -1
            tg = bezier(p0[sd], p0[sd] + [0, sgn * OUT, UP], hov[sd] + [0, 0, OVER], hov[sd], u)
            mode = _args.rot
            if mode == "slerp":
                rt = Slerp([0, 1], Rot.from_matrix([r0[sd], Rpitch(PITCH)]))(u).as_matrix()
            elif mode == "pitch":
                rt = Rpitch(_args.pitch0 + (PITCH - _args.pitch0) * u)
            else:
                rt = None if u < 0.6 else Rpitch(PITCH)
            q_prev = q.copy()
            if WARM == "zeros":
                q0 = q.copy()
                q0[ARM_MJ[sd]] = 0.0
                q, _, _ = ik.solve(sd, tg, rt, q0, iters=120, use_waist=False, point_offset=OFF)
            else:
                q, _, _ = ik.solve(
                    sd, tg, rt, q, iters=60, use_waist=False, point_offset=OFF, posture=q_h, posture_weight=PW
                )
            e = np.linalg.norm(pad(q, sd)[0] - tg)
            ikmax = max(ikmax, e)
            jump = float(np.degrees(np.abs(q[ARM_MJ[sd]] - q_prev[ARM_MJ[sd]]).max()))
            jumps.append(jump)
        place(q, so)
        hits.update(contacts())
        traj.append(q.copy())
        p = d.geom_xpos[finger_g]
        over = (
            (p[:, 0] + 0.015 > near_face - 0.01) & (np.abs(p[:, 1] - crate0[1]) < 0.22) & (p[:, 0] < near_face + 0.42)
        )
        if over.any():
            clr = float(p[over, 2].min()) - 0.015 - bar_top
            if worst is None or clr < worst[0]:
                worst = (clr, u)
        # table-edge plane check: any finger below the table top while within 3 cm of the edge plane
        near_edge = (p[:, 0] + 0.015 > edge_x - 0.03) & (p[:, 2] - 0.015 < top)
        if near_edge.any():
            hits.add(("finger", f"edge-plane@u={u:.2f}"))
    print(
        f"so {so:.2f} depth {depth:.2f}: path {L:.2f} m, IK max err {ikmax * 100:.1f} cm, max joint step {max(jumps[2:]):.0f} deg | min finger clearance over the bar {worst[0] * 100:+.1f} cm at u={worst[1]:.2f} (minus sag {(worst[0] - SAG) * 100:+.1f}) | contacts {sorted(hits) or '-'}"
    )
    print(
        "   final arm q L",
        np.degrees(q[ARM_MJ["left"]]).round(0).tolist(),
        " hover(zeros) L",
        np.degrees(q_h[ARM_MJ["left"]]).round(0).tolist(),
    )
    if _args.save:
        np.save(f"{_args.save}_so{so:.2f}.npy", np.array(traj))
