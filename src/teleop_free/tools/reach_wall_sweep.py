"""Reach sweep: grasp the crate's LONG walls (crate rotated 90 deg so they are
left/right of the robot), fingers forward pitched down, jaw straddling the wall
top. Waist locked, standing height. Grid over stand-off from the crate's near
face, grasp depth along the wall from the near face, pad-centre depth below
the wall top, and palm pitch. Prints one line per config and a summary."""

import itertools
import json
import os as _os
import sys

import mujoco
import numpy as np

sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_os.chdir(
    _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
)  # repo root: the scene/output paths below are repo-relative
from arm_ik import ArmIK
from primitives_const import GRASP_CENTRE_OFFSET_CONST as OFF

out = "outputs/scene/crate_bimanual"
meta = json.load(open(f"{out}/scene_meta.json"))
xml = f"{out}/scene_locomani.xml"
ik = ArmIK(xml)
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
top = meta["surfaces"]["A"]["top"]
hx = meta["surfaces"]["A"]["flat_half_extent"][0]
edge_x = meta["surfaces"]["A"]["xy"][0] - hx
ik.lo[12] = ik.hi[12] = 0.0
ik.lo[13] = ik.hi[13] = 0.0
ik.lo[14] = ik.hi[14] = 0.0
WALL_Y = 0.1267
CRATE_HALF_X = 0.2013 + 0.008
CRATE_TOP = top + 0.087 + 0.085
crate_x = edge_x + CRATE_HALF_X  # near face on the table edge
near_face = crate_x - CRATE_HALF_X
PZ = 0.793


def Rpitch(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, meta["objects"]["blue_crate"]["body"])
cadr = int(m.jnt_qposadr[m.body_jntadr[cid]])
pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
rb = {pel}
[rb.add(b) for b in range(m.nbody) if m.body_rootid[b] == m.body_rootid[pel]]
grip_q = [
    m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in (
        "left_hand_gripper_1_joint",
        "left_hand_gripper_2_joint",
        "right_hand_gripper_1_joint",
        "right_hand_gripper_2_joint",
    )
]


def targets(pelvis_x, depth, h):
    return {
        sd: np.array([near_face + depth, WALL_Y if sd == "left" else -WALL_Y, CRATE_TOP - h])
        - np.array([pelvis_x, 0, PZ])
        for sd in ("left", "right")
    }


def solve(tg, R):
    q = np.zeros(29)
    for it in range(8):
        for sd in ("left", "right"):
            q, _, _ = ik.solve(sd, tg[sd], R, q, iters=150, use_waist=True, waist_cost=3.0, point_offset=OFF)
    errs = {}
    for sd in ("left", "right"):
        pos, rot = ik.fk(q, sd)
        errs[sd] = float(np.linalg.norm(pos + rot @ np.asarray(OFF) - tg[sd]))
    return q, errs


def pen(q, pelvis_x):
    mujoco.mj_resetData(m, d)
    d.qpos[ik.base_qadr : ik.base_qadr + 3] = [pelvis_x, 0, PZ]
    d.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = [1, 0, 0, 0]
    d.qpos[ik.qadr] = q
    d.qpos[grip_q] = 0.0245
    d.qpos[cadr : cadr + 3] = [crate_x, 0, top + 0.087]
    d.qpos[cadr + 3 : cadr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    bad = set()
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in rb) != (b2 in rb) and c.dist < -0.002:
            mine, other = (b1, b2) if b1 in rb else (b2, b1)
            on = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other)
            if on != "world":
                bad.add((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, mine).replace("_link", ""), on.split("_")[0]))
    return bad


print(
    f"crate rotated: long walls at y=+-{WALL_Y}, near face on the table edge x={near_face:.3f}, wall top z={CRATE_TOP:.3f} (pelvis z {PZ})"
)
print(
    f"{'so':>5} {'depth':>5} {'h':>5} {'pitch':>5} | {'errL':>5} {'errR':>5} | {'wpL':>4} {'wpR':>4} | {'elL':>4} {'elR':>4} | pen"
)
rows = []
for so, depth, h, pdeg in itertools.product(
    (0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.44), (0.04, 0.08, 0.12, 0.16, 0.20), (0.025, 0.045), (45, 60, 75, 90)
):
    pelvis_x = near_face - so
    q, e = solve(targets(pelvis_x, depth, h), Rpitch(pdeg))
    bad = pen(q, pelvis_x)
    ok = max(e.values()) < 0.01 and not bad
    rows.append(
        dict(
            so=so,
            depth=depth,
            h=h,
            pitch=pdeg,
            errL=e["left"],
            errR=e["right"],
            wpL=float(np.degrees(q[20])),
            wpR=float(np.degrees(q[27])),
            elL=float(np.degrees(q[18])),
            elR=float(np.degrees(q[25])),
            pen=sorted(bad),
            ok=ok,
        )
    )
    print(
        f"{so:5.2f} {depth:5.2f} {h:5.3f} {pdeg:5d} | {e['left'] * 100:5.1f} {e['right'] * 100:5.1f} | {np.degrees(q[20]):4.0f} {np.degrees(q[27]):4.0f} | {np.degrees(q[18]):4.0f} {np.degrees(q[25]):4.0f} | {sorted(bad) if bad else '-'}{'  <== OK' if ok else ''}",
        flush=True,
    )
json.dump(rows, open("outputs/reach/wall_grasp_sweep.json", "w"), indent=0)
print("\n=== OK configs (IK < 1 cm both hands, no penetration) ===")
for r in rows:
    if r["ok"]:
        print(
            f"so {r['so']:.2f} depth {r['depth']:.2f} h {r['h']:.3f} pitch {r['pitch']}  wrist pitch L/R {r['wpL']:.0f}/{r['wpR']:.0f} elbow {r['elL']:.0f}/{r['elR']:.0f}"
        )
