"""Kinematic pre-flight for the 0.45 m table / set-back crate scene.

1. Default walking pose (policy-delivered q from pilot 21) at four stand-offs:
   finger-tip clearance to the table edge plane and the crate's near wall.
2. Raise path default -> via-point -> hover at each stand-off: contacts and
   the finger height over the bar top while the fingers are past the wall.
3. Ego-camera occlusion of the crate by the arms in the default pose.
No policy, no physics stepping: mj_forward only.
"""

import argparse
import json
import os as _os
import sys

import mujoco
import numpy as np

_SRC = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, _SRC)  # reach src/ for `python src/locomani/tasks/crate/walk_pose_preflight.py`
_os.chdir(_os.path.dirname(_SRC))  # repo root: the scene/output paths below are repo-relative

from locomani.backends.sonic.primitives_const import GRASP_CENTRE_OFFSET_CONST as OFF  # noqa: E402
from locomani.primitives.arm_ik import ArmIK  # noqa: E402
from locomani.primitives.g1_joints import ARM_MJ  # noqa: E402

# Argparse flags, same names/defaults as the old env vars (decision 8: a
# diagnostic tool's knobs are CLI flags, not config-dataclass fields).
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--scene-dir", default="outputs/scene/crate_bimanual")
_ap.add_argument("--pose-left-deg", default=None, help="hand-authored walking pose, mirrored to the right")
_ap.add_argument("--via-chain", default="-40,20,0,-30,0,-30,0")
_ap.add_argument(
    "--trace",
    default="outputs/videos/wall_pilot/trace/wall_s0_05_state_trace.npz",
    help="pilot state trace whose frame 10 supplies the policy-delivered body pose; "
    "'none' uses the scene's qpos0 (pelvis 0.793 m) instead",
)
_args = _ap.parse_args()
# POSE_LEFT_DEG stopped being read with the argparse conversion (refactor step 3);
# a run that still sets it silently checks the pilot-21 pose. Refuse, don't warn.
if _os.environ.get("POSE_LEFT_DEG"):
    sys.exit("refusing: POSE_LEFT_DEG is no longer read; pass --pose-left-deg 10,10,0,60,0,30,0")

out = _args.scene_dir
meta = json.load(open(f"{out}/scene_meta.json"))
xml = f"{out}/scene_locomani.xml"
ik = ArmIK(xml)
m = mujoco.MjModel.from_xml_path(xml)
d = mujoco.MjData(m)
ik.lo[12] = ik.hi[12] = 0.0
ik.lo[13] = ik.hi[13] = 0.0
ik.lo[14] = ik.hi[14] = 0.0
top = meta["surfaces"]["A"]["top"]
edge_x = meta["surfaces"]["A"]["xy"][0] - meta["surfaces"]["A"]["flat_half_extent"][0]
crate0 = np.asarray(meta["objects"]["blue_crate"]["init_pos"])
cq = meta["objects"]["blue_crate"]["init_quat"]
NEAR_X = 0.1347
WALL_Y = 0.1267
TOP_Z = 0.085
ACROSS_BIAS = 0.012
near_face = crate0[0] - NEAR_X
bar_top = crate0[2] + TOP_Z
PZ = 0.79
SAG = 0.08
cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, meta["objects"]["blue_crate"]["body"])
cadr = int(m.jnt_qposadr[m.body_jntadr[cid]])
pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
rb = {b for b in range(m.nbody) if m.body_rootid[b] == m.body_rootid[pel]}
bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
finger_g = [g for g in range(m.ngeom) if "dex1_finger" in bname(m.geom_bodyid[g])]
arm_g = [
    g
    for g in range(m.ngeom)
    if any(k in bname(m.geom_bodyid[g]) for k in ("shoulder", "elbow", "wrist", "dex1", "hand"))
]
crate_g = [g for g in range(m.ngeom) if m.geom_bodyid[g] == cid]
table_g = [g for g in range(m.ngeom) if "dining_table" in bname(m.geom_bodyid[g])]

if _args.trace == "none":
    # scene default; only the legs/waist come from here once --pose-left-deg is set
    q_default29 = m.qpos0[ik.qadr].copy()
    print("[body pose] --trace none: using the scene's qpos0, not a policy-delivered stand")
elif not _os.path.exists(_args.trace):
    sys.exit(
        f"missing pilot trace {_args.trace}: it is written by a wall pilot (scripts/run_wall_pilot.sh, "
        "$SIM_VIDEO_DIR/trace/wall_s0_NN_state_trace.npz). Copy one from the collection machine, "
        "point --trace at it, or pass --trace none to use the scene's qpos0."
    )
else:
    trace = np.load(_args.trace)["qpos"][10]
    q_default29 = trace[ik.qadr]  # policy-delivered default pose (29 dof)
if _args.pose_left_deg:  # hand-authored walking pose, mirrored to the right
    _l = np.radians([float(v) for v in _args.pose_left_deg.split(",")])
    q_default29 = q_default29.copy()
    q_default29[ARM_MJ["left"]] = _l
    q_default29[ARM_MJ["right"]] = _l * np.array([1, -1, -1, 1, -1, 1, -1])
grip_q = [
    m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
    for n in (
        "left_hand_gripper_1_joint",
        "left_hand_gripper_2_joint",
        "right_hand_gripper_1_joint",
        "right_hand_gripper_2_joint",
    )
]


def place(q29, so):
    mujoco.mj_resetData(m, d)
    d.qpos[ik.base_qadr : ik.base_qadr + 3] = [near_face - so, 0.0, PZ]
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


def finger_stats():
    p = d.geom_xpos[finger_g]
    # geom half-sizes are small; use the geom centre minus a 1.5 cm radius as the tip proxy
    return dict(x_max=float(p[:, 0].max()) + 0.015, z_min=float(p[:, 2].min()) - 0.015)


def Rpitch(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def solve(targets_pel, R, q0):
    q = q0.copy()
    for _ in range(6):
        for sd in ("left", "right"):
            q, _, _ = ik.solve(sd, targets_pel[sd], R, q, iters=150, use_waist=True, waist_cost=3.0, point_offset=OFF)
    errs = {
        sd: float(np.linalg.norm(ik.fk(q, sd)[0] + ik.fk(q, sd)[1] @ np.asarray(OFF) - targets_pel[sd]))
        for sd in ("left", "right")
    }
    return q, errs


print(
    f"table top {top:.2f} edge x {edge_x:.3f} | crate near face x {near_face:.3f} bar top z {bar_top:.3f} | pelvis z {PZ}"
)
print("\n=== 1. default walking pose at the stand ===")
for so in (0.13, 0.18, 0.24, 0.31):
    place(q_default29, so)
    fs = finger_stats()
    print(
        f"so {so:.2f}: finger tips x_max {fs['x_max'] - (near_face - so):+.3f} from pelvis -> {near_face - fs['x_max']:+.3f} m short of the wall, "
        f"{edge_x - fs['x_max']:+.3f} of the table edge; tip z_min {fs['z_min']:.3f} (table {top:.2f}, bar top {bar_top:.3f}); "
        f"contacts {contacts() or '-'}"
    )

print("\n=== 2. raise path default -> via -> hover ===")
report = {}
for so in (0.13, 0.18, 0.24, 0.31):
    depth = float(np.clip(0.38 - so, 0.06, 0.11))
    px = near_face - so
    hov = {
        sd: np.array([near_face + depth, (WALL_Y - ACROSS_BIAS) * (1 if sd == "left" else -1), bar_top + 0.06])
        - np.array([px, 0, PZ])
        for sd in ("left", "right")
    }
    via = {
        sd: np.array([px + 0.15, WALL_Y * (1 if sd == "left" else -1), bar_top + 0.20]) - np.array([px, 0, PZ])
        for sd in ("left", "right")
    }
    # via = the FK-authored carry pose (IK to a chest-height target picks a wrong branch); hover from zeros like the expert
    chain = [q_default29]
    for spec in _args.via_chain.split(";"):
        _v = np.radians([float(x) for x in spec.split(",")])
        qq = q_default29.copy()
        qq[ARM_MJ["left"]] = _v
        qq[ARM_MJ["right"]] = _v * np.array([1, -1, -1, 1, -1, 1, -1])
        chain.append(qq)
    q_via = chain[-1]
    e_via = {"left": 0.0}
    q_hov, e_hov = solve(hov, Rpitch(60), np.zeros(29))
    segs = [(chain[i], chain[i + 1], f"seg{i}") for i in range(len(chain) - 1)] + [(q_via, q_hov, "via->hover")]
    worst = None
    hits = set()
    for a, b, tag in segs:
        for u in np.linspace(0, 1, 26):
            place(a + (b - a) * u, so)
            hits.update((tag, c) for c in contacts())
            p = d.geom_xpos[finger_g]
            past = (
                (p[:, 0] + 0.015 > near_face - 0.01)
                & (np.abs(p[:, 1] - crate0[1]) < 0.22)
                & (p[:, 0] < near_face + 0.42)
            )  # over the crate footprint
            if past.any():
                clr = float(p[past, 2].min()) - 0.015 - bar_top
                if worst is None or clr < worst[0]:
                    worst = (clr, tag, float(u))
    print(
        f"so {so:.2f} depth {depth:.2f}: IK via {max(e_via.values()) * 100:.1f} cm hover {max(e_hov.values()) * 100:.1f} cm | "
        f"min finger clearance over the bar while past the wall {worst[0] * 100:+.1f} cm ({worst[1]} u={worst[2]:.2f}), minus {SAG * 100:.0f} cm sag {(worst[0] - SAG) * 100:+.1f} | "
        f"contacts {sorted(hits) or '-'}"
    )
    report[so] = dict(
        via_deg={sd: np.degrees(q_via[ARM_MJ[sd]]).round(0).tolist() for sd in ("left", "right")},
        hover_deg={sd: np.degrees(q_hov[ARM_MJ[sd]]).round(0).tolist() for sd in ("left", "right")},
    )
print("via pose (deg) at so 0.24:", report[0.24]["via_deg"])

print("\n=== 3. ego occlusion of the crate by the arms (default pose) ===")
import os  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")
r = mujoco.Renderer(m, height=240, width=320)
cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
r.enable_segmentation_rendering()
opt = mujoco.MjvOption()
mujoco.mjv_defaultOption(opt)
crate_set = set(crate_g)
for so in (0.60, 0.45, 0.35, 0.24):
    place(q_default29, so)
    r.update_scene(d, camera=cam, scene_option=opt)
    seg = r.render()[..., 0]
    with_arms = int(np.isin(seg, crate_g).sum())
    saved = m.geom_group[arm_g].copy()
    m.geom_group[arm_g] = 5
    opt.geomgroup[5] = 0
    r.update_scene(d, camera=cam, scene_option=opt)
    seg2 = r.render()[..., 0]
    m.geom_group[arm_g] = saved
    opt.geomgroup[5] = 1
    no_arms = int(np.isin(seg2, crate_g).sum())
    print(
        f"so {so:.2f}: crate pixels {with_arms} of {no_arms} visible -> {100 * (1 - with_arms / max(no_arms, 1)):.0f}% hidden by the arms"
    )
