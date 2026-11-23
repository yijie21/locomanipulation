"""selfmodel_via_20261120, fourth step, part two (the plan entry's "第四步第二部分"): narrow-gap scenes with real obstacles,
each planned with the obstacles thickened by every margin.

    cd $SIMPLE_ROOT && .venv/bin/python -I $REPO/scripts/experiments/selfmodel_via_20261120/gap_routes.py \
        <SIMPLE root> <stand_start.json> <out dir> <n base scenes> <scene seed> [--part k/K] [--levels 3,5,...] \
        [--margins 1,2,...] [--edge lo,hi] [--past lo,hi] [--bar w,t] [--room level|goal] [--keep all|reachable]

The settings the user chose on 2026-11-23 after the development probe: --edge 0.28,0.34 --past 0.04,0.10 --room goal
--keep reachable (see --room and --keep below); the defaults are the design as first confirmed.

The planner is fresh_routes.py's (SIMPLE's g1_29dof_with_dex3.yml, end link right_wrist_yaw_link, the right arm's 7 joints
free, waist and hand locked, the right arm and hand's 24 spheres, activation distance 3 cm), from the same stand. Frame: the
stand's heading frame (x ahead, y left, z above the floor, m).

A base scene: a board (the table: 4 cm thick, 1.2 m wide, 0.6 m deep; top 0.70-0.90 high, front edge --edge 0.20-0.26
ahead); the goal (palm centre) --past 8-14 cm past the edge, y -0.28 to -0.05, at mid-gap height +- 2 cm; the hand roughly
level and ahead: the goal attitude rot_z(dyaw) rot_y(pitch), pitch 0-30 deg (down), dyaw within 20 deg. A level f adds a bar
(--bar width x thickness, 3 x 3 cm, as wide as the table) over the table's front edge -- its front face flush with the edge --
its underside g = h + f above the table top. h: the 17 spheres of the wrist, palm and fingers (wrist_pitch, palm, thumb,
middle, index links), their vertical extent with the hand level (attitude identity) at a goal 0.30 ahead, 0.20 right, 0.85
up, measured here once with cuRobo's kinematics on an obstacle-free plan. One base scene for all levels: only the bar moves.
Every (base scene, level) is planned with the obstacles thickened by every margin m (--margins, cm) on every side; a plan that
fails is recorded as failed, nothing is redrawn. Scenes come from one generator seeded <scene seed>, all drawn before any is
planned, so --part k/K (base scenes s with s % K == k) splits the work without changing a scene.

-> <out>/via/via_<NAME>.npz, <out>/prep/prep_<NAME>.npz (fresh_routes.py's layout; NAME = Gap_s<sss>_f<ff>_m<mm>, f and m
in cm; prep's boxes are the REAL obstacles [centre, half sizes], boxes_plan the thickened ones), <out>/cands_<k>of<K>.json:
h, the draw of every base scene, and per (scene, level, margin) ok, status, plan seconds and the plan's spheres' smallest
clearance (cm) to the real obstacles (clear_arm: all 24; clear_thr: the 17) and to the thickened ones (clear_plan).
"""

import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
import via_gen as VG  # noqa: E402

S, START, OUT = Path(sys.argv[1]), json.load(open(sys.argv[2])), Path(sys.argv[3])
N, SEED = int(sys.argv[4]), int(sys.argv[5])
ARGS = sys.argv[6:]


def opt(name: str, default: str) -> str:
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


PART = tuple(int(x) for x in opt("--part", "0/1").split("/"))
LEVELS = [int(x) for x in opt("--levels", "3,5,7,9,11,14,17,20,24").split(",")]
MARGINS = [int(x) for x in opt("--margins", "1,2,3,4,6,8,10,12").split(",")]
EDGE = tuple(float(x) for x in opt("--edge", "0.20,0.26").split(","))
PAST = tuple(float(x) for x in opt("--past", "0.08,0.14").split(","))
BAR = tuple(float(x) for x in opt("--bar", "0.03,0.03").split(","))  # along x (depth), along z (thickness), m
# --room goal (2026-11-23, after the dev probe; NOT the confirmed design, a variant for the user to decide on): each base scene's
# own hand thickness -- the 17 spheres' vertical extent at its goal attitude, from an obstacle-free IK at the goal -- instead of
# the level hand's h: g = h_s + f, the hand's extent centred in the gap and its offset dz clipped to +- f/4. A goal the arm
# cannot reach without obstacles is recorded as such (every plan of the scene fails).
ROOM = opt("--room", "level")
assert ROOM in ("level", "goal"), ROOM
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
EE, OFF, ACT, TOL = "right_wrist_yaw_link", np.array([0.0415, 0.0, 0.0]), 0.03, 0.01
THROUGH = ["right_wrist_pitch_link", "right_hand_palm_link", "right_hand_thumb_0_link", "right_hand_thumb_1_link",
           "right_hand_thumb_2_link", "right_hand_middle_0_link", "right_hand_middle_1_link", "right_hand_index_0_link",
           "right_hand_index_1_link"]  # fmt: skip
BOARD_HALF = np.array([0.30, 0.60, 0.02])
td = TensorDeviceType()
(OUT / "via").mkdir(parents=True, exist_ok=True)
(OUT / "prep").mkdir(parents=True, exist_ok=True)


def rot(axis: str, a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = np.copysign(x, R[2, 1] - R[1, 2]), np.copysign(y, R[0, 2] - R[2, 0]), np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def mat_of(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


# the robot, as fresh_routes.py (plan2.py) builds it
cfg = load_yaml(str(S / "data/robots/g1/curobo/g1_29dof_with_dex3.yml"))["robot_cfg"]
kin = cfg["kinematics"]
kin["external_asset_path"] = str(S / "data/robots/g1")
kin["external_robot_configs_path"] = str(S / "data/robots/g1/curobo")
kin["ee_link"] = EE
kin["link_names"] = [EE] + [n for n in kin["collision_link_names"] if n.startswith("right_")]
reading = dict(zip(START["names"], START["q"]))
cs = kin["cspace"]
cspace = list(cs["joint_names"])
LOCK = [n for n in cspace if n not in ARM and (n.startswith("waist_") or n.startswith("right_hand_"))]
keep = [i for i, n in enumerate(cspace) if n in ARM or n in LOCK]
for k_, v_ in list(cs.items()):
    if isinstance(v_, list) and len(v_) == len(cspace):
        cs[k_] = [v_[i] for i in keep]
kin["lock_joints"] = {n: float(reading.get(n, 0.0)) for n in LOCK}
kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n.startswith("right_")]
_urdf = ET.parse(S / "data/robots/g1" / kin["urdf_path"]).getroot()
_lk = [x.get("link") for j in _urdf.iter("joint") if j.get("name") in LOCK for x in (j.find("parent"), j.find("child"))]
kin["link_names"] = list(dict.fromkeys(kin["link_names"] + _lk))
robot = RobotConfig.from_dict(cfg, td)

Rp = rot("z", START["yaw"]) @ rot("y", START["pitch"]) @ rot("x", START["roll"])
Rh = rot("z", START["yaw"])
pp = np.array([0.0, 0.0, START["pelvis_z"]])
to_b = lambda X: (np.asarray(X, float) @ Rh.T - pp) @ Rp  # heading frame -> pelvis frame (rows)  # noqa: E731
from_b = lambda X: (np.asarray(X, float) @ Rp.T + pp) @ Rh  # noqa: E731
qb = quat_wxyz((Rh.T @ Rp).T).tolist()


def world_of(boxes) -> WorldConfig:
    return WorldConfig(cuboid=[Cuboid(name=f"b{i}", pose=[*map(float, to_b(c)), *qb], dims=[float(2 * v) for v in h])
                               for i, (c, h) in enumerate(boxes)])  # fmt: skip


def clearance(Sw: np.ndarray, rad: np.ndarray, boxes) -> float:
    """The smallest distance (m) from spheres Sw (..., n, 3) of radii rad (n,) to axis-aligned boxes [(centre, half)]."""
    best = np.inf
    for c, h in boxes:
        dd = np.abs(Sw - c) - h
        best = min(best, float((np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0) - rad).min()))
    return best


mg = MotionGen(MotionGenConfig.load_from_robot_config(
    robot, world_of([(np.array([0.0, 0.0, -1.0]), np.array([0.05, 0.05, 0.05]))]), td,
    collision_checker_type=CollisionCheckerType.PRIMITIVE, collision_cache={"obb": 16}, collision_activation_distance=ACT,
    interpolation_dt=0.02, use_cuda_graph=False, num_trajopt_seeds=8, num_ik_seeds=64,
))  # fmt: skip
kc = mg.kinematics.kinematics_config
inv = {v: k for k, v in kc.link_name_to_idx_map.items()}
sph_link = np.array([inv[int(i)] for i in kc.link_sphere_idx_map.cpu().numpy()])
names = mg.kinematics.joint_names
q0 = torch.tensor([[reading.get(n, 0.0) for n in names]], **td.as_torch_dict())
start = JointState.from_position(q0, joint_names=names)
pc = MotionGenPlanConfig(max_attempts=4, enable_graph=True, timeout=20.0, enable_finetune_trajopt=True)


def spheres(Q):
    """Q (T, dof) -> (centres (T, n, 3) in the heading frame, radii (n,), link of each) of the live spheres."""
    ks = mg.kinematics.get_state(torch.as_tensor(Q, **td.as_torch_dict()))
    sph = ks.link_spheres_tensor.cpu().numpy()
    live = (sph[:, :, 3] > 0).all(0)
    return from_b(sph[:, live, :3].reshape(-1, 3)).reshape(len(Q), -1, 3), sph[0, live, 3], sph_link[live], ks


def goal_pose(g, R):
    return Pose(position=torch.tensor([to_b(g - R @ OFF)], **td.as_torch_dict()),
                quaternion=torch.tensor([quat_wxyz(Rp.T @ Rh @ R)], **td.as_torch_dict()))  # fmt: skip


# ---- h: the wrist, palm and fingers' vertical extent with the hand level
res = mg.plan_single(start, goal_pose(np.array([0.30, -0.20, 0.85]), np.eye(3)), pc)
assert bool(res.success.item()), f"the level-hand plan for h failed: {res.status}"
Qh = res.get_interpolated_plan().position.cpu().numpy()
Qh = Qh[0] if Qh.ndim == 3 else Qh
Sw, rad, lk, _ = spheres(Qh[-1:])
thr = np.isin(lk, THROUGH)
assert thr.sum() == 17 and len(rad) == 24, (thr.sum(), len(rad))
H = float((Sw[0, thr, 2] + rad[thr]).max() - (Sw[0, thr, 2] - rad[thr]).min())
print(f"h = {100 * H:.2f} cm (17 spheres, level hand); 24 spheres live: {dict(zip(*np.unique(lk, return_counts=True)))}",
      flush=True)  # fmt: skip

# ---- every base scene, drawn before any is planned


def draw(rng, s: int) -> dict:
    top, edge = float(rng.uniform(0.70, 0.90)), float(rng.uniform(*EDGE))
    past, y, dz = float(rng.uniform(*PAST)), float(rng.uniform(-0.28, -0.05)), float(rng.uniform(-0.02, 0.02))
    pitch, dyaw = float(rng.uniform(0, 30)), float(rng.uniform(-20, 20))
    return dict(s=s, top=top, edge=edge, past=past, y=y, dz=dz, pitch=pitch, dyaw=dyaw)


def boxes_of(b: dict, f_cm: int):
    """(real boxes [(centre, half)], goal, goal attitude) of base scene b at level f (cm)."""
    R = rot("z", math.radians(b["dyaw"])) @ rot("y", math.radians(b["pitch"]))
    if ROOM == "goal":
        g = b["h_s"] + f_cm / 100
        dz = float(np.clip(b["dz"], -f_cm / 400, f_cm / 400))
        gz = b["top"] + f_cm / 200 + dz - b["lo"]  # the hand's lowest point f/2 + dz over the table top
    else:
        g, gz = H + f_cm / 100, b["top"] + (H + f_cm / 100) / 2 + b["dz"]
    board = (np.array([b["edge"] + BOARD_HALF[0], -0.10, b["top"] - BOARD_HALF[2]]), BOARD_HALF.copy())
    bh = np.array([BAR[0] / 2, BOARD_HALF[1], BAR[1] / 2])
    bar = (np.array([b["edge"] + bh[0], -0.10, b["top"] + g + bh[2]]), bh)
    goal = np.array([b["edge"] + b["past"], b["y"], gz])
    return [board, bar], goal, R


def ik_ok(g, R) -> bool:
    return bool(mg.ik_solver.solve_single(goal_pose(g, R)).success.item())


def room(b: dict) -> bool:
    """--room goal: the scene's hand thickness at its goal attitude (obstacle-free IK, goal 10 cm over the top) into b;
    False when that goal cannot be reached."""
    R = rot("z", math.radians(b["dyaw"])) @ rot("y", math.radians(b["pitch"]))
    g0 = np.array([b["edge"] + b["past"], b["y"], b["top"] + 0.10])
    ik = mg.ik_solver.solve_single(goal_pose(g0, R))
    if not bool(ik.success.item()):
        b.update(lo=-H / 2, h_s=H)
        return False
    Sw, rad, lk, _ = spheres(ik.solution.view(-1, ik.solution.shape[-1])[:1].cpu().numpy())
    t = np.isin(lk, THROUGH)
    lo, hi = float((Sw[0, t, 2] - rad[t]).min() - g0[2]), float((Sw[0, t, 2] + rad[t]).max() - g0[2])
    b.update(lo=lo, h_s=hi - lo)
    return True


# --keep reachable (2026-11-23, the user's choice after the dev probe, with --room goal): draws from the one generator are
# kept, in order, until N base scenes whose goal is reached without obstacles at every level's goal position (IK only, no
# obstacle, the levels' goals differ in height) -- such a goal says nothing about the gap; every draw is logged, kept or not.
# A kept scene that the planner then fails on is recorded as failed, nothing is redrawn.
KEEP = opt("--keep", "all")
assert KEEP in ("all", "reachable") and (KEEP == "all" or ROOM == "goal"), (KEEP, ROOM)
rng = np.random.default_rng(SEED)
BASE, DRAWN = [], []
while len(BASE) < N:
    assert len(DRAWN) < 50 * N, f"--keep {KEEP}: {len(DRAWN)} draws for {len(BASE)} kept scenes"
    b = draw(rng, len(BASE))
    if ROOM == "goal":
        b["reach"] = room(b)
        if KEEP == "reachable" and b["reach"]:
            b["reach"] = all(ik_ok(*boxes_of(b, f)[1:]) for f in LEVELS)
    DRAWN.append(dict(b))
    if KEEP == "all" or b["reach"]:
        BASE.append(b)
if ROOM == "goal":
    hs = [b["h_s"] for b in BASE if b["reach"]]
    print(f"--room goal, --keep {KEEP}: {len(DRAWN)} drawn, {len(BASE)} kept, {len(hs)} reachable; hand thickness at the goal "
          f"attitude {100 * min(hs):.1f} to {100 * max(hs):.1f} cm (median {100 * float(np.median(hs)):.1f})", flush=True)  # fmt: skip


k_part, n_part = PART
log = dict(seed=SEED, n=N, part=list(PART), h_cm=round(100 * H, 3), levels=LEVELS, margins=MARGINS, edge=EDGE, past=PAST,
           bar=BAR, room=ROOM, keep=KEEP, activation=ACT, tol=TOL, base=BASE, drawn=DRAWN, plans=[])  # fmt: skip
t_all = time.time()
for b in BASE:
    if b["s"] % n_part != k_part:
        continue
    for f in LEVELS:
        real, g, R = boxes_of(b, f)
        for m in MARGINS:
            plan = [(c, h + m / 100) for c, h in real]
            name = f"Gap_s{b['s']:03d}_f{f:02d}_m{m:02d}"
            mg.update_world(world_of(plan))
            t0 = time.time()
            res = mg.plan_single(start, goal_pose(g, R), pc)
            ok = bool(res.success.item())
            row = dict(name=name, s=b["s"], f=f, m=m, ok=ok, status=str(res.status), sec=round(time.time() - t0, 2))
            if ok:
                Qp = res.get_interpolated_plan().position.cpu().numpy()
                Qp = Qp[0] if Qp.ndim == 3 else Qp
                Sw, rad, lk, ks = spheres(Qp)
                thr = np.isin(lk, THROUGH)
                Rw = np.array([Rh.T @ Rp @ mat_of(q) for q in ks.ee_quaternion.cpu().numpy()])
                J = from_b(ks.ee_position.cpu().numpy()) + Rw @ OFF
                idx = VG.rdp(J, TOL)
                inner = idx[1:-1]
                np.savez(OUT / "via" / f"via_{name}.npz", p=J[inner], R=Rw[inner].reshape(-1, 3, 3), goal=J[-1],
                         k=np.array(inner), max_dev=VG.poly_dev(J, idx))  # fmt: skip
                np.savez(OUT / "prep" / f"prep_{name}.npz", pelvis_p=pp, pelvis_R=Rh.T @ Rp, jaw_p=J[0], wrist_R=Rw[0],
                         goal_jaw=g, goal_R=R, path=J, boxes=np.array([np.r_[c, h] for c, h in real]),
                         boxes_plan=np.array([np.r_[c, h] for c, h in plan]))  # fmt: skip
                row.update(via=len(inner), path_cm=round(100 * float(np.linalg.norm(np.diff(J, axis=0), axis=1).sum()), 1),
                           clear_arm=round(100 * clearance(Sw, rad, real), 2),
                           clear_thr=round(100 * clearance(Sw[:, thr], rad[thr], real), 2),
                           clear_plan=round(100 * clearance(Sw, rad, plan), 2),
                           goal_miss_mm=round(1000 * float(np.linalg.norm(J[-1] - g)), 2))  # fmt: skip
            log["plans"].append(row)
            print(json.dumps(row), flush=True)
log["seconds"] = round(time.time() - t_all, 1)
json.dump(log, open(OUT / f"cands_{k_part}of{n_part}.json", "w"), indent=1)
P = log["plans"]
print(f"part {k_part}/{n_part}: planned {sum(r['ok'] for r in P)} of {len(P)} in {log['seconds']} s")
