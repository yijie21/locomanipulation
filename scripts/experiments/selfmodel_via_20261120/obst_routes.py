"""selfmodel_via_20261120, sixth step (the plan entry's "第六步"): fresh_routes.py's detour scenes with their real boxes, each
planned with the boxes thickened by every margin.

    cd $SIMPLE_ROOT && .venv/bin/python -I $REPO/scripts/experiments/selfmodel_via_20261120/obst_routes.py \
        <SIMPLE root> <stand_start.json> <out dir> <n scenes> <scene seed> [--part k/K] [--margins 0,1,2,3,4,6,8]

The planner is fresh_routes.py's (SIMPLE's g1_29dof_with_dex3.yml, end link right_wrist_yaw_link, the right arm's 7 joints
free, waist and hand locked, the right arm and hand's 24 spheres, activation distance 3 cm), from the same stand. Frame: the
stand's heading frame (x ahead, y left, z above the floor, m).

A scene is fresh_routes.py's `edge` (a board: 4 cm thick, 1.2 m wide, 0.6 m deep, top 0.70-0.90 high, front edge 18-26 cm
ahead; the goal 8-20 cm above it, 6-14 cm past the edge) or `box` (the same with a thin box 8-15 cm high on the board between
the edge and the goal), drawn with the same ranges in the same order; even scenes are edge, odd ones box. All scenes are drawn
from one generator seeded <scene seed> before any is planned, so --part k/K (scenes s with s % K == k) splits the work
without changing a scene.

Every scene is planned first with its real boxes (margin 0). One the planner does not solve there is recorded and left out
(two thirds of the draws: the goal is out of the arm's reach with the waist locked, or the stand is already in collision); the
others are the BASE scenes. Each base scene is then planned with every box thickened by every other margin m (--margins, cm)
on every side; a plan that fails is recorded as failed, nothing is redrawn.

-> <out>/via/via_<NAME>.npz, <out>/prep/prep_<NAME>.npz (fresh_routes.py's layout; NAME = Obst<Kind>_s<ssss>_f00_m<mm>, m in
cm; prep's boxes are the REAL boxes [centre, half sizes], boxes_plan the thickened ones), <out>/cands_<k>of<K>.json in
gap_routes.py's layout (gap_slots.py reads it; one level, 0): the draw of every scene and per (scene, margin) ok, status,
plan seconds and the plan's spheres' smallest clearance (cm) to the real boxes (clear_arm: all 24; clear_thr: the 17 of the
wrist, palm and fingers) and to the thickened ones (clear_plan).
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
MARGINS = [int(x) for x in opt("--margins", "0,1,2,3,4,6,8").split(",")]
assert MARGINS[0] == 0 and MARGINS == sorted(set(MARGINS)), "--margins: ascending, the first one 0 (the real boxes)"
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
EE, OFF, ACT, TOL = "right_wrist_yaw_link", np.array([0.0415, 0.0, 0.0]), 0.03, 0.01
THROUGH = ["right_wrist_pitch_link", "right_hand_palm_link", "right_hand_thumb_0_link", "right_hand_thumb_1_link",
           "right_hand_thumb_2_link", "right_hand_middle_0_link", "right_hand_middle_1_link", "right_hand_index_0_link",
           "right_hand_index_1_link"]  # fmt: skip
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


# the robot, as fresh_routes.py (plan2.py) builds it (gap_routes.py's lines)
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


# ---- every scene, drawn before any is planned (fresh_routes.py's scene(): the same ranges, the same order of draws)


def draw(rng, s: int) -> dict:
    kind = ("edge", "box")[s % 2]
    dyaw = float(rng.uniform(-30, 30))
    top, edge = float(rng.uniform(0.70, 0.90)), float(rng.uniform(0.18, 0.26))
    g = [edge + float(rng.uniform(0.06, 0.14)), float(rng.uniform(-0.28, -0.05)), top + float(rng.uniform(0.08, 0.20))]
    b = dict(s=s, kind=kind, top=top, edge=edge, dyaw=dyaw, pitch=float(rng.uniform(45, 90)))
    if kind == "box":
        b["h"], b["bx"] = float(rng.uniform(0.08, 0.15)), edge + float(rng.uniform(0.03, 0.05))
        g[0], g[2] = max(g[0], b["bx"] + 0.08), max(g[2], top + 0.08)
    return dict(b, goal=g)


def boxes_of(b: dict):
    """(real boxes [(centre, half)], goal, goal attitude) of scene b."""
    boxes = [(np.array([b["edge"] + 0.30, -0.10, b["top"] - 0.02]), np.array([0.30, 0.60, 0.02]))]
    if b["kind"] == "box":
        boxes.append(
            (np.array([b["bx"], b["goal"][1], b["top"] + 0.5 * b["h"]]), np.array([0.015, 0.10, 0.5 * b["h"]]))
        )
    return boxes, np.array(b["goal"]), rot("z", math.radians(b["dyaw"])) @ rot("y", math.radians(b["pitch"]))


rng = np.random.default_rng(SEED)
DRAWN = [draw(rng, s) for s in range(N)]
k_part, n_part = PART
log = dict(seed=SEED, n=N, part=list(PART), levels=[0], margins=MARGINS, activation=ACT, tol=TOL, base=DRAWN, plans=[])
t_all = time.time()
for b in DRAWN:
    if b["s"] % n_part != k_part:
        continue
    real, g, R = boxes_of(b)
    solved = True
    for m in MARGINS:
        name = f"Obst{b['kind'].capitalize()}_s{b['s']:04d}_f00_m{m:02d}"
        row = dict(
            name=name, s=b["s"], f=0, m=m, kind=b["kind"], ok=False, status="not planned: no plan with the real boxes"
        )
        if solved:
            plan = [(c, h + m / 100) for c, h in real]
            mg.update_world(world_of(plan))
            t0 = time.time()
            res = mg.plan_single(start, goal_pose(g, R), pc)
            row.update(ok=bool(res.success.item()), status=str(res.status), sec=round(time.time() - t0, 2))
        if row["ok"]:
            Qp = res.get_interpolated_plan().position.cpu().numpy()
            Qp = Qp[0] if Qp.ndim == 3 else Qp
            Sw, rad, lk, ks = spheres(Qp)
            thr = np.isin(lk, THROUGH)
            assert thr.sum() == 17 and len(rad) == 24, (thr.sum(), len(rad))
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
                       straight_cm=round(100 * float(np.linalg.norm(J[-1] - J[0])), 1),
                       clear_arm=round(100 * clearance(Sw, rad, real), 2),
                       clear_thr=round(100 * clearance(Sw[:, thr], rad[thr], real), 2),
                       clear_plan=round(100 * clearance(Sw, rad, plan), 2),
                       goal_miss_mm=round(1000 * float(np.linalg.norm(J[-1] - g)), 2))  # fmt: skip
        elif m == 0:
            solved = False  # not a base scene: the other margins are not planned
        log["plans"].append(row)
        print(json.dumps(row), flush=True)
log["seconds"] = round(time.time() - t_all, 1)
json.dump(log, open(OUT / f"cands_{k_part}of{n_part}.json", "w"), indent=1)
P = log["plans"]
base = sorted({r["s"] for r in P if r["m"] == 0 and r["ok"]})
print(f"part {k_part}/{n_part}: {len(base)} base scenes of {len({r['s'] for r in P})} drawn in {log['seconds']} s; planned by margin: "
      + ", ".join(f"{m} cm {sum(r['ok'] for r in P if r['m'] == m)}" for m in MARGINS))  # fmt: skip
