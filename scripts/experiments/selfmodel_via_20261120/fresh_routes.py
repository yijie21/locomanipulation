"""selfmodel_via_20261120, second round: a fresh set of planner-made paths for the final test.

    cd $REPO && third_party/SIMPLE/.venv/bin/python -I scripts/experiments/selfmodel_via_20261120/fresh_routes.py \
        <SIMPLE root> <stand_start.json> <out dir> [n scenes] [seed]

The 35 cuRobo paths of curobo_avoid_20261120 have been looked at many times by now. Here cuRobo plans the right arm (as
plan2.py sets it up: SIMPLE's g1_29dof_with_dex3.yml, end link right_wrist_yaw_link, the arm's 7 joints free, the waist
and the hand locked, collisions for the right arm and hand, activation distance 3 cm) from the robot's stand on the open
floor to a goal behind an obstacle, in scenes drawn here and nowhere else. Three kinds, in the heading frame of the stand
(x ahead, y left, z above the floor, m):
  edge    a board (4 cm thick, top 0.70-0.90 high) whose front edge is 18-26 cm ahead; the goal 8-20 cm above it, 6-14 cm
          past the edge: reach over an edge
  box     the same board with a thin box (8-15 cm high) on it between the edge and the goal: reach over something
  under   a board overhead (underside 0.98-1.12 high); the goal 8-18 cm below it, 25-38 cm ahead: reach in under
The goals stay within about 40 cm of the right shoulder: with the waist locked the arm reaches no further (a first try
with goals 40-60 cm ahead planned 1 scene of 8). Upright obstacles in front (a post 20-28 cm ahead, a wall 24-30 cm ahead) were
tried and dropped: they stand where the hanging hand is, and the planner refuses a start in collision.
The goal's attitude is gen_reach.py's rot_z(dyaw) rot_y(pitch): pitch 45-90 deg (down) for edge and box, 0-40 for under,
dyaw within 30 deg. Scenes and goals come from one seeded generator; a scene the planner fails on is counted and left out,
nothing is redrawn to make it plan.

-> <out>/via/via_<NAME>.npz and <out>/prep/prep_<NAME>.npz in via2.py's and prep2.py's layout (what via_gen.load_routes
reads; REACH_VIA_REPLAY=<out>/via:<out>/prep), <out>/scenes.json (every scene drawn, planned or not, with its boxes and
goal, and for the planned ones the path's length and its spheres' smallest clearance to the boxes).
"""

import json
import math
import sys
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
N, SEED = (int(sys.argv[4]) if len(sys.argv) > 4 else 60), (int(sys.argv[5]) if len(sys.argv) > 5 else 20261121)
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
EE, OFF, ACT, TOL = "right_wrist_yaw_link", np.array([0.0415, 0.0, 0.0]), 0.03, 0.01
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


# the robot, as plan2.py builds it
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

# the frame: the pelvis of the stand; the heading frame has its origin on the floor under it
Rp = rot("z", START["yaw"]) @ rot("y", START["pitch"]) @ rot("x", START["roll"])
Rh = rot("z", START["yaw"])
pp = np.array([0.0, 0.0, START["pelvis_z"]])
to_b = lambda X: (np.asarray(X, float) @ Rh.T - pp) @ Rp  # heading frame -> pelvis frame (rows)  # noqa: E731
from_b = lambda X: (np.asarray(X, float) @ Rp.T + pp) @ Rh  # noqa: E731
qb = quat_wxyz((Rh.T @ Rp).T).tolist()


def world_of(boxes) -> WorldConfig:
    return WorldConfig(cuboid=[Cuboid(name=f"b{i}", pose=[*map(float, to_b(c)), *qb], dims=[float(2 * v) for v in h])
                               for i, (c, h) in enumerate(boxes)])  # fmt: skip


def planner(boxes) -> MotionGen:
    return MotionGen(MotionGenConfig.load_from_robot_config(
        robot, world_of(boxes), td, collision_checker_type=CollisionCheckerType.PRIMITIVE, collision_cache={"obb": 16},
        collision_activation_distance=ACT, interpolation_dt=0.02, use_cuda_graph=False, num_trajopt_seeds=8, num_ik_seeds=64,
    ))  # fmt: skip


def scene(rng, kind: str):
    """(boxes [(centre, half sizes)], goal jaw, goal attitude) in the heading frame. The goals stay within about 40 cm of
    the right shoulder: the planner moves the arm only, the waist is locked."""
    dyaw = math.radians(rng.uniform(-30, 30))
    if kind in ("edge", "box"):
        top, edge = rng.uniform(0.70, 0.90), rng.uniform(0.18, 0.26)
        boxes = [(np.array([edge + 0.30, -0.10, top - 0.02]), np.array([0.30, 0.60, 0.02]))]
        g = np.array([edge + rng.uniform(0.06, 0.14), rng.uniform(-0.28, -0.05), top + rng.uniform(0.08, 0.20)])
        pitch = math.radians(rng.uniform(45, 90))
        if kind == "box":
            h = rng.uniform(0.08, 0.15)
            bx = edge + rng.uniform(0.03, 0.05)
            boxes.append((np.array([bx, g[1], top + 0.5 * h]), np.array([0.015, 0.10, 0.5 * h])))
            g[0], g[2] = max(g[0], bx + 0.08), max(g[2], top + 0.08)
    else:  # under
        low = rng.uniform(0.98, 1.12)
        boxes = [(np.array([0.40, -0.10, low + 0.02]), np.array([0.25, 0.60, 0.02]))]
        g = np.array([rng.uniform(0.25, 0.38), rng.uniform(-0.28, -0.05), low - rng.uniform(0.08, 0.18)])
        pitch = math.radians(rng.uniform(0, 40))
    return boxes, g, rot("z", dyaw) @ rot("y", pitch)


rng = np.random.default_rng(SEED)
mg = planner([(np.array([0.0, 0.0, -1.0]), np.array([0.05, 0.05, 0.05]))])
names = mg.kinematics.joint_names
q0 = torch.tensor([[reading.get(n, 0.0) for n in names]], **td.as_torch_dict())
start = JointState.from_position(q0, joint_names=names)
st = mg.kinematics.get_state(q0)
jaw0 = from_b(st.ee_position[0].cpu().numpy()) + (Rh.T @ Rp @ mat_of(st.ee_quaternion[0].cpu().numpy())) @ OFF
print("start jaw by cuRobo's kinematics", jaw0.round(4), "recorded", np.round(START["jaw_h"], 4),
      "difference mm", round(1000 * float(np.linalg.norm(jaw0 - np.array(START["jaw_h"]))), 2), flush=True)  # fmt: skip
pc = MotionGenPlanConfig(max_attempts=4, enable_graph=True, timeout=20.0, enable_finetune_trajopt=True)
log, kinds, made = [], ("edge", "box", "under"), 0
for k in range(N):
    kind = kinds[k % 3]
    boxes, g, R = scene(rng, kind)
    name = f"Fresh{kind.capitalize()}_L0_e{k}"
    mg.update_world(world_of(boxes))
    wr = g - R @ OFF
    goal = Pose(position=torch.tensor([to_b(wr)], **td.as_torch_dict()),
                quaternion=torch.tensor([quat_wxyz(Rp.T @ Rh @ R)], **td.as_torch_dict()))  # fmt: skip
    res = mg.plan_single(start, goal, pc)
    ok = bool(res.success.item())
    row = dict(name=name, kind=kind, ok=ok, status=str(res.status), goal=g.round(4).tolist(),
               boxes=[[c.round(4).tolist(), h.round(4).tolist()] for c, h in boxes])  # fmt: skip
    if ok:
        Qp = res.get_interpolated_plan().position.cpu().numpy()
        Qp = Qp[0] if Qp.ndim == 3 else Qp
        ks = mg.kinematics.get_state(torch.tensor(Qp, **td.as_torch_dict()))
        Rw = np.array([Rh.T @ Rp @ mat_of(q) for q in ks.ee_quaternion.cpu().numpy()])
        J = from_b(ks.ee_position.cpu().numpy()) + Rw @ OFF
        sph = ks.link_spheres_tensor.cpu().numpy()
        live = (sph[:, :, 3] > 0).all(0)
        Sw = from_b(sph[:, live, :3].reshape(-1, 3)).reshape(len(Qp), -1, 3)
        clear = np.inf
        for c, h in boxes:
            dd = np.abs(Sw - c) - h
            clear = min(
                clear,
                float((np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0) - sph[:, live, 3]).min()),
            )
        idx = VG.rdp(J, TOL)
        inner = idx[1:-1]
        np.savez(OUT / "via" / f"via_{name}.npz", p=J[inner], R=Rw[inner].reshape(-1, 3, 3), goal=J[-1], k=np.array(inner),
                 max_dev=VG.poly_dev(J, idx))  # fmt: skip
        np.savez(OUT / "prep" / f"prep_{name}.npz", pelvis_p=pp, pelvis_R=Rh.T @ Rp, jaw_p=J[0], wrist_R=Rw[0], goal_jaw=g,
                 goal_R=R, path=J, boxes=np.array([np.r_[c, h] for c, h in boxes]))  # fmt: skip
        row.update(via=len(inner), path_cm=round(100 * float(np.linalg.norm(np.diff(J, axis=0), axis=1).sum()), 1),
                   straight_cm=round(100 * float(np.linalg.norm(J[-1] - J[0])), 1), clear_cm=round(100 * clear, 2),
                   goal_miss_mm=round(1000 * float(np.linalg.norm(J[-1] - g)), 2))  # fmt: skip
        made += 1
    log.append(row)
    print(json.dumps(row), flush=True)
json.dump(
    dict(seed=SEED, n=N, planned=made, activation=ACT, tol=TOL, scenes=log), open(OUT / "scenes.json", "w"), indent=1
)
print(
    f"planned {made} of {N}: "
    + ", ".join(
        f"{kd} {sum(r['ok'] for r in log if r['kind'] == kd)}/{sum(r['kind'] == kd for r in log)}" for kd in kinds
    )
)
