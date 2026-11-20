"""curobo_avoid_20261120 step 2b: cuRobo plans the right arm from the reach's start to the pregrasp around the look's
obstacles (TabletopGrasp), offline; checked against the replica's true table.

    $SIMPLE/.venv/bin/python -I plan2.py <SIMPLE root> <step1 dir> <step2 dir> EP [--act 0.03]

Robot: SIMPLE's data/robots/g1/curobo/g1_29dof_with_dex3.yml with
  - the end link right_wrist_yaw_link (the bench commands that link's pose: the jaw centre is OFF from it, motor.jaw_centre)
  - only the right arm's 7 joints free; every other joint locked at the replica's reading at the reach's start (prep2.py)
  - world collisions for the right arm and hand only (the locked rest cannot move out of anything; the left wrist rests on
    the table's edge at the start in these runs, and the torso is ~5 cm from it)
World: step 1's voxels (the look's points less the robot's own pixels and the target's), each a VOX cube, in the pelvis
frame of the reach's start. Goal: the wanted pregrasp's wrist pose (prep2.py), in that frame.
Checks: the start's wrist pose by cuRobo's kinematics against the replica's (mm, deg); the plan's moving spheres' clearance to
the true table box (the replica's, world) and to the obstacle voxels, over the interpolated plan.
-> <step2>/plan_<NAME>.npz, one JSON line
"""

import argparse
import json
import time
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

ap = argparse.ArgumentParser()
ap.add_argument("simple")
ap.add_argument("s1")
ap.add_argument("s2")
ap.add_argument("ep", type=int)
ap.add_argument("--task", default="TabletopGraspMP")
ap.add_argument("--act", type=float, default=0.03, help="m, cuRobo's collision activation distance")
ap.add_argument("--tgt", type=float, default=0.02, help="m: voxels within this of the target's cloud are dropped (3D)")
ap.add_argument(
    "--cloud-dir",
    default="",
    help="the target clouds the bench picked its grasp from (cloud_eyes_final/<NAME>.npz, pts, world); the look dump's own mask and cloud are a table patch in 17 of 30 episodes",
)
ap.add_argument("--no-finetune", action="store_true", help="skip cuRobo's finetune trajopt")
ap.add_argument("--diag", action="store_true", help="on a failure: plan again without obstacles and with more tries")
ap.add_argument("--free", type=float, default=0.01,
                help="m: voxels nearer than this to the start pose's spheres are dropped (the start pose is where the body is)")  # fmt: skip
a = ap.parse_args()
S, S1, S2 = Path(a.simple), Path(a.s1), Path(a.s2)
e = a.ep
name = f"{a.task}_L{e // 10}_e{e % 10}"
P = S2 / f"prep_{name}.npz"
if not P.exists():
    print(json.dumps(dict(ep=e, name=name, skip="no prep (no grasp)")))
    raise SystemExit
pr, ob = np.load(P), np.load(S1 / f"obst_{name}.npz")
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
EE = "right_wrist_yaw_link"
td = TensorDeviceType()


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


# the robot
cfg = load_yaml(str(S / "data/robots/g1/curobo/g1_29dof_with_dex3.yml"))["robot_cfg"]
kin = cfg["kinematics"]
kin["external_asset_path"] = str(S / "data/robots/g1")
kin["external_robot_configs_path"] = str(S / "data/robots/g1/curobo")
kin["ee_link"] = EE
# SIMPLE's cuRobo builds the tree from the pelvis to the end link and link_names only (its cuda_robot_generator no longer adds
# the collision links, "different from official version"), and sizes its self-collision mask by link_names: every collision
# link goes in link_names, or the hand's joints are not in the tree (KeyError) and the mask is too small (IndexError)
kin["link_names"] = [EE] + [n for n in kin["collision_link_names"] if n.startswith("right_")]
reading = dict(zip([str(n) for n in pr["names"]], pr["q"].tolist()))
cs = kin["cspace"]
cspace = list(cs["joint_names"])
missing = [n for n in cspace if n not in reading]
# the tree cuRobo builds runs from the pelvis to the right wrist and the right arm's and hand's collision links: the waist,
# the right arm, the right hand; the legs and the left arm are not in it (locking them fails: KeyError in joint_data)
LOCK = [n for n in cspace if n not in ARM and (n.startswith("waist_") or n.startswith("right_hand_"))]
keep = [i for i, n in enumerate(cspace) if n in ARM or n in LOCK]
for k_, v_ in list(cs.items()):
    if isinstance(v_, list) and len(v_) == len(cspace):
        cs[k_] = [v_[i] for i in keep]
kin["lock_joints"] = {n: float(reading.get(n, 0.0)) for n in LOCK}
kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n.startswith("right_")]
# locking adds each locked joint's parent and child link to link_names after the self-collision mask is sized
# (IndexError 14 of 14): they go in link_names first, so the set it builds does not grow
import xml.etree.ElementTree as ET  # noqa: E402

_urdf = ET.parse(S / "data/robots/g1" / kin["urdf_path"]).getroot()
_lk = [x.get("link") for j in _urdf.iter("joint") if j.get("name") in LOCK for x in (j.find("parent"), j.find("child"))]
kin["link_names"] = list(dict.fromkeys(kin["link_names"] + _lk))
robot = RobotConfig.from_dict(cfg, td)

# the frame: the pelvis at the reach's start
Rp, pp = pr["pelvis_R"], pr["pelvis_p"]
to_b = lambda X: (np.asarray(X) - pp) @ Rp  # world -> pelvis frame (rows)
vs = float(ob["vox_size"])
qb = quat_wxyz(Rp.T).tolist()
from scipy.spatial import cKDTree  # noqa: E402


def box_sd(X, c, h):  # signed distance of points X to the axis-aligned box (centre c, half sizes h)
    dd = np.abs(X - c) - h
    return np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0)


def motion_gen(vox_w):
    vox = to_b(vox_w)
    world = WorldConfig(
        cuboid=[Cuboid(name=f"v{i}", pose=[*map(float, v), *qb], dims=[vs, vs, vs]) for i, v in enumerate(vox)]
    )
    return MotionGen(MotionGenConfig.load_from_robot_config(
        robot, world, td, collision_checker_type=CollisionCheckerType.PRIMITIVE, collision_cache={"obb": len(vox) + 16},
        collision_activation_distance=a.act, interpolation_dt=0.02, use_cuda_graph=False, num_trajopt_seeds=8, num_ik_seeds=64,
    ))  # fmt: skip


t0 = time.time()
mg = motion_gen(np.zeros((0, 3)))  # the robot alone first: the start pose's spheres
names = mg.kinematics.joint_names
q0 = torch.tensor([[reading[n] for n in names]], **td.as_torch_dict())
start = JointState.from_position(q0, joint_names=names)
st = mg.kinematics.get_state(q0)
s0 = st.link_spheres_tensor[0].cpu().numpy()
s0 = s0[s0[:, 3] > 0]
W0s = s0[:, :3] @ Rp.T + pp
# drop the voxels whose cube is within --free of a start sphere: the body is there now, so what the look put there is the
# body's own points (the robot mask's edge) or depth fill; cuRobo refuses a start in collision
V = ob["vox"]
near = cKDTree(V).query_ball_point(W0s, s0[:, 3] + a.free + vs)
drop = np.zeros(len(V), bool)
for i, js in enumerate(near):
    for j in js:
        if box_sd(W0s[i], V[j], 0.5 * vs) - s0[i, 3] < a.free:
            drop[j] = True
n_tgt = 0
if a.tgt > 0:
    tc = np.load(Path(a.cloud_dir) / f"{name}.npz")["pts"] if a.cloud_dir else np.asarray(ob["cloud"], float)
    dt_ = cKDTree(np.asarray(tc, float)).query(V)[0]
    n_tgt = int(((dt_ < a.tgt + 0.5 * vs) & ~drop).sum())
    drop |= dt_ < a.tgt + 0.5 * vs
V = V[~drop]
mg = motion_gen(V)
t_build = time.time() - t0

# check: the start's wrist by cuRobo's kinematics against the replica's
st = mg.kinematics.get_state(q0)
ee_p = st.ee_position[0].cpu().numpy()
ee_R = mat_of(st.ee_quaternion[0].cpu().numpy())
fk_mm = 1000 * float(np.linalg.norm(ee_p - to_b(pr["wrist_p"])))
fk_deg = float(np.degrees(np.arccos(np.clip((np.trace((Rp.T @ pr["wrist_R"]).T @ ee_R) - 1) / 2, -1, 1))))


# the start's spheres against the (kept) voxels and the true table (which link, how deep, where the voxel is)
sph0 = st.link_spheres_tensor[0].cpu().numpy()
v0 = sph0[:, 3] > 0
kc = mg.kinematics.kinematics_config
idx = kc.link_sphere_idx_map.cpu().numpy()
inv = {v: k for k, v in getattr(kc, "link_name_to_idx_map", {}).items()}
W0 = sph0[:, :3] @ Rp.T + pp
dv0, iv0 = cKDTree(V).query(W0)
dv0 = dv0 - sph0[:, 3] - 0.5 * vs
c_, h_ = pr["table_pos"], 0.5 * pr["table_size"]
dd_ = np.abs(W0 - c_) - h_
dt0 = np.linalg.norm(np.maximum(dd_, 0), axis=-1) + np.minimum(dd_.max(-1), 0) - sph0[:, 3]
hit = [dict(link=inv.get(int(idx[i]), int(idx[i])), d_vox_cm=round(100 * float(dv0[i]), 1), d_table_cm=round(100 * float(dt0[i]), 1),
            sphere=W0[i].round(3).tolist(), vox=V[iv0[i]].round(3).tolist())
       for i in np.argsort(dv0) if v0[i] and dv0[i] < a.act][:8]  # fmt: skip

gp = to_b(pr["goal_wrist"])
gR = Rp.T @ pr["goal_R"]
goal = Pose(
    position=torch.tensor([gp], **td.as_torch_dict()), quaternion=torch.tensor([quat_wxyz(gR)], **td.as_torch_dict())
)
t1 = time.time()
res = mg.plan_single(
    start,
    goal,
    MotionGenPlanConfig(max_attempts=4, enable_graph=True, timeout=20.0, enable_finetune_trajopt=not a.no_finetune),
)
t_first = time.time() - t1  # with the kernels' first compile
t1 = time.time()
res = mg.plan_single(
    start,
    goal,
    MotionGenPlanConfig(max_attempts=4, enable_graph=True, timeout=20.0, enable_finetune_trajopt=not a.no_finetune),
)
t_plan = time.time() - t1  # warm: as a bench process that built the planner once would see it
ok = bool(res.success.item())
out = dict(ep=e, name=name, ok=ok, status=str(res.status), t_build_s=round(t_build, 1), t_plan_s=round(t_plan, 2), t_first_s=round(t_first, 2),
           fk_mm=round(fk_mm, 1), fk_deg=round(fk_deg, 2), missing_joints=missing, vox=len(V), vox_dropped=int(drop.sum()), vox_dropped_tgt=n_tgt, tgt=a.tgt, act=a.act, free=a.free,
           pre_err_direct_cm=float(pr["pre_err_cm"]), start_vox_cm=round(100 * float(dv0[v0].min()), 2),
           start_table_cm=round(100 * float(dt0[v0].min()), 2), start_hits=hit)  # fmt: skip
save = dict(names=np.array(names), q0=q0.cpu().numpy()[0])
if ok:
    traj = res.get_interpolated_plan()
    Q = traj.position.cpu().numpy()
    if Q.ndim == 3:
        Q = Q[0]
    ks = mg.kinematics.get_state(torch.tensor(Q, **td.as_torch_dict()))
    sph = ks.link_spheres_tensor.cpu().numpy()  # (T, n, 4), pelvis frame
    moving = (sph[:, :, 3] > 0).all(0) & (np.linalg.norm(sph[-1, :, :3] - sph[0, :, :3], axis=1) > 1e-3)
    Sw = sph[:, :, :3] @ Rp.T + pp  # world
    c, h = pr["table_pos"], 0.5 * pr["table_size"]
    dd = np.abs(Sw - c) - h
    sd = np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0) - sph[:, :, 3]
    tree = cKDTree(V)
    dv = tree.query(Sw[:, moving].reshape(-1, 3))[0].reshape(len(Q), -1) - sph[:, moving, 3] - 0.5 * vs
    eeT = ks.ee_position.cpu().numpy() @ Rp.T + pp
    eeQ = ks.ee_quaternion.cpu().numpy()
    jaw = np.array([p + (Rp @ mat_of(q)) @ pr["OFF"] for p, q in zip(eeT, eeQ)])  # the jaw centre, world
    end_mm = 1000 * float(np.linalg.norm(ks.ee_position.cpu().numpy()[-1] - gp))
    ct = sd[:, moving].min(1)
    left = np.flatnonzero(ct >= a.act)  # the hand starts ~0 cm from the table: the clearance once it has left (>= act)
    after = float(ct[left[0] :].min()) if len(left) else float("nan")
    out.update(clear_after_cm=round(100 * after, 2), left_at=int(left[0]) if len(left) else -1)
    out.update(T=len(Q), dt=float(res.interpolation_dt), dur_s=round(len(Q) * float(res.interpolation_dt), 2),
               clear_table_cm=round(100 * float(sd[:, moving].min()), 2), clear_vox_cm=round(100 * float(dv.min()), 2),
               start_clear_table_cm=round(100 * float(sd[0, moving].min()), 2), end_mm=round(end_mm, 1),
               jaw_max_z=round(float(jaw[:, 2].max()), 3), n_moving=int(moving.sum()))  # fmt: skip
    save.update(Q=Q, jaw=jaw, wrist_p=eeT, wrist_q=eeQ, clear_table=sd[:, moving].min(1), clear_vox=dv.min(1))
if not ok and a.diag:
    # (a) without obstacles: is the goal within the arm's reach at all (waist locked)? the goal pose's spheres against the
    # voxels and the true table; (b) the same world with more tries
    cfg_ = MotionGenPlanConfig(
        max_attempts=12, enable_graph=True, timeout=60.0, enable_finetune_trajopt=not a.no_finetune
    )
    r0 = motion_gen(np.zeros((0, 3))).plan_single(start, goal, cfg_)
    out["empty_ok"], out["empty_status"] = bool(r0.success.item()), str(r0.status)
    if out["empty_ok"]:
        Qe = r0.get_interpolated_plan().position.cpu().numpy().reshape(-1, len(names))
        sg = mg.kinematics.get_state(torch.tensor(Qe[-1:], **td.as_torch_dict())).link_spheres_tensor[0].cpu().numpy()
        vg = sg[:, 3] > 0
        Wg = sg[:, :3] @ Rp.T + pp
        dvg, ivg = cKDTree(V).query(Wg)
        dvg = dvg - sg[:, 3] - 0.5 * vs
        dtg = box_sd(Wg, pr["table_pos"], 0.5 * pr["table_size"]) - sg[:, 3]
        out["goal_vox_cm"], out["goal_table_cm"] = (
            round(100 * float(dvg[vg].min()), 2),
            round(100 * float(dtg[vg].min()), 2),
        )
        out["goal_hits"] = [dict(link=inv.get(int(idx[i]), int(idx[i])), d_vox_cm=round(100 * float(dvg[i]), 1),
                                 d_table_cm=round(100 * float(dtg[i]), 1), sphere=Wg[i].round(3).tolist(), vox=V[ivg[i]].round(3).tolist())
                            for i in np.argsort(dvg) if vg[i] and dvg[i] < 0][:6]  # fmt: skip
    r1 = mg.plan_single(start, goal, cfg_)
    out["retry_ok"], out["retry_status"] = bool(r1.success.item()), str(r1.status)
    top = float(pr["table_pos"][2] + 0.5 * pr["table_size"][2])
    out["goal_jaw_above_top_cm"] = round(100 * float(pr["goal_jaw"][2] - top), 2)
    out["goal_jaw_ahead_of_edge_cm"] = round(
        100 * float(pr["goal_jaw"][0] - (pr["table_pos"][0] - 0.5 * pr["table_size"][0])), 2
    )
    out["goal_from_pelvis_cm"] = round(100 * float(np.linalg.norm(pr["goal_wrist"] - pp)), 1)
np.savez_compressed(
    S2 / f"plan_{name}.npz", **save, **{k: np.asarray(v) for k, v in out.items() if not isinstance(v, (list, str))}
)
print(json.dumps(out), flush=True)
