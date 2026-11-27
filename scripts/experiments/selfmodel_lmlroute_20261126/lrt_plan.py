"""selfmodel_lmlroute_20261126: the route planner (cuRobo, SIMPLE's venv, one GPU). A file queue: lrt_run.py writes one
request per (controller, task, episode) at the first reach command, this process answers it once; C and CR (and any
rerun) read the same answer. Robot knowledge only (the look, odometry, our IK); no simulator truth.

Request ($LRT_ROUTES/req/<be>__<task>__ep+0030.npz, from lrt_run.route_request; measured world = plan_rec's frame):
  start_P, start_Q, start_q   the measured pelvis (pos, quat wxyz) and joints (MuJoCo order) at the command
  start_hand                  the right hand point there (our planner's FK), for the kinematics check
  chain_RP, chain_RQ          A's straight reference at this command (our IK: to the via point when the plan has one,
                              then to the pre-grasp): pelvis pos + quat per waypoint, joints (MuJoCo order)
  chain_hand                  the right hand point per waypoint (our planner's FK)
  furn, obj                   the look's furniture and object points, mapped to the measured world through odometry
Answer ($LRT_ROUTES/routes/<be>/<task>/ep+0030.npz, written once: a second answer never replaces the first):
  meta (json): status, routed, gate_clear_cm (A's reference arm on the way to the look voxels), n_way, n_vox, plan
  diagnostics, req_md5; V: the look's voxels (centres); via_p / via_R: the simplified route (hand points, wrist
  attitudes; start .. pre-grasp; the inner points are the route command's via points); H: the dense route (hand points).

Rules (the entry's 做法, dev items 3, 4, 5, 10):
  voxels   the furniture points cropped to BOX in the heading frame of the start pelvis, 2 cm voxels, those within 2 cm of
           the object points dropped (route_feas.py);
  gate     cuRobo's sphere model of the right arm and hand (FK of the waist and right-arm joints on each waypoint's
           pelvis) along A's chain, the waypoints whose hand point is > WAY_M (10 cm) from the chain's last hand point
           (the pre-grasp as our IK reached it): least sphere-to-voxel distance < ACT (5 cm) -> plan, else no route;
  plan     base = the chain's last pose (our IK's pre-grasp pelvis); waist (3) and right arm (7) free, right hand locked
           open; start = cuRobo IK of the measured start wrist pose in the base frame (seeded with the measured joints);
           goal = the base pose's wrist pose (FK); the voxels within FREE (1 cm) of the start spheres and FREE_GOAL (2 cm)
           of an obstacle-free goal IK's spheres dropped, at most MAX_VOX; activation distance ACT; 8 trajopt / 64 IK
           seeds, 4 attempts, graph on, 10 s, finetune;
  route    the plan's hand points simplified by RDP at 1 cm (+0.5 cm until <= MAX_VIA inner points); no inner point ->
           no route (fallback, as a failed plan).

    CUDA_VISIBLE_DEVICES=n $SIMPLE/.venv/bin/python -I lrt_plan.py <routes dir> [--once]
"""

import argparse
import hashlib
import json
import math
import os
import socket
import time
import traceback
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
from scipy.spatial import cKDTree

S = Path("/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE")
MJ = ("left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
      "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
      "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
      "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
      "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
      "right_wrist_pitch_joint", "right_wrist_yaw_joint")  # fmt: skip  (MuJoCo order)
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
EE, MAX_VOX = "right_wrist_yaw_link", 8000
HAND_OFF = np.array([0.13, 0.0, 0.0])
BOX = ((-0.30, 1.20), (-1.00, 0.70), (0.03, 1.80))  # heading frame of the start pelvis: x, y; z world
VOX, TGT, FREE, FREE_GOAL = 0.02, 0.02, 0.01, 0.02
ACT, WAY_M = 0.05, 0.10
RDP_TOL, RDP_STEP, MAX_VIA = 0.01, 0.005, 16
FAR = Cuboid(name="far", pose=[0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], dims=[0.01, 0.01, 0.01])
REQ_KEYS = ("start_P", "start_Q", "start_q", "start_hand", "chain_RP", "chain_RQ", "chain_hand", "furn", "obj")


def req_md5(z) -> str:
    """md5 over the request's arrays (fixed key order, dtype and shape included); lrt_run computes the same."""
    h = hashlib.md5()
    for k in REQ_KEYS:
        a = np.ascontiguousarray(np.asarray(z[k], np.float64))
        h.update(k.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x, y, z = np.copysign(x, R[2, 1] - R[1, 2]), np.copysign(y, R[0, 2] - R[2, 0]), np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def mat_of(q):
    q = np.asarray(q, float) / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def box_sd(X, c, h):
    dd = np.abs(X - c) - h
    return np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0)


def near_spheres(V, sph, margin, vs):
    sph = sph[sph[:, 3] > 0]
    drop = np.zeros(len(V), bool)
    if not len(V) or not len(sph):
        return drop
    hits = cKDTree(V).query_ball_point(sph[:, :3], sph[:, 3] + margin + vs)
    for i, js in enumerate(hits):
        for j in js:
            if box_sd(sph[i, :3], V[j], 0.5 * vs) - sph[i, 3] < margin:
                drop[j] = True
    return drop


def clearance(C, r, tree, V, vs):
    """Least (sphere surface to voxel cube) distance over spheres C (n, 3) with radii r (n,), m; inf without voxels."""
    if not len(V):
        return float("inf")
    d, j = tree.query(C, k=min(8, len(V)))
    j = np.atleast_2d(j).reshape(len(C), -1)
    best = np.inf
    for i in range(len(C)):
        sd = box_sd(C[i], V[j[i]], 0.5 * vs) - r[i]
        best = min(best, float(sd.min()))
    return best


def rdp(X, tol):
    def seg(Y, a, b):
        ab = b - a
        u = np.clip(((Y - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        return np.linalg.norm(Y - (a + u[:, None] * ab), axis=1)

    keep, todo = [0, len(X) - 1], [(0, len(X) - 1)]
    while todo:
        i, j = todo.pop()
        if j <= i + 1:
            continue
        d = seg(X[i + 1 : j], X[i], X[j])
        k = int(np.argmax(d))
        if d[k] > tol:
            keep.append(i + 1 + k)
            todo += [(i, i + 1 + k), (i + 1 + k, j)]
    return sorted(set(keep))


def build():
    """SIMPLE's g1_29dof_with_dex3.yml as route_feas.py --waist free builds it."""
    td = TensorDeviceType()
    cfg = load_yaml(str(S / "data/robots/g1/curobo/g1_29dof_with_dex3.yml"))["robot_cfg"]
    kin = cfg["kinematics"]
    kin["external_asset_path"] = str(S / "data/robots/g1")
    kin["external_robot_configs_path"] = str(S / "data/robots/g1/curobo")
    kin["ee_link"] = EE
    kin["link_names"] = [EE] + [n for n in kin["collision_link_names"] if n.startswith("right_")]
    cs = kin["cspace"]
    cspace = list(cs["joint_names"])
    free = ARM + WAIST
    lock = [n for n in cspace if n not in free and (n.startswith("waist_") or n.startswith("right_hand_"))]
    keep = [i for i, n in enumerate(cspace) if n in free or n in lock]
    for k_, v_ in list(cs.items()):
        if isinstance(v_, list) and len(v_) == len(cspace):
            cs[k_] = [v_[i] for i in keep]
    kin["lock_joints"] = {n: 0.0 for n in lock}
    kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n.startswith("right_")]
    urdf = ET.parse(S / "data/robots/g1" / kin["urdf_path"]).getroot()
    lk = [x.get("link") for j in urdf.iter("joint") if j.get("name") in lock for x in (j.find("parent"), j.find("child"))]
    kin["link_names"] = list(dict.fromkeys(kin["link_names"] + lk + ["torso_link"]))
    robot = RobotConfig.from_dict(cfg, td)
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot, WorldConfig(cuboid=[FAR]), td, collision_checker_type=CollisionCheckerType.PRIMITIVE,
        collision_cache={"obb": MAX_VOX + 8}, collision_activation_distance=ACT, interpolation_dt=0.02,
        use_cuda_graph=False, num_trajopt_seeds=8, num_ik_seeds=64,
    ))  # fmt: skip
    return mg, td


class Plan:
    def __init__(self):
        self.mg, self.td = build()
        self.names = list(self.mg.kinematics.joint_names)
        self.tt = self.td.as_torch_dict()

    def qt(self, Q_mj):
        """MuJoCo-order joints (n, 29) -> the planner's joints (n, 10) tensor."""
        Q_mj = np.atleast_2d(np.asarray(Q_mj, float))
        return torch.tensor(np.stack([Q_mj[:, MJ.index(n)] for n in self.names], 1), **self.tt)

    def spheres(self, RP, RQ):
        st = self.mg.kinematics.get_state(self.qt(RQ))
        sph = st.link_spheres_tensor.cpu().numpy()
        ok = (sph[:, :, 3] > 0).all(0)
        return [(sph[t, ok, :3] @ mat_of(RP[t, 3:7]).T + RP[t, :3], sph[t, ok, 3]) for t in range(len(RP))]

    def ee(self, P, Rb, q_mj):
        st = self.mg.kinematics.get_state(self.qt(q_mj))
        return P + Rb @ st.ee_position[0].cpu().numpy(), Rb @ mat_of(st.ee_quaternion[0].cpu().numpy())

    def solve(self, z) -> tuple[dict, dict]:
        mg, tt, names = self.mg, self.tt, self.names
        meta, arr = dict(req_md5=req_md5(z)), {}
        pp, Rp = np.asarray(z["start_P"], float), mat_of(z["start_Q"])
        Fm, Om = np.asarray(z["furn"], float).reshape(-1, 3), np.asarray(z["obj"], float).reshape(-1, 3)
        # the look's voxels (route_feas.py)
        Rh = rz(yaw_of(Rp))
        hx = (Fm[:, :2] - pp[:2]) @ Rh[:2, :2]
        (x0, x1), (y0, y1), (z0, z1) = BOX
        inb = (hx[:, 0] > x0) & (hx[:, 0] < x1) & (hx[:, 1] > y0) & (hx[:, 1] < y1) & (Fm[:, 2] > z0) & (Fm[:, 2] < z1)
        P = Fm[inb]
        if len(Om) and len(P):
            P = P[cKDTree(Om).query(P)[0] >= TGT]
        V = (np.unique(np.floor(P / VOX).astype(np.int64), axis=0) + 0.5) * VOX if len(P) else np.zeros((0, 3))
        meta["n_vox"] = len(V)
        arr.update(V=V.astype(np.float32))  # the look's voxels (centres, measured world): M2 reads them
        # the gate: A's straight reference on the way
        RPc, RQc, Hc = (np.asarray(z[k], float) for k in ("chain_RP", "chain_RQ", "chain_hand"))
        way = np.linalg.norm(Hc - Hc[-1], axis=1) > WAY_M
        meta["n_way"] = int(way.sum())
        tree = cKDTree(V) if len(V) else None
        Sc = self.spheres(RPc, RQc)
        cl = [clearance(Sc[t][0], Sc[t][1], tree, V, VOX) for t in np.flatnonzero(way)]
        gate = min(cl) if cl else float("inf")
        meta["gate_clear_cm"] = round(100 * gate, 3) if np.isfinite(gate) else None
        meta["chain_clear_all_cm"] = round(100 * min(clearance(s[0], s[1], tree, V, VOX) for s in Sc), 3) if len(V) else None
        sh = self.ee(pp, Rp, z["start_q"])
        meta["fk_hand_err_mm"] = round(1000 * float(np.linalg.norm(sh[0] + sh[1] @ HAND_OFF - np.asarray(z["start_hand"], float))), 3)
        if not gate < ACT:
            meta.update(status="GATE_CLEAR", routed=False)
            return meta, arr
        # the plan: base = our IK's pre-grasp pose (the chain's last waypoint)
        t0 = time.time()
        Pb, Rb, qb = RPc[-1, :3], mat_of(RPc[-1, 3:7]), RQc[-1]
        wg_p, wg_R = self.ee(Pb, Rb, qb)
        pre = wg_p + wg_R @ HAND_OFF
        meta["pre_vs_chain_mm"] = round(1000 * float(np.linalg.norm(pre - Hc[-1])), 3)
        gp, gR = Rb.T @ (wg_p - Pb), Rb.T @ wg_R
        sp_, sR_ = Rb.T @ (sh[0] - Pb), Rb.T @ sh[1]
        mg.update_world(WorldConfig(cuboid=[FAR]))
        sgoal = Pose(position=torch.tensor([sp_.tolist()], **tt), quaternion=torch.tensor([quat_wxyz(sR_).tolist()], **tt))
        seed = self.qt(z["start_q"])
        iks = mg.ik_solver.solve_single(sgoal, retract_config=seed, seed_config=seed.view(1, 1, -1))
        if not bool(iks.success.item()):
            meta.update(status="START_IK_FAIL", routed=False, start_ik_pos_mm=round(1000 * float(iks.position_error.min()), 2),
                        t_plan_s=round(time.time() - t0, 2))  # fmt: skip
            return meta, arr
        q0 = iks.solution[0].view(1, -1)
        meta["start_arm_change_deg"] = round(float(np.degrees((q0 - seed).abs().max().item())), 2)
        Vb = (V - Pb) @ Rb  # the base frame
        goal = Pose(position=torch.tensor([gp.tolist()], **tt), quaternion=torch.tensor([quat_wxyz(gR).tolist()], **tt))
        s0 = mg.kinematics.get_state(q0).link_spheres_tensor[0].cpu().numpy()
        d_start = near_spheres(Vb, s0, FREE, VOX)
        ik = mg.ik_solver.solve_single(goal)
        meta["goal_ik_ok"] = bool(ik.success.item())
        d_goal = np.zeros(len(Vb), bool)
        if meta["goal_ik_ok"]:
            sg = mg.kinematics.get_state(ik.solution[0]).link_spheres_tensor[0].cpu().numpy()
            d_goal = near_spheres(Vb, sg, FREE_GOAL, VOX)
        Vk = Vb[~(d_start | d_goal)][:MAX_VOX]
        meta.update(n_vox_used=len(Vk), n_drop_start=int(d_start.sum()), n_drop_goal=int((d_goal & ~d_start).sum()))
        boxes = [Cuboid(name=f"v{i}", pose=[*map(float, c), 1.0, 0.0, 0.0, 0.0], dims=[VOX, VOX, VOX]) for i, c in enumerate(Vk)]
        mg.update_world(WorldConfig(cuboid=boxes or [FAR]))
        cfgp = MotionGenPlanConfig(max_attempts=4, enable_graph=True, timeout=10.0, enable_finetune_trajopt=True)
        res = mg.plan_single(JointState.from_position(q0, joint_names=names), goal, cfgp)
        ok = bool(res.success.item())
        meta.update(t_plan_s=round(time.time() - t0, 2), plan_ok=ok, plan_status=str(res.status).split(".")[-1])
        mg.update_world(WorldConfig(cuboid=[FAR]))
        if not ok:
            meta.update(status="PLAN_FAIL", routed=False)
            return meta, arr
        Q = res.get_interpolated_plan().position.cpu().numpy()
        Q = Q[0] if Q.ndim == 3 else Q
        ks = mg.kinematics.get_state(torch.tensor(Q, **tt))
        eP, eQ = ks.ee_position.cpu().numpy(), ks.ee_quaternion.cpu().numpy()
        H = np.array([Pb + Rb @ (p_ + mat_of(q_) @ HAND_OFF) for p_, q_ in zip(eP, eQ)])
        tol = RDP_TOL
        idx = rdp(H, tol)
        while len(idx) - 2 > MAX_VIA:
            tol += RDP_STEP
            idx = rdp(H, tol)
        sph = ks.link_spheres_tensor.cpu().numpy()
        mv = (sph[:, :, 3] > 0).all(0)
        Sw = sph[:, mv, :3] @ Rb.T + Pb
        Vkw = Vk @ Rb.T + Pb if len(Vk) else Vk
        tk = cKDTree(Vkw) if len(Vkw) else None
        a_, b_ = H[0], H[-1]
        ab = b_ - a_
        u = np.clip(((H - a_) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
        dev = np.linalg.norm(H - (a_ + u[:, None] * ab), axis=1)
        meta.update(rdp_tol_cm=round(100 * tol, 2), n_via=len(idx) - 2, dur_s=round(len(Q) * 0.02, 2),
                    path_cm=round(100 * float(np.linalg.norm(np.diff(H, axis=0), axis=1).sum()), 2),
                    straight_cm=round(100 * float(np.linalg.norm(H[-1] - H[0])), 2), max_dev_cm=round(100 * float(dev.max()), 2),
                    start_vs_hand_mm=round(1000 * float(np.linalg.norm(H[0] - (sh[0] + sh[1] @ HAND_OFF))), 2),
                    end_vs_pre_mm=round(1000 * float(np.linalg.norm(H[-1] - pre)), 2),
                    plan_clear_kept_cm=round(100 * min(clearance(Sw[t], sph[t, mv, 3], tk, Vkw, VOX) for t in range(len(Sw))), 3) if len(Vkw) else None,
                    plan_clear_cm=round(100 * min(clearance(Sw[t], sph[t, mv, 3], tree, V, VOX) for t in range(len(Sw))), 3) if len(V) else None,
                    waist_move_deg=round(float(np.degrees(np.abs(Q[-1] - Q[0])[[names.index(n) for n in WAIST]].max())), 2))  # fmt: skip
        arr.update(via_p=H[idx], via_R=np.array([Rb @ mat_of(eQ[i]) for i in idx]), H=H)
        if len(idx) <= 2:
            meta.update(status="NO_INNER_VIA", routed=False)
        else:
            meta.update(status="ROUTED", routed=True)
        return meta, arr


def write_once(path: Path, meta: dict, arr: dict) -> bool:
    """The answer, atomically, only if none exists (os.link fails on an existing name): the first answer is final."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{socket.gethostname()}.{os.getpid()}.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, meta=np.array(json.dumps(meta)), **arr)
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("routes")
    ap.add_argument("--once", action="store_true", help="stop when the queue is empty")
    a = ap.parse_args()
    root = Path(a.routes)
    req, work = root / "req", root / "work"
    req.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    tag = f"{socket.gethostname()}.{os.getpid()}.gpu{os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    pl = Plan()
    hb = root / f"planner_{tag}.json"
    n, n_routed, t_start = 0, 0, time.time()
    print(f"lrt_plan {tag}: ready, joints {pl.names}, act {ACT}", flush=True)
    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except FileNotFoundError:
            return 0.0

    while True:
        items = sorted((p for p in req.glob("*.npz") if not p.name.startswith(".")), key=mtime)  # finished requests only
        if not items:
            hb.write_text(json.dumps(dict(tag=tag, n=n, routed=n_routed, up_s=round(time.time() - t_start), idle=True, t=time.time(),
                                          gpu_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20))))  # fmt: skip
            if a.once:
                break
            time.sleep(0.2)
            continue
        for p in items:
            w = work / f"{p.stem}.{tag}.npz"
            try:
                os.rename(p, w)
            except FileNotFoundError:
                continue  # another planner took it
            be, task, eps = p.stem.split("__")
            out = root / "routes" / be / task / f"{eps}.npz"
            t0 = time.time()
            try:
                z = dict(np.load(w, allow_pickle=False))
                meta, arr = pl.solve(z)
            except Exception as e:  # recorded as a failed plan (fallback), never retried
                meta, arr = dict(status="ERROR", routed=False, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:]), {}
                try:
                    meta["req_md5"] = req_md5(z)
                except Exception:
                    pass
            meta.update(planner=tag, wall_s=round(time.time() - t0, 2), gpu_reserved_mb=round(torch.cuda.max_memory_reserved() / 2**20))
            new = write_once(out, meta, arr)
            os.unlink(w)
            n += 1
            n_routed += bool(meta.get("routed"))
            print(json.dumps(dict(be=be, task=task, ep=eps, new=new, **{k: meta.get(k) for k in (
                "status", "gate_clear_cm", "n_way", "n_vox", "n_via", "t_plan_s", "path_cm", "plan_clear_cm", "fk_hand_err_mm",
                "wall_s", "gpu_reserved_mb", "error")})), flush=True)  # fmt: skip
            hb.write_text(json.dumps(dict(tag=tag, n=n, routed=n_routed, up_s=round(time.time() - t_start), idle=False, t=time.time(),
                                          gpu_reserved_mb=meta["gpu_reserved_mb"])))  # fmt: skip
            break  # re-list: oldest first


if __name__ == "__main__":
    main()
