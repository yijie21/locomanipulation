"""bigym_20261124: a cuRobo planner for BiGym's G1 right arm, kept up and serving plans (method J of bigym_bench.py, and
phase 2's self-model data the jrefwm way). The robot is make_config.py's (BiGym's own kinematics, Dex1-1 spheres).

    cd <loco checkout> && CUDA_VISIBLE_DEVICES=0 $SIMPLE/.venv/bin/python -I scripts/experiments/bigym_20261124/curobo/plan_server.py \
        outputs/bigym_20261124/plan.sock [--cfg outputs/bigym_20261124/curobo] [--act 0.03]
    ... plan_server.py --selftest      # build, plan two reaches (empty world; around a block), print, exit

Built as curobo_avoid_20261120/plan2.py builds SIMPLE's robot, with the waist locked once (jref_plan_server.py's way):
  - end link right_end_effector (BiGym's pinch-centre site, 0.148 m out of right_wrist_yaw_link)
  - only the 7 right-arm joints free; the waist locked at 0 and both right Dex1 finger joints at the open 0.0245
  - world collisions for the right arm's and right gripper's links only (the locked body is not in the world check;
    as in plan2.py, so the plans do not see the torso or the left arm)
The client gives everything in the frame of the VIRTUAL pelvis: the pelvis that would carry the robot's actual torso
with the waist at 0 (T_virtual = T_torso . T_torso<-pelvis(waist 0)^-1). The arm's chain starts at the torso, so this
is exact for every right-arm link, whatever the real waist reads.

Obstacles are voxel cubes (and optional boxes) in that frame, PRIMITIVE checker, one world per request (update_world).
Before planning, voxels are dropped (counted in the reply) when
  - farther than REACH from the right shoulder (no arm pose comes near them), and
  - their cube is within `free` of a start sphere (plan2.py's rule: the body is there now, so what the look put there is
    the robot's own points; cuRobo refuses a start in collision);
  beyond MAX_OBB the nearest to the shoulder are kept.

AF_UNIX socket, multiprocessing.connection, authkey b"bigym-jref"; several clients, one request at a time.
request  dict(q={the 7 right-arm joint names: rad}, vox=(N,3) voxel centres, vox_size=0.02, goal_p=(3,),
              goal_q=(4,) wxyz of right_end_effector, attempts=4, timeout=10.0, free=0.01,
              boxes=[(centre (3), quat wxyz (4), dims (3)), ...] optional,
              pos_only=False: True reaches goal_p with any orientation (cuRobo's reach_partial_pose, the orientation
              weights 0; goal_q is then only the IK seed's hint and end_err_deg is reported, not required))
reply    dict(ok, status, names (the 7, cuRobo's order), Q (T,7) at 0.02 s or None, t_plan, fk_p, fk_q (the start's
              right_end_effector by cuRobo), n_vox (kept), n_vox_dropped (the start's spheres), n_vox_far, n_vox_cap,
              start_clear_cm (start spheres to the kept voxels and boxes; inf without), and when ok: end_err_mm,
              end_err_deg (cuRobo's FK of Q[-1] against the goal), plan_clear_cm (the plan's spheres, every 0.02 s step))
"""

from __future__ import annotations

import argparse
import threading
import time
import xml.etree.ElementTree as ET
from multiprocessing.connection import Listener, wait
from pathlib import Path

import numpy as np
import torch
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from scipy.spatial import cKDTree

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
WAIST = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
FINGERS = ["right_dex1_finger_joint_1", "right_dex1_finger_joint_2"]
OPEN = 0.0245  # bigym.robots.configs.g1.G1_DEX1_QPOS_OPEN
EE, NAME = "right_end_effector", "g1_29dof_with_dex1_1"
AUTHKEY = b"bigym-jref"
MAX_OBB = 6000
REACH = 0.85  # m from the right shoulder: the arm (0.56 m to the pinch centre) plus its spheres, with room
SHOULDER = np.array([0.0, -0.10021, 0.29178])  # the right shoulder pitch joint in the pelvis frame at waist 0 (URDF)
FAR = Cuboid(name="far", pose=[0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], dims=[0.01, 0.01, 0.01])
REPO = Path(__file__).resolve().parents[4]


def build(cfg_dir: Path, act: float):
    td = TensorDeviceType()
    cfg = load_yaml(str(cfg_dir / f"{NAME}.yml"))["robot_cfg"]
    kin = cfg["kinematics"]
    kin["external_asset_path"] = str(cfg_dir)
    kin["external_robot_configs_path"] = str(cfg_dir)
    kin["ee_link"] = EE
    right = [n for n in kin["collision_link_names"] if n.startswith("right_")]
    kin["link_names"] = [EE] + right
    cs = kin["cspace"]
    lock = WAIST + FINGERS
    keep = [i for i, n in enumerate(cs["joint_names"]) if n in ARM or n in lock]
    for k_, v_ in list(cs.items()):
        if isinstance(v_, list) and len(v_) == len(cs["joint_names"]):
            cs[k_] = [v_[i] for i in keep]
    kin["lock_joints"] = {n: (OPEN if n in FINGERS else 0.0) for n in lock}
    kin["collision_link_names"] = right
    # SIMPLE's patched cuRobo (plan2.py): the locked joints' links go in link_names first, or the self-collision mask
    # it sizes from link_names is too small
    urdf = ET.parse(cfg_dir / kin["urdf_path"]).getroot()
    lk = [
        x.get("link") for j in urdf.iter("joint") if j.get("name") in lock for x in (j.find("parent"), j.find("child"))
    ]
    kin["link_names"] = list(dict.fromkeys(kin["link_names"] + lk))
    robot = RobotConfig.from_dict(cfg, td)
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot, WorldConfig(cuboid=[FAR]), td, collision_checker_type=CollisionCheckerType.PRIMITIVE,
        collision_cache={"obb": MAX_OBB + 64}, collision_activation_distance=act, interpolation_dt=0.02,
        use_cuda_graph=False, num_trajopt_seeds=8, num_ik_seeds=64,
    ))  # fmt: skip
    return mg, td


def mat_of(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def box_sd(X: np.ndarray, c: np.ndarray, h: float) -> np.ndarray:
    """Signed distance of points X (n,3) to axis-aligned cubes (centres c (n,3), half size h)."""
    dd = np.abs(X - c) - h
    return np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0)


def clearance(S: np.ndarray, V: np.ndarray, h: float, boxes: list) -> float:
    """Min over spheres S (n,4) of the distance to the voxel cubes V (centres, half size h) and boxes, minus the radius."""
    best = np.inf
    if len(V):
        k = min(8, len(V))
        _, idx = cKDTree(V).query(S[:, :3], k=k)
        idx = idx.reshape(len(S), k)
        sd = np.stack([box_sd(S[:, :3], V[idx[:, j]], h) for j in range(k)], 1).min(1) - S[:, 3]
        best = float(sd.min())
    for c, q, dims in boxes:
        R = mat_of(np.asarray(q, float))
        local = (S[:, :3] - np.asarray(c, float)) @ R
        dd = np.abs(local) - 0.5 * np.asarray(dims, float)
        sd = np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0) - S[:, 3]
        best = min(best, float(sd.min()))
    return best


def plan(mg, td, r: dict) -> dict:
    names = list(mg.kinematics.joint_names)
    q = torch.tensor([[float(r["q"][n]) for n in names]], **td.as_torch_dict())
    st = mg.kinematics.get_state(q)
    s0 = st.link_spheres_tensor[0].cpu().numpy()
    s0 = s0[s0[:, 3] > 0]
    out = dict(ok=False, status="", names=names, Q=None, t_plan=0.0, fk_p=st.ee_position[0].cpu().numpy(),
               fk_q=st.ee_quaternion[0].cpu().numpy())  # fmt: skip
    vs = float(r.get("vox_size", 0.02))
    V = np.asarray(r.get("vox", np.zeros((0, 3))), float).reshape(-1, 3)
    far = np.linalg.norm(V - SHOULDER, axis=1) > REACH
    V = V[~far]
    drop = np.zeros(len(V), bool)
    free = float(r.get("free", 0.01))
    if len(V):
        near = cKDTree(V).query_ball_point(s0[:, :3], s0[:, 3] + free + vs)
        for i, js in enumerate(near):
            if js:
                js = np.array(js)
                drop[js[box_sd(s0[i, :3][None], V[js], 0.5 * vs) - s0[i, 3] < free]] = True
    V = V[~drop]
    n_cap = 0
    boxes = list(r.get("boxes", []))
    if len(V) + len(boxes) > MAX_OBB:
        order = np.argsort(np.linalg.norm(V - SHOULDER, axis=1))
        n_cap = len(V) - (MAX_OBB - len(boxes))
        V = V[np.sort(order[: MAX_OBB - len(boxes)])]
    out.update(n_vox=len(V), n_vox_dropped=int(drop.sum()), n_vox_far=int(far.sum()), n_vox_cap=int(n_cap),
               start_clear_cm=round(100 * clearance(s0, V, 0.5 * vs, boxes), 2))  # fmt: skip
    cub = [Cuboid(name=f"v{i}", pose=[float(x), float(y), float(z), 1.0, 0.0, 0.0, 0.0], dims=[vs, vs, vs])
           for i, (x, y, z) in enumerate(V)]  # fmt: skip
    cub += [Cuboid(name=f"b{i}", pose=[*map(float, c), *map(float, qq)], dims=[*map(float, d)])
            for i, (c, qq, d) in enumerate(boxes)]  # fmt: skip
    mg.update_world(WorldConfig(cuboid=cub or [FAR]))
    gp, gq = np.asarray(r["goal_p"], float), np.asarray(r["goal_q"], float)
    goal = Pose(position=torch.tensor([gp.tolist()], **td.as_torch_dict()),
                quaternion=torch.tensor([gq.tolist()], **td.as_torch_dict()))  # fmt: skip
    t0 = time.time()
    metric = None
    if r.get("pos_only", False):
        w = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], **td.as_torch_dict())
        metric = PoseCostMetric(reach_partial_pose=True, reach_vec_weight=w)
    res = mg.plan_single(JointState.from_position(q, joint_names=names), goal, MotionGenPlanConfig(
        max_attempts=int(r.get("attempts", 4)), enable_graph=True, timeout=float(r.get("timeout", 10.0)),
        enable_finetune_trajopt=True, pose_cost_metric=metric))  # fmt: skip
    out.update(t_plan=round(time.time() - t0, 3), status=str(res.status), ok=bool(res.success.item()))
    if out["ok"]:
        Q = res.get_interpolated_plan().position.cpu().numpy()
        Q = Q[0] if Q.ndim == 3 else Q
        out["Q"] = Q
        ks = mg.kinematics.get_state(torch.tensor(Q, **td.as_torch_dict()))
        p_end, q_end = ks.ee_position[-1].cpu().numpy(), ks.ee_quaternion[-1].cpu().numpy()
        c = (np.trace(mat_of(gq / np.linalg.norm(gq)).T @ mat_of(q_end)) - 1) / 2
        out.update(end_err_mm=round(1000 * float(np.linalg.norm(p_end - gp)), 2),
                   end_err_deg=round(float(np.degrees(np.arccos(np.clip(c, -1, 1)))), 3))  # fmt: skip
        sph = ks.link_spheres_tensor.cpu().numpy()
        out["plan_clear_cm"] = round(
            100 * min(clearance(s[s[:, 3] > 0], V, 0.5 * vs, boxes) for s in sph), 2
        )  # every step
    return out


def selftest(mg, td) -> None:
    # BiGym's rest after the reset warmup (probe arm.json: targets 0, the arm sags to these)
    q0 = dict(zip(ARM, [0.018, 0.0016, 0.0001, 0.0276, -0.0001, 0.0382, 0.0001]))
    names = list(mg.kinematics.joint_names)
    st = mg.kinematics.get_state(torch.tensor([[q0[n] for n in names]], **td.as_torch_dict()))
    p0, qq = st.ee_position[0].cpu().numpy(), st.ee_quaternion[0].cpu().numpy()
    show = lambda tag, r: print(tag, {k: (v.shape if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in r.items()}, flush=True)  # noqa: E731  # fmt: skip
    r = plan(mg, td, dict(q=q0, vox=np.zeros((0, 3)), goal_p=p0 + [0.15, 0, 0], goal_q=qq))
    show("empty world, 15 cm ahead, the start's orientation:", r)
    r = plan(mg, td, dict(q=q0, vox=np.zeros((0, 3)), goal_p=p0 + [0.15, 0, 0], goal_q=qq, pos_only=True))
    show("empty world, 15 cm ahead, position only:", r)
    r = plan(mg, td, dict(q=q0, vox=np.zeros((0, 3)), goal_p=p0 + [0.08, 0.05, 0.08], goal_q=qq))
    show("empty world, (8, 5, 8) cm, the start's orientation:", r)
    # a 6 cm block of 2 cm voxels 10 cm over the hand, the goal 20 cm up: the path must go around it
    g = np.arange(-1, 2) * 0.02
    blk = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3) + p0 + [0.0, 0.0, 0.10]
    r0 = plan(mg, td, dict(q=q0, vox=np.zeros((0, 3)), goal_p=p0 + [0, 0, 0.20], goal_q=qq))
    if r0["ok"]:
        ks = mg.kinematics.get_state(torch.tensor(r0["Q"], **td.as_torch_dict())).link_spheres_tensor.cpu().numpy()
        print("  empty-world path 20 cm up, its clearance to the block (cm):",
              round(100 * min(clearance(s[s[:, 3] > 0], blk, 0.01, []) for s in ks), 2))  # fmt: skip
    r = plan(mg, td, dict(q=q0, vox=blk, vox_size=0.02, goal_p=p0 + [0, 0, 0.20], goal_q=qq))
    show("block over the hand, goal 20 cm up:", r)
    if r["ok"]:
        print("  with the block: path length (joint-space rad)", round(float(np.abs(np.diff(r["Q"], axis=0)).sum()), 3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("socket", nargs="?", default=str(REPO / "outputs/bigym_20261124/plan.sock"))
    ap.add_argument("--cfg", default=str(REPO / "outputs/bigym_20261124/curobo"))
    ap.add_argument("--act", type=float, default=0.03, help="m, cuRobo's collision activation distance (plan2.py's)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    mg, td = build(Path(a.cfg), a.act)
    print(f"built in {time.time() - t0:.1f} s", flush=True)
    if a.selftest:
        selftest(mg, td)
        return
    selftest(mg, td)  # warm the kernels before the first client
    Path(a.socket).unlink(missing_ok=True)
    lis = Listener(a.socket, "AF_UNIX", authkey=AUTHKEY)
    conns, lock = [], threading.Lock()

    def accept():
        while True:
            c = lis.accept()
            with lock:
                conns.append(c)

    threading.Thread(target=accept, daemon=True).start()
    print(f"ready {a.socket}", flush=True)
    n = 0
    while True:
        with lock:
            cs = list(conns)
        if not cs:
            time.sleep(0.05)
            continue
        for c in wait(cs, timeout=0.5):
            try:
                r = c.recv()
            except (EOFError, OSError):
                with lock:
                    conns.remove(c)
                continue
            try:
                c.send(plan(mg, td, r))
            except Exception as e:  # one bad request must not take the server down
                c.send(dict(ok=False, status=f"error {type(e).__name__}: {e}", names=[], Q=None, t_plan=0.0))
            n += 1
            if n % 100 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n} requests, {len(cs)} clients", flush=True)


if __name__ == "__main__":
    main()
