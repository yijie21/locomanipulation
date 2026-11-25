"""cuRobo for method J on HumanoidArena: the right arm's 7 joints (waist and right hand locked at 0) from the measured
start to the pregrasp, the look's voxels as obstacles; kept up as an AF_UNIX socket server so the bench builds it once.

The robot: SIMPLE's Dex3 G1 config adjusted to HumanoidArena's (curobo/g1_29dof_with_dex3_ha.yml, 0.001 mm off Isaac
Lab's kinematics; kin_compare.py), end link right_hand_palm_link, the right arm's and hand's collision spheres, as
curobo_avoid_20261120/plan2.py builds it. As simple_eval_20261112/jref_plan_server.py does, the waist is locked at 0
once and the client gives everything in a VIRTUAL pelvis: the frame in which a pelvis with the waist at 0 carries the
robot's actual torso (ha_bench.virtual_pelvis), so the measured waist costs nothing.

World: the request's voxels (centres, one cube of `vox` each), less those within `free` of the start pose's spheres
(the body is there now: its own points or depth fill; cuRobo refuses a start in collision) and within `free_goal` of the
goal pose's spheres (an obstacle-free IK of the goal first: the hand has to be there). Up to MAX_VOX cubes (a sparser
grid is the client's job).

    cd $SIMPLE && CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file> <socket> [--act 0.03]
request  dict(q={7 arm joint: angle}, vox=(N, 3), vox_size, goal_p (3), goal_q (4, wxyz), attempts=4, timeout=10.0,
              free=0.01, free_goal=0.02)
reply    dict(ok, status, names, Q (T, 7) at 0.02 s or None, t_plan, n_vox, n_drop_start, n_drop_goal, goal_ik_ok)
"""

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
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
S = Path("/wuji-vefps-D/wuji-il/yijie/tools/simple_isaac/SIMPLE")
ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
EE, MAX_VOX = "right_hand_palm_link", 6000
FAR = Cuboid(name="far", pose=[0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], dims=[0.01, 0.01, 0.01])


def build(act: float):
    td = TensorDeviceType()
    cfg = load_yaml(str(HERE / "curobo/g1_29dof_with_dex3_ha.yml"))["robot_cfg"]
    kin = cfg["kinematics"]
    kin["external_asset_path"] = str(S / "data/robots/g1")
    kin["external_robot_configs_path"] = str(S / "data/robots/g1/curobo")
    kin["ee_link"] = EE
    kin["link_names"] = [EE] + [n for n in kin["collision_link_names"] if n.startswith("right_") and n != EE]
    cs = kin["cspace"]
    cspace = list(cs["joint_names"])
    lock = [n for n in cspace if n not in ARM and (n.startswith("waist_") or n.startswith("right_hand_"))]
    keep = [i for i, n in enumerate(cspace) if n in ARM or n in lock]
    for k_, v_ in list(cs.items()):
        if isinstance(v_, list) and len(v_) == len(cspace):
            cs[k_] = [v_[i] for i in keep]
    kin["lock_joints"] = {n: 0.0 for n in lock}
    kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n.startswith("right_")]
    urdf = ET.parse(kin["urdf_path"]).getroot()
    lk = [
        x.get("link") for j in urdf.iter("joint") if j.get("name") in lock for x in (j.find("parent"), j.find("child"))
    ]
    kin["link_names"] = list(dict.fromkeys(kin["link_names"] + lk))
    robot = RobotConfig.from_dict(cfg, td)
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot, WorldConfig(cuboid=[FAR]), td, collision_checker_type=CollisionCheckerType.PRIMITIVE,
        collision_cache={"obb": MAX_VOX + 8}, collision_activation_distance=act, interpolation_dt=0.02,
        use_cuda_graph=False, num_trajopt_seeds=8, num_ik_seeds=64,
    ))  # fmt: skip
    return mg, td


def box_sd(X, c, h):
    dd = np.abs(X - c) - h
    return np.linalg.norm(np.maximum(dd, 0), axis=-1) + np.minimum(dd.max(-1), 0)


def near_spheres(V, sph, margin, vs):
    """Voxels whose cube is within `margin` of any sphere."""
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


def plan(mg, td, r: dict) -> dict:
    names = list(mg.kinematics.joint_names)
    q = torch.tensor([[float(r["q"][n]) for n in names]], **td.as_torch_dict())
    vs = float(r.get("vox_size", 0.02))
    V = np.asarray(r.get("vox", np.zeros((0, 3))), float).reshape(-1, 3)
    out = dict(ok=False, status="", names=names, Q=None, t_plan=0.0, n_vox_in=len(V))
    goal = Pose(position=torch.tensor([list(map(float, r["goal_p"]))], **td.as_torch_dict()),
                quaternion=torch.tensor([list(map(float, r["goal_q"]))], **td.as_torch_dict()))  # fmt: skip
    t0 = time.time()
    s0 = mg.kinematics.get_state(q).link_spheres_tensor[0].cpu().numpy()
    d_start = near_spheres(V, s0, float(r.get("free", 0.01)), vs)
    mg.update_world(WorldConfig(cuboid=[FAR]))
    ik = mg.ik_solver.solve_single(goal)
    out["goal_ik_ok"] = bool(ik.success.item())
    d_goal = np.zeros(len(V), bool)
    if out["goal_ik_ok"]:
        sg = mg.kinematics.get_state(ik.solution[0]).link_spheres_tensor[0].cpu().numpy()
        d_goal = near_spheres(V, sg, float(r.get("free_goal", 0.02)), vs)
    keep = ~(d_start | d_goal)
    Vk = V[keep][:MAX_VOX]
    out.update(n_vox=len(Vk), n_drop_start=int(d_start.sum()), n_drop_goal=int((d_goal & ~d_start).sum()))
    boxes = [
        Cuboid(name=f"v{i}", pose=[*map(float, c), 1.0, 0.0, 0.0, 0.0], dims=[vs, vs, vs]) for i, c in enumerate(Vk)
    ]
    mg.update_world(WorldConfig(cuboid=boxes or [FAR]))
    res = mg.plan_single(JointState.from_position(q, joint_names=names), goal, MotionGenPlanConfig(
        max_attempts=int(r.get("attempts", 4)), enable_graph=True, timeout=float(r.get("timeout", 10.0)),
        enable_finetune_trajopt=True))  # fmt: skip
    out.update(t_plan=round(time.time() - t0, 3), status=str(res.status), ok=bool(res.success.item()))
    if out["ok"]:
        Q = res.get_interpolated_plan().position.cpu().numpy()
        out["Q"] = Q[0] if Q.ndim == 3 else Q
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("socket")
    ap.add_argument("--act", type=float, default=0.03, help="m, cuRobo's collision activation distance (plan2.py's)")
    a = ap.parse_args()
    mg, td = build(a.act)
    Path(a.socket).unlink(missing_ok=True)
    lis = Listener(a.socket, "AF_UNIX", authkey=b"jref")
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
            print(f"{time.strftime('%H:%M:%S')} request {n}", flush=True)


if __name__ == "__main__":
    main()
