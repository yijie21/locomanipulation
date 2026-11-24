"""selfmodel_jref 38.10.3: a cuRobo planner that stays up and plans the right arm from many start states (the data of the
self model whose reference is cuRobo's joint trajectory; TODO 38.10.3, jref_collect.py is the client).

    cd $SIMPLE_ROOT && CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -I <this file> <SIMPLE root> <socket> [--act 0.03]

The robot as curobo_avoid_20261120/plan2.py builds it: SIMPLE's g1_29dof_with_dex3.yml, end link right_wrist_yaw_link,
the right arm's 7 joints free, the right side's collision links. One difference: the waist and the right hand are locked
at 0 once, so the planner is built once and serves every start. The client gives the world and the goal in the frame of
a pelvis that would carry the robot's ACTUAL torso with the waist at 0 (jref_collect.virtual_pelvis), so the real waist
angles cost nothing; the hand's spheres are the zero hand's (the boxes are virtual: they shape the paths, nothing in the
simulation touches them).

Several clients may connect (multiprocessing.connection, AF_UNIX, authkey b"jref"); requests are served one at a time.
request  dict(q={joint name: angle} for the 7 arm joints, boxes=[(centre (3), quat wxyz (4), dims (3)), ...],
              goal_p (3), goal_q (4) wxyz, attempts=2, timeout=5.0)
reply    dict(ok, status, names, Q (T, 7) at 0.02 s or None, t_plan, fk_p, fk_q: the start's wrist by cuRobo)
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

ARM = ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
       "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]  # fmt: skip
EE, MAX_BOXES = "right_wrist_yaw_link", 16
FAR = Cuboid(name="far", pose=[0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], dims=[0.01, 0.01, 0.01])


def build(S: Path, act: float):
    td = TensorDeviceType()
    cfg = load_yaml(str(S / "data/robots/g1/curobo/g1_29dof_with_dex3.yml"))["robot_cfg"]
    kin = cfg["kinematics"]
    kin["external_asset_path"] = str(S / "data/robots/g1")
    kin["external_robot_configs_path"] = str(S / "data/robots/g1/curobo")
    kin["ee_link"] = EE
    # SIMPLE's patched cuRobo: every collision link in link_names, and the locked joints' links added first (plan2.py)
    kin["link_names"] = [EE] + [n for n in kin["collision_link_names"] if n.startswith("right_")]
    cs = kin["cspace"]
    cspace = list(cs["joint_names"])
    lock = [n for n in cspace if n not in ARM and (n.startswith("waist_") or n.startswith("right_hand_"))]
    keep = [i for i, n in enumerate(cspace) if n in ARM or n in lock]
    for k_, v_ in list(cs.items()):
        if isinstance(v_, list) and len(v_) == len(cspace):
            cs[k_] = [v_[i] for i in keep]
    kin["lock_joints"] = {n: 0.0 for n in lock}
    kin["collision_link_names"] = [n for n in kin["collision_link_names"] if n.startswith("right_")]
    urdf = ET.parse(S / "data/robots/g1" / kin["urdf_path"]).getroot()
    lk = [
        x.get("link") for j in urdf.iter("joint") if j.get("name") in lock for x in (j.find("parent"), j.find("child"))
    ]
    kin["link_names"] = list(dict.fromkeys(kin["link_names"] + lk))
    robot = RobotConfig.from_dict(cfg, td)
    mg = MotionGen(MotionGenConfig.load_from_robot_config(
        robot, WorldConfig(cuboid=[FAR]), td, collision_checker_type=CollisionCheckerType.PRIMITIVE,
        collision_cache={"obb": MAX_BOXES}, collision_activation_distance=act, interpolation_dt=0.02,
        use_cuda_graph=False, num_trajopt_seeds=8, num_ik_seeds=64,
    ))  # fmt: skip
    return mg, td


def plan(mg, td, r: dict) -> dict:
    names = list(mg.kinematics.joint_names)
    q = torch.tensor([[float(r["q"][n]) for n in names]], **td.as_torch_dict())
    st = mg.kinematics.get_state(q)
    out = dict(ok=False, status="", names=names, Q=None, t_plan=0.0,
               fk_p=st.ee_position[0].cpu().numpy(), fk_q=st.ee_quaternion[0].cpu().numpy())  # fmt: skip
    boxes = [Cuboid(name=f"b{i}", pose=[*map(float, c), *map(float, qq)], dims=[*map(float, d)])
             for i, (c, qq, d) in enumerate(r["boxes"][:MAX_BOXES])] or [FAR]  # fmt: skip
    mg.update_world(WorldConfig(cuboid=boxes))
    goal = Pose(position=torch.tensor([list(map(float, r["goal_p"]))], **td.as_torch_dict()),
                quaternion=torch.tensor([list(map(float, r["goal_q"]))], **td.as_torch_dict()))  # fmt: skip
    t0 = time.time()
    res = mg.plan_single(JointState.from_position(q, joint_names=names), goal, MotionGenPlanConfig(
        max_attempts=int(r.get("attempts", 2)), enable_graph=True, timeout=float(r.get("timeout", 5.0)),
        enable_finetune_trajopt=True))  # fmt: skip
    out.update(t_plan=round(time.time() - t0, 3), status=str(res.status), ok=bool(res.success.item()))
    if out["ok"]:
        Q = res.get_interpolated_plan().position.cpu().numpy()
        out["Q"] = Q[0] if Q.ndim == 3 else Q
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("simple")
    ap.add_argument("socket")
    ap.add_argument("--act", type=float, default=0.03, help="m, cuRobo's collision activation distance (plan2.py's)")
    a = ap.parse_args()
    mg, td = build(Path(a.simple), a.act)
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
            if n % 200 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n} requests, {len(cs)} clients", flush=True)


if __name__ == "__main__":
    main()
