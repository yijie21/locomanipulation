"""CuRobo warehouse study, stage 3b: CuRobo motion generation on the same requests.

Runs in SIMPLE's venv. Robot: mars_curobo.json (pelvis base, waist + both arms, the fitted spheres), with the
URDF's joint limits narrowed to the expert's (requests.json `lower`/`upper`, written to a URDF copy here).
World per request: the carton's walls and the platform as cuboids in the pelvis frame. Goal: both palm poses,
from the measured start joints. Output mirrors baseline_ik.json for evaluate.py.

    CUDA_VISIBLE_DEVICES=0 third_party/SIMPLE/.venv/bin/python \
        scripts/experiments/curobo_warehouse_20261008/curobo_plan.py outputs/curobo_warehouse_20261008
"""

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

MAX_WAYPOINTS = 60


def limited_urdf(src, dst, names, lower, upper):
    """A copy of the URDF whose listed joints keep only the intersection with [lower, upper]."""
    tree = ET.parse(src)
    limits = dict(zip(names, zip(lower, upper)))
    for joint in tree.getroot().iter("joint"):
        name = joint.get("name")
        if name not in limits:
            continue
        limit = joint.find("limit")
        lo, hi = limits.pop(name)
        limit.set("lower", repr(max(float(limit.get("lower")), lo)))
        limit.set("upper", repr(min(float(limit.get("upper")), hi)))
    if limits:
        raise ValueError(f"joints missing from the URDF: {sorted(limits)}")
    tree.write(dst)


def pose(p, R):
    from curobo.types.math import Pose

    x, y, z, w = Rotation.from_matrix(np.asarray(R)).as_quat()
    return Pose(
        position=torch.tensor([p], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[w, x, y, z]], device="cuda", dtype=torch.float32),
    )


def world(obstacles):
    from curobo.geom.types import Cuboid, WorldConfig

    cuboids = []
    for o in obstacles:
        x, y, z, w = Rotation.from_matrix(np.asarray(o["rotation"])).as_quat()
        cuboids.append(Cuboid(name=o["name"], pose=[*o["center"], w, x, y, z], dims=[2 * h for h in o["half"]]))
    return WorldConfig(cuboid=cuboids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--activation", type=float, default=0.01, help="collision_activation_distance, m")
    ap.add_argument("--timeout", type=float, default=10.0, help="per plan, s")
    ap.add_argument("--attempts", type=int, default=4, help="MotionGenPlanConfig.max_attempts (CuRobo default 60)")
    args = ap.parse_args()

    from curobo.types.base import TensorDeviceType
    from curobo.types.state import JointState
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

    req = json.load(open(f"{args.dir}/requests.json"))
    robot = json.load(open(f"{args.dir}/mars_curobo.json"))["robot_cfg"]
    kin = dict(robot["kinematics"])
    urdf = f"{args.dir}/mars_limited.urdf"
    limited_urdf(kin["urdf_path"], urdf, req["active_joints"], req["lower"], req["upper"])
    kin["urdf_path"] = os.path.abspath(urdf)
    first = req["requests"][0]
    motion_gen = MotionGen(
        MotionGenConfig.load_from_robot_config(
            {"kinematics": kin},
            world(first["obstacles"]),
            TensorDeviceType(),
            collision_cache={"obb": 16},
            collision_activation_distance=args.activation,
            interpolation_dt=0.02,
            use_cuda_graph=False,
        )
    )
    names = list(motion_gen.kinematics.joint_names)
    order = [names.index(n) for n in req["active_joints"]]  # curobo order -> requests' active order
    t_warm = time.time()
    motion_gen.warmup(enable_graph=True)
    print(f"curobo joints {names}; warmup {time.time() - t_warm:.1f}s", flush=True)
    plan_config = MotionGenPlanConfig(enable_graph=True, max_attempts=args.attempts, timeout=args.timeout)

    variants = args.variants or list(req["variants"])
    requests = req["requests"][: args.limit or None]
    results = []
    t0 = time.time()
    for k, r in enumerate(requests):
        motion_gen.update_world(world(r["obstacles"]))
        start_active = np.asarray(r["start"])
        start = np.zeros(len(names))
        start[order] = start_active
        start_state = JointState.from_position(
            torch.tensor([start], device="cuda", dtype=torch.float32), joint_names=names
        )
        row = dict(label=r["label"], base=r["base"], ok=r["ok"], why=r["why"], variants={})
        for name in variants:
            left, right = r["goals"][name]
            t = time.time()
            result = motion_gen.plan_single(
                start_state,
                pose(right["pos"], right["rot"]),
                plan_config,
                link_poses={"left_wrist_yaw_link": pose(left["pos"], left["rot"])},
            )
            entry = dict(
                success=bool(result.success.item()),
                status=str(result.status),
                seconds=time.time() - t,
                position_error=float(result.position_error.item()) if result.position_error is not None else None,
                rotation_error=float(result.rotation_error.item()) if result.rotation_error is not None else None,
            )
            if entry["success"]:
                plan = result.get_interpolated_plan().position.cpu().numpy()
                step = max(1, int(np.ceil(len(plan) / MAX_WAYPOINTS)))
                waypoints = np.vstack([plan[::step], plan[-1:]])
                entry["trajectory"] = waypoints[:, order].tolist()
                entry["final"] = plan[-1, order].tolist()
                entry["duration"] = float(len(plan) * result.interpolation_dt)
            row["variants"][name] = entry
        results.append(row)
        summary = " ".join(
            f"{n}:{'ok' if e['success'] else e['status'].split('.')[-1]}" for n, e in row["variants"].items()
        )
        print(f"[{k + 1}/{len(requests)}] {r['label']} {summary} ({time.time() - t0:.0f}s)", flush=True)
    with open(f"{args.dir}/curobo_plan.json", "w") as f:
        json.dump(
            dict(
                method="CuRobo MotionGen, pelvis fixed",
                activation=args.activation,
                attempts=args.attempts,
                results=results,
            ),
            f,
        )
    print("wrote", f"{args.dir}/curobo_plan.json")


if __name__ == "__main__":
    main()
