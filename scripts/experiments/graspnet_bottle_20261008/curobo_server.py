"""TODO 21.3.4: SIMPLE's CuRobo planner for the bottle robot, served over ZMQ to the episode process.

What SIMPLE's MotionPlannerAgent does for a parallel-gripper grasp (third_party/SIMPLE/src/simple/mp/curobo.py,
`batch_plan_for_approach` and `batch_plan_for_lift`), on our robot (curobo_robot_right.py: waist + right arm, pelvis
fixed, inscribed spheres) and in our world (the table top as a cuboid, the target's CoACD hull as a mesh, both in the
pelvis frame):
- one batched MotionGen plan from the current joints to every candidate's pre-grasp wrist pose, SIMPLE's plan config
  (no finetune trajopt, 12 trajopt seeds, 10 attempts), the shortest successful trajectory kept;
- the straight entry from the pre-grasp to the grasp and the lift as IK steps of 1 cm, each seeded with the last
  (SIMPLE's lift, without world collision: the jaw has to reach the object).
Joint limits are the episode's (the IK planner's), sent with every request, so both planners search the same space.

    request  (JSON): {"start": [10], "names": [10], "lower": [10], "upper": [10], "pregrasp": [[x,y,z,qw,qx,qy,qz]...],
                      "grasp": [[...]...], "table": {"center": [3], "half": [3], "quat": [4]},
                      "object": {"vertices": [[3]...], "faces": [[3]...]}, "lift_m": 0.10}
    reply    (JSON): {"ok", "index", "pregrasp_path", "entry_path", "lift_path", "seconds", "status", "successes",
                      "success": [bool per candidate]}

    CUDA_VISIBLE_DEVICES=1 third_party/SIMPLE/.venv/bin/python scripts/experiments/graspnet_bottle_20261008/curobo_server.py \
        outputs/graspnet_bottle_20261008/curobo/g1_right_curobo.json --port 5791
"""

from __future__ import annotations

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

STEP_M = 0.01


def limited_urdf(src, dst, limits):
    tree = ET.parse(src)
    for joint in tree.getroot().iter("joint"):
        if joint.get("name") in limits:
            lo, hi = limits[joint.get("name")]
            lim = joint.find("limit")
            lim.set("lower", repr(float(lo)))
            lim.set("upper", repr(float(hi)))
    tree.write(dst)


class Planner:
    def __init__(self, robot_json: str, work: str):
        self.robot = json.load(open(robot_json))["robot_cfg"]
        self.work = work
        self.key = None

    def configure(self, names, lower, upper, retract):
        key = (tuple(names), tuple(np.round(lower, 5)), tuple(np.round(upper, 5)))
        if key == self.key:
            return
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import Cuboid, WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        kin = dict(self.robot["kinematics"])
        urdf = os.path.join(self.work, "g1_right_limited.urdf")
        limited_urdf(kin["urdf_path"], urdf, {n: (lo, hi) for n, lo, hi in zip(names, lower, upper)})
        kin["urdf_path"] = urdf
        # IK and trajopt regularise toward the retract configuration: the episode's first start (the hanging arm)
        # rather than all zeros, which with no posture cost parked the waist on its limits (2026-10-08)
        cspace = dict(kin["cspace"])
        order = [cspace["joint_names"].index(n) for n in names]
        retract_cs = [0.0] * len(cspace["joint_names"])
        for i, v in zip(order, retract):
            retract_cs[i] = float(v)
        cspace["retract_config"] = retract_cs
        kin["cspace"] = cspace
        cfg = {"kinematics": kin}
        placeholder = WorldConfig(cuboid=[Cuboid(name="table", pose=[1, 0, -2, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])])
        t = TensorDeviceType()
        self.motion_gen = MotionGen(
            MotionGenConfig.load_from_robot_config(
                cfg,
                placeholder,
                t,
                interpolation_dt=0.02,
                interpolation_steps=1000,
                collision_checker_type=CollisionCheckerType.MESH,
                collision_cache={"obb": 10, "mesh": 10},
                collision_activation_distance=0.01,
                ik_opt_iters=200,
                grad_trajopt_iters=200,
                use_cuda_graph=False,
            )
        )
        self.ik = IKSolver(
            IKSolverConfig.load_from_robot_config(
                cfg,
                None,
                t,
                position_threshold=0.001,
                rotation_threshold=0.01,
                num_seeds=2,
                self_collision_check=False,
                self_collision_opt=False,
                use_cuda_graph=False,
            )
        )
        self.names = list(self.motion_gen.kinematics.joint_names)
        self.order = [self.names.index(n) for n in names]  # curobo index of each request joint
        self.key = key
        print(f"[curobo] configured {self.names}", flush=True)

    def _to_curobo(self, q):
        out = np.zeros(len(self.names))
        out[self.order] = q
        return out

    def _ik_line(self, q0_curobo, start_pose, end_pose):
        """IK steps of STEP_M from start_pose to end_pose (positions straight, orientation fixed to end's)."""
        from curobo.types.math import Pose

        p0, p1 = np.asarray(start_pose[:3]), np.asarray(end_pose[:3])
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / STEP_M)))
        q = torch.tensor([q0_curobo], device="cuda", dtype=torch.float32)
        path = []
        for k in range(1, n + 1):
            p = p0 + (p1 - p0) * k / n
            goal = Pose(
                position=torch.tensor([p], device="cuda", dtype=torch.float32),
                quaternion=torch.tensor([end_pose[3:]], device="cuda", dtype=torch.float32),
            )
            r = self.ik.solve_single(goal, retract_config=q, seed_config=q[:, None, :].repeat(1, 2, 1))
            if not bool(r.success.any()):
                return None, f"ik step {k}/{n} failed"
            q = r.solution[:, 0, :]
            path.append(q[0].cpu().numpy()[self.order].tolist())
        return path, ""

    def plan(self, req: dict) -> dict:
        from curobo.geom.types import Cuboid, Mesh, WorldConfig
        from curobo.types.math import Pose
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        t0 = time.time()
        self.configure(req["names"], req["lower"], req["upper"], req["start"])
        table = req["table"]
        world = WorldConfig(
            cuboid=[
                Cuboid(
                    name="table",
                    pose=[*table["center"], *table.get("quat", [1, 0, 0, 0])],
                    dims=[2 * h for h in table["half"]],
                )
            ],
            mesh=[
                Mesh(
                    name="target",
                    pose=[0, 0, 0, 1, 0, 0, 0],
                    vertices=req["object"]["vertices"],
                    faces=req["object"]["faces"],
                )
            ],
        )
        self.motion_gen.update_world(world)
        pre = np.asarray(req["pregrasp"], dtype=np.float32)
        b = len(pre)
        start = torch.tensor([self._to_curobo(req["start"])] * b, device="cuda", dtype=torch.float32)
        goals = Pose(
            position=torch.tensor(pre[:, :3], device="cuda"), quaternion=torch.tensor(pre[:, 3:], device="cuda")
        )
        result = self.motion_gen.plan_batch(
            JointState.from_position(start, joint_names=self.names),
            goals,
            MotionGenPlanConfig(enable_finetune_trajopt=False, num_trajopt_seeds=12, max_attempts=10),
        )
        success = result.success.cpu().numpy().astype(bool)
        out = dict(
            ok=False, successes=int(success.sum()), success=success.tolist(), status=str(result.status), index=None
        )
        if not success.any():
            out["seconds"] = time.time() - t0
            return out
        lengths = result.path_buffer_last_tstep
        lengths = [int(v) for v in (lengths.tolist() if torch.is_tensor(lengths) else lengths)]
        tried = []
        for i in sorted(np.flatnonzero(success), key=lambda i: lengths[i]):  # SIMPLE keeps the shortest success
            # a batch of one comes back without its batch axis; indexing it raised "JointState does not have horizon"
            # and turned a success into "no path" (objects 053 and 060, 2026-10-08)
            traj = result.interpolated_plan
            traj = traj[int(i)] if traj.position.ndim == 3 else traj
            plan = traj.trim_trajectory(0, lengths[i]).position.cpu().numpy()
            path = plan[:, self.order]
            # SIMPLE's straight entry and lift as CuRobo IK steps: recorded, not required -- the episode executes the
            # entry and the lift with its shared closed-loop servo for both planners
            entry, entry_why = self._ik_line(plan[-1], req["pregrasp"][i], req["grasp"][i])
            lift, lift_why = None, "no entry"
            if entry is not None:
                lift_from = list(req["grasp"][i])
                lift_to = [lift_from[0], lift_from[1], lift_from[2] + float(req.get("lift_m", 0.10)), *lift_from[3:]]
                lift, lift_why = self._ik_line(self._to_curobo(entry[-1]), lift_from, lift_to)
            out.update(
                ok=True, index=int(i), pregrasp_path=path.tolist(), entry_path=entry or [], lift_path=lift or [],
                entry_status=entry_why or "ok", lift_status=lift_why or "ok",
            )  # fmt: skip
            break
        out.update(seconds=time.time() - t0, rejected=tried)
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("robot")
    ap.add_argument("--port", type=int, default=5791)
    args = ap.parse_args()
    import zmq

    planner = Planner(args.robot, os.path.dirname(os.path.abspath(args.robot)))
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[curobo] listening on :{args.port}", flush=True)
    while True:
        req = json.loads(sock.recv())
        try:
            rep = planner.plan(req)
        except Exception as exc:  # noqa: BLE001 -- the episode records why the planner gave no path
            import traceback

            traceback.print_exc()
            rep = dict(ok=False, status=f"error: {exc!r}", seconds=0.0, successes=0, index=None)
        sock.send_string(json.dumps(rep))
        print(
            f"[curobo] {len(req.get('pregrasp', []))} candidates -> ok {rep['ok']} index {rep.get('index')} "
            f"successes {rep.get('successes')} {rep.get('seconds', 0):.1f} s {rep.get('status', '')[:80]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
