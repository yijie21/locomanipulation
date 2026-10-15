"""CuRobo arm-route service: a collision-free joint path from the current arm to an end-effector goal, over ZMQ.

Runs as its own process in SIMPLE's venv (CuRobo 0.7.4); the episode process talks to it through
`primitives/arm_route.py` and never imports this module's CuRobo side. Item 4 of the plan in
docs/handoff/single-arm-curobo-route-plan-20261009.md; the planner settings are the ones the GraspNet comparison
ran (scripts/experiments/graspnet_bottle_20261008/curobo_server.py, which stays as it was for that record).

Messages (JSON, REQ/REP):

    {"cmd": "configure", "names": [n], "lower": [n], "upper": [n], "retract": [n]}
        -> {"ok": true, "built": bool, "seconds": s}
    {"cmd": "plan", "start": [n], "goals": [[x, y, z, qw, qx, qy, qz], ...],
     "world": {"cuboids": [{"name", "pose": [7], "dims": [3]}], "meshes": [{"name", "pose": [7], "vertices", "faces"}]}}
        -> {"ok", "index", "path": [[n]...], "success": [bool per goal], "status", "seconds",
            "canary": bool, "rebuilt": bool, "poisoning": bool (the last three on failed plans only)}
    {"cmd": "plan", "start": [n], "goal_joints": [n], "world": {...}}
        -> the same reply for one joint-space goal (MotionGen.plan_single_js, collision-checked like a pose plan)

A plan request may carry `"check_start_validity": false` (default true): CuRobo then plans from a start that already
touches an obstacle instead of refusing it (MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION). The warehouse reach
route retries with it once, for a hanging hand already inside its inflated clearance (tasks/crate/reach_route.py).

Poses are in the robot base frame CuRobo plans in (the pelvis); joint vectors follow the request's `names`. `retract` is
the posture the solvers regularise toward: with none the waist parked on its limits (2026-10-08). It is part of the
configuration, so a changed retract rebuilds; the GraspNet server kept the first request's start for every later one.

Repeatability (measured 2026-10-09). Without a seed reset, two plain repeats of one request differed by up to 0.67 rad
and could pick another goal: the solvers' random streams advance between calls and a new batch shape reallocates
them. So every plan pads its goals to the service's fixed batch (copies of the first goal, whose results are
discarded) and resets the seeds first, and `configure` ends with a warm-up plan. That makes a request repeated
straight after itself identical, but not one repeated after other requests or a rebuild: replaying the GraspNet
comparison's 119 saved requests twice, in opposite orders (scripts/experiments/curobo_route_20261009/replay_saved.py),
the success flags agreed on 117/119 and the paths differed by a median 0.029 rad (p90 0.37, max 0.51). Success is what
the service holds still -- on the 57 single-goal requests of the fixed-grasp run it solved 54, as the GraspNet server
did live (one swap each way) -- not the path. The shortest successful trajectory among the real goals is returned
(SIMPLE's `batch_plan_for_approach` keeps the shortest per group).

Self-healing (2026-10-09): one failed request (a GraspNet debug run's, gnb_obj_000_smoke10) left the solvers failing
IK on every later request, including ones a fresh build solves, at a fixed batch; clearing the world cache, resetting
the seeds and dropping the solvers' goal buffers did not recover it, a new MotionGen did. The GraspNet server never
showed it because its batch shape changed almost every call. So a plan where no goal succeeds is followed by the
warm-up plan (the canary, solvable by construction); if that fails too the service rebuilds and plans the request
once more, and says so in the reply (`canary`, `rebuilt`); if the retry fails and breaks the canary again, the request
itself is the cause (`poisoning`) and the service rebuilds once more so the next request starts clean. A canary that
fails straight after a build is an error.

Joint-space goals (protocol 2, 2026-10-10): the single-arm route asks for its lifted-ready joints, not the wrist pose, so
the loop after it starts from the configuration the lifted ready gives it. A pose goal reached the same wrist with the
waist yawed and rolled and the shoulder and wrist in another solution, and the grasp loop's posture prior carried that
into the entry (route plan section 6). A joint goal outside the configured limits is an error, not clipped here.

Batch: at least 2, even for one goal (pose goals). CuRobo 0.7.4 mishandles a batch of one twice: the plan comes back without its
batch axis (the GraspNet server's "no horizon" failure), and a goal that fails its first attempt and succeeds a later
one crashes merging the attempts (`copy_idx` indexes a 0-d tensor; replay of gnb_obj_008, 2026-10-09).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src third_party/SIMPLE/.venv/bin/python -m locomani.primitives.curobo_service \
        <robot.json> --port 5795 --batch 2
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

PROTOCOL = 2


def pad_goals(goals, batch: int) -> tuple[list, int]:
    """(goals padded to `batch` with copies of the first, number of real goals). More goals than the batch is an error:
    a second batch shape is exactly what the fixed batch exists to avoid."""
    goals = [list(map(float, g)) for g in goals]
    n = len(goals)
    if n == 0:
        raise ValueError("plan needs at least one goal")
    if n > batch:
        raise ValueError(f"{n} goals exceed the service batch {batch}")
    if any(len(g) != 7 for g in goals):
        raise ValueError("every goal is [x, y, z, qw, qx, qy, qz]")
    return goals + [goals[0]] * (batch - n), n


def check_goal_joints(goal, lower, upper, tol: float = 1e-6) -> np.ndarray:
    """A joint-space goal as a float vector, refused when it is not finite, has the wrong length or leaves the
    configured limits (by more than `tol`)."""
    goal = np.asarray(goal, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    if goal.shape != lower.shape or not np.isfinite(goal).all():
        raise ValueError(f"goal_joints must be {lower.shape[0]} finite values")
    outside = np.flatnonzero((goal < lower - tol) | (goal > upper + tol))
    if len(outside):
        raise ValueError(f"goal_joints outside the configured limits at {outside.tolist()}")
    return goal


def shortest_success(success, lengths, n: int) -> int | None:
    """Index of the shortest successful trajectory among the first `n` (real) goals; None if none succeeded.
    `lengths` is CuRobo's per-goal trajectory length, which it leaves None when nothing succeeded."""
    ok = [i for i in range(n) if bool(success[i])]
    if not ok:
        return None
    if lengths is None:
        raise ValueError("successful goals without trajectory lengths")
    lengths = [int(v) for v in (lengths.tolist() if hasattr(lengths, "tolist") else lengths)]
    return min(ok, key=lambda i: (lengths[i], i))


def limited_urdf(src: str, dst: str, limits: dict) -> None:
    """Copy `src` with the given joints' limits replaced, so CuRobo searches the episode's joint ranges."""
    tree = ET.parse(src)
    for joint in tree.getroot().iter("joint"):
        if joint.get("name") in limits:
            lo, hi = limits[joint.get("name")]
            lim = joint.find("limit")
            lim.set("lower", repr(float(lo)))
            lim.set("upper", repr(float(hi)))
    tree.write(dst)


@dataclasses.dataclass(frozen=True)
class Settings:
    """MotionGen settings. The defaults are the GraspNet comparison's, under which every route up to 2026-10-15 was
    planned; `primitive` checks boxes only and refuses a request that carries meshes. `self_collision_opt` False drops
    the self-collision cost from the optimisation but keeps the self-collision check on the result (a colliding path
    fails): with the mars model's 722 spheres that cost's kernel was 99.5% of a warehouse route's GPU time
    (TODO 26.5.5, 2026-10-15)."""

    cuda_graph: bool = False
    checker: str = "mesh"  # mesh | primitive
    ik_iters: int = 200
    trajopt_iters: int = 200
    self_collision_opt: bool = True

    def __post_init__(self):
        if self.checker not in ("mesh", "primitive"):
            raise ValueError(f"checker must be mesh or primitive, not {self.checker!r}")
        if self.ik_iters < 1 or self.trajopt_iters < 1:
            raise ValueError("ik_iters and trajopt_iters must be positive")


class RoutePlanner:
    """One MotionGen per configuration, warmed up at the fixed batch."""

    def __init__(self, robot_json: str, batch: int, work_dir: str, settings: Settings | None = None):
        with open(robot_json, "rb") as f:
            raw = f.read()
        self.robot = json.loads(raw)["robot_cfg"]
        self.robot_digest = hashlib.sha256(raw).hexdigest()[:16]
        self.settings = Settings() if settings is None else settings
        self.batch = int(batch)
        if self.batch < 2:
            raise ValueError("the route service's batch must be at least 2 (CuRobo's batch of one, module doc)")
        self.work = work_dir
        self.key = None
        self.motion_gen = None
        self.config_args = None  # what the current build was configured with, for a rebuild
        self.canary = None  # the warm-up request
        self.rebuilds = 0

    def configure(self, names, lower, upper, retract) -> bool:
        """Build for these joints, limits and retract posture; False when already built for them."""
        names = list(names)
        lower, upper, retract = (np.asarray(v, dtype=float) for v in (lower, upper, retract))
        if not (len(names) == len(lower) == len(upper) == len(retract)):
            raise ValueError("names, lower, upper and retract differ in length")
        if np.any(lower > upper) or np.any(retract < lower - 1e-6) or np.any(retract > upper + 1e-6):
            raise ValueError("retract outside [lower, upper], or lower > upper")
        key = (tuple(names), tuple(np.round(lower, 6)), tuple(np.round(upper, 6)), tuple(np.round(retract, 6)))
        if key == self.key:
            return False
        self.config_args = (names, lower, upper, retract)
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import Cuboid, WorldConfig
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        kin = dict(self.robot["kinematics"])
        cspace = dict(kin["cspace"])
        missing = [n for n in names if n not in cspace["joint_names"]]
        if missing:
            raise ValueError(f"joints not in the robot's chain: {missing}")
        urdf = os.path.join(self.work, "limited.urdf")
        limited_urdf(kin["urdf_path"], urdf, {n: (lo, hi) for n, lo, hi in zip(names, lower, upper)})
        kin["urdf_path"] = urdf
        retract_cs = [0.0] * len(cspace["joint_names"])
        for name, value in zip(names, retract):
            retract_cs[cspace["joint_names"].index(name)] = float(value)
        cspace["retract_config"] = retract_cs
        kin["cspace"] = cspace
        placeholder = WorldConfig(
            cuboid=[Cuboid(name="placeholder", pose=[1, 0, -2, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]
        )
        self.motion_gen = MotionGen(
            MotionGenConfig.load_from_robot_config(
                {"kinematics": kin},
                placeholder,
                TensorDeviceType(),
                interpolation_dt=0.02,
                interpolation_steps=1000,
                collision_checker_type=(
                    CollisionCheckerType.PRIMITIVE
                    if self.settings.checker == "primitive"
                    else CollisionCheckerType.MESH
                ),
                collision_cache={"obb": 10} if self.settings.checker == "primitive" else {"obb": 10, "mesh": 10},
                collision_activation_distance=0.01,
                ik_opt_iters=self.settings.ik_iters,
                grad_trajopt_iters=self.settings.trajopt_iters,
                use_cuda_graph=self.settings.cuda_graph,
                self_collision_opt=self.settings.self_collision_opt,
            )
        )
        chain = list(self.motion_gen.kinematics.joint_names)
        if sorted(chain) != sorted(names):
            raise ValueError(f"request joints {names} are not the planned chain {chain}")
        self.names, self.order = names, [chain.index(n) for n in names]
        self.key = key
        self._warm_up(retract)
        return True

    def _warm_up(self, retract) -> None:
        """One plan at the fixed batch, to the retract posture's own end-effector pose plus 5 cm up, empty world; kept
        as the canary. It has to succeed: a build that cannot plan it is broken."""
        import torch

        state = self.motion_gen.kinematics.get_state(
            torch.tensor([self._to_chain(retract)], device="cuda", dtype=torch.float32)
        )
        pos = state.ee_position[0].cpu().numpy() + [0.0, 0.0, 0.05]
        goal = [*pos.tolist(), *state.ee_quaternion[0].cpu().numpy().tolist()]
        self.canary = dict(start=list(map(float, retract)), goals=[goal], world={})
        if not self._plan_once(self.canary)["ok"]:
            raise RuntimeError("the warm-up plan failed right after a build")

    def fingerprint(self) -> dict:
        """Everything besides the request that decides a plan; a client's route cache keys on it."""
        return dict(settings=dataclasses.asdict(self.settings), batch=self.batch, robot=self.robot_digest,
                    protocol=PROTOCOL)  # fmt: skip

    def rebuild(self) -> None:
        self.key = None
        self.configure(*self.config_args)
        self.rebuilds += 1

    def _to_chain(self, q):
        out = np.zeros(len(self.names))
        out[self.order] = q
        return out

    def _world(self, world: dict):
        from curobo.geom.types import Cuboid, Mesh, WorldConfig

        cuboids = [Cuboid(name=c["name"], pose=list(c["pose"]), dims=list(c["dims"])) for c in world.get("cuboids", [])]
        if world.get("meshes") and self.settings.checker == "primitive":
            raise ValueError("mesh obstacles need the mesh checker (this service runs --checker primitive)")
        meshes = [
            Mesh(name=m["name"], pose=list(m["pose"]), vertices=m["vertices"], faces=m["faces"])
            for m in world.get("meshes", [])
        ]
        if not cuboids and not meshes:
            # an empty world still needs one obstacle for the mesh checker's cache; far below the robot
            cuboids = [Cuboid(name="placeholder", pose=[1, 0, -2, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]
        return WorldConfig(cuboid=cuboids, mesh=meshes)

    def plan(self, req: dict) -> dict:
        """`_plan_once`; when nothing succeeded, the canary decides whether the solvers are still sound (module doc)."""
        out = self._plan_once(req)
        if out["ok"]:
            return out
        canary = self._plan_once(self.canary)["ok"]
        out.update(canary=canary, rebuilt=False)
        if canary:
            return out
        self.rebuild()
        again = self._plan_once(req)
        again.update(canary=False, rebuilt=True, first_status=out["status"], poisoning=False)
        if not again["ok"] and not self._plan_once(self.canary)["ok"]:
            # the request itself breaks the solvers: leave a clean build for the next one
            self.rebuild()
            again["poisoning"] = True
        return again

    def _plan_once(self, req: dict) -> dict:
        import torch
        from curobo.types.math import Pose
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        if self.motion_gen is None:
            raise RuntimeError("plan before configure")
        if req.get("goal_joints") is not None:
            return self._plan_joints(req)
        t0 = time.time()
        goals, n = pad_goals(req["goals"], self.batch)
        start = np.asarray(req["start"], dtype=float)
        if start.shape != (len(self.names),):
            raise ValueError(f"start has {start.shape[0]} joints, configured for {len(self.names)}")
        self.motion_gen.update_world(self._world(req.get("world") or {}))
        self.motion_gen.reset_seed()
        g = np.asarray(goals, dtype=np.float32)
        result = self.motion_gen.plan_batch(
            JointState.from_position(
                torch.tensor([self._to_chain(start)] * self.batch, device="cuda", dtype=torch.float32),
                joint_names=list(self.motion_gen.kinematics.joint_names),
            ),
            Pose(position=torch.tensor(g[:, :3], device="cuda"), quaternion=torch.tensor(g[:, 3:], device="cuda")),
            MotionGenPlanConfig(
                enable_finetune_trajopt=False,
                num_trajopt_seeds=12,
                max_attempts=10,
                check_start_validity=bool(req.get("check_start_validity", True)),
            ),
        )
        success = result.success.cpu().numpy().astype(bool).reshape(-1)[:n]
        out = dict(ok=False, index=None, path=[], success=success.tolist(), status=str(result.status))
        lengths = result.path_buffer_last_tstep  # None when no goal succeeded
        i = shortest_success(success, lengths, n)
        if i is not None:
            traj = result.interpolated_plan
            # a batch of one comes back without its batch axis (the GraspNet server's "no horizon" bug, 2026-10-09)
            traj = traj[i] if traj.position.ndim == 3 else traj
            path = traj.trim_trajectory(0, lengths[i]).position.cpu().numpy()[:, self.order]
            out.update(ok=True, index=i, path=path.tolist())
        out["seconds"] = time.time() - t0
        return out

    def _plan_joints(self, req: dict) -> dict:
        import torch
        from curobo.types.state import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        t0 = time.time()
        _, lower, upper, _ = self.config_args
        start = np.asarray(req["start"], dtype=float)
        if start.shape != (len(self.names),):
            raise ValueError(f"start has {start.shape[0]} joints, configured for {len(self.names)}")
        goal = check_goal_joints(req["goal_joints"], lower, upper)
        self.motion_gen.update_world(self._world(req.get("world") or {}))
        self.motion_gen.reset_seed()
        chain = list(self.motion_gen.kinematics.joint_names)

        def state(q):
            return JointState.from_position(
                torch.tensor([self._to_chain(q)], device="cuda", dtype=torch.float32), joint_names=chain
            )

        result = self.motion_gen.plan_single_js(
            state(start),
            state(goal),
            MotionGenPlanConfig(
                enable_finetune_trajopt=False,
                max_attempts=10,
                check_start_validity=bool(req.get("check_start_validity", True)),
            ),
        )
        ok = bool(result.success.reshape(-1)[0].item())
        out = dict(ok=ok, index=0 if ok else None, path=[], success=[ok], status=str(result.status))
        if ok:
            out["path"] = (
                result.get_interpolated_plan().position.reshape(-1, len(chain)).cpu().numpy()[:, self.order].tolist()
            )
        out["seconds"] = time.time() - t0
        return out


def handle(planner: RoutePlanner, req: dict) -> dict:
    cmd = req.get("cmd")
    t0 = time.time()
    if cmd == "configure":
        built = planner.configure(req["names"], req["lower"], req["upper"], req["retract"])
        return dict(ok=True, built=built, seconds=time.time() - t0, protocol=PROTOCOL, rebuilds=planner.rebuilds,
                    planner=planner.fingerprint())  # fmt: skip
    if cmd == "plan":
        return planner.plan(req)
    raise ValueError(f"unknown cmd {cmd!r}")


def main(argv=None) -> int:
    import zmq

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("robot_json")
    ap.add_argument("--port", type=int, default=5795)
    ap.add_argument("--batch", type=int, default=2, help="fixed planning batch, at least 2; requests pad up to it")
    ap.add_argument("--cuda-graph", action="store_true", help="MotionGen use_cuda_graph (Settings)")
    ap.add_argument("--checker", choices=("mesh", "primitive"), default=Settings.checker)
    ap.add_argument("--ik-iters", type=int, default=Settings.ik_iters)
    ap.add_argument("--trajopt-iters", type=int, default=Settings.trajopt_iters)
    ap.add_argument("--no-self-collision-opt", action="store_true", help="check, but do not optimise, self-collision")
    args = ap.parse_args(argv)
    settings = Settings(
        args.cuda_graph, args.checker, args.ik_iters, args.trajopt_iters, not args.no_self_collision_opt
    )
    work = tempfile.mkdtemp(prefix="curobo_service_", dir=os.environ.get("TMPDIR"))
    planner = RoutePlanner(args.robot_json, args.batch, work, settings)
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"[curobo_service] listening on :{args.port} batch {args.batch} protocol {PROTOCOL} {settings}", flush=True)
    while True:
        req = json.loads(sock.recv())
        try:
            rep = handle(planner, req)
        except Exception as exc:  # noqa: BLE001 -- the reply carries the error; the client raises on it
            traceback.print_exc()
            rep = dict(ok=False, error=f"{type(exc).__name__}: {exc}", status="error")
        sock.send_string(json.dumps(rep))
        print(
            f"[curobo_service] {req.get('cmd')}: ok {rep.get('ok')} index {rep.get('index')} "
            f"{rep.get('seconds', 0):.2f} s canary {rep.get('canary')} rebuilt {rep.get('rebuilt')} "
            f"poisoning {rep.get('poisoning')} {rep.get('error', '')}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
