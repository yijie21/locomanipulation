"""TODO 40 (user 2026-10-27): the CuRobo pregrasp route for the scene line -- `Motor.reach_route`, installed by
`SceneWorld.route_reaches` when MotorConfig.reach_route is on (RSI_REACH_ROUTE=1 for scripts/rsi's rounds).

The route service (`primitives/curobo_service.py`, SIMPLE's venv) plans the waist and both arms (crate reach_route's
17 joints and the mars model of the 10-08 study) with the pelvis held at the reach's goal pose, around the scene's
collidable geoms as they are now: every box within RADIUS_M of the pelvis (the apartment's furniture is 12,025 boxes;
up to ~380 lie that near an object) grown by 1.5 cm on each side, as the warehouse route grows its box (ScaleBFM
tracks the hands with a few centimetres of error, tasks/crate/reach_route.py), and every mesh body that near (the free
objects) pushed out as much from each geom's centre; a plan that fails so is tried once more with CuRobo's own margin
only (CLEARANCE_M). The service's robot model should leave out the idle arm (the skills reach with the right hand;
`outputs/scene_rsi_20261016/curobo_route_20261027/mars_curobo_right.json`): resting in a shelf, it sat inside the
margin at both ends of the route. Keep the self-collision cost in the optimisation (no --no-self-collision-opt, the
warehouse default): from the hanging hand to a pregrasp the right shoulder pitches 1.2-2.2 rad past the torso, and
without it 24 of 26 pregrasp plans failed even in an empty world; with it 5 of 6 of those planned in ~1.6 s. The joints the service does not plan (legs) move linearly
along the route's arc. Start the services with a collision cache that holds that many boxes:

    ROBOT=<torso + right arm model> N=2 GPUS=0,1 FLAGS="--obb-cache 512 --mesh-cache 32" \\
        bash scripts/experiments/scalebfm_warehouse_rsi_20261014/route_service.sh <log>
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_route import ArmRouteClient, Box, MeshObstacle
from locomani.tasks.crate.reach_route import PLANNED, PLANNED_INDEX

# tried in order: a goal inside the grown boxes (a pregrasp in a 5-6 cm shelf) gets CuRobo's own margin
CLEARANCE_M = (0.015, 0.0)
RADIUS_M = 1.1  # geom surface (bounding sphere) to the pelvis
TARGET_M = 0.15  # an approach's target: the free object nearest its jaw goal, this near (TODO 40.4)
# s a plan may wait for its reply: 12 rounds queue on 4 services and a failing plan takes up to ~25 s per try, so 180 s
# timed out 48 plans in 33 of 240 rounds (2026-10-27), each falling back to the straight legs
ROUTE_TIMEOUT_S = 1800.0


def _pose7(p, rot) -> tuple:
    x, y, z, w = Rotation.from_matrix(rot).as_quat()
    return (float(p[0]), float(p[1]), float(p[2]), float(w), float(x), float(y), float(z))


class SceneRoute:
    """Callable(start joints, goal joints, pelvis position, pelvis quaternion wxyz) -> (T, 29) path or None; `last`
    keeps the latest plan's facts for the round record. `pose_goal` (TODO 40.4): a joint goal that cannot be planned
    is tried as the wrist pose (the jaw goal less `tool_offset`, the jaw centre in the wrist frame), CuRobo choosing
    the arm."""

    def __init__(
        self, sim, client: ArmRouteClient, *, clearance=CLEARANCE_M, radius: float = RADIUS_M, dump_dir=None,
        pose_goal: bool = False, tool_offset=None,
    ):  # fmt: skip
        m = sim.m
        self.m, self.d, self.client = m, sim.d, client
        self.dump_dir = dump_dir  # a failed request is written here, for offline bisection of its obstacles
        if pose_goal and tool_offset is None:
            raise ValueError("a pose goal needs the jaw centre's offset in the wrist frame")
        self.pose_goal, self.tool_offset = bool(pose_goal), None if tool_offset is None else np.asarray(tool_offset)
        self.clearance, self.radius = tuple(float(c) for c in clearance), float(radius)
        rng = np.array([m.jnt_range[m.joint(n).id] for n in PLANNED], dtype=float)
        self.lower, self.upper = rng[:, 0], rng[:, 1]
        root = m.body("pelvis").id
        robot = {root}
        for b in range(root + 1, m.nbody):
            if int(m.body_parentid[b]) in robot:
                robot.add(b)
        self.geoms = np.array([
            g for g in range(m.ngeom)
            if int(m.geom_bodyid[g]) not in robot and (m.geom_contype[g] or m.geom_conaffinity[g])
            and int(m.geom_type[g]) != mujoco.mjtGeom.mjGEOM_PLANE
        ])  # fmt: skip
        self.mesh = {}
        for g in self.geoms[m.geom_type[self.geoms] == mujoco.mjtGeom.mjGEOM_MESH]:
            mid = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
            f = m.mesh_face[m.mesh_faceadr[mid] : m.mesh_faceadr[mid] + m.mesh_facenum[mid]].astype(int)
            self.mesh[int(g)] = (v, f, np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6))
        free = m.jnt_bodyid[m.jnt_type == mujoco.mjtJoint.mjJNT_FREE]
        self.free = np.array([int(b) for b in free if int(b) not in robot], dtype=int)
        self.configured = False
        self.last = {}

    @classmethod
    def connect(cls, sim, cache_dir=None, dump_dir=None, **kw) -> SceneRoute:
        """A client on one of the route services (sim.ports.curobo_routes, by process id), its replies cached in
        `cache_dir` when given."""
        from locomani.sim.ports import curobo_routes

        ports = curobo_routes()
        client = ArmRouteClient(ports[os.getpid() % len(ports)], plan_timeout_s=ROUTE_TIMEOUT_S, cache_dir=cache_dir)
        return cls(sim, client, dump_dir=dump_dir, **kw)

    def target(self, jaw) -> int | None:
        """The free body nearest the jaw goal, if within TARGET_M."""
        if not len(self.free):
            return None
        dist = np.linalg.norm(self.d.xpos[self.free] - np.asarray(jaw, dtype=float), axis=1)
        return int(self.free[int(np.argmin(dist))]) if float(dist.min()) <= TARGET_M else None

    def obstacles(self, pos, quat, clearance: float, exclude: int | None = None) -> list:
        m, d = self.m, self.d
        frame = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        pos = np.asarray(pos, dtype=float)
        near = self.geoms[
            np.linalg.norm(d.geom_xpos[self.geoms] - pos, axis=1) - m.geom_rbound[self.geoms] < self.radius
        ]
        boxes, meshes = [], {}
        for g in near:
            g = int(g)
            if exclude is not None and int(m.geom_bodyid[g]) == exclude:
                continue
            gp, gr = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
            kind, size = int(m.geom_type[g]), m.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                v, f, n = self.mesh[g]
                v = v * (1.0 + clearance / n)  # grown about the geom centre
                meshes.setdefault(m.body(int(m.geom_bodyid[g])).name, []).append(((v @ gr.T + gp - pos) @ frame, f))
                continue
            dims = {
                mujoco.mjtGeom.mjGEOM_BOX: 2 * size,
                mujoco.mjtGeom.mjGEOM_SPHERE: np.full(3, 2 * size[0]),
                mujoco.mjtGeom.mjGEOM_CAPSULE: np.array([2 * size[0], 2 * size[0], 2 * (size[1] + size[0])]),
                mujoco.mjtGeom.mjGEOM_CYLINDER: np.array([2 * size[0], 2 * size[0], 2 * size[1]]),
                mujoco.mjtGeom.mjGEOM_ELLIPSOID: 2 * size,
            }.get(kind)
            if dims is None:
                raise ValueError(f"geom {m.geom(g).name or g}: type {kind} is not supported")
            name = m.geom(g).name or f"geom{g}"
            boxes.append(Box(name, _pose7(frame.T @ (gp - pos), frame.T @ gr), tuple(dims + 2 * clearance)))
        out = list(boxes)
        for name, parts in meshes.items():
            offset, vs, fs = 0, [], []
            for v, f in parts:
                vs.append(v)
                fs.append(f + offset)
                offset += len(v)
            out.append(MeshObstacle(name, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), np.vstack(vs), np.vstack(fs)))
        return out

    def _plan(self, call):
        """(route, seconds, start in collision): `call`, once more without the start check when the start touches."""
        route = call()
        seconds, touching = route.seconds, "INVALID_START_STATE_WORLD_COLLISION" in route.status
        if touching:
            route = call(check_start_validity=False)
            seconds += route.seconds
        return route, seconds, touching

    def _wrist(self, jaw, pelvis_pos, pelvis_quat) -> tuple:
        """The wrist pose for the jaw goal (position, rotation, world), in the planning frame."""
        frame = Rotation.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
        rot = np.asarray(jaw[1], dtype=float)
        wrist = np.asarray(jaw[0], dtype=float) - rot @ self.tool_offset
        return _pose7(frame.T @ (wrist - np.asarray(pelvis_pos, dtype=float)), frame.T @ rot)

    def __call__(self, start, goal, pelvis_pos, pelvis_quat, *, jaw=None, approach=False, pose_ok=False):
        """`jaw` (position, rotation, world): where the reach puts the jaw centre at this goal (TODO 40.4). With
        `approach` the free object nearest it (TARGET_M) is no obstacle; with `pose_ok` and pose_goal a joint goal that
        cannot be planned is tried as that wrist pose."""
        if not self.configured:
            self.client.configure(PLANNED, self.lower, self.upper, np.clip(0.0, self.lower, self.upper))
            self.configured = True
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        exclude = self.target(jaw[0]) if approach and jaw is not None else None
        seconds, kind = 0.0, "joints"
        for clearance in self.clearance:
            obstacles = self.obstacles(pelvis_pos, pelvis_quat, clearance, exclude)
            request = (
                np.clip(start[PLANNED_INDEX], self.lower, self.upper),
                np.clip(goal[PLANNED_INDEX], self.lower, self.upper),
                obstacles,
            )
            route, dt, start_in_collision = self._plan(lambda **kw: self.client.plan_to_joints(*request, **kw))
            seconds, kind = seconds + dt, "joints"
            if not route.ok and self.pose_goal and pose_ok and jaw is not None:
                wrist = self._wrist(jaw, pelvis_pos, pelvis_quat)
                route, dt, start_in_collision = self._plan(
                    lambda **kw: self.client.plan(request[0], [wrist], obstacles, **kw)
                )
                seconds, kind = seconds + dt, "pose"
            if route.ok:
                break
        self.last = dict(
            goal=kind,
            excluded=None if exclude is None else self.m.body(exclude).name,
            clearance_cm=round(100 * clearance, 1),
            ok=bool(route.ok),
            plan_s=round(float(seconds), 3),
            waypoints=len(route.path),
            status=route.status,
            start_in_collision=start_in_collision,
            boxes=sum(isinstance(o, Box) for o in obstacles),
            meshes=sum(isinstance(o, MeshObstacle) for o in obstacles),
            cached=bool(route.cached),
        )
        if not route.ok:
            dump = self.dump_dir
            if dump:
                import json
                import time

                from locomani.primitives.arm_route import world_message

                os.makedirs(dump, exist_ok=True)
                with open(os.path.join(dump, f"{os.getpid()}_{time.time():.3f}.json"), "w") as f:
                    json.dump(dict(start=request[0].tolist(), goal=request[1].tolist(), status=route.status,
                                   pelvis=[list(map(float, pelvis_pos)), list(map(float, pelvis_quat))],
                                   world=world_message(obstacles)), f)  # fmt: skip
            return None
        planned = np.asarray(route.path, dtype=float)
        arc = np.r_[0.0, np.cumsum(np.abs(np.diff(planned, axis=0)).max(axis=1))]
        frac = arc / max(float(arc[-1]), 1e-9)
        path = start + frac[:, None] * (goal - start)
        path[:, PLANNED_INDEX] = planned
        return path
