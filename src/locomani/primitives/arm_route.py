"""The episode side of the CuRobo arm-route service (`primitives/curobo_service.py`, its own process and venv).

`ArmRouteClient.configure` names the planned joints, their limits and the retract posture (the service builds and
warms up once per configuration, then no-ops); `plan` sends the start, the goals and the obstacles in the planning
frame and returns a `Route`. Any service error, timeout or protocol mismatch raises `RouteServiceError`: a run that
asked for a planned route must not quietly fall back to another way of moving the arm.

`scene_obstacles` is the collision world from the scene itself (item 1 of
docs/handoff/single-arm-curobo-route-plan-20261009.md, SIMPLE's `create_collision_world_cfg`): every collidable geom
outside the robot, boxes as boxes and each body's meshes merged into one mesh (the service caches ten of each), static
bodies where the model puts them and free bodies where the state says they are now. `retime_path` / `sample_path`
turn the planned joint path into a smoothstep timing at a joint speed and sample it on the physical clock.

Route cache (`cache_dir`, TODO 26.5.5): the service's plan for a request depends on the requests it answered before
(curobo_service module doc), so a re-run round (a fork's prefix, a gate sentinel, a candidate re-run) would not get
its first plan back. With a cache directory the client keys every plan request on its exact content plus the
configure request and the service's planner fingerprint, answers a repeat from disk and stores each new reply; a
re-run then gets the identical path, whichever service instance and history planned it first.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import zmq
from scipy.spatial.transform import Rotation

PROTOCOL = 2  # curobo_service.PROTOCOL; that module is not importable here without its venv's torch


class RouteServiceError(RuntimeError):
    """The route service failed, timed out or speaks another protocol."""


@dataclass(frozen=True)
class Box:
    name: str
    pose: tuple  # x, y, z, qw, qx, qy, qz in the planning frame
    dims: tuple  # full extents, m


@dataclass(frozen=True)
class MeshObstacle:
    name: str
    pose: tuple
    vertices: np.ndarray  # (V, 3), mesh frame
    faces: np.ndarray  # (F, 3) int


@dataclass(frozen=True)
class Route:
    ok: bool
    index: int | None  # which goal the path reaches
    path: np.ndarray  # (T, n) joint positions in the configured order, 0.02 s apart as planned
    success: tuple  # per goal
    seconds: float
    status: str = ""
    rebuilt: bool = False  # the service found its solvers broken (canary failed) and rebuilt before this answer
    poisoning: bool = False  # this request broke them (curobo_service module doc); the service rebuilt again after it
    world: dict = field(default_factory=dict, repr=False)  # what was sent, for the round record
    cached: bool = False  # answered from the client's route cache (module doc), not planned now


def world_message(obstacles) -> dict:
    cuboids, meshes = [], []
    for o in obstacles:
        if isinstance(o, Box):
            cuboids.append(dict(name=o.name, pose=[float(v) for v in o.pose], dims=[float(v) for v in o.dims]))
        elif isinstance(o, MeshObstacle):
            meshes.append(
                dict(
                    name=o.name,
                    pose=[float(v) for v in o.pose],
                    vertices=np.asarray(o.vertices, dtype=float).tolist(),
                    faces=np.asarray(o.faces, dtype=int).tolist(),
                )
            )
        else:
            raise TypeError(f"not an obstacle: {o!r}")
    return dict(cuboids=cuboids, meshes=meshes)


class ArmRouteClient:
    def __init__(
        self,
        port: int,
        *,
        host: str = "127.0.0.1",
        configure_timeout_s: float = 180.0,
        plan_timeout_s=60.0,
        cache_dir: str | os.PathLike | None = None,
    ):
        self.address = f"tcp://{host}:{port}"
        self.configure_timeout_s, self.plan_timeout_s = configure_timeout_s, plan_timeout_s
        self.ctx = zmq.Context.instance()
        self.sock = None
        self.names = None
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.cache_scope = None  # the configure request and the service's planner fingerprint, set by configure

    def _call(self, req: dict, timeout_s: float) -> dict:
        if self.sock is None:
            self.sock = self.ctx.socket(zmq.REQ)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(self.address)
        self.sock.send_string(json.dumps(req))
        if not self.sock.poll(int(timeout_s * 1000)):
            # a REQ socket that missed its reply cannot send again: drop it
            self.sock.close()
            self.sock = None
            raise RouteServiceError(f"no reply from the route service at {self.address} in {timeout_s:.0f} s")
        rep = json.loads(self.sock.recv())
        if rep.get("status") == "error":
            raise RouteServiceError(f"route service error: {rep.get('error')}")
        return rep

    def configure(self, names, lower, upper, retract) -> dict:
        rep = self._call(
            dict(
                cmd="configure",
                names=list(names),
                lower=[float(v) for v in lower],
                upper=[float(v) for v in upper],
                retract=[float(v) for v in retract],
            ),
            self.configure_timeout_s,
        )
        if rep.get("protocol") != PROTOCOL:
            raise RouteServiceError(f"route service protocol {rep.get('protocol')}, client {PROTOCOL}")
        if self.cache_dir is not None and not rep.get("planner"):
            raise RouteServiceError("the route service sends no planner fingerprint, so its plans cannot be cached")
        self.names = list(names)
        self.cache_scope = dict(
            names=self.names,
            lower=[float(v) for v in lower],
            upper=[float(v) for v in upper],
            retract=[float(v) for v in retract],
            planner=rep.get("planner"),
        )
        return rep

    def plan(self, start, goals, obstacles=()) -> Route:
        """To any of the end-effector `goals` (x, y, z, qw, qx, qy, qz in the planning frame)."""
        return self._plan(start, obstacles, goals=[[float(v) for v in g] for g in goals])

    def plan_to_joints(self, start, goal, obstacles=(), *, check_start_validity: bool = True) -> Route:
        """To the joint vector `goal`, in the configured joint order (inside the configured limits). With
        `check_start_validity` False the service plans from a start that already touches an obstacle."""
        goal = [float(v) for v in goal]
        if self.names is not None and len(goal) != len(self.names):
            raise ValueError(f"goal has {len(goal)} joints, configured {len(self.names)}")
        extra = {} if check_start_validity else {"check_start_validity": False}
        return self._plan(start, obstacles, goal_joints=goal, **extra)

    def _plan(self, start, obstacles, **goal) -> Route:
        if self.names is None:
            raise RouteServiceError("plan before configure")
        start = [float(v) for v in start]
        if len(start) != len(self.names):
            raise ValueError(f"start has {len(start)} joints, configured {len(self.names)}")
        world = world_message(obstacles)
        req = dict(cmd="plan", start=start, world=world, **goal)
        rep, cached = self._cached(req)
        path = np.asarray(rep.get("path") or np.zeros((0, len(self.names))), dtype=float)
        return Route(
            ok=bool(rep["ok"]),
            index=rep.get("index"),
            path=path.reshape(-1, len(self.names)),
            success=tuple(bool(s) for s in rep.get("success", [])),
            seconds=float(rep.get("seconds", 0.0)),
            status=str(rep.get("status", "")),
            rebuilt=bool(rep.get("rebuilt", False)),
            poisoning=bool(rep.get("poisoning", False)),
            world=world,
            cached=cached,
        )

    def cache_path(self, req: dict) -> Path:
        key = hashlib.sha256(json.dumps([self.cache_scope, req], sort_keys=True).encode()).hexdigest()
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cached(self, req: dict) -> tuple[dict, bool]:
        """(reply, from the cache): the stored reply for this exact request, else the service's, stored."""
        if self.cache_dir is None:
            return self._call(req, self.plan_timeout_s), False
        path = self.cache_path(req)
        if path.is_file():
            return json.loads(path.read_text()), True
        path.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent rounds with the same prefix (probe forks after the reach) send the same request at once: the
        # per-key lock makes the second wait for the first's reply instead of planning a possibly different one.
        with open(path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.is_file():
                return json.loads(path.read_text()), True
            rep = self._call(req, self.plan_timeout_s)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as f:
                json.dump(rep, f)
            os.replace(tmp, path)
        return rep, False

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def _wxyz_matrix(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=float)
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _pose7(position, rotation) -> tuple:
    q = Rotation.from_matrix(rotation).as_quat()
    return (*[float(v) for v in position], float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def scene_obstacles(model, free_poses: dict, frame_position, frame_rotation, *, robot_root: str = "pelvis") -> list:
    """The collidable geoms outside `robot_root`'s subtree, in the frame (`frame_position`, `frame_rotation`: the
    planning frame's origin and axes in the world). `free_poses` maps each free body's name to its current
    (position, wxyz quaternion); a free body missing from it is an error, not an omission. Planes are skipped (the
    planned arm is far above the floor); any other unsupported shape is an error."""
    root = model.body(robot_root).id
    robot = {root}
    for b in range(root + 1, model.nbody):
        if int(model.body_parentid[b]) in robot:
            robot.add(b)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)  # static bodies' world poses; free bodies are overridden below
    frame_r = np.asarray(frame_rotation, dtype=float)
    frame_p = np.asarray(frame_position, dtype=float)

    def body_root(b):
        while int(model.body_parentid[b]) != 0:
            b = int(model.body_parentid[b])
        return b

    boxes, meshes = [], {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b in robot or not (model.geom_contype[g] or model.geom_conaffinity[g]):
            continue
        kind = int(model.geom_type[g])
        if kind == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        top = body_root(b)
        name = model.body(b).name
        if top != 0 and model.body_jntnum[top] > 0:
            if model.body(top).name not in free_poses:
                raise ValueError(f"free body {model.body(top).name} has no current pose")
            if top != b:
                raise ValueError(f"free body {model.body(top).name} has collidable children; not supported")
            pos, quat = free_poses[model.body(top).name]
            body_p, body_r = np.asarray(pos, dtype=float), _wxyz_matrix(quat)
        else:
            body_p, body_r = data.xpos[b].copy(), data.xmat[b].reshape(3, 3).copy()
        geom_r = body_r @ _wxyz_matrix(model.geom_quat[g])
        geom_p = body_p + body_r @ model.geom_pos[g]
        local_p, local_r = frame_r.T @ (geom_p - frame_p), frame_r.T @ geom_r
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            boxes.append(
                Box(model.geom(g).name or f"{name}_box{g}", _pose7(local_p, local_r), tuple(2 * model.geom_size[g]))
            )
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            f = model.mesh_face[model.mesh_faceadr[mid] : model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
            # one mesh per body, in the planning frame (identity pose), so the service sees each object once
            verts = np.asarray(v, dtype=float) @ local_r.T + local_p
            meshes.setdefault(name, []).append((verts, np.asarray(f, dtype=int)))
        else:
            raise ValueError(f"geom {model.geom(g).name or g} of body {name}: type {kind} is not supported")
    out = list(boxes)
    for name, parts in meshes.items():
        offset, vs, fs = 0, [], []
        for v, f in parts:
            vs.append(v)
            fs.append(f + offset)
            offset += len(v)
        out.append(MeshObstacle(name, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0), np.vstack(vs), np.vstack(fs)))
    return out


def retime_path(path, speed: float, min_seconds: float = 0.4) -> tuple[np.ndarray, np.ndarray, float]:
    """(path, arc fraction of each waypoint, duration): one smoothstep over the whole path whose peak speed on the most
    moved joint is `speed` rad/s (the mean is two thirds of it); the GraspNet comparison's retiming, 2026-10-08."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or len(path) < 2:
        raise ValueError("a route needs at least two waypoints")
    arc = np.r_[0.0, np.cumsum(np.abs(np.diff(path, axis=0)).max(axis=1))]
    total = max(float(min_seconds), 1.5 * float(arc[-1]) / float(speed))
    return path, arc / max(float(arc[-1]), 1e-9), total


def sample_path(path, fraction, total: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Joint position and velocity `t` seconds into a retimed path; clamped to its ends."""
    u = min(1.0, max(0.0, t / total))
    s = 3 * u * u - 2 * u**3
    ds = (6 * u - 6 * u * u) / total
    k = int(np.clip(np.searchsorted(fraction, s, side="right") - 1, 0, len(path) - 2))
    span = max(float(fraction[k + 1] - fraction[k]), 1e-9)
    w = min(1.0, max(0.0, (s - fraction[k]) / span))
    q = path[k] + w * (path[k + 1] - path[k])
    v = (path[k + 1] - path[k]) / span * ds
    return q, v
