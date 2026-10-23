"""What a skill may read (docs/handoff/vision-rsi-design-20261023.md section 2): the world and the robot through two
views, never the simulator. A skills package that sets `WORLD_VIEW = True` gets them as `ctx.world` and `ctx.motor`
(scripts/rsi/scene_round.py -> episode.run_episode); a package without it still gets the SceneWorld and the Motor.

`TruthView` answers every question from the simulator's true state, so a package ported to it behaves exactly as it
did on the SceneWorld (step 0 of the design: the sealed pool reproduces round by round). A perception-backed view
answers the same questions from sensors and models instead. What the world view offers:

- `meta`: the scene description (objects with their boxes and supports, surfaces, rooms), the scene file's schema;
  `occ`, `nav`: the occupancy map and navigator built from it;
- `free`: the movable objects' keys, in scene-file order;
- `object_pose(key)`: (pos, quat) now; `object_geoms(key)`: a free object's collision shapes in its own frame;
- `static_colliders()`: world boxes of the fixed scene the arms can hit;
- `held(key)`: the sides holding the object with both fingers; `finger_contacts(side, key)`: which finger touches it;
- `state(keys)`, `check(goal)`: the goal predicates' state and their judge.

`RobotView` is the motor without the simulator handle and without its contact readers (`between_fingers`, `gripping`,
`touching`): contacts reach a skill only through the world view, which a perception view answers from the gripper's
opening and effort. Everything else (commands, joint and jaw readings, the robot's own pose) passes through.

`PerturbedView` is level L1 of the design (section 6): the TruthView with errors added to the free objects' poses
(`Perturbation`: a fixed offset and heading error per object and round, fresh noise at every read, a delay), the
truth underneath and every other answer unchanged."""

from __future__ import annotations

import bisect
import zlib
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

GEOM_TYPES = {int(v): k[len("mjGEOM_") :].lower() for k, v in vars(mujoco.mjtGeom).items() if k.startswith("mjGEOM_")}


class TruthView:
    """The world view over a SceneWorld's true state."""

    __slots__ = ("_world", "_robot")

    def __init__(self, world):
        self._world = world
        self._robot = RobotView(world.motor)

    @property
    def motor(self) -> RobotView:
        return self._robot

    @property
    def meta(self) -> dict:
        return self._world.meta

    @property
    def occ(self):
        return self._world.occ

    @property
    def nav(self):
        return self._world.nav

    @property
    def free(self) -> tuple[str, ...]:
        return tuple(self._world.free)

    def object_pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        return self._world.object_pose(key)

    def object_geoms(self, key: str) -> list[dict]:
        """The collision geoms (contype or conaffinity set) of free object `key`, in model order: id, type ("box",
        "mesh", ...), size, pos and quat in the body frame, and for a mesh its vertices and faces (the model's arrays,
        unconverted). Read-only."""
        m, b = self._world.sim.m, self._world.free[key]
        out = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            row = dict(id=g, type=GEOM_TYPES[int(m.geom_type[g])], size=m.geom_size[g], pos=m.geom_pos[g],
                       quat=m.geom_quat[g])  # fmt: skip
            if row["type"] == "mesh":
                mid = int(m.geom_dataid[g])
                a, k = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                row.update(vert=m.mesh_vert[a : a + k], face=m.mesh_face[fa : fa + fn])
            out.append(row)
        return out

    def static_colliders(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """(centre, half extent) world AABBs of the fixed scene's box colliders that the robot's arms collide with
        (children of the world body, not mocap), in model order. From the model's body poses, not geom_xpos (a
        sleeping island leaves those zero)."""
        m = self._world.sim.m
        wrist = m.body("right_wrist_yaw_link").id
        arm = {
            (int(m.geom_contype[g]), int(m.geom_conaffinity[g]))
            for g in range(m.ngeom)
            if int(m.geom_bodyid[g]) == wrist
        }
        out = []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX) or int(m.body_jntnum[b]):
                continue
            if int(m.body_parentid[b]) != 0 or int(m.body_mocapid[b]) >= 0:
                continue
            ct, ca = int(m.geom_contype[g]), int(m.geom_conaffinity[g])
            if not any((ct & a) or (ca & c) for c, a in arm):
                continue
            Rb = _quat_mat(m.body_quat[b])
            out.append((m.body_pos[b] + Rb @ m.geom_pos[g], np.abs(Rb @ _quat_mat(m.geom_quat[g])) @ m.geom_size[g]))
        return out

    def held(self, key: str) -> set[str]:
        return self._world.motor.gripping(self._world.free[key])

    def finger_contacts(self, side: str, key: str) -> tuple[bool, bool]:
        """Whether `side`'s finger 1 (on the wrist's -y side) and finger 2 (+y) touch free object `key` now."""
        motor, body = self._world.motor, self._world.free[key]
        m, d = motor.sim.m, motor.sim.d
        f1, f2 = motor.fingers[side]
        hit = [False, False]
        for i in range(d.ncon):
            b1, b2 = int(m.geom_bodyid[d.contact[i].geom1]), int(m.geom_bodyid[d.contact[i].geom2])
            if body in (b1, b2):
                other = b2 if b1 == body else b1
                hit[0] |= other == f1
                hit[1] |= other == f2
        return hit[0], hit[1]

    def state(self, keys=None) -> dict:
        return self._world.state(keys)

    def check(self, goal):
        return self._world.check(goal)


class RobotView:
    """The motor for a skill: every attribute but the simulator handle, the policy and the contact readers."""

    __slots__ = ("_motor",)
    DENIED = frozenset({"sim", "pol", "between_fingers", "gripping", "touching", "place", "reset"})

    def __init__(self, motor):
        object.__setattr__(self, "_motor", motor)

    def __getattr__(self, name):
        if name in RobotView.DENIED:
            raise AttributeError(f"RobotView: {name!r} is not available to a skill (locomani.tasks.scene.view)")
        return getattr(self._motor, name)

    def __setattr__(self, name, value):
        if name in RobotView.DENIED:
            raise AttributeError(f"RobotView: {name!r} is not available to a skill (locomani.tasks.scene.view)")
        setattr(self._motor, name, value)


def _quat_mat(q) -> np.ndarray:
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, float))
    return out.reshape(3, 3)


@dataclass(frozen=True)
class Perturbation:
    """What `PerturbedView` adds to a free object's pose, all zero by default (= TruthView):
    - obj_bias_m: the position off by this distance, in one random 3-D direction per object and round;
    - obj_jitter_m: plus fresh Gaussian noise of this standard deviation per axis at every read;
    - obj_yaw_deg: the heading off by this angle about the vertical, random sign per object and round;
    - latency_s: a tracked object's (the task's objects: the motor trace) true pose this long ago;
    - seed: mixed with the round's start seed, so a condition's errors repeat round by round."""

    obj_bias_m: float = 0.0
    obj_jitter_m: float = 0.0
    obj_yaw_deg: float = 0.0
    latency_s: float = 0.0
    seed: int = 0

    @classmethod
    def of(cls, spec: dict | None) -> Perturbation | None:
        """The perturbation a round record's `perception` spec names; None for none (unknown keys raise)."""
        return cls(**spec) if spec else None

    def spec(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v}


class PerturbedView(TruthView):
    """TruthView with `Perturbation`'s errors on the free objects' poses. Static objects, shapes in the object frame,
    the scene description, maps, colliders and contacts are the truth."""

    __slots__ = ("_p", "_round", "_err", "_rng", "_trace", "_trace_t")

    def __init__(self, world, perturbation: Perturbation, round_seed: int):
        super().__init__(world)
        self._p, self._round, self._err = perturbation, int(round_seed), {}
        self._rng = np.random.default_rng([self._round, int(perturbation.seed), 1])
        self._trace, self._trace_t = None, []  # the motor trace list and its rows' times, extended as it grows

    def errors(self, key: str) -> tuple[np.ndarray, float]:
        """(position offset, heading error in rad) of free object `key` this round, drawn on first use."""
        if key not in self._err:
            rng = np.random.default_rng([self._round, int(self._p.seed), zlib.crc32(key.encode())])
            d = rng.normal(size=3)
            off = float(self._p.obj_bias_m) * d / max(float(np.linalg.norm(d)), 1e-12)
            yaw = float(np.radians(self._p.obj_yaw_deg)) * (1.0 if rng.random() < 0.5 else -1.0)
            self._err[key] = (off, yaw)
        return self._err[key]

    def object_pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        if key not in self._world.free:
            return self._world.object_pose(key)
        pos, quat = self._delayed(key) if self._p.latency_s > 0 else self._world.object_pose(key)
        off, yaw = self.errors(key)
        pos = np.asarray(pos, float) + off
        if self._p.obj_jitter_m > 0:
            pos = pos + self._rng.normal(scale=float(self._p.obj_jitter_m), size=3)
        if yaw:
            quat = _quat_mul(np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]), np.asarray(quat, float))
        return pos, quat

    def _delayed(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """The true pose `latency_s` ago from the motor trace (its rows carry the tracked objects), the oldest row when
        the round is younger than that, and the pose now for an object the trace does not carry."""
        motor = self._world.motor
        body = self._world.meta["objects"][key].get("body")
        trace = motor.trace
        if not trace or body not in trace[-1]["objects"]:
            return self._world.object_pose(key)
        if trace is not self._trace:  # Motor.reset starts a new list
            self._trace, self._trace_t = trace, []
        ts = self._trace_t
        ts.extend(float(r["t"]) for r in trace[len(ts) :])
        i = max(bisect.bisect_right(ts, float(motor.t) - float(self._p.latency_s)) - 1, 0)
        row = trace[i]["objects"][body]
        return np.array(row[:3], float), np.array(row[3:7], float)


def _quat_mul(a, b) -> np.ndarray:
    """Hamilton product a * b of wxyz quaternions."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])  # fmt: skip
