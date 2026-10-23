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
opening and effort. Everything else (commands, joint and jaw readings, the robot's own pose) passes through."""

from __future__ import annotations

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
