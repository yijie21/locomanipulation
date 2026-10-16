"""One scene, one robot: the scene file, the ScaleBFM sim and motor layer on it, the occupancy map, and the privileged
state the goal predicates read (TODO 27.2.3).

The state is the predicates' format (scene_discovery/predicates.py): `objects[key]` = pos (body origin = the box
centre, world), quat (wxyz) and `touching` (the hands whose fingers touch it); static objects carry their scene-file
pose; `pelvis` = pos, quat; `fallen`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.motor import Motor, MotorConfig
from locomani.primitives.navigator import build as build_navigation
from locomani.scene_discovery.predicates import compile_goal, parse


@dataclass(frozen=True)
class Task:
    """A goal (predicate conjunction) with its phrasings; `objects` = the movable objects the goal names."""

    id: str
    goal: tuple[str, ...]
    prompts: tuple[str, ...] = ()
    room: str | None = None
    objects: tuple[str, ...] = field(default=())

    @classmethod
    def from_row(cls, row: dict, meta: dict) -> "Task":
        """A `tasks.jsonl` row (scene_discovery.proposals)."""
        goal = tuple(row["goal"])
        movable = []
        for g in goal:
            for a in parse(g).args:
                if isinstance(a, str) and (meta["objects"].get(a) or {}).get("free") and a not in movable:
                    movable.append(a)
        return cls(str(row["id"]), goal, tuple(row.get("prompts") or ()), row.get("room"), tuple(movable))


class SceneWorld:
    """`scene_dir` holds scene_meta.json + scene_locomani.xml (scripts/adt_to_scene.py mujoco)."""

    def __init__(self, scene_dir, policy: R.Policy, *, motor_config: MotorConfig | None = None, jacobian="analytic"):
        self.dir = Path(scene_dir)
        self.meta = json.loads((self.dir / "scene_meta.json").read_text())
        backend = ScaleBfmBackendConfig(scene_dir=str(self.dir), object_name="")
        self.sim = R.Sim(policy.meta, backend=backend)
        self.motor = Motor(
            self.sim, policy, motor_config, model_xml=str(self.dir / "scene_locomani.xml"), jacobian=jacobian
        )
        m = self.sim.m
        self.free = {k: m.body(o["body"]).id for k, o in self.meta["objects"].items() if o.get("free")}
        self.occ, self.nav = build_navigation(self.meta)
        self._goals: dict = {}

    def check(self, goal) -> callable:
        """compile_goal, cached per goal."""
        key = tuple(goal)
        if key not in self._goals:
            self._goals[key] = compile_goal(list(goal), self.meta)
        return self._goals[key]

    def object_pose(self, key: str) -> tuple[np.ndarray, np.ndarray]:
        """(pos, quat) of a free object now, or a static one's scene-file pose."""
        if key in self.free:
            b = self.free[key]
            return self.sim.d.xpos[b].copy(), self.sim.d.xquat[b].copy()
        o = self.meta["objects"][key]
        return np.asarray(o["init_pos"], float), np.asarray(o["init_quat"], float)

    def state(self, keys=None) -> dict:
        """The predicates' state for `keys` (default: every object of the scene)."""
        keys = self.meta["objects"] if keys is None else keys
        objects = {}
        for k in keys:
            pos, quat = self.object_pose(k)
            touching = sorted(self.motor.touching(self.free[k])) if k in self.free else []
            objects[k] = dict(pos=pos, quat=quat, touching=touching)
        d = self.sim.d
        return dict(
            objects=objects, pelvis=dict(pos=d.qpos[:3].copy(), quat=d.qpos[3:7].copy()), fallen=self.sim.fallen()
        )
