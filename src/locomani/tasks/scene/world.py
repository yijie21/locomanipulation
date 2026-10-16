"""One scene, one robot: the scene file, the ScaleBFM sim and motor layer on it, the occupancy map, and the privileged
state the goal predicates read (TODO 27.2.3).

The state is the predicates' format (scene_discovery/predicates.py): `objects[key]` = pos (body origin = the box
centre, world), quat (wxyz), `touching` (the hands with a finger on it) and `gripped` (the hands with both fingers on
it); static objects carry their scene-file pose; `pelvis` = pos, quat; `fallen`.
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

GRIP_LAW = dict(grip_kp=1.5, grip_kd=0.1, grip_clip=20.0)
# navigation inflation: the navigator's 0.26 m (arms tucked, SONIC) let a path run 0.33 m from the kitchen island's
# hull and the swinging arm caught it -- the robot was thrown and fell (episode_check seed 1, 2026-10-16)
BODY_RADIUS_M = 0.35


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


def physics_scene(scene_dir) -> Path:
    """`<scene_dir>_physics/`: the scene's MJCF without the visual-only geoms outside the robot (contype and
    conaffinity 0: no contact, no mass) and the meshes / materials / textures nothing references any more, plus
    `robot.xml` (the robot alone, for whole-body IK) and a copy of scene_meta.json. The ADT apartment's full MJCF is
    5.1 GB in memory (9.6 M mesh vertices, nearly all visual) and the IK loaded a second copy: 12 round processes ran
    the container out of memory (15 OOM kills, 2026-10-16). Physics is unchanged; renders replay the trace on the full
    scene. Rebuilt when the source is newer."""
    import xml.etree.ElementTree as ET

    src = Path(scene_dir)
    dst = src.parent / f"{src.name}_physics"
    xml = src / "scene_locomani.xml"
    if (dst / "robot.xml").exists() and (dst / "robot.xml").stat().st_mtime >= xml.stat().st_mtime:
        return dst
    dst.mkdir(exist_ok=True)

    def prune_assets(root) -> None:
        geoms = list(root.iter("geom"))
        meshes = {g.get("mesh") for g in geoms if g.get("mesh")}
        mats = {g.get("material") for g in geoms if g.get("material")}
        asset = root.find("asset")
        for el in list(asset):
            if el.tag == "mesh" and el.get("name") not in meshes:
                asset.remove(el)
            elif el.tag == "material" and el.get("name") not in mats:
                asset.remove(el)
        texs = {m.get("texture") for m in asset.iter("material") if m.get("texture")}
        for el in list(asset):
            if el.tag == "texture" and el.get("type") != "skybox" and el.get("name") not in texs:
                asset.remove(el)

    tree = ET.parse(xml)
    root = tree.getroot()
    world = root.find("worldbody")
    robot = world.find("body[@name='pelvis']")
    robot_geoms = {id(g) for g in robot.iter("geom")}
    for parent in list(root.iter()):
        for g in [c for c in parent if c.tag == "geom"]:
            if id(g) not in robot_geoms and g.get("contype") == "0" and g.get("conaffinity") == "0":
                parent.remove(g)
    prune_assets(root)
    tree.write(dst / "scene_locomani.xml")
    for el in [c for c in world if c.tag == "body" and c.get("name") != "pelvis"]:
        world.remove(el)
    prune_assets(root)
    tree.write(dst / "robot.xml")
    meta = json.loads((src / "scene_meta.json").read_text())
    meta["physics_of"] = str(src)
    (dst / "scene_meta.json").write_text(json.dumps(meta))
    return dst


class SceneWorld:
    """`scene_dir` holds scene_meta.json + scene_locomani.xml (scripts/adt_to_scene.py mujoco); the sim runs its
    physics copy (`physics_scene`)."""

    def __init__(
        self,
        scene_dir,
        policy: R.Policy,
        *,
        motor_config: MotorConfig | None = None,
        jacobian="analytic",
        grip_law: dict | None = None,
    ):
        self.dir = Path(scene_dir)
        self.physics = physics_scene(self.dir)
        self.meta = json.loads((self.dir / "scene_meta.json").read_text())
        # the finger law: the SONIC line's (kp 1.5, kd 0.1, ctrl clamp 20), which the 65-object comparison ran on both
        # controllers and the ported hand program (skills/grasp_program.py) closes with
        law = GRIP_LAW if grip_law is None else grip_law
        backend = ScaleBfmBackendConfig(scene_dir=str(self.physics), object_name="", **law)
        self.sim = R.Sim(policy.meta, backend=backend)
        self.motor = Motor(self.sim, policy, motor_config, model_xml=str(self.physics / "robot.xml"), jacobian=jacobian)
        m = self.sim.m
        self.free = {k: m.body(o["body"]).id for k, o in self.meta["objects"].items() if o.get("free")}
        self.occ, self.nav = build_navigation(self.meta, body_radius=BODY_RADIUS_M)
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
            gripped = sorted(self.motor.gripping(self.free[k])) if k in self.free else []
            objects[k] = dict(pos=pos, quat=quat, touching=touching, gripped=gripped)
        d = self.sim.d
        return dict(
            objects=objects, pelvis=dict(pos=d.qpos[:3].copy(), quat=d.qpos[3:7].copy()), fallen=self.sim.fallen()
        )
