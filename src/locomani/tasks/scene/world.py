"""One scene, one robot: the scene file, the ScaleBFM sim and motor layer on it, the occupancy map, and the privileged
state the goal predicates read (TODO 27.2.3).

The state is the predicates' format (scene_discovery/predicates.py): `objects[key]` = pos (body origin = the box
centre, world), quat (wxyz), `touching` (the hands with a finger on it) and `gripped` (the hands with both fingers on
it); static objects carry their scene-file pose; `pelvis` = pos, quat; `fallen`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.scalebfm.motor import Motor, MotorConfig
from locomani.primitives.navigator import build as build_navigation
from locomani.scene_discovery.predicates import compile_goal, parse
from locomani.scene_discovery.scene_file import to_surface_frame
from locomani.tasks.scene import body

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


SEAT_BAND_M = 0.05  # SceneWorld.seat lifts an object whose underside is up to this far into a surface it is over
# ... and deeper than this: the pools' placements already start their hulls up to 0.9 cm into the slab (the hull is a
# little larger than the object; the settle hold pushes them out), and at 1.2 cm none of 1496 is lifted (2026-10-20)
SEAT_TOLERANCE_M = 0.012


class SceneWorld:
    """`scene_dir` holds scene_meta.json + scene_locomani.xml (scripts/adt_to_scene.py mujoco); the sim runs its
    physics copy (`physics_scene`). `controller` "amo" runs AMO (backends/amo, TODO 28) under the same motor
    interface; `policy` is then an AmoPolicy."""

    def __init__(
        self,
        scene_dir,
        policy: R.Policy,
        *,
        motor_config: MotorConfig | None = None,
        jacobian="analytic",
        grip_law: dict | None = None,
        controller: str = "scalebfm",
        sleep: bool = True,
    ):
        self.dir = Path(scene_dir)
        self.physics = physics_scene(self.dir)
        self.meta = json.loads((self.dir / "scene_meta.json").read_text())
        # the finger law: the SONIC line's (kp 1.5, kd 0.1, ctrl clamp 20), which the 65-object comparison ran on both
        # controllers and the ported hand program (skills/grasp_program.py) closes with
        law = GRIP_LAW if grip_law is None else grip_law
        # sleep: resting objects leave the physics until touched (MuJoCo island sleeping): the apartment's step 1.82 ->
        # 0.62 ms; outcomes moved with it, so every record was re-run (TODO 32.4)
        backend = ScaleBfmBackendConfig(scene_dir=str(self.physics), object_name="", sleep=sleep, **law)
        self.controller = controller
        if controller.startswith("gr00t_wbc"):  # TODO 34: `policy` is a backends.gr00t_wbc.runtime.Gr00tPolicy
            from locomani.backends.gr00t_wbc.motor import VARIANTS, Gr00tMotor
            from locomani.backends.gr00t_wbc.runtime import Gr00tSim

            self.sim = Gr00tSim(backend)
            self.motor = Gr00tMotor(self.sim, policy, motor_config, model_xml=str(self.physics / "robot.xml"),
                                    jacobian=jacobian, gr00t=VARIANTS[controller])  # fmt: skip
        elif controller == "amo":  # TODO 28: `policy` is a backends.amo.runtime.AmoPolicy
            from locomani.backends.amo.motor import AmoMotor
            from locomani.backends.amo.runtime import AmoSim

            self.sim = AmoSim(backend)
            self.motor = AmoMotor(
                self.sim, policy, motor_config, model_xml=str(self.physics / "robot.xml"), jacobian=jacobian
            )
        else:
            self.sim = R.Sim(policy.meta, backend=backend)
            self.motor = Motor(
                self.sim, policy, motor_config, model_xml=str(self.physics / "robot.xml"), jacobian=jacobian
            )
        # TODO 33.1.2: every one-hand reach's record carries the body model's prediction for its command
        self.motor.reach_note = body.reach_note()
        if getattr(self.motor.cfg, "reach_compensation", False):  # TODO 33.1.4 (2): default off
            self.motor.reach_offset = body.reach_offset()
        if getattr(self.motor.cfg, "reach_route", False):  # TODO 40: default off
            self.route_reaches(scope=self.motor.cfg.reach_route_scope, pose_goal=self.motor.cfg.reach_route_pose_goal)
        m = self.sim.m
        self.free = {k: m.body(o["body"]).id for k, o in self.meta["objects"].items() if o.get("free")}
        self.occ, self.nav = build_navigation(self.meta, body_radius=BODY_RADIUS_M)
        self._goals: dict = {}

    def compensate_reaches(self) -> None:
        """TODO 33.1.4 (2): turn the motor's reach compensation on (MotorConfig.reach_compensation, default off) and
        install the body model's free-space miss as its `reach_offset` hook, keeping the rest of the motor's own config
        (scripts/rsi/scene_round.make_world under RSI_REACH_COMPENSATION=1)."""
        self.motor.cfg = dataclasses.replace(self.motor.cfg, reach_compensation=True)
        self.motor.reach_offset = body.reach_offset()

    def route_reaches(self, cache_dir=None, dump_dir=None, scope: str = "via", pose_goal: bool = False) -> None:
        """TODO 40: turn the motor's pregrasp route on (MotorConfig.reach_route, default off; `scope` and `pose_goal`:
        reach_route_scope and reach_route_pose_goal, TODO 40.4) and install the CuRobo route around this scene
        (tasks/scene/route.py) as its `reach_route` hook, its replies cached in `cache_dir` and its failed requests
        written to `dump_dir` when given (scripts/rsi/scene_round.make_world under RSI_REACH_ROUTE=1; needs the route
        services on TF_CUROBO_PORTS)."""
        from locomani.tasks.scene.route import SceneRoute

        self.motor.cfg = dataclasses.replace(
            self.motor.cfg, reach_route=True, reach_route_scope=scope, reach_route_pose_goal=pose_goal
        )
        self.motor.reach_route = SceneRoute.connect(
            self.sim, cache_dir, dump_dir, pose_goal=pose_goal, tool_offset=R.OFF if pose_goal else None
        )

    def check(self, goal) -> callable:
        """compile_goal, cached per goal."""
        key = tuple(goal)
        if key not in self._goals:
            self._goals[key] = compile_goal(list(goal), self.meta)
        return self._goals[key]

    def underside(self, key: str) -> float:
        """World z of the lowest vertex of a free object's colliders (mesh hulls; other shapes by their bounding box)."""
        m, d = self.sim.m, self.sim.d
        mujoco.mj_kinematics(m, d)
        low = np.inf
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != self.free[key] or not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            R_g = d.geom_xmat[g].reshape(3, 3)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                a, k = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                low = min(low, float((d.geom_xpos[g] + m.mesh_vert[a : a + k] @ R_g.T)[:, 2].min()))
            else:
                low = min(
                    low, float(d.geom_xpos[g][2] + R_g[2] @ m.geom_aabb[g, :3] - np.abs(R_g[2]) @ m.geom_aabb[g, 3:])
                )
        return low

    def seat(self, key: str) -> float:
        """Lift a free object a start placed into a surface's slab onto the slab's top; returns the lift (m, 0 when it
        rests on or above every surface it is over). The pools' placements were drawn on the surface heights of their
        day: 17 of them were 0.5-3.7 cm low until 2026-10-20 (TODO 27.14), and a placement at the old height starts
        inside the raised slab, from which the solver may push it either way."""
        low = self.underside(key)
        b = self.free[key]
        xy = self.sim.d.xpos[b][:2]
        tops = [
            float(s["top"])
            for s in self.meta["surfaces"].values()
            if np.all(np.abs(to_surface_frame(xy, s)) <= np.asarray(s["half_extent"][:2], float))
            and float(s["top"]) - SEAT_BAND_M < low < float(s["top"]) - SEAT_TOLERANCE_M
        ]
        if not tops:
            return 0.0
        lift = max(tops) - low + 0.0005
        m = self.sim.m
        adr = int(m.jnt_qposadr[m.body_jntadr[b]])
        self.sim.d.qpos[adr + 2] += lift
        mujoco.mj_forward(m, self.sim.d)
        return float(lift)

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
