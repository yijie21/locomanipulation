"""The scene as the discovery layer sees it (design doc §8): objects, surfaces, affordances, task instances.

Built from `scene_meta.json` alone; positions at run time come from the privileged state dict
(`state["objects"][key]["pos"/"quat"]`). `instances()` is the agent's menu -- the Q6 template grammar
`put <obj> in <container> | on <surface>` -- and `criterion(instance)` compiles the template into the
success predicate the executor evaluates, so the agent never judges success itself. Geometry is looked
up, never guessed: a single-hand pick needs a box that fits the open Dex1 jaw, a bimanual pick needs the
scene's `bimanual` flag, a place needs the scene's `container` flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from locomani.tasks.bottle.place_phase import in_container

DEX1_OPENING_M = 0.095  # open jaw width (visual_config.close_trigger note)
PUT_ON_HEIGHT_M = 0.30  # an object counts as "on" a surface within this height above its top
PUT_ON_SLACK_M = (
    0.02  # underside may sit this far below the top (contact penetration) and the centre this far past the edge
)


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    phrase: str
    half_extent: tuple[float, float, float] | None  # box half-extent, body frame
    centre: tuple[float, float, float] | None  # box centre, body frame
    wall: float
    bimanual: bool
    container: bool
    on_surface: str | None
    height: float | None
    init_pos: tuple[float, float, float] | None

    @property
    def single_hand_graspable(self) -> bool:
        if self.bimanual or self.container or self.half_extent is None:
            return False
        return 2 * min(self.half_extent[0], self.half_extent[1]) <= DEX1_OPENING_M


@dataclass(frozen=True)
class SurfaceInfo:
    key: str
    top: float
    xy: tuple[float, float]
    half_extent: tuple[float, float]
    yaw: float = 0.0  # scene_discovery surfaces are turned; the hand-built ones are axis-aligned
    phrase: str = ""  # scene_discovery surfaces carry Claude's phrase; "" = the key, underscores as spaces


@dataclass(frozen=True)
class Affordance:
    kind: str  # pick_single | pick_bimanual | place_into | put_on
    obj: str
    target: str | None
    hands: str  # single | bimanual


@dataclass(frozen=True)
class TaskInstance:
    template: str  # "put <obj> in <container>" | "put <obj> on <surface>"
    obj: str
    target: str
    kind: str  # in | on
    profile: str
    text: str


def _t(value, n):
    return None if value is None else tuple(float(v) for v in list(value)[:n])


class SceneModel:
    def __init__(self, meta: dict, reach=None):
        self.meta, self.reach = meta, reach
        phrases = meta.get("object_phrase") or {}
        phrases = phrases if isinstance(phrases, dict) else {}
        self.surfaces = {
            k: SurfaceInfo(
                k,
                float(s["top"]),
                _t(s["xy"], 2),
                _t(s.get("flat_half_extent") or s["half_extent"], 2),
                float(s.get("yaw", 0.0)),
                str(s.get("phrase") or k.replace("_", " ")),
            )
            for k, s in (meta.get("surfaces") or {}).items()
        }
        self.objects = {}
        for k, o in (meta.get("objects") or {}).items():
            box = o.get("box") or {}
            self.objects[k] = ObjectInfo(
                key=k,
                # a scene_discovery file carries Claude's phrase per object; the hand-built scenes a top-level map
                phrase=phrases.get(k) or (o.get("semantic") or {}).get("phrase") or k.replace("_", " "),
                half_extent=_t(box.get("half_extent"), 3),
                centre=_t(box.get("centre"), 3) or ((0.0, 0.0, 0.0) if box else None),
                wall=float(box.get("wall", 0.01)),
                bimanual=bool(o.get("bimanual", False)),
                container=bool(o.get("container", False)),
                on_surface=o.get("surface"),
                height=None if o.get("height") is None else float(o["height"]),
                init_pos=_t(o.get("init_pos"), 3),
            )
        self.robot_start = dict(meta.get("robot_start") or {})

    # -- geometry ---------------------------------------------------------------------------------
    def heading(self) -> float:
        return float(np.radians(self.robot_start.get("yaw_deg", 0.0)))

    def front_line(self, target: str) -> dict:
        """The brake line an approach to `target` must not cross: its surface's near edge, or its own box front.

        Returned as {"kind", "key", "xy", "half_extent", "top"} in world xy; the crate line's `near_edge` /
        `box_front_line` compute the scalar along the walk-in axis from the same numbers.
        """
        if target in self.surfaces:
            s = self.surfaces[target]
            return {
                "kind": "surface",
                "key": target,
                "xy": list(s.xy),
                "half_extent": list(s.half_extent),
                "top": s.top,
                "yaw": s.yaw,
            }
        obj = self.objects[target]
        if obj.half_extent is not None and obj.init_pos is not None:
            xy = [obj.init_pos[0] + obj.centre[0], obj.init_pos[1] + obj.centre[1]]
            top = obj.init_pos[2] + obj.centre[2] + obj.half_extent[2]
            return {"kind": "box", "key": target, "xy": xy, "half_extent": list(obj.half_extent[:2]), "top": top}
        if obj.on_surface in self.surfaces:
            return self.front_line(obj.on_surface)
        raise ValueError(f"{target!r} has neither a box footprint nor a surface")

    # -- discovery --------------------------------------------------------------------------------
    def profile_for(self, obj: str) -> str:
        o = self.objects[obj]
        if not o.bimanual:
            return "bottle_simple"
        top = self.surfaces[o.on_surface].top if o.on_surface in self.surfaces else 0.0
        return "crate_low_table" if top >= 0.25 else "warehouse_box"

    def affordances(self) -> list[Affordance]:
        out = []
        pickable = []
        for k, o in self.objects.items():
            if o.single_hand_graspable:
                out.append(Affordance("pick_single", k, None, "single"))
                pickable.append((k, "single"))
            elif o.bimanual:
                out.append(Affordance("pick_bimanual", k, None, "bimanual"))
                pickable.append((k, "bimanual"))
        for k, hands in pickable:
            for c, o in self.objects.items():
                if o.container and c != k:
                    out.append(Affordance("place_into", k, c, hands))
            for s in self.surfaces:
                if s != self.objects[k].on_surface:
                    out.append(Affordance("put_on", k, s, hands))
        return out

    def instances(self) -> list[TaskInstance]:
        out = []
        for a in self.affordances():
            if a.kind == "place_into":
                text = f"put {self.objects[a.obj].phrase} in {self.objects[a.target].phrase}"
                out.append(
                    TaskInstance("put <obj> in <container>", a.obj, a.target, "in", self.profile_for(a.obj), text)
                )
            elif a.kind == "put_on":
                text = f"put {self.objects[a.obj].phrase} on {self.surfaces[a.target].phrase}"
                out.append(TaskInstance("put <obj> on <surface>", a.obj, a.target, "on", self.profile_for(a.obj), text))
        return out

    # -- criteria ---------------------------------------------------------------------------------
    def criterion(self, inst: TaskInstance) -> Callable[[dict], tuple[bool, str]]:
        """Compile a template instance into a predicate over the privileged state: (ok, "[tag] why")."""
        if inst.kind == "in":
            c = self.objects[inst.target]
            inner = (c.half_extent[0] - c.wall, c.half_extent[1] - c.wall)

            def inside(state: dict) -> tuple[bool, str]:
                if state.get("fallen"):
                    return False, "[place] robot fell"
                goal = state["objects"][inst.target]
                gz = float(goal["pos"][2]) + c.centre[2]
                goal_pos = [goal["pos"][0] + c.centre[0], goal["pos"][1] + c.centre[1], gz]
                ok, why = in_container(
                    state["objects"][inst.obj]["pos"],
                    goal_pos,
                    goal["quat"],
                    inner,
                    gz - c.half_extent[2] + c.wall,
                    gz + c.half_extent[2],
                )
                return ok, "" if ok else f"[place] {why}"

            return inside
        s = self.surfaces[inst.target]
        o = self.objects[inst.obj]
        centre = np.asarray(o.centre if o.centre is not None else (0.0, 0.0, 0.0), dtype=float)
        half_z = float(o.half_extent[2]) if o.half_extent is not None else 0.0

        def on_top(state: dict) -> tuple[bool, str]:
            if state.get("fallen"):
                return False, "[put_on] robot fell"
            pos = np.asarray(state["objects"][inst.obj]["pos"], dtype=float)
            bottom = float(pos[2] + centre[2] - half_z)  # the box's underside, not its origin
            if not (s.top - PUT_ON_SLACK_M <= bottom <= s.top + PUT_ON_HEIGHT_M):
                return False, f"[put_on] underside z {bottom:.3f} not on the top {s.top:.3f} (+{PUT_ON_HEIGHT_M} m)"
            xy = pos[:2] + centre[:2]
            d = xy - np.asarray(s.xy)
            c, sn = np.cos(-s.yaw), np.sin(-s.yaw)
            over = np.abs([c * d[0] - sn * d[1], sn * d[0] + c * d[1]]) - np.asarray(s.half_extent)
            if np.any(over > PUT_ON_SLACK_M):
                return (
                    False,
                    f"[put_on] centre xy {xy.round(3).tolist()} is {over.max():.2f} m outside the surface footprint",
                )
            return True, ""

        return on_top
