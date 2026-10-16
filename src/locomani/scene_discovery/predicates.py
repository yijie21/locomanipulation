"""Goal predicates: a task's goal is a conjunction like ["in(Banana_A, WhiteBowl)", "upright(Mug_1)"].

Plan decision 7: success is judged from the simulator's privileged state, never by the planner. `compile_goal`
checks every name, arity and key against the scene file up front -- a typo fails when the task is loaded, not as
a silent False at the end of an episode -- and returns one callable `check(state) -> (ok, "[goal] why")` that
reports the first predicate that does not hold. `in` reuses the bottle line's `in_container`; `on` is
`SceneModel`'s put-on rule with the surface's yaw applied. A goal that needs a predicate not listed here goes into
the task's `missing` field (docs/handoff/scene-file-format-20261003.md §7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from locomani.scene_discovery.scene_file import to_surface_frame
from locomani.skills.scene_model import PUT_ON_HEIGHT_M, PUT_ON_SLACK_M
from locomani.tasks.bottle.place_phase import in_container

ROBOT = "robot"
UPRIGHT_MAX_DEG = 15.0
RESTING_GAP_M = 0.02  # an underside this close to a surface top under it is resting, not held
ARITY = {"on": 2, "in": 2, "held": 1, "upright": 1, "near": 3, "in_room": 2}
_FORM = re.compile(r"^\s*([a-z_]+)\s*\((.*)\)\s*$")


@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple

    def __str__(self) -> str:
        return f"{self.name}({', '.join(str(a) for a in self.args)})"


def parse(text: str) -> Predicate:
    m = _FORM.match(text)
    if m is None:
        raise ValueError(f"{text!r}: expected name(arg, ...)")
    name, raw = m.group(1), [a.strip() for a in m.group(2).split(",") if a.strip()]
    if name not in ARITY:
        raise ValueError(f"{text!r}: unknown predicate {name!r}; known {sorted(ARITY)}")
    if len(raw) != ARITY[name]:
        raise ValueError(f"{text!r}: {name} takes {ARITY[name]} arguments, got {len(raw)}")
    if name == "near":
        raw[2] = float(raw[2])
    return Predicate(name, tuple(raw))


def _rotate(quat, v) -> np.ndarray:
    """Rotate `v` by the (w, x, y, z) quaternion `quat`."""
    w, x, y, z = (float(c) for c in quat)
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


class _Scene:
    """The scene-file numbers the predicates read, looked up once."""

    def __init__(self, meta: dict):
        self.objects, self.surfaces = meta.get("objects") or {}, meta.get("surfaces") or {}
        self.rooms = (meta.get("room") or {}).get("rooms") or {}

    def box(self, key: str) -> tuple[np.ndarray, np.ndarray, float]:
        box = self.objects[key].get("box") or {}
        half = np.asarray(box.get("half_extent", (0.0, 0.0, 0.0)), dtype=float)
        centre = np.asarray(box.get("centre", (0.0, 0.0, 0.0)), dtype=float)
        return half, centre, float(box.get("wall", 0.01))

    def underside(self, key: str, state: dict) -> tuple[np.ndarray, float]:
        """World xy of the box centre and z of its underside (object rotation ignored, as SceneModel does)."""
        half, centre, _ = self.box(key)
        pos = np.asarray(state["objects"][key]["pos"], dtype=float)
        return pos[:2] + centre[:2], float(pos[2] + centre[2] - half[2])

    def xy(self, key: str, state: dict) -> np.ndarray:
        if key == ROBOT:
            return np.asarray(state["pelvis"]["pos"][:2], dtype=float)
        if key in self.objects:
            return self.underside(key, state)[0]
        return np.asarray(self.surfaces[key]["xy"], dtype=float)


def _require(scene: _Scene, pred: Predicate, i: int, *kinds: str) -> None:
    key = pred.args[i]
    ok = (
        ("object" in kinds and key in scene.objects)
        or ("surface" in kinds and key in scene.surfaces)
        or ("container" in kinds and key in scene.objects and scene.objects[key].get("container"))
        or ("room" in kinds and key in scene.rooms)
        or ("robot" in kinds and key == ROBOT)
    )
    if not ok:
        raise ValueError(f"{pred}: argument {i + 1} {key!r} must be a {' or '.join(kinds)} of this scene")


def _on(scene: _Scene, obj: str, surface: str, state: dict) -> tuple[bool, str]:
    s = scene.surfaces[surface]
    xy, bottom = scene.underside(obj, state)
    top = float(s["top"])
    if not (top - PUT_ON_SLACK_M <= bottom <= top + PUT_ON_HEIGHT_M):
        return False, f"underside z {bottom:.3f} not on {surface} (top {top:.3f}, +{PUT_ON_HEIGHT_M} m)"
    half = np.asarray(s.get("flat_half_extent") or s["half_extent"], dtype=float)
    over = np.abs(to_surface_frame(xy, s)) - half
    if np.any(over > PUT_ON_SLACK_M):
        return False, f"{over.max():.2f} m outside {surface}'s footprint"
    return True, ""


def _in(scene: _Scene, obj: str, container: str, state: dict) -> tuple[bool, str]:
    half, centre, wall = scene.box(container)
    goal = state["objects"][container]
    gz = float(goal["pos"][2]) + centre[2]
    goal_pos = [goal["pos"][0] + centre[0], goal["pos"][1] + centre[1], gz]
    inner = (half[0] - wall, half[1] - wall)
    return in_container(state["objects"][obj]["pos"], goal_pos, goal["quat"], inner, gz - half[2] + wall, gz + half[2])


def _held(scene: _Scene, obj: str, state: dict) -> tuple[bool, str]:
    """Off every surface and gripped: both fingers of one hand on it, or both hands on it (a two-hand carry). One
    finger is not a grip -- a mug hanging on closed fingers rose 2.5 cm and was scored held (TODO 27.9, 2026-10-16)."""
    o = state["objects"][obj]
    touching, gripped = set(o.get("touching") or ()), set(o.get("gripped") or ())
    if not touching:
        return False, "no hand touches it"
    if not gripped and touching != {"left", "right"}:
        return False, f"not between two fingers ({'/'.join(sorted(touching))} hand touches it)"
    xy, bottom = scene.underside(obj, state)
    for key, s in scene.surfaces.items():
        half = np.asarray(s.get("flat_half_extent") or s["half_extent"], dtype=float)
        if np.all(np.abs(to_surface_frame(xy, s)) <= half) and bottom - float(s["top"]) <= RESTING_GAP_M:
            return False, f"resting on {key} (underside {bottom - float(s['top']):+.3f} m)"
    return True, ""


def _upright(scene: _Scene, obj: str, state: dict) -> tuple[bool, str]:
    axis = (scene.objects[obj].get("semantic") or {}).get("upright_axis") or (0.0, 0.0, 1.0)
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    up = _rotate(state["objects"][obj]["quat"], axis)
    tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
    return (True, "") if tilt < UPRIGHT_MAX_DEG else (False, f"tilted {tilt:.0f} deg (limit {UPRIGHT_MAX_DEG:.0f})")


def _near(scene: _Scene, a: str, b: str, d: float, state: dict) -> tuple[bool, str]:
    dist = float(np.linalg.norm(scene.xy(a, state) - scene.xy(b, state)))
    return (True, "") if dist < d else (False, f"{dist:.2f} m apart (limit {d:.2f})")


def _in_room(scene: _Scene, who: str, room: str, state: dict) -> tuple[bool, str]:
    poly = np.asarray(scene.rooms[room]["polygon"], dtype=float)
    x, y = scene.xy(who, state)
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return (True, "") if inside else (False, f"at ({x:.2f}, {y:.2f}), outside {room}")


def compile_goal(goal: list[str], meta: dict) -> Callable[[dict], tuple[bool, str]]:
    """Validate the goal against the scene file and return check(state) -> (ok, "[goal] <predicate>: why")."""
    scene = _Scene(meta)
    preds = [parse(g) for g in goal]
    if not preds:
        raise ValueError("a goal needs at least one predicate")
    for p in preds:
        if p.name == "on":
            _require(scene, p, 0, "object")
            _require(scene, p, 1, "surface")
        elif p.name == "in":
            _require(scene, p, 0, "object")
            _require(scene, p, 1, "container")
        elif p.name in ("held", "upright"):
            _require(scene, p, 0, "object")
        elif p.name == "near":
            _require(scene, p, 0, "object", "robot")
            _require(scene, p, 1, "object", "surface", "robot")
        elif p.name == "in_room":
            _require(scene, p, 0, "object", "robot")
            _require(scene, p, 1, "room")
    rules = {"on": _on, "in": _in, "held": _held, "upright": _upright, "near": _near, "in_room": _in_room}

    def check(state: dict) -> tuple[bool, str]:
        if state.get("fallen"):
            return False, "[goal] robot fell"
        for p in preds:
            ok, why = rules[p.name](scene, *p.args, state)
            if not ok:
                return False, f"[goal] {p}: {why}"
        return True, ""

    return check
