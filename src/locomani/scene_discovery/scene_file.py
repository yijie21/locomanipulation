"""Scene file v2: `scene_meta.json` plus what discovery needs (docs/handoff/scene-file-format-20261003.md).

A v2 file keeps every legacy key with its old meaning, so the ~40 existing readers (navigation, approach, grasps,
criteria, `SceneModel`) work unchanged; the new information lives in new keys: `schema_version`, `source`, `room`,
`randomization`, and per object `semantic` + `provenance`, per surface `yaw` / `of_object` / `approach_edges` /
`room` / `provenance`. `validate` lists every problem at once instead of stopping at the first, and `load` refuses a
file with any: a half-valid scene is the partially-applied-config failure this repo keeps paying for.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2
SOURCES = frozenset({"adt", "rule", "claude", "measured"})  # where an inferred value came from
GRASPABLE = frozenset({"single", "bimanual", "none"})
EDGES = frozenset({"+x", "-x", "+y", "-y"})  # surface frame: +x is the surface's own x axis after `yaw`


def _is_xy(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(c, (int, float)) for c in v)


def _check_provenance(where: str, block: dict, provenance, out: list[str]) -> None:
    if not isinstance(provenance, dict):
        out.append(f"{where}: provenance missing")
        return
    for field in block:
        src = provenance.get(field)
        if src is None:
            out.append(f"{where}: no provenance for {field!r}")
        elif src not in SOURCES:
            out.append(f"{where}: provenance of {field!r} is {src!r}, not one of {sorted(SOURCES)}")


def _check_object(key: str, obj: dict, rooms: set, out: list[str]) -> None:
    where = f"objects.{key}"
    sem = obj.get("semantic")
    if not isinstance(sem, dict):
        out.append(f"{where}: semantic missing")
        return
    _check_provenance(where, sem, obj.get("provenance"), out)
    if sem.get("graspable", "none") not in GRASPABLE:
        out.append(f"{where}: graspable {sem.get('graspable')!r} not one of {sorted(GRASPABLE)}")
    for field in ("mass_kg", "friction"):
        if field in sem and not (isinstance(sem[field], (int, float)) and sem[field] > 0):
            out.append(f"{where}: {field} must be a positive number, got {sem[field]!r}")
    if "movable" in sem and not isinstance(sem["movable"], bool):
        out.append(f"{where}: movable must be true/false")
    if "room" in sem and sem["room"] not in rooms:
        out.append(f"{where}: room {sem['room']!r} is not in room.rooms")
    axis = sem.get("upright_axis")
    if axis is not None and not (len(axis) == 3 and np.linalg.norm(np.asarray(axis, dtype=float)) > 0):
        out.append(f"{where}: upright_axis must be a non-zero 3-vector")


def _check_surface(key: str, surf: dict, objects: dict, rooms: set, out: list[str]) -> None:
    where = f"surfaces.{key}"
    if not isinstance(surf.get("yaw"), (int, float)):
        out.append(f"{where}: yaw missing (radians; 0 for an axis-aligned surface)")
    if not _is_xy(surf.get("xy")) or not _is_xy(surf.get("half_extent")):
        out.append(f"{where}: xy and half_extent must be [x, y]")
    if not isinstance(surf.get("top"), (int, float)):
        out.append(f"{where}: top missing")
    if surf.get("of_object") is not None and surf["of_object"] not in objects:
        out.append(f"{where}: of_object {surf['of_object']!r} is not an object")
    bad = set(surf.get("approach_edges") or ()) - EDGES
    if bad:
        out.append(f"{where}: approach_edges {sorted(bad)} not in {sorted(EDGES)}")
    if "room" in surf and surf["room"] not in rooms:
        out.append(f"{where}: room {surf['room']!r} is not in room.rooms")
    inferred = {k: surf[k] for k in ("yaw", "of_object", "approach_edges", "room") if k in surf}
    _check_provenance(where, inferred, surf.get("provenance"), out)


def _check_room(room, out: list[str]) -> set:
    if not isinstance(room, dict):
        out.append("room missing")
        return set()
    bounds = room.get("bounds") or {}
    for axis in ("x", "y"):
        lim = bounds.get(axis)
        if not (_is_xy(lim) and lim[0] < lim[1]):
            out.append(f"room.bounds.{axis} must be [min, max]")
    for i, seg in enumerate(room.get("walls") or ()):
        if not (len(seg) == 2 and all(_is_xy(p) for p in seg)):
            out.append(f"room.walls[{i}] must be [[x0, y0], [x1, y1]]")
    for i, poly in enumerate(room.get("obstacles") or ()):
        if len(poly) < 3 or not all(_is_xy(p) for p in poly):
            out.append(f"room.obstacles[{i}] needs >= 3 [x, y] points")
    rooms = room.get("rooms") or {}
    for name, r in rooms.items():
        poly = (r or {}).get("polygon") or ()
        if len(poly) < 3 or not all(_is_xy(p) for p in poly):
            out.append(f"room.rooms.{name}: polygon needs >= 3 [x, y] points")
        if (r or {}).get("name_source") not in SOURCES:
            out.append(f"room.rooms.{name}: name_source must be one of {sorted(SOURCES)}")
    for name in room.get("spawn") or {}:
        if name not in rooms:
            out.append(f"room.spawn.{name}: not a room")
    return set(rooms)


def _check_randomization(rnd, objects: dict, surfaces: dict, rooms: set, out: list[str]) -> None:
    if rnd is None:
        return
    for pool in ("movable_pool", "distractor_pool"):
        for key in rnd.get(pool) or ():
            if key not in objects:
                out.append(f"randomization.{pool}: {key!r} is not an object")
    for key, targets in (rnd.get("placements") or {}).items():
        if key not in objects:
            out.append(f"randomization.placements: {key!r} is not an object")
        for s in targets or ():
            if s not in surfaces:
                out.append(f"randomization.placements.{key}: {s!r} is not a surface")
    for name in rnd.get("spawn_rooms") or ():
        if name not in rooms:
            out.append(f"randomization.spawn_rooms: {name!r} is not a room")


def validate(meta: dict) -> list[str]:
    """Every problem in a v2 scene file, as one line each; empty when the file is usable."""
    out: list[str] = []
    if meta.get("schema_version") != SCHEMA_VERSION:
        out.append(f"schema_version must be {SCHEMA_VERSION}, got {meta.get('schema_version')!r}")
    for key in ("surfaces", "objects", "robot_start", "source"):
        if not isinstance(meta.get(key), dict):
            out.append(f"{key} missing")
    objects, surfaces = meta.get("objects") or {}, meta.get("surfaces") or {}
    rooms = _check_room(meta.get("room"), out)
    for key, obj in objects.items():
        _check_object(key, obj, rooms, out)
        if obj.get("surface") is not None and obj["surface"] not in surfaces:
            out.append(f"objects.{key}: surface {obj['surface']!r} is not a surface")
    for key, surf in surfaces.items():
        _check_surface(key, surf, objects, rooms, out)
    _check_randomization(meta.get("randomization"), objects, surfaces, rooms, out)
    return out


def load(path) -> dict:
    """Read and validate a v2 scene file; raises ValueError listing every problem."""
    meta = json.loads(Path(path).read_text())
    problems = validate(meta)
    if problems:
        raise ValueError(f"{path}: {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return meta


def with_approach_edges(meta: dict, overlay) -> dict:
    """A copy of `meta` whose surfaces take `approach_edges` from `overlay` (a path to, or the dict of, a
    `{"surfaces": {key: {"approach_edges": [...]}}}` file; scripts/experiments/scene_rsi_20261016/stance_edges.py
    writes one from the scene runner's stance search, TODO 27.3.1). Which edges a robot can stand at depends on the
    robot and the runner, not only the scene, so the file stays as it is and the checker and the proposer read the
    overlay; surfaces it leaves out keep their own. Raises ValueError on a surface the scene lacks or an unknown edge."""
    if not isinstance(overlay, dict):
        overlay = json.loads(Path(overlay).read_text())
    out = json.loads(json.dumps(meta))
    problems = [f"surfaces.{k}: not in the scene" for k in sorted(set(overlay["surfaces"]) - set(out["surfaces"]))]
    for key, row in sorted(overlay["surfaces"].items()):
        bad = set(row["approach_edges"]) - EDGES
        if bad:
            problems.append(f"surfaces.{key}: approach_edges {sorted(bad)} not in {sorted(EDGES)}")
        elif key in out["surfaces"]:
            out["surfaces"][key]["approach_edges"] = list(row["approach_edges"])
    if problems:
        raise ValueError(f"approach-edge overlay: {len(problems)} problem(s):\n  " + "\n  ".join(problems))
    return out


def to_surface_frame(xy, surf: dict) -> np.ndarray:
    """World xy -> the surface's own frame (origin at its centre, axes turned by its yaw)."""
    d = np.asarray(xy, dtype=float)[:2] - np.asarray(surf["xy"], dtype=float)
    c, s = np.cos(-float(surf.get("yaw", 0.0))), np.sin(-float(surf.get("yaw", 0.0)))
    return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
