"""TODO 33.24.3 (user 2026-10-24: chairs and stools in front of a counter can be pushed aside before a grasp): floor
furniture that moves, for the navigation map and the preparatory-action layer.

In a scene built by scripts/rsi/movable_scene.py the scene file's `movable` names each piece of floor furniture that
is a free body, with its navigation footprint in its own frame and the room obstacle that footprint is. The planners'
map (navigator.OccupancyMap) is built once from the scene file, so it keeps a piece where it stood. Here:

  navigation(world)   the map with every piece's footprint where it is now (exactly the scene file's map while they all
                      stand at their initial poses)
  refresh(world)      rebuild world.occ / world.nav that way -- a skill calls it after it moved a piece; reset(world)
                      puts back the map the world was built with (episode.run_episode calls it at every round's start,
                      when the pieces are back where they began)
  goals(world, key)   the preparatory goals of kind "floor" for a grasp of `key`: each piece within NEAR_M of the object
                      moved MOVES_M away from the object or to either side, on free floor clear of the other pieces --
                      the selector (prep.propose) keeps those after which the chooser finds a standing option

The 2026-10-24 push probe (outputs/scene_rsi_20261016/push_probe_20261024/full): in 36 rounds the robot never fell;
walking into a kitchen chair moved it 19-25 cm (tilted 2-19 deg), into the step stool 20-35 cm; both hands on the
chair's back moved it about 13 cm; gripping a top edge and stepping back never held; the bar stool barely moved. So
the goals offer 20-30 cm, and the action -- how to move it -- is the skills' (prep.register("floor")).
A scene without `movable` has no floor pieces: every function here is then a no-op and the map is untouched."""

from __future__ import annotations

import copy
import math

import numpy as np

from locomani.primitives.navigator import Navigator
from locomani.primitives.navigator import build as build_navigation

BODY_RADIUS_M = 0.35  # world.BODY_RADIUS_M, choose.BODY_RADIUS_M
NEAR_M = 1.2  # a piece this close to the object (centre to centre) may be in the way of its stances
MOVES_M = (0.20, 0.30)  # how far a piece is offered to move (the probe moved chairs and the step stool 20-35 cm)
SIDE_DEG = (0.0, 90.0, -90.0)  # away from the object, or to either side
FREE_CELL_FRACTION = 0.9  # the moved footprint must lie this much on cells the map without the pieces calls free


def pieces(world) -> dict:
    """{key: {"obstacle": index, "footprint": [[x, y], ...] in the piece's frame}} -- empty for a scene without any."""
    return dict((world.meta or {}).get("movable") or {})


def _yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def footprint(world, key: str, pose=None) -> np.ndarray:
    """The piece's footprint polygon (world xy) at `pose` (pos, quat), or where it is now."""
    p, q = pose if pose is not None else world.object_pose(key)
    yaw = _yaw(q)
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.asarray(pieces(world)[key]["footprint"], float)
    return local @ np.array([[c, s], [-s, c]]) + np.asarray(p, float)[:2]


def _meta_with(world, footprints: dict) -> dict:
    meta = dict(world.meta)
    room = dict(meta.get("room") or {})
    drop = {int(v["obstacle"]) for v in pieces(world).values()}
    obs = [poly for i, poly in enumerate(room.get("obstacles") or []) if i not in drop]
    room["obstacles"] = obs + [fp.tolist() for fp in footprints.values()]
    meta["room"] = room
    return meta


_BASE: dict = {}  # id(meta) -> (meta, the map without the pieces): the walls alone take about 1.2 s to draw


def _base(world, radius: float = BODY_RADIUS_M):
    k = (id(world.meta), radius)
    if k not in _BASE or _BASE[k][0] is not world.meta:
        _BASE[k] = (world.meta, build_navigation(_meta_with(world, {}), body_radius=radius)[0])
    return _BASE[k][1]


def navigation(world, poses: dict | None = None):
    """(occupancy map, navigator) with every piece where it is now (or at `poses`, {key: (pos, quat)}): a copy of the
    cached map without the pieces, their footprints marked, the clearance recomputed."""
    if not pieces(world):
        return build_navigation(world.meta, body_radius=BODY_RADIUS_M)
    poses = poses or {}
    base = _base(world)
    occ = copy.copy(base)
    occ.occupied = base.occupied.copy()
    occ.surfaces = dict(base.surfaces)
    for k in pieces(world):
        occ.add_polygon(footprint(world, k, poses.get(k)))
    occ.recompute()
    return occ, Navigator(occ)


def refresh(world) -> None:
    """Rebuild world.occ / world.nav with the pieces where they are now; the first call keeps the map it replaces."""
    if not pieces(world):
        return
    if getattr(world, "_floor_base", None) is None:
        world._floor_base = (getattr(world, "occ", None), getattr(world, "nav", None))
    world.occ, world.nav = navigation(world)
    for attr in ("_choose_floor_statics",):  # the chooser's cache of the low fixed objects the map leaves out
        if hasattr(world, attr):
            try:
                delattr(world, attr)
            except AttributeError:
                pass


def reset(world) -> None:
    """Put back the map the world was built with (the pieces are back at their initial poses each round)."""
    base = getattr(world, "_floor_base", None)
    if base is not None:
        world.occ, world.nav = base
        world._floor_base = None


def _free_fraction(occ, polygon: np.ndarray) -> float:
    lo, hi = polygon.min(0), polygon.max(0)
    xs = np.arange(lo[0], hi[0] + 1e-9, occ.res)
    ys = np.arange(lo[1], hi[1] + 1e-9, occ.res)
    inside, free = 0, 0
    for x in xs:
        for y in ys:
            if not _inside(polygon, (x, y)):
                continue
            inside += 1
            c = occ.cell_of(np.array([x, y]))
            free += bool(occ.in_bounds(c) and not occ.occupied[c[0], c[1]])
    return free / inside if inside else 0.0


def _inside(poly, xy) -> bool:
    poly = np.asarray(poly, float)
    x, y = xy
    ins = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            ins = not ins
    return ins


def _overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Two convex polygons overlap (separating axes)."""
    for poly in (a, b):
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def goals(world, key: str) -> list[dict]:
    """Candidate moves of the pieces near `key` (module docstring): dicts with key, pos, quat, move_m, direction_deg,
    note -- prep.goals turns them into PrepGoals of kind "floor". Each new footprint lies on free floor of the map
    without the pieces and clear of every other piece where it stands now."""
    if not pieces(world):
        return []
    p_obj = np.asarray(world.object_pose(key)[0], float)[:2]
    static = _base(world, 0.0)
    now = {k: footprint(world, k) for k in pieces(world)}
    out = []
    for k in pieces(world):
        p, q = world.object_pose(k)
        p = np.asarray(p, float)
        d = p[:2] - p_obj
        dist = float(np.linalg.norm(d))
        if dist > NEAR_M:
            continue
        away = d / (dist or 1.0)
        for side in SIDE_DEG:
            a = math.radians(side)
            u = np.array([math.cos(a) * away[0] - math.sin(a) * away[1], math.sin(a) * away[0] + math.cos(a) * away[1]])
            for s in MOVES_M:
                pos = np.r_[p[:2] + s * u, p[2]]
                fp = footprint(world, k, (pos, q))
                if _free_fraction(static, fp) < FREE_CELL_FRACTION:
                    continue
                if any(_overlap(fp, other) for kk, other in now.items() if kk != k):
                    continue
                how = (
                    "away from the object" if side == 0.0 else f"to the {'left' if side > 0 else 'right'} of that line"
                )
                out.append(dict(key=k, pos=tuple(float(v) for v in pos), quat=tuple(float(v) for v in q),
                                move_m=float(s), direction_deg=math.degrees(math.atan2(u[1], u[0])),
                                note=f"move {k} {100 * s:.0f} cm {how}"))  # fmt: skip
    return out
