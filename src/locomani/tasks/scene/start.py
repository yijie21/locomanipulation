"""Start states of a scene task (TODO 27.2.3, plan D7): the task's movable objects moved to a random spot (and yaw) on
the surface they stand on, clear of the other free objects there; the robot at a random free spot of the room's spawn
region, 1-3 m from the first task object, facing a random way, from where the object's surroundings are reachable on
the occupancy map. Everything else stays where the scene file has it. A start is a plain record, so pools of starts
can be drawn once and replayed."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

SURFACE_MARGIN_M = 0.02  # object footprint kept this far inside the surface's flat top
OBJECT_GAP_M = 0.02  # between footprints of objects on one surface
SPAWN_DISTANCE_M = (1.0, 3.0)  # robot pelvis to the first task object
APPROACH_RING_M = (0.35, 0.80)  # where the robot must be able to stand near the object (reachability check)
TRIES = 2000  # a crowded top (the utility cart: a tray across most of it) leaves a ~10 cm strip


@dataclass(frozen=True)
class Start:
    seed: int
    robot_xy: tuple[float, float]
    robot_yaw: float
    placements: dict  # key -> (pos xyz, quat wxyz)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["placements"] = {k: [list(map(float, p)), list(map(float, q))] for k, (p, q) in self.placements.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Start":
        return cls(int(d["seed"]), tuple(d["robot_xy"]), float(d["robot_yaw"]),
                   {k: (np.asarray(p, float), np.asarray(q, float)) for k, (p, q) in d["placements"].items()})  # fmt: skip


def _rot2(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])  # fmt: skip


def footprint_radius(obj: dict) -> float:
    half = obj["box"]["half_extent"]
    return float(np.hypot(half[0], half[1]))


def footprint(obj: dict, xy, quat) -> tuple[np.ndarray, np.ndarray, float]:
    """(centre xy, half extents xy, yaw) of an object's box seen from above."""
    w, x, y, z = (float(c) for c in quat)
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return np.asarray(xy, float)[:2], np.asarray(obj["box"]["half_extent"][:2], float), yaw


def overlap(a, b, gap: float) -> bool:
    """Two footprints (footprint()) closer than `gap` (separating axes of the two rectangles)."""
    (ca, ha, ya), (cb, hb, yb) = a, b
    Ra, Rb = _rot2(ya), _rot2(yb)
    d = cb - ca
    for axis in (Ra[:, 0], Ra[:, 1], Rb[:, 0], Rb[:, 1]):
        ra = abs(Ra[:, 0] @ axis) * ha[0] + abs(Ra[:, 1] @ axis) * ha[1]
        rb = abs(Rb[:, 0] @ axis) * hb[0] + abs(Rb[:, 1] @ axis) * hb[1]
        if abs(d @ axis) > ra + rb + gap:
            return False
    return True


def place_on_surface(meta: dict, key: str, rng, taken: list) -> tuple[np.ndarray, np.ndarray] | None:
    """A random spot and yaw for `key` on its own surface: its footprint inside the flat top (SURFACE_MARGIN_M) and
    clear of the `taken` footprints by OBJECT_GAP_M. The height is the scene file's (the object already rests on that
    surface); the yaw turns its initial attitude about the world vertical."""
    obj = meta["objects"][key]
    surf = meta["surfaces"][obj["surface"]]
    flat = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)
    yaw_s = float(surf.get("yaw", 0.0))
    top = (np.asarray(surf["xy"], float), flat - SURFACE_MARGIN_M, yaw_s)
    room = np.maximum(flat - SURFACE_MARGIN_M, 0.0)
    for _ in range(TRIES):
        xy = top[0] + _rot2(yaw_s) @ rng.uniform(-room, room)
        quat = _qmul(_yaw_quat(float(rng.uniform(-np.pi, np.pi))), np.asarray(obj["init_quat"], float))
        fp = footprint(obj, xy, quat)
        if not inside_top(fp, top) or any(overlap(fp, t, OBJECT_GAP_M) for t in taken):
            continue
        return np.array([xy[0], xy[1], float(obj["init_pos"][2])]), quat
    return None


def inside_top(fp, top) -> bool:
    """Every corner of footprint `fp` inside the rectangle `top`."""
    (c, h, y), (ct, ht, yt) = fp, top
    corners = c + (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * h) @ _rot2(y).T
    local = (corners - ct) @ _rot2(yt)
    return bool(np.all(np.abs(local) <= ht + 1e-9))


def _inside(poly, xy) -> bool:
    poly = np.asarray(poly, float)
    x, y = xy
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def spawn_polygons(meta: dict, room: str | None) -> list:
    spawn = (meta.get("room") or {}).get("spawn") or {}
    if room is not None and room in spawn:
        return list(spawn[room])
    return [p for polys in spawn.values() for p in polys]


def sample_start(meta: dict, occ, nav, objects, rng, *, seed: int, room: str | None = None) -> Start:
    """`objects` = the task's movable objects (moved), the first one anchors the robot's spawn."""
    placements, taken = {}, []
    for k, o in meta["objects"].items():  # every object resting on a surface and not moved keeps its spot -- free or
        # not: a movable object that is not free still collides (its hull), and the canister placed on the utility
        # cart's tray was thrown off within a second (2026-10-16)
        if o.get("surface") and k not in objects:
            taken.append(footprint(o, o["init_pos"], o["init_quat"]))
    for k in objects:
        spot = place_on_surface(meta, k, rng, taken)
        if spot is None:  # a crowded top (the bedroom's display shelves): the scene file's own spot is a valid start
            o = meta["objects"][k]
            spot = (np.asarray(o["init_pos"], float), np.asarray(o["init_quat"], float))
        placements[k] = spot
        taken.append(footprint(meta["objects"][k], spot[0], spot[1]))
    anchor = placements[objects[0]][0][:2] if objects else None
    polys = spawn_polygons(
        meta, room or (meta["surfaces"][meta["objects"][objects[0]]["surface"]].get("room") if objects else None)
    )
    if not polys:
        raise RuntimeError("the scene file has no spawn region")
    pts = np.concatenate([np.asarray(p, float) for p in polys])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    if anchor is not None:
        nav.set_goal_ring(anchor, *APPROACH_RING_M)
    for _ in range(TRIES):
        xy = rng.uniform(lo, hi)
        if not any(_inside(p, xy) for p in polys) or not occ.is_free(xy):
            continue
        if anchor is not None:
            d = float(np.linalg.norm(xy - anchor))
            if not (SPAWN_DISTANCE_M[0] <= d <= SPAWN_DISTANCE_M[1]) or not nav.reachable(xy):
                continue
        return Start(seed, (float(xy[0]), float(xy[1])), float(rng.uniform(-np.pi, np.pi)), placements)
    raise RuntimeError("no reachable spawn point in the room's spawn region")
