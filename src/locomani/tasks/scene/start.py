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
# the scene file's spawn rule (adt_to_scene: navigator body radius 0.26 + spawn margin 0.15) on the occupancy map: the
# spawn regions are outer contours with the furniture inside them, and `occ.is_free` asks only for the walking radius,
# counted cell to cell (5 cm) -- a start 0.28 m from a kitchen chair's footprint passed and fell (TODO 42.2)
SPAWN_CLEAR_M = 0.41
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


NORMALS = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-y": (0.0, -1.0), "+y": (0.0, 1.0)}  # a surface's edges, its frame
PLACEMENT_KEYS = ("depth_m", "yaw_deg", "blocker", "edge")  # the `placement` option (TODO 33.5, D5)
BLOCKER_GAP_M = 0.03
BLOCKER_SLIDES_M = (0.0, 0.03, -0.03, 0.06, -0.06)  # a blocker slid along the edge still sits in the hand's corridor


def edge_depths(surf: dict, xy) -> dict[str, float]:
    """Distance from a world point to each approach edge of the surface (all four when the scene file lists none)."""
    flat = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    local = _rot2(float(surf.get("yaw", 0.0))).T @ (np.asarray(xy, float)[:2] - np.asarray(surf["xy"], float))
    out = {}
    for name in surf.get("approach_edges") or list(NORMALS):
        axis = 0 if name[1] == "x" else 1
        out[name] = float(flat[axis] - local[axis] if name[0] == "+" else local[axis] + flat[axis])
    return out


def placement_ok(surf: dict, xy, yaw: float, want: dict) -> bool:
    """The `placement` option's depth and yaw conditions at a candidate spot: `depth_m` [lo, hi] is the distance to the
    edge -- `edge` when the option names one ("-x", "+x", "-y", "+y" in the surface's frame), else the nearest approach
    edge -- and `yaw_deg` [lo, hi] the angle between the object's box x axis and that edge's outward normal, folded to
    0-90 (0: the box's x axis points at the edge, 90: it lies along it)."""
    depths = edge_depths(surf, xy)
    name = want["edge"] if want.get("edge") else min(depths, key=depths.get)
    if name not in depths:
        depths[name] = edge_depths(dict(surf, approach_edges=[name]), xy)[name]
    if "depth_m" in want and not (want["depth_m"][0] <= depths[name] <= want["depth_m"][1]):
        return False
    if "yaw_deg" in want:
        n = NORMALS[name]
        rel = (yaw - float(surf.get("yaw", 0.0)) - np.arctan2(n[1], n[0]) + np.pi / 2) % np.pi - np.pi / 2
        if not (want["yaw_deg"][0] <= abs(np.degrees(rel)) <= want["yaw_deg"][1]):
            return False
    return True


def place_on_surface(meta: dict, key: str, rng, taken: list, placement: dict | None = None):
    """A random spot and yaw for `key` on its own surface: its footprint inside the flat top (SURFACE_MARGIN_M) and
    clear of the `taken` footprints by OBJECT_GAP_M. The height is the scene file's (the object already rests on that
    surface); the yaw turns its initial attitude about the world vertical. `placement` (TODO 33.5, D5) adds the
    `depth_m` / `yaw_deg` conditions of `placement_ok`. None when TRIES spots gave nothing."""
    obj = meta["objects"][key]
    surf = meta["surfaces"][obj["surface"]]
    flat = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    yaw_s = float(surf.get("yaw", 0.0))
    top = (np.asarray(surf["xy"], float), flat - SURFACE_MARGIN_M, yaw_s)
    room = np.maximum(flat - SURFACE_MARGIN_M, 0.0)
    want = {k: v for k, v in (placement or {}).items() if k in ("depth_m", "yaw_deg", "edge")}
    if want.get("edge") and want["edge"] not in NORMALS:
        raise KeyError(f"placement edge {want['edge']!r}: one of {sorted(NORMALS)}")
    want = {k: v for k, v in want.items() if k != "edge" or "depth_m" in want or "yaw_deg" in want}
    for _ in range(TRIES):
        xy = top[0] + _rot2(yaw_s) @ rng.uniform(-room, room)
        quat = _qmul(_yaw_quat(float(rng.uniform(-np.pi, np.pi))), np.asarray(obj["init_quat"], float))
        fp = footprint(obj, xy, quat)
        if not inside_top(fp, top) or any(overlap(fp, t, OBJECT_GAP_M) for t in taken):
            continue
        if want and not placement_ok(surf, xy, fp[2], want):
            continue
        return np.array([xy[0], xy[1], float(obj["init_pos"][2])]), quat
    return None


def place_blocker(meta: dict, key: str, anchor: str, anchor_spot, taken: list, gap: float, edge: str | None = None):
    """`key` set between `anchor` (at `anchor_spot`) and an edge of their shared surface -- `edge` when named, else
    the nearest approach edge, then the next -- its footprint `gap` from the anchor's, in its initial attitude, slid
    along the edge by up to BLOCKER_SLIDE_M when it does not fit straight ahead. None when no edge takes it. The hand's
    corridor to the anchor is then blocked (the `blocker` placement, TODO 33.5)."""
    obj, aobj = meta["objects"][key], meta["objects"][anchor]
    if obj.get("surface") != aobj.get("surface"):
        raise KeyError(f"placement blocker {key} is not on {anchor}'s surface")
    surf = meta["surfaces"][obj["surface"]]
    flat = np.asarray(surf.get("flat_half_extent") or surf["half_extent"], float)[:2]
    yaw_s = float(surf.get("yaw", 0.0))
    top = (np.asarray(surf["xy"], float), flat - SURFACE_MARGIN_M, yaw_s)
    axy = np.asarray(anchor_spot[0], float)[:2]
    quat = np.asarray(obj["init_quat"], float)
    dist = footprint_radius(aobj) + footprint_radius(obj) + gap
    depths = edge_depths(surf, axy)
    names = [edge] if edge else sorted(depths, key=depths.get)
    for name in names:
        n = _rot2(yaw_s) @ np.asarray(NORMALS[name], float)
        side = np.array([-n[1], n[0]])
        for slide in BLOCKER_SLIDES_M:
            xy = axy + dist * n + slide * side
            fp = footprint(obj, xy, quat)
            if inside_top(fp, top) and not any(overlap(fp, t, OBJECT_GAP_M) for t in taken):
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


def _clearance(occ, xy) -> float:
    c = occ.cell_of(xy)
    return float(occ.clearance[c[0], c[1]])


def spawn_polygons(meta: dict, room: str | None) -> list:
    spawn = (meta.get("room") or {}).get("spawn") or {}
    if room is not None and room in spawn:
        return list(spawn[room])
    return [p for polys in spawn.values() for p in polys]


def sample_start(
    meta: dict, occ, nav, objects, rng, *, seed: int, room: str | None = None, spawn: dict | None = None,
    placement: dict | None = None,
) -> Start:  # fmt: skip
    """`objects` = the task's movable objects (moved), the first one anchors the robot's spawn. `spawn` (a task
    family's option, TODO 31): {"rooms": "all"} draws the robot from every room's spawn region (the reachability check
    keeps only the rooms the navigation map connects to the object), {"distance_m": [lo, hi]} replaces
    SPAWN_DISTANCE_M (hi None: no upper limit). `placement` (a family's or a task's option, TODO 33.5, D5):
    {"depth_m": [lo, hi]} and {"yaw_deg": [lo, hi]} constrain the task objects' spots (placement_ok), {"blocker":
    {"key": K, "gap_m": g}} sets the free object K between the first task object and its nearest approach edge
    (place_blocker); a condition the surface cannot meet raises RuntimeError, so the pool skips that seed. Without
    them every start is drawn as before."""
    spawn = dict(spawn or {})
    unknown = set(spawn) - {"rooms", "distance_m"}
    if unknown or spawn.get("rooms", "all") != "all":
        raise KeyError(f"spawn option: unknown {sorted(unknown) or spawn.get('rooms')}")
    want = dict(placement or {})
    unknown = set(want) - set(PLACEMENT_KEYS)
    if unknown:
        raise KeyError(f"placement option: unknown {sorted(unknown)}")
    blocker = (want.get("blocker") or {}).get("key")
    if blocker and (blocker in objects or not meta["objects"].get(blocker, {}).get("free")):
        raise KeyError(f"placement blocker {blocker}: not a free object outside the task")
    d_lo, d_hi = spawn.get("distance_m") or SPAWN_DISTANCE_M
    d_hi = np.inf if d_hi is None else d_hi
    placements, taken = {}, []
    # with a placement option only the objects on the task object's own surface count as taken: the footprints are
    # flat, so without that filter a shelf above or below (a display unit's) crowds this one out of the spots the
    # option asks for; the plain draw keeps every resting object, as every pool so far was drawn (TODO 33.5)
    own = {meta["objects"][k].get("surface") for k in objects} if want else None
    for k, o in meta["objects"].items():  # every object resting on a surface and not moved keeps its spot -- free or
        # not: a movable object that is not free still collides (its hull), and the canister placed on the utility
        # cart's tray was thrown off within a second (2026-10-16)
        if o.get("surface") and k not in objects and k != blocker and (own is None or o["surface"] in own):
            taken.append(footprint(o, o["init_pos"], o["init_quat"]))
    for k in objects:
        spot = place_on_surface(meta, k, rng, taken, want)
        if spot is None and want:
            raise RuntimeError(f"no spot on {meta['objects'][k]['surface']} meets the placement option for {k}")
        if spot is None:  # a crowded top (the bedroom's display shelves): the scene file's own spot is a valid start
            o = meta["objects"][k]
            spot = (np.asarray(o["init_pos"], float), np.asarray(o["init_quat"], float))
        placements[k] = spot
        taken.append(footprint(meta["objects"][k], spot[0], spot[1]))
    if blocker and objects:
        spot = place_blocker(meta, blocker, objects[0], placements[objects[0]], taken,
                             float(want["blocker"].get("gap_m", BLOCKER_GAP_M)), want.get("edge"))  # fmt: skip
        if spot is None:
            raise RuntimeError(f"no spot for the blocker {blocker} beside {objects[0]}")
        placements[blocker] = spot
        taken.append(footprint(meta["objects"][blocker], spot[0], spot[1]))
    anchor = placements[objects[0]][0][:2] if objects else None
    if spawn.get("rooms") == "all":
        polys = spawn_polygons(meta, None)
    else:
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
        if not any(_inside(p, xy) for p in polys) or not occ.is_free(xy) or _clearance(occ, xy) < SPAWN_CLEAR_M:
            continue
        if anchor is not None:
            d = float(np.linalg.norm(xy - anchor))
            if not (d_lo <= d <= d_hi) or not nav.reachable(xy):
                continue
        return Start(seed, (float(xy[0]), float(xy[1])), float(rng.uniform(-np.pi, np.pi)), placements)
    raise RuntimeError("no reachable spawn point in the room's spawn region")
