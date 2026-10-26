"""Vision RSI D2 step 2d (docs/handoff/vision-rsi-design-20261023.md section 17.7): the scene description the robot
would have after its mapping tour, in the scene file's own shape, so the skills read it as `world.meta` unchanged.

Each part replaces its truth independently (`parts`), for the swap-back matrix of section 6:
  grid      the walking map: the room's walls and obstacles give way to the perceived blocked cells (obstacles, and
            cells never seen as floor: section 17.1, rule 4) as a raster (room.blocked, which the navigator reads);
            static objects leave the description (the grid holds what was seen of them)
  surfaces  every surface the map found takes its perceived centre, height, extent and heading, and approach edges by
            the scene file's rule (scene_discovery.floorplan.approach_edges) on the perceived clearance; a surface the
            map did not find keeps the truth, listed in `meta["perceived"]["surfaces_kept"]`
  objects   each free object's `init_pos` becomes its object node's centre (the map's memory of where it is), where
            the map has one; the others keep the truth, listed in `meta["perceived"]["objects_kept"]`
"""

from __future__ import annotations

import base64
import copy

import numpy as np
from scipy.ndimage import distance_transform_edt

PARTS = ("grid", "surfaces", "objects")


def raster(blocked: np.ndarray, origin, res: float) -> dict:
    """The navigator's raster form of a blocked-cell mask ([x, y]-indexed): navigator.OccupancyMap.add_raster."""
    bits = np.packbits(np.asarray(blocked, bool).ravel())
    return dict(origin=[float(v) for v in origin], res=float(res), shape=list(blocked.shape),
                bits=base64.b64encode(bits.tobytes()).decode())  # fmt: skip


class _Grid:
    """scene_discovery.floorplan's grid interface (to_px: xy -> (column, row)) over an [x, y]-indexed array."""

    def __init__(self, origin, res: float):
        self.origin, self.res = np.asarray(origin, float), float(res)

    def to_px(self, xy):
        return (np.asarray(xy, float) - self.origin) / self.res


def edges_of(surface: dict, clear: np.ndarray, origin, res: float, body_radius: float) -> list[str]:
    from locomani.scene_discovery.floorplan import approach_edges

    rect = dict(xy=surface["xy"], half_extent=surface["half_extent"], yaw=float(surface.get("yaw", 0.0)))
    return approach_edges(rect, clear.T, _Grid(origin, res), body_radius)  # floorplan's images are [row=y, col=x]


def perceived_meta(meta: dict, grid: dict, surfaces: list[dict], match: dict[str, int], nodes: dict[str, list],
                   *, parts=PARTS, body_radius: float = 0.35) -> dict:  # fmt: skip
    """`grid`: map_build's grid.npz arrays (obstacle, free, origin, res); `surfaces`: its perceived patches;
    `match`: true surface key -> index of the perceived patch matched to it; `nodes`: object key -> [x, y, z]."""
    bad = set(parts) - set(PARTS)
    if bad:
        raise ValueError(f"unknown parts {sorted(bad)}; known {PARTS}")
    out = copy.deepcopy(meta)
    origin, res = np.asarray(grid["origin"], float), float(grid["res"])
    blocked = np.asarray(grid["obstacle"], bool) | ~np.asarray(grid["free"], bool)
    clear = distance_transform_edt(~blocked, sampling=res)
    info = dict(parts=list(parts), surfaces_kept=[], objects_kept=[])
    if "grid" in parts:
        out["room"] = dict(out["room"], walls=[], obstacles=[], blocked=raster(blocked, origin, res))
        out["objects"] = {k: o for k, o in out["objects"].items() if o.get("free")}
    if "surfaces" in parts:
        for k, s in out["surfaces"].items():
            if k not in match:
                info["surfaces_kept"].append(k)
                continue
            p = surfaces[match[k]]
            he = [float(p["half_extent"][0]), float(p["half_extent"][1])]
            s.update(xy=list(p["xy"]), top=float(p["top"]), half_extent=he, flat_half_extent=he, yaw=float(p["yaw"]),
                     circumradius=float(np.hypot(*he)), area=float(4 * he[0] * he[1]))  # fmt: skip
            s["approach_edges"] = edges_of(s, clear, origin, res, body_radius)
    if "objects" in parts:
        for k, o in out["objects"].items():
            if not o.get("free"):
                continue
            if k in nodes:
                o["init_pos"] = [float(v) for v in nodes[k]]
            else:
                info["objects_kept"].append(k)
    out["perceived"] = info
    return out


def with_free_disc(meta: dict, xy, radius: float) -> dict:
    """A copy of a perceived description whose raster has the cells within `radius` of `xy` free: the floor a round
    starts on (the robot stands there and its first looks see around it; the tour need not have)."""
    r = meta["room"].get("blocked")
    if not r:
        return meta
    nx, ny = (int(v) for v in r["shape"])
    mask = np.unpackbits(np.frombuffer(base64.b64decode(r["bits"]), np.uint8), count=nx * ny).reshape(nx, ny)
    ij = np.stack(np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij"), -1)
    centres = np.asarray(r["origin"], float) + (ij + 0.5) * float(r["res"])
    mask[np.linalg.norm(centres - np.asarray(xy, float)[:2], axis=-1) <= radius] = 0
    out = dict(meta, room=dict(meta["room"], blocked=raster(mask.astype(bool), r["origin"], r["res"])))
    return out
