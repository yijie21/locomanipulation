"""Vision RSI D2 step 2d (docs/handoff/vision-rsi-design-20261023.md section 17.7): the scene description the robot
would have after its mapping tour, in the scene file's own shape, so the skills read it as `world.meta` unchanged.

Each part replaces its truth independently (`parts`), for the swap-back matrix of section 6:
  grid      the walking map: the room's walls and obstacles become the perceived blocked cells (obstacles, and cells
            never seen as floor: section 17.1, rule 4) as polygons; static objects leave the description (the grid
            holds what was seen of them)
  surfaces  every surface the map found takes its perceived centre, height, extent and heading, and approach edges by
            the scene file's rule (scene_discovery.floorplan.approach_edges) on the perceived clearance; a surface the
            map did not find keeps the truth, listed in `meta["perceived"]["surfaces_kept"]`
  objects   each free object's `init_pos` becomes its object node's centre (the map's memory of where it is), where
            the map has one; the others keep the truth, listed in `meta["perceived"]["objects_kept"]`
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

PARTS = ("grid", "surfaces", "objects")


def blocked_polygons(blocked: np.ndarray, origin, res: float) -> list[list[list[float]]]:
    """Outlines of the blocked cells' connected regions, world xy (a polygon per outer contour; holes are free)."""
    img = np.ascontiguousarray(blocked.T.astype(np.uint8))  # rows y, columns x
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if len(c) < 3:
            ij = c[:, 0, :]
            lo, hi = ij.min(0), ij.max(0) + 1
            c = np.array([[lo], [[hi[0], lo[1]]], [hi], [[lo[0], hi[1]]]])
        xy = np.asarray(origin, float) + (c[:, 0, :] + 0.5) * res
        out.append(xy.round(4).tolist())
    return out


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
        out["room"] = dict(out["room"], walls=[], obstacles=blocked_polygons(blocked, origin, res))
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
