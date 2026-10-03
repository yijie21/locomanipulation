"""Floor plan from wall segments: rooms, spawn regions and which side of a surface a robot can stand at.

Pure raster geometry (numpy / scipy / cv2) so it runs and is tested in the sim venv; the mesh slicing that
produces the wall segments lives in `scripts/adt_to_scene.py`. Arrays are images: row = y index, column = x
index, `origin` is the world xy of cell (0, 0)'s corner. Rooms are the classic segmentation (plan decision 9,
format decision 3): close doorways by eroding the interior, take the large components as room cores, grow
each core back through the free space it can reach; names come later from Claude.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

RES = 0.05  # m per cell, the navigator's resolution


class Grid:
    def __init__(self, bounds: dict, res: float = RES):
        self.res = res
        self.origin = np.array([bounds["x"][0], bounds["y"][0]], dtype=float)
        self.shape = (
            int(np.ceil((bounds["y"][1] - bounds["y"][0]) / res)),
            int(np.ceil((bounds["x"][1] - bounds["x"][0]) / res)),
        )

    def to_px(self, xy) -> np.ndarray:
        """World xy (..., 2) -> pixel (col, row) as float."""
        return (np.asarray(xy, dtype=float) - self.origin) / self.res

    def to_world(self, px) -> np.ndarray:
        """Pixel (col, row) of a cell centre -> world xy."""
        return (np.asarray(px, dtype=float) + 0.5) * self.res + self.origin

    def segments(self, segments, thickness: float) -> np.ndarray:
        img = np.zeros(self.shape, np.uint8)
        width = max(1, int(round(thickness / self.res)))
        for a, b in segments:
            pa, pb = np.round(self.to_px(a)).astype(int), np.round(self.to_px(b)).astype(int)
            cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), 1, width)
        return img.astype(bool)

    def polygons(self, polygons) -> np.ndarray:
        img = np.zeros(self.shape, np.uint8)
        for poly in polygons:
            cv2.fillPoly(img, [np.round(self.to_px(poly)).astype(np.int32)], 1)
        return img.astype(bool)

    def triangles(self, tris) -> np.ndarray:
        """Cells covered by world triangles (..., 3, 2+), one convex fill each (fillPoly over a list is even-odd)."""
        img = np.zeros(self.shape, np.uint8)
        for tri in np.round(self.to_px(np.asarray(tris, dtype=float)[:, :, :2])).astype(np.int32):
            cv2.fillConvexPoly(img, tri, 1)
        return img.astype(bool)

    def contour(self, mask: np.ndarray, simplify: float = RES) -> list[list[float]]:
        """The largest outer contour of `mask` as a world polygon, simplified to `simplify` metres."""
        found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not found:
            return []
        c = max(found, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, simplify / self.res, True)[:, 0, :]
        return [[round(float(v), 3) for v in p] for p in self.to_world(c)]


def interior(walls: np.ndarray) -> np.ndarray:
    """Free cells not connected to the grid border: the inside of a closed outer wall."""
    labels, _ = ndimage.label(~walls)
    border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    return ~walls & ~np.isin(labels, list(border - {0}))


def segment_rooms(
    walls: np.ndarray,
    inside: np.ndarray | None = None,
    res: float = RES,
    door_close: float = 1.0,
    min_area: float = 2.0,
) -> np.ndarray:
    """Room labels (0 = wall / outside / unassigned). Passages narrower than `door_close` m split rooms.

    `inside` (e.g. the rasterised floor) bounds the free space; without it the free space not connected to the
    grid border is used, which fails as soon as the outer wall has an opening (an entrance door)."""
    inside = interior(walls) if inside is None else inside & ~walls
    radius = int(np.ceil(door_close / 2 / res))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    cores = cv2.erode(inside.astype(np.uint8), disk).astype(bool)
    labels, n = ndimage.label(cores)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=np.arange(1, n + 1)) * res * res
    keep = np.zeros(n + 1, dtype=np.int32)
    for new, old in enumerate(np.flatnonzero(sizes >= min_area) + 1, start=1):
        keep[old] = new
    labels = keep[labels]
    cross = ndimage.generate_binary_structure(2, 1)
    while True:  # grow every core back through the free space it reaches, one cell ring per step
        grown = ndimage.grey_dilation(labels, footprint=cross)
        fill = (labels == 0) & inside & (grown > 0)
        if not fill.any():
            return labels
        labels = np.where(fill, grown, labels)


def clearance(obstacles: np.ndarray, res: float = RES) -> np.ndarray:
    """Distance in metres from each cell to the nearest obstacle cell."""
    return ndimage.distance_transform_edt(~obstacles) * res


def spawn_regions(labels: np.ndarray, clear: np.ndarray, need: float, grid: Grid, min_area: float = 0.25) -> dict:
    """Per room label, polygons of cells with at least `need` m of clearance (robot body radius + margin)."""
    out = {}
    for k in np.unique(labels[labels > 0]):
        mask = (labels == k) & (clear >= need)
        comps, n = ndimage.label(mask)
        polys = []
        for c in range(1, n + 1):
            part = comps == c
            if part.sum() * grid.res**2 >= min_area:
                polys.append(grid.contour(part))
        out[int(k)] = polys
    return out


def approach_edges(
    rect: dict,
    clear: np.ndarray,
    grid: Grid,
    body_radius: float,
    standoff: float = 0.05,
    own: np.ndarray | None = None,
    max_overhang: float = 0.6,
) -> list:
    """Edges of a surface rectangle ({xy, half_extent, yaw}) with room for the robot in front of them.

    An edge counts when at least 3 of 5 points spread along it have `body_radius` of clearance once moved out
    past the furniture's own footprint `own` (a seat sits inside a sofa's outline; up to `max_overhang` m) and
    then `body_radius + standoff` further: the robot can stand square to that edge."""
    c, s = np.cos(rect["yaw"]), np.sin(rect["yaw"])
    rot = np.array([[c, -s], [s, c]])
    hx, hy = rect["half_extent"]
    out = []
    for name, normal, along, half_along, half_across in (
        ("+x", (1, 0), (0, 1), hy, hx),
        ("-x", (-1, 0), (0, 1), hy, hx),
        ("+y", (0, 1), (1, 0), hx, hy),
        ("-y", (0, -1), (1, 0), hx, hy),
    ):
        n, a = np.asarray(normal, float), np.asarray(along, float)
        ok = 0
        for t in np.linspace(-0.8, 0.8, 5) * half_along:
            edge = np.asarray(rect["xy"], dtype=float) + (n * half_across + a * t) @ rot.T
            out_dir = n @ rot.T
            past = 0.0
            while own is not None and past < max_overhang and _cell(own, grid, edge + out_dir * past, False):
                past += grid.res
            probe = edge + out_dir * (past + body_radius + standoff)
            if _cell(clear, grid, probe, 0.0) >= body_radius:
                ok += 1
        if ok >= 3:
            out.append(name)
    return out


def _cell(img: np.ndarray, grid: Grid, xy, default):
    col, row = np.floor(grid.to_px(xy)).astype(int)
    return img[row, col] if 0 <= row < img.shape[0] and 0 <= col < img.shape[1] else default
