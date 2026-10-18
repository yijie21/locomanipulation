"""Support surfaces from a mesh's upward-facing triangles: table tops, counter tops, shelf boards.

Input is world-frame triangles (z up) of one piece of furniture; output is one oriented rectangle per flat,
large-enough horizontal region: `{top, xy, half_extent, yaw, area, fill}`. Heights are clustered (a top
modelled as several slightly offset sheets is one surface), each height level is rasterised at `res` and split
into connected regions, and each region gets its minimum-area rectangle. `fill` (region area over rectangle
area) says how well the rectangle describes it; an L-shaped desk scores low and is dropped below `min_fill`.
Pure numpy / cv2; the glb reading is in `scripts/adt_to_scene.py`.
"""

from __future__ import annotations

import cv2
import numpy as np


def horizontal_regions(
    tris: np.ndarray,
    min_normal_z: float = 0.97,
    gap: float = 0.015,
    res: float = 0.01,
    min_area: float = 0.04,
    min_side: float = 0.15,
    z_range: tuple = (0.10, 1.30),
    min_fill: float = 0.6,
) -> list[dict]:
    tris = np.asarray(tris, dtype=float).reshape(-1, 3, 3)
    if len(tris) == 0:
        return []
    normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(normal, axis=1)
    up = (length > 0) & (normal[:, 2] / np.maximum(length, 1e-12) >= min_normal_z)
    z = tris[:, :, 2].mean(axis=1)
    up &= (z >= z_range[0]) & (z <= z_range[1])
    tris, z = tris[up], z[up]
    if len(tris) == 0:
        return []
    order = np.argsort(z)
    breaks = np.flatnonzero(np.diff(z[order]) > gap) + 1
    out = []
    for group in np.split(order, breaks):
        out.extend(_regions(tris[group], float(np.median(z[group])), res, min_area, min_side, min_fill))
    return out


def _regions(tris, top, res, min_area, min_side, min_fill) -> list[dict]:
    xy = tris[:, :, :2]
    lo = xy.reshape(-1, 2).min(axis=0) - 2 * res
    shape = tuple((np.ceil((xy.reshape(-1, 2).max(axis=0) - lo) / res) + 4).astype(int))[::-1]
    img = np.zeros(shape, np.uint8)
    for tri in np.round((xy - lo) / res).astype(np.int32):  # one call per triangle: fillPoly is even-odd over a list
        cv2.fillConvexPoly(img, tri, 1)
    n, comps, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    out = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA] * res * res
        if area < min_area:
            continue
        pts = np.column_stack(np.nonzero(comps == k)[::-1]).astype(np.float32)  # (col, row)
        (cx, cy), (w, h), angle = cv2.minAreaRect(pts)
        w, h = (w + 1) * res, (h + 1) * res  # pixel centres -> covered extent
        if min(w, h) < min_side or area / (w * h) < min_fill:
            continue
        yaw = float(np.radians(angle))
        if w < h:  # x along the long side, yaw in (-pi/2, pi/2]
            w, h, yaw = h, w, yaw + np.pi / 2
        yaw = float((yaw + np.pi / 2) % np.pi - np.pi / 2)
        out.append(
            {
                "top": round(top, 4),
                "xy": [round(float(v), 4) for v in (np.array([cx, cy]) + 0.5) * res + lo],
                "half_extent": [round(w / 2, 4), round(h / 2, 4)],
                "yaw": round(yaw, 4),
                "area": round(float(area), 4),
                "fill": round(float(area / (w * h)), 3),
            }
        )
    return out


def _corners(r: dict) -> np.ndarray:
    c, s = np.cos(r["yaw"]), np.sin(r["yaw"])
    hx, hy = r["half_extent"]
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    return local @ np.array([[c, s], [-s, c]]) + np.asarray(r["xy"], dtype=float)


def _gap(a: dict, b: dict) -> float:
    """Distance between two oriented rectangles' outlines (0 when they touch or overlap)."""
    pa, pb = _corners(a), _corners(b)
    if cv2.intersectConvexConvex(pa.astype(np.float32), pb.astype(np.float32))[0] > 0:
        return 0.0
    return min(
        abs(cv2.pointPolygonTest(pb.astype(np.float32), (float(x), float(y)), True)) for x, y in pa
    )  # corner-to-outline is enough for the adjacent steps this is for


def drop_stairs(
    regions: list[dict], riser: tuple = (0.10, 0.25), touch: float = 0.05, min_steps: int = 3
) -> list[dict]:
    """Remove flights of stairs: >= `min_steps` touching regions that, sorted by height, rise one riser at a time.

    A pair alone is not enough: a sink basin sits a riser below the counter on both sides of it, and that
    counter is exactly the surface this module exists to find."""
    n = len(regions)
    linked = [[False] * n for _ in range(n)]
    for i, a in enumerate(regions):
        for j in range(i + 1, n):
            b = regions[j]
            if riser[0] <= abs(a["top"] - b["top"]) <= riser[1] and _gap(a, b) <= touch:
                linked[i][j] = linked[j][i] = True
    seen, stairs = set(), set()
    for start in range(n):
        if start in seen:
            continue
        comp, todo = set(), [start]
        while todo:
            k = todo.pop()
            if k not in comp:
                comp.add(k)
                todo.extend(j for j in range(n) if linked[k][j])
        seen |= comp
        flight = sorted(comp, key=lambda k: regions[k]["top"])
        if len(flight) >= min_steps and all(linked[a][b] for a, b in zip(flight, flight[1:])):
            stairs |= comp
    return [r for k, r in enumerate(regions) if k not in stairs]


SHADOW_SHARE = 0.9  # of a surface's area under another surface ...
SHADOW_GAP = 0.10  # ... less than this many metres higher: nothing can rest on it


def shadowed(surfaces: dict) -> dict[str, str]:
    """{key: the key of the surface over it} for every surface of `surfaces` (scene-file rows: xy, half_extent, yaw,
    top) that lies SHADOW_SHARE or more under another one less than SHADOW_GAP higher (2026-10-18): the ADT kitchen's
    short and medium counters are cabinet tops 3.6 cm under the long counter top, which covers them whole. An object
    put there rests on the higher surface, and on(X, lower) held at the start of 3 of 5 starts of a round-2 task."""
    out = {}
    u = (np.arange(20) + 0.5) / 20 * 2 - 1
    gu, gv = np.meshgrid(u, u)
    for key, s in sorted(surfaces.items()):
        hx, hy = (float(v) for v in s["half_extent"][:2])
        c, sn = np.cos(float(s.get("yaw", 0.0))), np.sin(float(s.get("yaw", 0.0)))
        lx, ly = gu.ravel() * hx, gv.ravel() * hy
        pts = np.stack([s["xy"][0] + c * lx - sn * ly, s["xy"][1] + sn * lx + c * ly], axis=1)
        for k2, t in sorted(surfaces.items()):
            if k2 == key or not 0 < float(t["top"]) - float(s["top"]) < SHADOW_GAP:
                continue
            c2, s2 = np.cos(-float(t.get("yaw", 0.0))), np.sin(-float(t.get("yaw", 0.0)))
            dx, dy = pts[:, 0] - t["xy"][0], pts[:, 1] - t["xy"][1]
            inside = (np.abs(c2 * dx - s2 * dy) <= t["half_extent"][0]) & (
                np.abs(s2 * dx + c2 * dy) <= t["half_extent"][1]
            )
            if inside.mean() >= SHADOW_SHARE:
                out[key] = k2
                break
    return out
