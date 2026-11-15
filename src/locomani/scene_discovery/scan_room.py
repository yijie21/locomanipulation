"""The `room` block of a scanned scene (scene-file format §4, docs/handoff/scene-file-format-20261003.md) from the
meshes themselves (TODO 38.1). The ADT apartment gets its walls, obstacles and spawn regions from the dataset's
floor plan; a scanned room (the logistics room, assets/scene/logistics_room_box_v1) has nothing but visual meshes,
none of which collide, so the robot would walk through the racks and the start sampler would have no spawn region.

`slice_occupancy` cuts every non-robot, non-free geom at the heights the body sweeps (BAND_M) into a 5 cm top-down
grid, sampling the mesh triangles' edges (a rack's side panel is two triangles: its vertices alone leave the middle
empty). `room_block` turns that grid into the scene-file fields: the occupied cells as `obstacles` (axis-aligned
rectangles the navigator rasterizes back), the bounds, and as `spawn` the free cells at least body radius + margin
from anything, in the connected region the robot start stands in. `walls` stays empty: a scan does not tell a wall
from a rack, both are in `obstacles`."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

RES_M = 0.05  # the navigator's cell (primitives.navigator.RESOLUTION)
BAND_M = (0.10, 1.50)  # heights the standing robot's body and arms sweep; floors below, lights and ceiling above
STEP_M = 0.025  # along a triangle edge: half a cell, so no cell between two samples is skipped
SPAWN_MARGIN_M = 0.10  # beyond the body radius, so a spawn is never on the C-space boundary
MAX_SAMPLES = 400  # per edge (10 m at STEP_M); longer edges are the shell's and get this many


def _sample_edges(a: np.ndarray, b: np.ndarray, band: tuple[float, float], step: float) -> np.ndarray:
    """Points every `step` along the segments a[i] -> b[i] that cross the height band, those inside the band."""
    zlo, zhi = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
    keep = (zhi > band[0]) & (zlo < band[1])
    a, b = a[keep], b[keep]
    if not len(a):
        return np.zeros((0, 3))
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / step)).astype(int)
    out = [a, b]
    for k in range(1, min(int(n.max()), MAX_SAMPLES)):
        m = n > k
        out.append(a[m] + (b[m] - a[m]) * (k / n[m])[:, None])
    p = np.concatenate(out)
    return p[(p[:, 2] > band[0]) & (p[:, 2] < band[1])]


_BOX_EDGES = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]


def slice_occupancy(model, data, *, band=BAND_M, res: float = RES_M, step: float = STEP_M):
    """(occupied cells [nx, ny] bool, grid origin xy) of the scene's fixed geometry inside the height band: every geom
    outside the robot (the `pelvis` subtree) and outside free bodies (they move; the start sampler places them),
    meshes along their triangle edges, boxes along their 12 edges. Other primitives (planes, spheres, capsules) are
    skipped. The origin is a multiple of `res` so the navigator's cells line up with these."""
    import mujoco

    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    free = {int(model.jnt_bodyid[j]) for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE}

    def skipped(b: int) -> bool:
        while b != 0:
            if b == pelvis or b in free:
                return True
            b = int(model.body_parentid[b])
        return False

    pts = []
    for g in range(model.ngeom):
        if skipped(int(model.geom_bodyid[g])):
            continue
        rot, pos = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(model.geom_dataid[g])
            v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            f = model.mesh_face[model.mesh_faceadr[mid] : model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
            w = v @ rot.T + pos
            a = np.concatenate([w[f[:, 0]], w[f[:, 1]], w[f[:, 2]]])
            b = np.concatenate([w[f[:, 1]], w[f[:, 2]], w[f[:, 0]]])
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            s = model.geom_size[g]
            c = np.array([[sx, sy, sz] for sx in (-s[0], s[0]) for sy in (-s[1], s[1]) for sz in (-s[2], s[2])])
            w = c @ rot.T + pos
            a, b = w[[i for i, _ in _BOX_EDGES]], w[[j for _, j in _BOX_EDGES]]
        else:
            continue
        pts.append(_sample_edges(a, b, band, step))
    p = np.concatenate(pts) if pts else np.zeros((0, 3))
    if not len(p):
        raise ValueError("no fixed geometry inside the height band")
    lo = np.floor(p[:, :2].min(axis=0) / res) * res - res
    hi = np.ceil(p[:, :2].max(axis=0) / res) * res + res
    nx, ny = int(round((hi[0] - lo[0]) / res)), int(round((hi[1] - lo[1]) / res))
    occ = np.zeros((nx, ny), dtype=bool)
    ij = ((p[:, :2] - lo) / res).astype(int)
    occ[ij[:, 0], ij[:, 1]] = True
    return occ, lo


def rectangles(mask: np.ndarray, lo, res: float) -> list[list[list[float]]]:
    """Axis-aligned rectangles (4-point polygons, metres) covering exactly the true cells of `mask`: runs along y in
    each x column, merged across neighbouring columns with the same run."""
    rects: list[tuple[int, int, int, int]] = []  # i0, i1 (inclusive), j0, j1 (inclusive)
    open_runs: dict[tuple[int, int], int] = {}
    nx = mask.shape[0]
    for i in range(nx + 1):
        runs = set()
        if i < nx:
            col = np.flatnonzero(mask[i])
            if len(col):
                breaks = np.flatnonzero(np.diff(col) > 1)
                starts = np.concatenate([[col[0]], col[breaks + 1]])
                ends = np.concatenate([col[breaks], [col[-1]]])
                runs = set(zip(starts.tolist(), ends.tolist()))
        nxt = {r: open_runs.pop(r, i) for r in runs}
        rects += [(i0, i - 1, j0, j1) for (j0, j1), i0 in open_runs.items()]
        open_runs = nxt
    x0, y0 = float(lo[0]), float(lo[1])
    out = []
    for i0, i1, j0, j1 in sorted(rects):
        xa, xb = round(x0 + i0 * res, 4), round(x0 + (i1 + 1) * res, 4)
        ya, yb = round(y0 + j0 * res, 4), round(y0 + (j1 + 1) * res, 4)
        out.append([[xa, ya], [xb, ya], [xb, yb], [xa, yb]])
    return out


def room_block(
    occ: np.ndarray, lo, res: float, start_xy, *, body_radius: float, margin: float = SPAWN_MARGIN_M,
    name: str = "room", band=BAND_M,
) -> dict:  # fmt: skip
    """The scene file's `room` from a top-down occupancy grid: `bounds` the occupied extent, `obstacles` the occupied
    cells as rectangles, `spawn[name]` the cells at least body_radius + margin clear of them in the connected free
    region holding `start_xy` (or the free cell nearest it), `rooms[name]` that region's bounding rectangle. `walls`
    and `doors` are empty: a scan's walls are in `obstacles`."""
    from scipy.ndimage import distance_transform_edt, label

    lo = np.asarray(lo, float)
    clear = distance_transform_edt(~occ, sampling=res)
    free = clear >= body_radius + margin
    labels, n = label(free)
    if n == 0:
        raise ValueError("no cell is clear enough to stand on")
    cell = np.clip(((np.asarray(start_xy, float) - lo) / res).astype(int), 0, np.array(occ.shape) - 1)
    k = int(labels[cell[0], cell[1]])
    if k == 0:  # the start itself is not clear enough: the nearest clear cell's region
        fi, fj = np.nonzero(free)
        nearest = int(np.argmin((fi - cell[0]) ** 2 + (fj - cell[1]) ** 2))
        k = int(labels[fi[nearest], fj[nearest]])
    region = labels == k
    oi, oj = np.nonzero(occ)
    ri, rj = np.nonzero(region)
    bounds = {"x": [round(float(lo[0] + oi.min() * res), 4), round(float(lo[0] + (oi.max() + 1) * res), 4)],
              "y": [round(float(lo[1] + oj.min() * res), 4), round(float(lo[1] + (oj.max() + 1) * res), 4)]}  # fmt: skip
    xa, xb = lo[0] + ri.min() * res - body_radius, lo[0] + (ri.max() + 1) * res + body_radius
    ya, yb = lo[1] + rj.min() * res - body_radius, lo[1] + (rj.max() + 1) * res + body_radius
    polygon = [[round(float(x), 4), round(float(y), 4)] for x, y in ((xa, ya), (xb, ya), (xb, yb), (xa, yb))]
    return {
        "bounds": bounds,
        "walls": [],
        "doors": [],
        "obstacles": rectangles(occ, lo, res),
        "rooms": {name: {"polygon": polygon, "name_source": "rule"}},
        "spawn": {name: rectangles(region, lo, res)},
        "slice": {
            "band_m": list(band),
            "res_m": res,
            "body_radius_m": body_radius,
            "margin_m": margin,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },  # fmt: skip
    }
