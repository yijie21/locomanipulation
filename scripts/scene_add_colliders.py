#!/usr/bin/env python3
"""Give a built ADT scene colliders for its walls and for its furniture above the lowest surface (user 2026-10-19,
TODO 27.13; plan docs/handoff/scene-rsi-plan-20261016.md §14; scene file format §3). Runs in /venv/main (trimesh), after
`scene_add_panels.py`.

    /venv/main/bin/python scripts/scene_add_colliders.py --scene outputs/scene_discovery/adt_apartment_mars \\
        [--approach-edges outputs/scene_rsi_20261016/capability/stance/adt_apartment_mars.json]

The robot passed through both in the report videos: an arm through the bathroom wall to the sink, the head into a
bookcase, an arm through a chair back. The shell (`adt_shell_*`) collides with nothing, and a furniture piece that holds
surfaces collides with the robot only below the lowest one (`adt_to_scene.furniture_solid`: a convex hull above it fills
the shelves). Both get boxes on a voxel grid, from the source meshes (not the exported visuals: a piece over
FURNITURE_FACES is drawn as its hull):
* walls: the shell's surface between WALL_Z, at WALL_PITCH in the world frame (the apartment's walls are axis-aligned),
  colliding with the robot and the free objects (contype ROBOT_ONLY_BIT | OBJECT_ONLY_BIT);
* furniture: every piece that holds a surface (doors and flat pieces aside, as adt_to_scene), in its body frame at
  FURN_PITCH, from one cell under the top of its collider (`adt_fixcol_*`; a floating shelf has none: from
  SURFACE_CLEAR_M under its lowest surface) up, colliding with the robot only (ROBOT_ONLY_BIT), as D3.
Only the surface is voxelized (no fill), so a shelf's or a cabinet's inside stays open. Around every surface the slab
stands for the board (the cells from under the slab to just over its top are cleared), and where the piece closes the
space over the surface (a shelf inside a bookcase or a cabinet: `open_space`), the space in front of each approach edge
is cleared up to that ceiling: the scene treats doors as open (format §4), the glass cabinet's glass doors too. A
surface open above keeps what stands at its edges: a chair's back, the cart's rim and handles. A cell is occupied when the surface passes through it; occupied cells are merged
into boxes greedily (runs along x, then y, then z), and each box is cut down to the surface it holds plus a quarter
cell, at least 2 * MIN_HALF thick (boxes stood up to a cell proud of the surface: 2-3 cm into the objects resting
against a back panel). The walls also stop free objects, so `scene_settle.py --write` runs after this: an object
standing against a wall (the hand soap at the bathroom sink) is pushed clear while damped.

Writes the boxes into `scene_locomani.xml` (walls also into `env_only.xml`) after the shell's last geom and after each
piece's collider, and `<scene>/colliders.json` (counts per piece). Refuses a scene that already has them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh import remesh

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]

import adt_to_scene as A  # noqa: E402

from locomani.scene_discovery import adt, mjcf  # noqa: E402

WALL_Z = (0.10, 2.0)  # m: over door sills and skirting the feet step across, above the robot's raised hands
WALL_PITCH = 0.02  # m
FURN_PITCH = 0.015  # m
FRONT_OPEN_M = 0.06  # m past an approach edge kept open: a door or a glass pane in front of a shelf
OPEN_ABOVE_M = 0.6  # m: how far over a surface its ceiling is looked for
CEIL_MIN = 0.05  # m: nearer than this over a surface is the surface's own lip or an object's, not a ceiling
CEIL_CORE = 0.6  # share of a surface's half extents where its ceiling is looked for (a chair's back is at an edge)
CEIL_COVER = 0.75  # share of that core the ceiling must cover
WALL_BITS = A.ROBOT_ONLY_BIT | A.OBJECT_ONLY_BIT
MIN_HALF = 0.0075  # m: a box is at least 1.5 cm thick, so a finger pressing into it does not come out behind
CHUNK_POINTS = 4_000_000  # subdivided points held at once: the shell's walls are ~2.4e8 (a 6 m sliver face alone 5e5)


def surface_chunks(mesh: trimesh.Trimesh, pitch: float):
    """The mesh's vertices after subdividing it to pitch / 2 (every cell its surface passes through holds some), in
    chunks of whole faces: a face is split into 4 until its longest edge fits, whatever its neighbours do, so the chunks
    together hold the same points as the whole mesh subdivided at once (a shared edge's points once per side)."""
    tri = mesh.vertices[mesh.faces]
    longest = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).max(1)
    n = 2.0 ** np.ceil(np.log2(np.maximum(longest / (pitch / 2), 1)))
    count = np.cumsum((n + 1) * (n + 2) / 2)
    starts = np.searchsorted(count, np.arange(0, count[-1], CHUNK_POINTS), side="right") if len(n) else []
    for s, e in zip(starts, [*starts[1:], len(n)]):
        if e > s:
            yield remesh.subdivide_to_size(mesh.vertices, mesh.faces[s:e], max_edge=pitch / 2, max_iter=30)[0]


def cell_bounds(cell: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per distinct cell: the cell, and the min of `lo` / max of `hi` over its rows."""
    idx, inv = np.unique(cell, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cmin = np.full((len(idx), 3), np.inf)
    cmax = np.full((len(idx), 3), -np.inf)
    np.minimum.at(cmin, inv, lo)
    np.maximum.at(cmax, inv, hi)
    return idx, cmin, cmax


def scan(mesh: trimesh.Trimesh, pitch: float, origin: np.ndarray, frames: list[tuple], to_world):
    """One pass over the mesh's surface points (surface_chunks), keeping only what boxes() and open_space() read: the
    occupied cells with their points' bounds (the points' frame), and per surface in `frames` the lowest point
    `to_world(points)` puts in each of its 4 x 4 core cells (ceiling_lows)."""
    cells, lows = [], np.full((len(frames), 16), np.inf)
    for pts in surface_chunks(mesh, pitch):
        cells.append(cell_bounds(np.floor((pts - origin) / pitch).astype(np.int64), pts, pts))
        lows = np.minimum(lows, ceiling_lows(frames, to_world(pts)))
    if not cells:
        return (np.zeros((0, 3), np.int64), np.zeros((0, 3)), np.zeros((0, 3))), lows
    return cell_bounds(*(np.concatenate(c) for c in zip(*cells))), lows


def merge(idx: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Greedy boxes over occupied cells: [(lo, hi)] cell ranges (hi exclusive; runs along x, then y, then z) and each
    cell's box."""
    if len(idx) == 0:
        return [], np.zeros(0, np.int64)
    base = idx.min(0)
    shape = idx.max(0) - base + 1
    at = tuple((idx - base)[:, ::-1].T)
    occ = np.zeros(shape[::-1], bool)  # (z, y, x): x varies fastest, so it is extended first
    occ[at] = True
    free = occ.copy()
    label = np.full(occ.shape, -1, np.int64)
    nz, ny, nx = occ.shape
    out = []
    for z, y, x in np.argwhere(occ):
        if not free[z, y, x]:
            continue
        x1 = x + 1
        while x1 < nx and free[z, y, x1]:
            x1 += 1
        y1 = y + 1
        while y1 < ny and free[z, y1, x:x1].all():
            y1 += 1
        z1 = z + 1
        while z1 < nz and free[z1, y:y1, x:x1].all():
            z1 += 1
        free[z:z1, y:y1, x:x1] = False
        label[z:z1, y:y1, x:x1] = len(out)
        out.append((base + [x, y, z], base + [x1, y1, z1]))
    return out, label[at]


def boxes(cells: tuple, pitch: float, origin: np.ndarray, keep) -> list[tuple[np.ndarray, np.ndarray]]:
    """(centre, half extents) of the boxes over the occupied cells (scan: cells and their points' bounds) whose centres
    `keep` (a function of the centres) keeps, in the points' frame. A box is its cells cut down to its points' bounds
    plus pitch / 4 (closing the gaps to the boxes beside it), and at least 2 * MIN_HALF thick: one face of a wall alone
    is a flat set of points."""
    idx, cmin, cmax = cells
    kept = keep(origin + (idx + 0.5) * pitch)
    merged, ids = merge(idx[kept])
    if not merged:
        return []
    vmin = np.full((len(merged), 3), np.inf)
    vmax = np.full((len(merged), 3), -np.inf)
    np.minimum.at(vmin, ids, cmin[kept])
    np.maximum.at(vmax, ids, cmax[kept])
    out = []
    for (lo, hi), a, b in zip(merged, vmin, vmax):
        lo = np.maximum(origin + lo * pitch, a - pitch / 4)
        hi = np.minimum(origin + hi * pitch, b + pitch / 4)
        out.append(((lo + hi) / 2, np.maximum((hi - lo) / 2, MIN_HALF)))
    return out


def surface_frames(meta: dict, of: str) -> list[tuple]:
    """Per surface of `of`: (key, xy, yaw, half extents, top)."""
    return [
        (sk, np.asarray(sf["xy"], float), float(sf.get("yaw", 0.0)), np.asarray(sf["half_extent"][:2], float),
         float(sf["top"]))
        for sk, sf in sorted(meta["surfaces"].items())
        if sf.get("of_object") == of
    ]  # fmt: skip


def ceiling_lows(frames: list[tuple], points: np.ndarray) -> np.ndarray:
    """(len(frames), 16): per surface, the lowest of `points` (world frame) over the middle of the surface (CEIL_CORE of
    its half extents), CEIL_MIN to OPEN_ABOVE_M above it, in each cell of a 4 x 4 grid there (inf: none)."""
    out = np.full((len(frames), 16), np.inf)
    for low, (_sk, xy, yaw, half, top) in zip(out, frames):
        lx, ly = local(points, xy, yaw)
        z = points[:, 2]
        core = (np.abs(lx) <= CEIL_CORE * half[0]) & (np.abs(ly) <= CEIL_CORE * half[1])
        core &= (z > top + CEIL_MIN) & (z <= top + OPEN_ABOVE_M)
        gx = np.clip(((lx[core] / (CEIL_CORE * half[0]) + 1) * 2).astype(int), 0, 3)
        gy = np.clip(((ly[core] / (CEIL_CORE * half[1]) + 1) * 2).astype(int), 0, 3)
        np.minimum.at(low, gx * 4 + gy, z[core])
    return out


def open_space(meta: dict, overlay: dict, frames: list[tuple], lows: np.ndarray) -> list[tuple]:
    """Per surface in `frames`: (key, xy, yaw, half extents, top, ceiling, approach edges). `ceiling` is how high the
    space over the surface is closed: where the piece's surface reaches at least CEIL_COVER of the surface's 4 x 4 grid
    (`lows`, ceiling_lows), the median of their lowest points, else the surface listed over it (mjcf.ceiling) -- None
    for one open above: a seat with its back at an edge, a table, the cart's top tier."""
    out = []
    for (sk, xy, yaw, half, top), low in zip(frames, lows):
        ceil = float(np.median(low[np.isfinite(low)])) if np.isfinite(low).mean() >= CEIL_COVER else None
        if ceil is None and mjcf.ceiling(meta["surfaces"], sk) is not None:
            ceil = mjcf.ceiling(meta["surfaces"], sk) - mjcf.SLAB
        sf = meta["surfaces"][sk]
        edges = set(sf.get("approach_edges") or ()) | set((overlay.get(sk) or {}).get("approach_edges") or ())
        out.append((sk, xy, yaw, half, top, ceil, sorted(edges)))
    return out


def local(points: np.ndarray, xy: np.ndarray, yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Points (world frame) in a slab's frame: (x, y)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = points[:, :2] - xy
    return c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]


def keep_mask(world: np.ndarray, pitch: float, spaces: list[tuple]) -> np.ndarray:
    """False for cell centres (world frame) inside a surface's board, or in front of its approach edges up to its
    ceiling."""
    keep = np.ones(len(world), bool)
    for _sk, xy, yaw, half, top, ceil, edges in spaces:
        (lx, ly), z = local(world, xy, yaw), world[:, 2]
        board = (np.abs(lx) <= half[0] + pitch) & (np.abs(ly) <= half[1] + pitch)
        board &= (z >= top - mjcf.SLAB - pitch) & (z <= top + pitch)
        keep &= ~board
        for e in edges if ceil is not None else ():  # a door only closes a space with something over it
            ax, sign = (lx, ly) if e[1] == "x" else (ly, lx), 1.0 if e[0] == "+" else -1.0
            h_along, h_across = (half[0], half[1]) if e[1] == "x" else (half[1], half[0])
            front = (sign * ax[0] >= h_along - pitch) & (sign * ax[0] <= h_along + FRONT_OPEN_M)
            front &= np.abs(ax[1]) <= h_across
            front &= (z > top + pitch) & (z <= ceil - pitch)
            keep &= ~front
    return keep


def wall_boxes(objs: dict, meta: dict, overlay: dict) -> tuple[list, list]:
    """The walls' boxes (world frame) and the shell's surfaces' open spaces."""
    shell = objs[adt.SHELL]
    parts = []
    for part in A._parts(shell):
        lo, hi = part.bounds
        if hi[2] < WALL_Z[0] or lo[2] > WALL_Z[1]:
            continue
        z = part.vertices[part.faces][:, :, 2]  # the faces that reach into the band (cells outside it are dropped)
        part = trimesh.Trimesh(
            part.vertices, part.faces[(z.max(1) >= WALL_Z[0]) & (z.min(1) <= WALL_Z[1])], process=False
        )
        if len(part.faces):
            parts.append(part)
    shell = trimesh.util.concatenate(parts)
    origin = np.floor(shell.vertices[shell.faces].reshape(-1, 3).min(0) / WALL_PITCH) * WALL_PITCH  # = the points' min
    frames = surface_frames(meta, adt.SHELL)
    cells, lows = scan(shell, WALL_PITCH, origin, frames, lambda p: p)
    spaces = open_space(meta, overlay, frames, lows)

    def keep(c):
        return keep_mask(c, WALL_PITCH, spaces) & (c[:, 2] >= WALL_Z[0]) & (c[:, 2] <= WALL_Z[1])

    return boxes(cells, WALL_PITCH, origin, keep), spaces


def furniture_boxes(key: str, a: adt.AdtObject, o: dict, meta: dict, overlay: dict, floor_z: float):
    """Body-frame boxes of piece `key` above `floor_z` (body frame: the top of its existing collider), and its surfaces'
    open spaces."""
    raw = A.load_mesh(key)
    body = raw.copy()
    body.vertices = raw.vertices @ a.body_axes.T + np.asarray(a.body()["mesh_offset"])
    origin = np.floor(body.bounds[0] / FURN_PITCH) * FURN_PITCH
    rot, pos = adt.quat_to_matrix(o["init_quat"]), np.asarray(o["init_pos"], float)
    frames = surface_frames(meta, key)
    cells, lows = scan(body, FURN_PITCH, origin, frames, lambda p: pos + p @ rot.T)
    spaces = open_space(meta, overlay, frames, lows)

    def keep(c):
        return (c[:, 2] >= floor_z - FURN_PITCH) & keep_mask(pos + c @ rot.T, FURN_PITCH, spaces)

    return boxes(cells, FURN_PITCH, origin, keep), spaces


def geom_xml(name: str, centre, half, bits: int, rgba: str, quat=None) -> str:
    q = f' quat="{mjcf.fmt(quat)}"' if quat is not None else ""
    return (
        f'<geom name="{name}" type="box" pos="{mjcf.fmt(np.round(centre, 4))}"{q} size="{mjcf.fmt(np.round(half, 4))}" '
        f'contype="{bits}" conaffinity="0" group="3" density="0" rgba="{rgba}" />'
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--approach-edges", type=Path, help="the stance overlay (stance_edges.py): more edges kept open")
    a = ap.parse_args(argv)
    meta = json.loads((a.scene / "scene_meta.json").read_text())
    overlay = json.loads(a.approach_edges.read_text())["surfaces"] if a.approach_edges else {}
    objs = {o.name: o for o in adt.read_sequence(A.DATA / "sequences" / meta["source"]["sequence"])}
    xml = (a.scene / "scene_locomani.xml").read_text()
    if "adt_wallcol_" in xml or "adt_upcol_" in xml:
        raise SystemExit(f"{a.scene / 'scene_locomani.xml'} already has wall / furniture colliders")

    walls, spaces = wall_boxes(objs, meta, overlay)
    ceilings = {sp[0]: sp[5] for sp in spaces}
    print(f"[colliders] walls: {len(walls)} boxes", flush=True)
    wall_xml = "".join(
        geom_xml(f"adt_wallcol_{i:04d}", c, h, WALL_BITS, "0.6 0.6 0.9 0.3") for i, (c, h) in enumerate(walls)
    )
    report: dict = {"walls": len(walls), "furniture": {}, "params": dict(
        wall_z=WALL_Z, wall_pitch=WALL_PITCH, furn_pitch=FURN_PITCH, front_open_m=FRONT_OPEN_M,
        open_above_m=OPEN_ABOVE_M, approach_edges=str(a.approach_edges) if a.approach_edges else None)}  # fmt: skip
    pieces: dict[str, str] = {}
    hosts = sorted({s["of_object"] for s in meta["surfaces"].values()} - {adt.SHELL})
    for key in hosts:
        o = meta["objects"][key]
        if objs[key].category == "door" or 2 * o["box"]["half_extent"][2] < A.FURNITURE_MIN_THICKNESS_M:
            continue  # adt_to_scene gives these no collider either
        col = a.scene / "mesh" / f"adt_fixcol_{key}.obj"
        if col.exists():  # from the top of the collider under the lowest surface
            floor_z = float(trimesh.load(col, force="mesh", process=False).bounds[1][2])
        else:  # nothing under the lowest surface (a floating shelf): from just under it
            lowest = min(float(s["top"]) for s in meta["surfaces"].values() if s["of_object"] == key)
            floor_z = lowest - A.SURFACE_CLEAR_M - float(o["init_pos"][2])
        bx, spaces = furniture_boxes(key, objs[key], o, meta, overlay, floor_z)
        ceilings.update({sp[0]: sp[5] for sp in spaces})
        rot, pos = adt.quat_to_matrix(o["init_quat"]), np.asarray(o["init_pos"], float)
        pieces[key] = "".join(
            geom_xml(f"adt_upcol_{key}_{i:03d}", pos + rot @ c, h, A.ROBOT_ONLY_BIT, "0.9 0.6 0.2 0.3", o["init_quat"])
            for i, (c, h) in enumerate(bx)
        )
        report["furniture"][key] = len(bx)
        print(f"[colliders] {key}: {len(bx)} boxes", flush=True)

    for name in ("scene_locomani.xml", "env_only.xml"):
        path = a.scene / name
        s = path.read_text()
        last = list(re.finditer(r'<geom name="adt_shell_\d+" [^>]*/>', s))[-1]
        s = s[: last.end()] + wall_xml + s[last.end() :]
        if name == "scene_locomani.xml":
            for key, g in pieces.items():
                m = re.search(rf'<geom name="adt_fixcol_{re.escape(key)}" [^>]*/>', s) or re.search(
                    rf'<geom name="adt_fix_{re.escape(key)}" [^>]*/>', s
                )
                s = s[: m.end()] + g + s[m.end() :]
        path.write_text(s)
        print(f"[colliders] {path}: {len(walls)} wall boxes" + (f", {sum(report['furniture'].values())} furniture boxes "
              f"on {len(pieces)} pieces" if name == "scene_locomani.xml" else ""), flush=True)  # fmt: skip
    report["ceilings"] = {k: None if v is None else round(v, 3) for k, v in sorted(ceilings.items())}
    (a.scene / "colliders.json").write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
