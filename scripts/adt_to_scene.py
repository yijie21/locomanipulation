#!/usr/bin/env python3
"""ADT apartment -> scene file v2 (TODO 9.6; docs/handoff/scene-file-format-20261003.md). Runs in /venv/main (trimesh).

    /venv/main/bin/python scripts/adt_to_scene.py geometry \\
        --sequence Apartment_release_clean_seq131_M1292 --out outputs/scene_discovery/adt_apartment

`geometry` writes `<out>/scene_file.json` and `<out>/floorplan.png`: walls from horizontal sections of the
`ApartmentEnv` shell, rooms and spawn regions (`scene_discovery.floorplan`), support surfaces from the upward
faces of every fixed piece of furniture (`scene_discovery.surfaces`), and every object with its rule-derived
fields (`scene_discovery.adt`). Fields Claude fills (names, mass, friction, containers, room names) come in the
next stage; until then the file carries only adt / rule provenance. The output is ADT-derived and stays under
`outputs/` (plan decision 17).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.scene_discovery import adt, floorplan, scene_file, surfaces  # noqa: E402

DATA = Path("/workspace/datasets/adt")
SECTION_HEIGHTS = (0.3, 0.6, 0.9, 1.2, 1.5)  # below windows, above skirting; a wall at any of them blocks
OBSTACLE_Z = (0.05, 1.8)  # furniture that occupies this height band blocks walking
SUPPORT_SKIP = frozenset(
    {
        "shelter",
        "door",
        "wall artwork",
        "curtain",
        "picture",
        "wall clock",
        "hook",
        "thermostat",
        "part of a cabinet/wardrobe",
    }
)  # drawer tops sit inside their cabinet
BODY_RADIUS = 0.26  # navigator.BODY_RADIUS
SPAWN_MARGIN = 0.15


def load_mesh(name: str) -> trimesh.Trimesh:
    return trimesh.load(DATA / "objects" / name / "3d-asset.glb").to_geometry()


def world_mesh(obj: adt.AdtObject) -> trimesh.Trimesh:
    m = load_mesh(obj.name)
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = obj.rot, obj.pos
    m.apply_transform(t)
    return m


def shell_walls(shell: adt.AdtObject):
    """Wall segments from horizontal sections, the floor's triangles, and the shell mesh (sim frame)."""
    mesh = world_mesh(shell)
    segs = []
    for h in SECTION_HEIGHTS:
        lines = trimesh.intersections.mesh_plane(mesh, [0, 0, 1], [0, 0, h])
        segs.extend(lines[:, :, :2].round(3).tolist())
    keep = {tuple(map(tuple, s)) for s in segs if np.hypot(*np.subtract(s[1], s[0])) > 1e-3}
    up = (mesh.face_normals[:, 2] > 0.97) & (np.abs(mesh.triangles_center[:, 2]) < 0.1)
    return [list(map(list, s)) for s in sorted(keep)], mesh.triangles[up], mesh


def hull(points) -> list:
    import cv2

    h = cv2.convexHull(np.asarray(points, dtype=np.float32))[:, 0, :]
    return h.round(4).tolist()


def room_at(labels, grid: floorplan.Grid, xy, names: dict):
    col, row = np.floor(grid.to_px(xy)).astype(int)
    r = int(np.ceil(0.6 / grid.res))
    win = labels[max(row - r, 0) : row + r + 1, max(col - r, 0) : col + r + 1]
    if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1] and labels[row, col] > 0:
        return names[int(labels[row, col])]
    vals = win[win > 0]
    return names[int(np.bincount(vals).argmax())] if vals.size else None


def geometry(args) -> None:
    t0 = time.time()
    capability = yaml.safe_load((REPO / "configs/scene_discovery/capability.yaml").read_text())
    objs = adt.read_sequence(DATA / "sequences" / args.sequence)
    shell = next(o for o in objs if o.name == adt.SHELL)
    objs = [o for o in objs if o.name != adt.SHELL]
    walls, floor_tris, shell_mesh = shell_walls(shell)
    lo, hi = shell_mesh.bounds[0][:2] - 0.5, shell_mesh.bounds[1][:2] + 0.5
    bounds = {
        "x": [round(float(lo[0]), 2), round(float(hi[0]), 2)],
        "y": [round(float(lo[1]), 2), round(float(hi[1]), 2)],
    }
    grid = floorplan.Grid(bounds)
    wall_img = grid.segments(walls, thickness=0.10)
    floor = grid.triangles(floor_tris)
    labels = floorplan.segment_rooms(wall_img, inside=floor)
    room_ids = sorted(int(k) for k in np.unique(labels[labels > 0]))
    names = {k: f"room_{i + 1}" for i, k in enumerate(room_ids)}
    print(f"[adt] {len(walls)} wall segments, {len(room_ids)} rooms, {len(objs)} objects", flush=True)

    fixed = [o for o in objs if not o.movable]
    blocking = [o for o in fixed if o.world_z()[0] < OBSTACLE_Z[1] and o.world_z()[1] > OBSTACLE_Z[0]]
    furniture = grid.polygons([hull(o.footprint()) for o in blocking])
    clear = floorplan.clearance(wall_img | furniture | ~floor)  # off the floor is outside the apartment
    spawn = floorplan.spawn_regions(labels, clear, BODY_RADIUS + SPAWN_MARGIN, grid)

    # the shell carries the built-in counters (and the stairs, dropped); its "own" footprint is the wall raster
    shell_regions = surfaces.drop_stairs(surfaces.horizontal_regions(shell_mesh.triangles, z_range=(0.3, 1.3)))
    candidates, mesh_check = [(shell, shell_regions, wall_img)], []
    for o in fixed:
        if o.category in SUPPORT_SKIP or max(np.ptp(o.footprint(), axis=0)) < 0.3 or o.world_z()[1] < 0.15:
            continue
        m = load_mesh(o.name)
        mesh_check.append(float(np.abs(np.concatenate([m.bounds[0] - o.box_lo, m.bounds[1] - o.box_hi])).max()))
        t = np.eye(4)
        t[:3, :3], t[:3, 3] = o.rot, o.pos
        m.apply_transform(t)
        candidates.append((o, surfaces.horizontal_regions(m.triangles), grid.polygons([hull(o.footprint())])))
    surfs = {}
    for o, regions, own in candidates:
        for i, r in enumerate(regions):
            key = f"{o.name}__{i}"
            r_edges = floorplan.approach_edges(r, clear, grid, BODY_RADIUS, own=own)
            surfs[key] = {
                "name": f"{o.category} {o.name}",
                "xy": r["xy"],
                "top": r["top"],
                "half_extent": r["half_extent"],
                "flat_half_extent": r["half_extent"],
                "circumradius": round(float(np.hypot(*r["half_extent"])), 4),
                "yaw": r["yaw"],
                "of_object": o.name,
                "approach_edges": r_edges,
                "room": room_at(labels, grid, r["xy"], names),
                "area": r["area"],
                "fill": r["fill"],
                "provenance": {"yaw": "rule", "of_object": "rule", "approach_edges": "rule", "room": "rule"},
            }
    print(
        f"[adt] {len(surfs)} surfaces from {len(mesh_check)} furniture meshes; "
        f"mesh vs ADT box max mismatch {max(mesh_check):.3f} m",
        flush=True,
    )

    objects = {}
    for o in [shell] + objs:
        b = o.body()
        centre_world = o.pos + o.rot @ (o.body_axes.T @ np.asarray(b["centre"]))
        underside = o.world_z()[0]
        on = None
        best = 0.05
        for key, s in surfs.items():
            local = np.abs(scene_file.to_surface_frame(centre_world[:2], s)) - np.asarray(s["half_extent"])
            gap = underside - s["top"]
            if np.all(local <= 0.02) and -0.03 <= gap <= best:
                on, best = key, gap
        g = adt.graspable(b["half_extent"], o.movable, capability)
        sem = {
            "asset": f"objects/{o.name}/3d-asset.glb",
            "adt": {
                "instance_name": o.name,
                "prototype_name": o.prototype,
                "category": o.category,
                "motion_type": o.motion,
                "rigidity": o.rigidity,
            },  # fmt: skip
            "room": room_at(labels, grid, centre_world[:2], names),
            "movable": o.movable,
            "upright_axis": b["upright_axis"],
            "graspable": g,
            "support": any(s["of_object"] == o.name for s in surfs.values()),
        }
        prov = {"asset": "adt", "adt": "adt", "room": "rule", "movable": "rule", "upright_axis": "adt",
                "graspable": "rule", "support": "rule"}  # fmt: skip
        if sem["room"] is None:
            del sem["room"], prov["room"]
        objects[o.name] = {
            "surface": on,
            "init_pos": o.pos.round(4).tolist(),
            "init_quat": b["init_quat"],
            "height": round(2 * b["half_extent"][2], 4),
            "box": {"half_extent": b["half_extent"], "centre": b["centre"]},
            "bimanual": g == "bimanual",
            "container": False,  # Claude decides in the properties stage
            "free": o.movable,
            "semantic": sem,
            "provenance": prov,
        }

    rooms = {names[k]: {"polygon": grid.contour(labels == k), "name_source": "rule"} for k in room_ids}
    spawn_named = {names[k]: v for k, v in spawn.items() if v}
    largest = max(spawn_named.items(), key=lambda kv: sum(len(p) for p in kv[1]))
    start = np.asarray(max(largest[1], key=len), dtype=float).mean(axis=0)
    meta = {
        "schema_version": scene_file.SCHEMA_VERSION,
        "source": {
            "kind": "adt",
            "dataset_root": str(DATA),
            "sequence": args.sequence,
            "frame": "first row per object in scene_objects.csv",
            "transform": {"adt_to_sim": adt.ADT_TO_SIM.tolist()},
            "built_by": "scripts/adt_to_scene.py geometry",
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "surfaces": surfs,
        "objects": objects,
        "robot_start": {"xy": start.round(3).tolist(), "yaw_deg": 0.0, "pelvis_z": 0.793},
        "room": {"bounds": bounds, "walls": walls, "doors": [], "rooms": rooms, "spawn": spawn_named},
        "randomization": {
            "movable_pool": sorted(k for k, o in objects.items() if o["free"]),
            "placements": {
                k: sorted(s for s, v in surfs.items() if v["room"] == o["semantic"].get("room"))
                for k, o in objects.items()
                if o["free"]
            },
            "distractor_pool": sorted(k for k, o in objects.items() if o["free"]),
            "spawn_rooms": sorted(spawn_named),
        },
    }
    problems = scene_file.validate(meta)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scene_file.json").write_text(json.dumps(meta, indent=1))
    preview(meta, grid, wall_img, labels, out / "floorplan.png")
    print(f"[adt] wrote {out / 'scene_file.json'} in {time.time() - t0:.0f} s; {len(problems)} validation problem(s)")
    for p in problems[:20]:
        print("   ", p)


def preview(meta, grid, wall_img, labels, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ext = [
        grid.origin[0],
        grid.origin[0] + grid.shape[1] * grid.res,
        grid.origin[1],
        grid.origin[1] + grid.shape[0] * grid.res,
    ]
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.imshow(np.ma.masked_equal(labels, 0), origin="lower", extent=ext, cmap="Pastel1", alpha=0.6)
    ax.imshow(np.ma.masked_equal(wall_img.astype(int), 0), origin="lower", extent=ext, cmap="gray_r", vmin=0, vmax=1)
    for name, r in meta["room"]["rooms"].items():
        p = np.asarray(r["polygon"])
        ax.text(p[:, 0].mean(), p[:, 1].mean(), name, fontsize=14, weight="bold", ha="center")
    for polys in meta["room"]["spawn"].values():
        for p in polys:
            p = np.asarray(p + p[:1])
            ax.plot(p[:, 0], p[:, 1], "g-", lw=0.8)
    for key, s in meta["surfaces"].items():
        c, sn = np.cos(s["yaw"]), np.sin(s["yaw"])
        hx, hy = s["half_extent"]
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]])
        w = local @ np.array([[c, sn], [-sn, c]]) + s["xy"]
        ax.plot(w[:, 0], w[:, 1], "b-", lw=0.8)
        ax.text(s["xy"][0], s["xy"][1], f"{s['top']:.2f}", fontsize=6, color="b", ha="center")
    for o in meta["objects"].values():
        x, y = o["init_pos"][:2]
        ax.plot(x, y, "r." if o["free"] else "k.", ms=3 if o["free"] else 1.5)
    ax.set_aspect("equal")
    ax.set_title("rooms (colour), walls (black), spawn (green), surfaces (blue, top height), movable objects (red)")
    fig.savefig(path, dpi=90, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("geometry", help="walls, rooms, spawn, surfaces, objects -> scene_file.json")
    g.add_argument("--sequence", required=True)
    g.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.stage == "geometry":
        geometry(args)


if __name__ == "__main__":
    main()
