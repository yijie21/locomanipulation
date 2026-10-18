#!/usr/bin/env python3
"""ADT apartment -> scene file v2 (TODO 19.6; docs/handoff/scene-file-format-20261003.md). Runs in /venv/main (trimesh).

    /venv/main/bin/python scripts/adt_to_scene.py geometry \\
        --sequence Apartment_release_clean_seq131_M1292 --out outputs/scene_discovery/adt_apartment

`geometry` writes `<out>/scene_meta.json` (a v2 scene file keeps the legacy name, which the stack and the Isaac
stage look for next to the scene xml) and `<out>/floorplan.png`: walls from horizontal sections of the
`ApartmentEnv` shell, rooms and spawn regions (`scene_discovery.floorplan`), support surfaces from the upward
faces of every fixed piece of furniture (`scene_discovery.surfaces`), and every object with its rule-derived
fields (`scene_discovery.adt`). `estimate --scene <out>/scene_meta.json` then asks Claude (Opus by default) for
names, mass, friction, containers and room names and applies the rules of `scene_discovery.properties`, in place;
every exchange is in `<out>/claude_properties.jsonl`, what the rules changed in `<out>/properties_report.json`.
`mujoco --scene <out>/scene_meta.json` writes `<out>/scene_locomani.xml` (robot from `--base`, made the lab's
real G1 with `--robot mars`, `make_dex1_model.apply_mars`) and `<out>/env_only.xml` (no robot, for
`scripts/scene_drop_test.py`), meshes under `<out>/mesh/`: the shell and furniture as scenery (furniture also as a
convex hull that only the robot collides with, `--furniture-collision`), support surfaces as
slabs, every movable object as a hull (containers as five boxes), free bodies only for `--free` (default: every
graspable movable object resting on a surface). A free body is drawn with its own textured mesh (`free_visuals`);
its hull only collides. Then `scripts/scene_add_panels.py` (the furniture panels free objects rest against) and
`scripts/scene_settle.py --write` (poses the simulator keeps) finish the scene (format §3). A scene on another robot is another directory: copy `scene_meta.json` (the geometry and
estimate stages do not depend on the robot) and run this stage there. The output is ADT-derived and stays under
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

from locomani.scene_discovery import adt, floorplan, mjcf, properties, scene_file, surfaces  # noqa: E402
from locomani.sim import make_dex1_model  # noqa: E402

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
RUNUP = 0.6  # m of clear floor an approach edge needs behind the stand: the shortest spawn the plan runner takes
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


def obstacle_polygons(fixed: list) -> list:
    """Footprints of the fixed objects that occupy the walking height band. Doors are left out: round 1 treats
    every door as open (format §4), and in MuJoCo a door is scenery without contact anyway."""
    return [
        hull(o.footprint())
        for o in fixed
        if o.category != "door" and o.world_z()[0] < OBSTACLE_Z[1] and o.world_z()[1] > OBSTACLE_Z[0]
    ]


def refresh(args) -> None:
    """Recompute the rule-derived object poses, room obstacles and surface approach edges of a scene file in place.

    Everything else -- Claude's names and estimates, the drop test, the MuJoCo indices -- is kept, which a rerun
    of `geometry` would discard. Use it after changing OBSTACLE_Z, RUNUP or the approach-edge rule."""
    path = Path(args.scene)
    meta = json.loads(path.read_text())
    objs = {
        o.name: o
        for o in adt.read_sequence(DATA / "sequences" / meta["source"]["sequence"])
        if o.name in meta["objects"]
    }
    shell = objs.pop(adt.SHELL)
    walls, floor_tris, _shell_mesh = shell_walls(shell)
    grid = floorplan.Grid(meta["room"]["bounds"])
    wall_img = grid.segments(walls, thickness=0.10)
    floor = grid.triangles(floor_tris)
    before = (
        len(meta["room"].get("obstacles") or []),
        sum(bool(s.get("approach_edges")) for s in meta["surfaces"].values()),
    )
    meta["room"]["obstacles"] = obstacle_polygons([o for o in objs.values() if not o.movable])
    for key, o in objs.items():  # the body frame rule (origin at the box centre, adt.AdtObject.body)
        b = o.body()
        meta["objects"][key]["init_pos"] = b["init_pos"]
        meta["objects"][key]["init_quat"] = b["init_quat"]
        meta["objects"][key]["box"].update(half_extent=b["half_extent"], centre=b["centre"])
    clear = floorplan.clearance(wall_img | grid.polygons(meta["room"]["obstacles"]) | ~floor)
    changed = []
    for key, s in meta["surfaces"].items():
        of = s.get("of_object")
        own = wall_img if of == adt.SHELL else grid.polygons([hull(objs[of].footprint())])
        edges = floorplan.approach_edges(s, clear, grid, BODY_RADIUS, own=own, runup=RUNUP)
        if edges != s.get("approach_edges"):
            changed.append(f"{key}: {s.get('approach_edges')} -> {edges}")
        s["approach_edges"] = edges
    path.write_text(json.dumps(meta, indent=1))
    after = (len(meta["room"]["obstacles"]), sum(bool(s["approach_edges"]) for s in meta["surfaces"].values()))
    print(f"[adt] obstacles {before[0]} -> {after[0]}, approachable surfaces {before[1]} -> {after[1]} in {path}")
    for line in changed:
        print("   ", line)


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
    no_model = sorted(o.name for o in objs if not (DATA / "objects" / o.name / "3d-asset.glb").exists())
    if no_model:  # the Digital Twin Catalog has no model for a few instances; they cannot exist in the sim
        print(f"[adt] skipped {len(no_model)} objects without a catalog model: {no_model}", flush=True)
    objs = [o for o in objs if o.name not in no_model]
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
    obstacles = obstacle_polygons(fixed)
    furniture = grid.polygons(obstacles)
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
            r_edges = floorplan.approach_edges(r, clear, grid, BODY_RADIUS, own=own, runup=RUNUP)
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
        centre_world = np.asarray(b["init_pos"])
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
            "init_pos": b["init_pos"],
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
        "room": {
            "bounds": bounds,
            "walls": walls,
            "obstacles": obstacles,
            "doors": [],
            "rooms": rooms,
            "spawn": spawn_named,
        },
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
    (out / "scene_meta.json").write_text(json.dumps(meta, indent=1))
    preview(meta, grid, wall_img, labels, out / "floorplan.png")
    print(f"[adt] wrote {out / 'scene_meta.json'} in {time.time() - t0:.0f} s; {len(problems)} validation problem(s)")
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


FURNITURE_FACES = 60_000  # above this a furniture visual is its hull: MuJoCo only previews, Isaac/Unreal render
# TODO 27.2.5 (scene RSI plan D3): furniture collides with the robot only. Its convex hull fills a shelf or a cabinet
# interior, so it must not touch the free objects resting there (they stay on their surface slabs): the hull takes a
# contype bit nothing else sets and only the robot's collision geoms gain it in conaffinity (1 | 4 = 5).
ROBOT_ONLY_BIT = 4
FURNITURE_MIN_THICKNESS_M = 0.05  # flatter furniture (rugs, mats) is walked over, not bumped into
SURFACE_CLEAR_M = 0.02  # a holding piece's collider stops this far under its lowest surface
# Contact margin (m) of the surface slabs. Without it an object's hull on a slab gets 1-4 contacts that change from step
# to step at the runner's 5 ms, and resting objects turned by themselves (the wallet 2 deg/s, potted plants 4 deg/s;
# torsional friction, elliptic cones and noslip did not stop it). Contacts found 1 mm early keep a steady set: it rests
# 0.8 mm above the slab instead of 0.75 mm in it. On the slabs, not the objects, so jaw-object contacts are unchanged
# (a contact takes the larger margin of its two geoms). scripts/experiments/scene_rsi_20261016/spin_check.py measures it.
SLAB_MARGIN = "0.001"
OBJECT_ONLY_BIT = 2  # side panels (contype; scripts/scene_add_panels.py) and free objects (conaffinity): a contact the
# robot never makes


def colour(mesh: trimesh.Trimesh) -> list[float]:
    """One RGBA for a MuJoCo preview: the mean of the base-colour texture, else the material's factor."""
    vis = getattr(mesh, "visual", None)
    mat = getattr(vis, "material", None)
    img = getattr(mat, "baseColorTexture", None)
    if img is not None:
        arr = np.asarray(img.convert("RGB").resize((32, 32)), dtype=float) / 255.0
        return [*arr.reshape(-1, 3).mean(axis=0).round(3).tolist(), 1.0]
    factor = getattr(mat, "baseColorFactor", None)
    if factor is not None:
        f = np.asarray(factor, dtype=float).ravel()
        f = f / 255.0 if f.max() > 1.0 else f
        return [*f[:3].round(3).tolist(), 1.0]
    return [0.7, 0.7, 0.7, 1.0]


# MuJoCo renders a mesh over 349,525 faces about 12x slower (2^20 indices): split, not decimate
FACES_PER_MESH = 300_000
TEXTURE_MAX = 1024  # px; the dataset cameras are 640 wide and a free object fills a fraction of them


def free_visuals(key: str, a: adt.AdtObject, out: Path) -> list[dict]:
    """A free body's own meshes in its body frame, for a visual geom each: `{name, obj, png | None, rgba}`.

    The glb's geometries (graph transforms applied) moved like the hull (`AdtObject.body`), written as OBJ with
    texture coordinates, and the base-colour texture as a PNG at most TEXTURE_MAX wide; a geometry without one keeps
    the preview colour. Its hull is the collision shape (group 3), so MuJoCo's cameras, the Unreal import
    (`unreal.export_mjcf` drops colliding geoms of a body that has visuals) and Isaac (which loads the glb itself)
    all draw the object as scanned.
    """
    scene = trimesh.load(DATA / "objects" / key / "3d-asset.glb")
    geoms = scene.dump() if isinstance(scene, trimesh.Scene) else [scene]
    offset = np.asarray(a.body()["mesh_offset"], dtype=float)
    parts = []
    for g, geom in enumerate(geoms):
        verts = np.asarray(geom.vertices) @ a.body_axes.T + offset
        uv = getattr(geom.visual, "uv", None)
        img = getattr(getattr(geom.visual, "material", None), "baseColorTexture", None)
        png = None
        if img is not None and uv is not None:
            png = out / "mesh" / f"adt_vis_{key}_{g}.png"
            img = img.convert("RGB")
            if max(img.size) > TEXTURE_MAX:
                img = img.resize((TEXTURE_MAX, TEXTURE_MAX * img.size[1] // img.size[0]))
            png.parent.mkdir(parents=True, exist_ok=True)
            img.save(png)
        faces = np.asarray(geom.faces)
        for c, start in enumerate(range(0, len(faces), FACES_PER_MESH)):
            chunk = faces[start : start + FACES_PER_MESH]
            used, local = np.unique(chunk, return_inverse=True)
            name = f"adt_vis_{key}_{g}_{c}"
            obj = out / "mesh" / f"{name}.obj"
            lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in verts[used]]
            if png is not None:
                lines += [f"vt {u:.6f} {v:.6f}" for u, v in np.asarray(uv)[used]]
                lines += [f"f {i + 1}/{i + 1} {j + 1}/{j + 1} {k + 1}/{k + 1}" for i, j, k in local.reshape(-1, 3)]
            else:
                lines += [f"f {i + 1} {j + 1} {k + 1}" for i, j, k in local.reshape(-1, 3)]
            obj.write_text("\n".join(lines) + "\n")
            parts.append(
                {"name": name, "obj": str(obj), "png": None if png is None else str(png), "rgba": colour(geom)}
            )
    return parts


def export(mesh: trimesh.Trimesh, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False).export(path)
    return str(path)


def furniture_solid(body_mesh, lowest_top: float | None, origin_z: float):
    """The furniture piece's robot collider (TODO 27.2.5): its convex hull, or for a piece that holds surfaces the hull
    of its part below the lowest one (2 cm under it). A whole hull fills the space above a tier up to the piece's top
    -- the utility cart's handles stand 9 cm over its top tier and the hull swallowed the coffee canister standing
    there (2026-10-16) -- so pieces with surfaces only keep the feet and legs out. None when nothing is left."""
    if lowest_top is None:
        return body_mesh.convex_hull
    below = body_mesh.vertices[body_mesh.vertices[:, 2] < lowest_top - SURFACE_CLEAR_M - origin_z]
    if len(below) < 4:
        return None
    try:
        return trimesh.convex.convex_hull(below)
    except Exception:  # degenerate (flat) point set: no solid below the surfaces
        return None


def mujoco(args) -> None:
    """Scene file -> MJCF: scenery, slabs, furniture hulls (robot only), static and free objects; the robot tree comes
    from --base, its colliding geoms gaining the furniture bit."""
    import xml.etree.ElementTree as ET

    path = Path(args.scene).resolve()  # MuJoCo resolves mesh files from its own cwd: write absolute paths
    meta = json.loads(path.read_text())
    out = path.parent
    objs = {o.name: o for o in adt.read_sequence(DATA / "sequences" / meta["source"]["sequence"])}
    missing = sorted(k for k in meta["objects"] if not (DATA / "objects" / k / "3d-asset.glb").exists())
    if missing:
        raise SystemExit(f"objects without a catalog model (rebuild the geometry stage, it skips them): {missing}")
    # default free bodies: graspable movable objects resting on an extracted surface; anything else (in a rack,
    # inside a cabinet, stacked on another object) has nothing that collides under it and would fall
    free = set(args.free) if args.free else {
        k for k, o in meta["objects"].items()
        if o["semantic"]["movable"] and o["semantic"]["graspable"] != "none" and o.get("surface")
    }  # fmt: skip
    tree = ET.parse(args.base)
    if args.ego_camera and args.robot != "mars":
        raise SystemExit(f"--ego-camera picks one of mars's head cameras; robot {args.robot} has one ego camera")
    ego_camera = (args.ego_camera or make_dex1_model.MARS_EGO_DEFAULT) if args.robot == "mars" else None
    if args.robot == "mars":
        make_dex1_model.write_mars()  # the meshes the scene references (and the robot model) are gitignored
        tree = make_dex1_model.apply_mars(tree, make_dex1_model.MARS_MESHES, ego_camera)
    root = tree.getroot()
    asset, world = root.find("asset"), root.find("worldbody")
    for el in list(world):
        if el.tag in ("camera",) or (el.tag == "body" and el.get("name") != "pelvis"):
            world.remove(el)
    pelvis = world.find("body[@name='pelvis']")
    x, y = meta["robot_start"]["xy"]
    # qpos0 is the reset's nominal pose and the plan runner's spawn origin: put it in the free spawn region
    pelvis.set("pos", mjcf.fmt([x, y, meta["robot_start"].get("pelvis_z", 0.793)]))
    floor = world.find("geom[@name='floor']")
    if floor is not None:
        floor.set("rgba", "0 0 0 0")  # the shell has its own floor mesh; the plane still collides
    env = ET.SubElement(world, "body", name="adt_env")  # scenery and colliders, so env_only.xml can copy it

    def mesh_asset(name: str, file: str) -> None:
        ET.SubElement(asset, "mesh", name=name, file=file)

    # the shell: one visual geom per scene-graph part, world frame
    shell = objs[adt.SHELL]
    for i, part in enumerate(_parts(shell)):
        name = f"adt_shell_{i:02d}"
        mesh_asset(name, export(part, out / "mesh" / f"{name}.obj"))
        ET.SubElement(env, "geom", name=name, type="mesh", mesh=name, contype="0", conaffinity="0", group="1",
                      density="0", rgba=mjcf.fmt(colour(part)))  # fmt: skip
    for key, s in meta["surfaces"].items():
        s["body"] = "adt_env"  # every slab lives in the env body: contacts with it read as furniture contacts
        sl = mjcf.slab(s)
        ET.SubElement(env, "geom", name=f"adt_slab_{key}", type="box", pos=mjcf.fmt(sl["pos"]), quat=mjcf.fmt(sl["quat"]),
                      size=mjcf.fmt(sl["size"]), rgba="0.5 0.5 0.9 0.25", group="3", margin=SLAB_MARGIN)  # fmt: skip
    # the lowest surface each furniture piece holds (furniture_solid keeps only what is under it)
    hosts: dict[str, float] = {}
    for sf in meta["surfaces"].values():
        hosts[sf["of_object"]] = min(hosts.get(sf["of_object"], np.inf), float(sf["top"]))
    n_free = n_hull = 0
    for key, o in sorted(meta["objects"].items()):
        if key == adt.SHELL:
            continue
        a = objs[key]
        raw = load_mesh(key)
        rgba = mjcf.fmt(colour(raw))
        body_mesh = raw.copy()
        # object frame -> body frame: z up at the initial pose, origin at the box centre (adt.AdtObject.body)
        body_mesh.vertices = raw.vertices @ a.body_axes.T + np.asarray(a.body()["mesh_offset"])
        sem = o["semantic"]
        if not sem["movable"]:  # furniture: objects rest on its surface slabs; the robot also bumps its hull
            vis = body_mesh if len(body_mesh.faces) <= FURNITURE_FACES else body_mesh.convex_hull
            name = f"adt_fix_{key}"
            mesh_asset(name, export(vis, out / "mesh" / f"{name}.obj"))
            ET.SubElement(env, "geom", name=name, type="mesh", mesh=name, pos=mjcf.fmt(o["init_pos"]),
                          quat=mjcf.fmt(o["init_quat"]), contype="0", conaffinity="0", group="1", density="0", rgba=rgba)  # fmt: skip
            solid = furniture_solid(body_mesh, hosts.get(key), float(o["init_pos"][2]))
            # no door gets a collider: the scene format treats every door as open (format §4) and the occupancy map
            # leaves doors out (obstacle_polygons); the bathroom door's hull (a leaf ajar) closed half its doorway
            # off the map and the robot fell walking into it (TODO 27.3.1, 2026-10-16)
            if (
                args.furniture_collision == "robot"
                and a.category != "door"
                and 2 * o["box"]["half_extent"][2] >= FURNITURE_MIN_THICKNESS_M
                and solid
            ):
                hull = f"adt_fixcol_{key}"
                mesh_asset(hull, export(solid, out / "mesh" / f"{hull}.obj"))
                ET.SubElement(env, "geom", name=hull, type="mesh", mesh=hull, pos=mjcf.fmt(o["init_pos"]),
                              quat=mjcf.fmt(o["init_quat"]), contype=str(ROBOT_ONLY_BIT), conaffinity="0", group="3",
                              density="0", rgba="0.9 0.3 0.3 0.15")  # fmt: skip
                n_hull += 1
            continue
        hull = body_mesh.convex_hull
        name = f"adt_obj_{key}"
        o["mesh"] = export(hull, out / "mesh" / f"{name}.obj")
        mesh_asset(name, o["mesh"])
        is_free = key in free
        o["free"] = is_free
        parent = ET.SubElement(world, "body", name=f"{key}_object", pos=mjcf.fmt(o["init_pos"]), quat=mjcf.fmt(o["init_quat"])) \
            if is_free else None  # fmt: skip
        mu = sem.get("friction", 0.8)
        friction = mjcf.fmt([mu, 0.005, 0.0001])
        if is_free:
            n_free += 1
            o["body"] = f"{key}_object"
            ET.SubElement(parent, "freejoint", name=f"{key}_free")
            for part in free_visuals(key, a, out):
                vis = {"rgba": mjcf.fmt(part["rgba"])}
                if part["png"]:
                    ET.SubElement(asset, "texture", name=part["name"], type="2d", file=part["png"])
                    ET.SubElement(asset, "material", name=part["name"], texture=part["name"])
                    vis = {"material": part["name"]}
                mesh_asset(part["name"], part["obj"])
                ET.SubElement(parent, "geom", name=part["name"], type="mesh", mesh=part["name"], contype="0",
                              conaffinity="0", group="1", density="0", **vis)  # fmt: skip
            inert = mjcf.box_inertia(sem.get("mass_kg", 0.2), o["box"])
            ET.SubElement(
                parent,
                "inertial",
                mass=str(inert["mass"]),
                pos=mjcf.fmt(inert["pos"]),
                diaginertia=mjcf.fmt(inert["diaginertia"]),
            )
            holder, at = parent, {}
        else:
            holder, at = env, {"pos": mjcf.fmt(o["init_pos"]), "quat": mjcf.fmt(o["init_quat"])}
        if o.get("container"):
            if not is_free:  # a free container is drawn by its free_visuals
                ET.SubElement(holder, "geom", name=f"{name}_vis", type="mesh", mesh=name, contype="0", conaffinity="0",
                              group="1", density="0", rgba=rgba, **at)  # fmt: skip
            for b in mjcf.container_boxes(o["box"]):
                pos = b["pos"]
                if not is_free:  # a static container's boxes are placed in the world frame
                    rot = adt.quat_to_matrix(o["init_quat"])
                    pos = (np.asarray(o["init_pos"]) + rot @ np.asarray(pos)).tolist()
                    extra = {"quat": mjcf.fmt(o["init_quat"])}
                else:
                    extra = {"conaffinity": str(1 | OBJECT_ONLY_BIT)}  # a free container meets the side panels too
                ET.SubElement(holder, "geom", name=f"{name}_{b['name']}", type="box", pos=mjcf.fmt(pos), size=mjcf.fmt(b["size"]),
                              friction=friction, density="0", group="3", rgba="0.9 0.5 0.2 0.3", **extra)  # fmt: skip
        else:
            free_bits = {"group": "3", "conaffinity": str(1 | OBJECT_ONLY_BIT)} if is_free else {}
            ET.SubElement(holder, "geom", name=f"{name}_col", type="mesh", mesh=name, friction=friction, density="0",
                          rgba=rgba, **free_bits, **at)  # fmt: skip
    meta["scene_xml"] = str(out / "scene_locomani.xml")
    meta["robot"] = args.robot  # robot_look.scene_robot; the Unreal level and Isaac stage builders record it
    meta["furniture_collision"] = args.furniture_collision
    if args.furniture_collision == "robot":
        for g in pelvis.iter("geom"):  # the robot's colliding geoms (default contype / conaffinity 1) gain the bit
            if g.get("contype", "1") != "0" and g.get("conaffinity", "1") != "0":
                g.set("conaffinity", str(int(g.get("conaffinity", "1")) | ROBOT_ONLY_BIT))
    meta.pop("ego_camera", None)
    if ego_camera:
        meta["ego_camera"] = ego_camera
    tree.write(out / "scene_locomani.xml")
    if args.robot == "mars":
        make_dex1_model.check_mars(str(out / "scene_locomani.xml"))
    # the robot-free scene for the drop test, written from scratch: the robot's equalities, actuators, sensors and
    # contact excludes all name robot elements, so a copy minus the pelvis tree does not compile
    env_root = ET.Element("mujoco", model="adt_env_only")
    for tag in ("compiler", "option"):
        if root.find(tag) is not None:
            env_root.append(root.find(tag))
    env_asset = ET.SubElement(env_root, "asset")
    for el in asset.findall("mesh"):
        if (el.get("name") or "").startswith("adt_"):
            env_asset.append(el)
    env_world = ET.SubElement(env_root, "worldbody")
    ET.SubElement(env_world, "geom", name="floor", type="plane", size="0 0 0.05")
    ET.SubElement(env_world, "light", pos="0 0 4", dir="0 0 -1")
    for el in world.findall("body"):
        if el.get("name") != "pelvis":
            env_world.append(el)
    ET.ElementTree(env_root).write(out / "env_only.xml")
    path.write_text(json.dumps(meta, indent=1))
    print(
        f"[mujoco] {out / 'scene_locomani.xml'}: {len(meta['surfaces'])} slabs, {n_free} free bodies, {n_hull} furniture "
        "hulls (robot only); env_only.xml for the drop test"
    )


def _parts(shell: adt.AdtObject):
    """The shell's scene-graph geometries in the sim world frame."""
    scene = trimesh.load(DATA / "objects" / adt.SHELL / "3d-asset.glb")
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = shell.rot, shell.pos
    for node in scene.graph.nodes_geometry:
        transform, gname = scene.graph[node]
        part = scene.geometry[gname].copy()
        part.apply_transform(t @ transform)
        yield part


def estimate(args) -> None:
    """Claude's fields: object names / mass / friction / containers, then surface names, then room names."""
    path = Path(args.scene)
    meta = json.loads(path.read_text())
    log = path.parent / "claude_properties.jsonl"
    rows = properties.object_rows(meta)
    notes, estimates = [], []
    for k in range(0, len(rows), args.batch):
        part = rows[k : k + args.batch]
        reply = properties.ask({"objects": part}, properties.OBJECT_SCHEMA, properties.OBJECT_SYSTEM, args.model, log)
        estimates.extend(reply["objects"])
        print(f"[props] objects {k + len(part)}/{len(rows)}", flush=True)
    notes += properties.apply_objects(meta, estimates)
    phrase = {key: o["semantic"].get("phrase", key) for key, o in meta["objects"].items()}
    phrase[properties.SHELL_KEY] = "the apartment's built-in fittings"
    reply = properties.ask(
        {"surfaces": properties.surface_rows(meta, phrase)}, properties.SURFACE_SCHEMA, properties.SURFACE_SYSTEM,
        args.model, log,
    )  # fmt: skip
    notes += properties.apply_surfaces(meta, reply["surfaces"])
    reply = properties.ask(
        {"rooms": properties.room_rows(meta)}, properties.ROOM_SCHEMA, properties.ROOM_SYSTEM, args.model, log
    )
    notes += properties.rename_rooms(meta, reply["rooms"])
    meta["source"]["properties_by"] = {"model": args.model, "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "log": log.name}
    problems = scene_file.validate(meta)
    path.write_text(json.dumps(meta, indent=1))
    (path.parent / "properties_report.json").write_text(json.dumps({"notes": notes, "problems": problems}, indent=1))
    print(
        f"[props] {len(notes)} rule notes, {len(problems)} validation problem(s); rooms {sorted(meta['room']['rooms'])}"
    )
    for n in notes[:25]:
        print("   ", n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    g = sub.add_parser("geometry", help="walls, rooms, spawn, surfaces, objects -> scene_meta.json")
    g.add_argument("--sequence", required=True)
    g.add_argument("--out", required=True)
    e = sub.add_parser("estimate", help="Claude's names, mass, friction, containers, room names (in place)")
    e.add_argument("--scene", required=True, help="the scene_meta.json the geometry stage wrote")
    e.add_argument("--model", default="opus")
    e.add_argument("--batch", type=int, default=60)
    j = sub.add_parser("mujoco", help="MJCF from the scene file (robot from --base)")
    j.add_argument("--scene", required=True)
    j.add_argument("--base", default=str(REPO / "outputs/scene/walk/scene_locomani.xml"))
    j.add_argument("--robot", choices=("stock", "mars"), default="stock", help="mars: the lab's real G1 (apply_mars)")
    j.add_argument("--ego-camera", choices=("d455_head", "d435_head"), default=None, help="mars only: the ego camera")
    j.add_argument(
        "--furniture-collision",
        choices=("robot", "none"),
        default="robot",
        help="robot: each furniture piece's convex hull collides with the robot only (TODO 27.2.5); none: scenery only",
    )
    j.add_argument(
        "--free",
        nargs="*",
        help="object keys that get a free joint (default: every graspable movable one on a surface)",
    )
    b = sub.add_parser("refresh", help="recompute obstacles and approach edges in place (keeps Claude's fields)")
    b.add_argument("--scene", required=True)
    args = ap.parse_args()
    {"geometry": geometry, "estimate": estimate, "mujoco": mujoco, "refresh": refresh}[args.stage](args)


if __name__ == "__main__":
    main()
