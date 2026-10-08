#!/usr/bin/env python3
"""A built ADT scene file -> the Isaac scene layout, or the Unreal room bake (TODO 19.6.5; docs/rendering.md "资产构建").
Runs inside Blender.

    blender --background --python scripts/adt_to_usd.py -- \\
        outputs/scene_discovery/adt_apartment/scene_meta.json $ISAAC_ASSETS/scene/adt_apartment
    blender --background --python scripts/adt_to_usd.py -- --unreal \\
        outputs/scene_discovery/adt_apartment_mars/scene_meta.json outputs/scene_discovery/adt_apartment_mars/unreal_room

Writes `<out>/apartment.usdc` (the shell, every fixed object and every movable object that is not a free body,
at its world pose; PBR materials from the glb), `<out>/scene.usda` referencing it at /World/room with interior
lights (`scene_discovery.usd_scene`, which also rewrites it alone for light tuning), and per free
body `<out>/usd/<key>/<key>.usd` + `<key>.json` in the MuJoCo body frame, so `stage.assemble_stage` drives
/World/<key> with the body pose straight from a replay. Blender's glTF importer already turns ADT's y-up into
z-up (the same +90 deg about x as `scene_discovery.adt.ADT_TO_SIM`), so an object's Blender world matrix is
ADT_TO_SIM . R . ADT_TO_SIM^T and a free body's mesh is moved by body_axes . ADT_TO_SIM^T.

`--unreal` writes what `unreal.build_level --room <out> --room-level <name>` builds a room level from, the layout
scripts/unreal/warehouse_bake.py gives the warehouse: `<out>/room.glb` (the same room, world transforms applied to
the meshes, textures at most TEXTURE_MAX) and `<out>/room_scene.json` (an area light under the ceiling at every
`usd_scene.light_points`, facing down, and the exposure). The free bodies are not in it: the level imports them
with the robot (`unreal.export_mjcf`), drawn with the textured meshes `adt_to_scene.py mujoco` gave them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from locomani.scene_discovery import adt  # noqa: E402  (numpy-only module; Blender bundles numpy)

DATA = Path("/workspace/datasets/adt")
TEXTURE_MAX = 1024  # px; 4.9 GB of 2-4K scans is more than the room level needs at the dataset cameras' 640
# The Unreal room's lights and exposure start at the warehouse bake's (0.57 m panels at 65 W, exposure -0.25);
# in_editor/warehouse_level.py turns them into Rect Lights and the manual exposure the same way.
LIGHT_W = 65.0
LIGHT_SIZE = 0.5  # m
EXPOSURE = -0.25


def clear() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_glb(name: str) -> list:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(DATA / "objects" / name / "3d-asset.glb"))
    new = [o for o in bpy.data.objects if o not in before]
    roots = [o for o in new if o.parent is None or o.parent not in new]
    return new, roots


def matrix(m3, t=(0.0, 0.0, 0.0)) -> Matrix:
    m = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            m[i][j] = float(m3[i][j])
        m[i][3] = float(t[i])
    return m


def export(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = dict(filepath=str(path), export_materials=True, generate_preview_surface=True, export_uvmaps=True,
                   export_normals=True, export_cameras=False, export_lights=False, selected_objects_only=False,
                   root_prim_path="/root")  # fmt: skip
    accepted = bpy.ops.wm.usd_export.get_rna_type().properties.keys()
    bpy.ops.wm.usd_export(**{k: v for k, v in options.items() if k in accepted})


def shrink_images(images, size: int) -> None:
    for img in images:
        w, h = img.size
        if max(w, h) > size:
            img.scale(size * w // max(w, h), size * h // max(w, h))


def place_room(meta: dict, objs: dict, free: list, texture_max: int | None = None) -> None:
    """Every object of the scene file but the free bodies, at its world pose; with `texture_max`, each object's
    textures shrunk as it comes in (decoded all at once, the scans' textures alone are tens of GB)."""
    clear()
    for key in sorted(meta["objects"]):  # the scene file, not the sequence: it drops objects without a model
        if key in free:
            continue
        o = objs[key]
        before = set(bpy.data.images)
        _, roots = import_glb(key)
        if texture_max:
            shrink_images([i for i in bpy.data.images if i not in before], texture_max)
        world = matrix(o.rot @ adt.ADT_TO_SIM.T, o.pos)  # o.rot already includes ADT_TO_SIM on the left
        for r in roots:
            r.matrix_world = world @ r.matrix_world


def bake_unreal(meta: dict, out: Path) -> None:
    """room.glb and room_scene.json for in_editor/warehouse_level.py (see the module docstring)."""
    from locomani.scene_discovery import usd_scene

    # Interchange places a glb's static meshes at the origin, so the world transforms go into the mesh data
    for o in [o for o in bpy.data.objects if o.type == "MESH"]:
        world = o.matrix_world.copy()
        o.parent = None
        o.data = o.data.copy()  # a mesh several objects share would be moved once per user
        o.data.transform(world)
        o.matrix_world = Matrix.Identity(4)
    for o in [o for o in bpy.data.objects if o.type != "MESH"]:
        bpy.data.objects.remove(o)
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out / "room.glb"), export_format="GLB", use_selection=False,
                              export_apply=True, export_yup=True, export_texcoords=True, export_normals=True,
                              export_materials="EXPORT", export_image_format="AUTO", export_lights=False,
                              export_cameras=False)  # fmt: skip
    points = usd_scene.light_points(meta["room"]["rooms"])
    lights = [
        {
            "name": f"ceiling_{i:02d}_{room}",
            "type": "AREA",
            "shape": "SQUARE",
            "size": LIGHT_SIZE,
            "size_y": LIGHT_SIZE,
            "energy_W": LIGHT_W,
            "color": [1.0, 1.0, 1.0],
            # identity rotation: an area light emits along its local -Z, straight down
            "matrix_world": [[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, usd_scene.HEIGHT], [0, 0, 0, 1]],
        }
        for i, (room, x, y) in enumerate(points)
    ]
    record = {
        "frame": "Blender world = MuJoCo world (Z up, metres); Unreal = (x, -y, z) * 100 cm",
        "source": "scripts/adt_to_usd.py --unreal",
        "lights": lights,
        "emissive_materials": [],
        "world": {"color": None, "strength": None},
        "render": {"exposure": EXPOSURE},
        "cameras": [],
    }
    (out / "room_scene.json").write_text(json.dumps(record, indent=1))
    print(f"[unreal] {out / 'room.glb'}: {len(bpy.data.meshes)} meshes; {len(lights)} ceiling lights", flush=True)


def main() -> None:
    unreal = "--unreal" in sys.argv
    scene_path, out = Path(sys.argv[-2]), Path(sys.argv[-1])
    meta = json.loads(scene_path.read_text())
    objs = {o.name: o for o in adt.read_sequence(DATA / "sequences" / meta["source"]["sequence"])}
    a2s = adt.ADT_TO_SIM
    free = sorted(k for k, o in meta["objects"].items() if o.get("free") and o.get("body"))

    place_room(meta, objs, free, TEXTURE_MAX if unreal else None)
    if unreal:
        bake_unreal(meta, out)
        return
    export(out / "apartment.usdc")
    print(f"[usd] apartment.usdc: {len(meta['objects']) - len(free)} objects", flush=True)

    for key in free:
        clear()
        o = objs[key]
        _, roots = import_glb(key)
        # the MuJoCo body frame: z up at the initial pose, origin at the box centre (adt.AdtObject.body)
        to_body = matrix(o.body_axes @ a2s.T, o.body()["mesh_offset"])
        for r in roots:
            r.matrix_world = to_body @ r.matrix_world
        export(out / "usd" / key / f"{key}.usd")
        (out / "usd" / key / f"{key}.json").write_text(
            json.dumps(
                {"scale": 1.0, "offset_pos": [0.0, 0.0, 0.0], "offset_quat": [1.0, 0.0, 0.0, 0.0], "source": "adt"}
            )
        )
    print(f"[usd] {len(free)} free bodies under usd/", flush=True)

    from locomani.scene_discovery import usd_scene

    rooms = meta["room"]["rooms"]
    points = usd_scene.light_points(rooms)
    (out / "scene.usda").write_text(usd_scene.scene_usda(str((out / "apartment.usdc").resolve()), points))
    print(f"[usd] {out / 'scene.usda'}: {len(points)} sphere lights", flush=True)


if __name__ == "__main__":
    main()
