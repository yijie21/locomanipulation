#!/usr/bin/env python3
"""A built ADT scene file -> the Isaac scene layout (TODO 9.6.5; docs/rendering.md "资产构建"). Runs inside Blender.

    blender --background --python scripts/adt_to_usd.py -- \\
        outputs/scene_discovery/adt_apartment/scene_meta.json $ISAAC_ASSETS/scene/adt_apartment

Writes `<out>/apartment.usdc` (the shell, every fixed object and every movable object that is not a free body,
at its world pose; PBR materials from the glb), `<out>/scene.usda` referencing it at /World/room with interior
lights (`scene_discovery.usd_scene`, which also rewrites it alone for light tuning), and per free
body `<out>/usd/<key>/<key>.usd` + `<key>.json` in the MuJoCo body frame, so `stage.assemble_stage` drives
/World/<key> with the body pose straight from a replay. Blender's glTF importer already turns ADT's y-up into
z-up (the same +90 deg about x as `scene_discovery.adt.ADT_TO_SIM`), so an object's Blender world matrix is
ADT_TO_SIM . R . ADT_TO_SIM^T and a free body's mesh is moved by body_axes . ADT_TO_SIM^T.
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


def main() -> None:
    scene_path, out = Path(sys.argv[-2]), Path(sys.argv[-1])
    meta = json.loads(scene_path.read_text())
    objs = {o.name: o for o in adt.read_sequence(DATA / "sequences" / meta["source"]["sequence"])}
    a2s = adt.ADT_TO_SIM
    free = sorted(k for k, o in meta["objects"].items() if o.get("free") and o.get("body"))

    clear()
    for key in sorted(meta["objects"]):  # the scene file, not the sequence: it drops objects without a model
        if key in free:
            continue
        o = objs[key]
        _, roots = import_glb(key)
        world = matrix(o.rot @ a2s.T, o.pos)  # o.rot already includes ADT_TO_SIM on the left
        for r in roots:
            r.matrix_world = world @ r.matrix_world
    export(out / "apartment.usdc")
    print(f"[usd] apartment.usdc: {len(meta['objects']) - len(free)} objects", flush=True)

    for key in free:
        clear()
        o = objs[key]
        _, roots = import_glb(key)
        to_body = matrix(o.body_axes @ a2s.T)
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
