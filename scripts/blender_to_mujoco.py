#!/usr/bin/env python3
"""Export a .blend room as the meshes a MuJoCo scene needs.

    blender --background room.blend --python scripts/blender_to_mujoco.py -- outputs/scene/<name>

Writes one OBJ per material into <out>/mesh plus room_parts.json and
carton_parts.json. Per material, not per object: MuJoCo binds one material to
one geom, the source room is thousands of objects sharing a few dozen
materials, and the carton is one object whose six faces are six photographs.

The room is exported for LOOKS ONLY. Collision comes from a ground plane and a
few boxes that scripts/build_mujoco_scene.py adds -- 489k triangles as contact
geometry would not survive a 50 Hz whole-body controller, and nothing in the
task touches the walls or the racks.
"""

import json
import os
import sys
from pathlib import Path

import bpy
import mathutils

ROOT = Path(__file__).resolve().parents[1]
BOX_BLEND = str(ROOT / "assets/blender/NOR_CAL_Box_Refined_V2.blend")
OBJ = "NOR-CAL | complete single carton mesh"
PLATFORM_TOP, NEAR_EDGE_Y = 0.371, -0.555
outdir = sys.argv[-1]
os.makedirs(outdir, exist_ok=True)
meshdir = os.path.join(outdir, "mesh")
os.makedirs(meshdir, exist_ok=True)

# 1. place the carton exactly as the Isaac scene has it
before = set(bpy.data.objects)
bpy.ops.wm.append(filepath=f"{BOX_BLEND}/Object/{OBJ}", directory=f"{BOX_BLEND}/Object/", filename=OBJ)
box = [o for o in bpy.data.objects if o not in before and o.type == "MESH"][0]
if box.name not in bpy.context.scene.objects:
    bpy.context.scene.collection.objects.link(box)
box.name = "nor_cal_box"
box.rotation_euler = (0, 0, 0)
box.location = (0.0, NEAR_EDGE_Y + 0.32 / 2 + 0.005, PLATFORM_TOP)
bpy.context.view_layer.update()


def export_obj(objs, path):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.wm.obj_export(
        filepath=path,
        export_selected_objects=True,
        export_materials=False,
        export_uv=True,
        export_normals=True,
        forward_axis="Y",
        up_axis="Z",
    )


def material_record(mat, base_color=False):
    """(texture image name or None, rgb) of a material.

    The room reports the viewport colour, as it always has. The carton asks for
    the Principled base colour: its viewport colours are all the 0.8 default,
    while the kraft, liner and seam materials only exist as node colours.
    """
    tex = None
    if mat and mat.use_nodes:
        for n in mat.node_tree.nodes:
            if n.type == "TEX_IMAGE" and n.image and "color" in n.image.name.lower():
                tex = n.image.name
                break
        if tex is None:
            for n in mat.node_tree.nodes:
                if n.type == "TEX_IMAGE" and n.image:
                    tex = n.image.name
                    break
    rgb = (0.7, 0.7, 0.7)
    if mat:
        rgb = tuple(round(c, 4) for c in mat.diffuse_color[:3])
        if base_color and mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    rgb = tuple(round(c, 4) for c in n.inputs["Base Color"].default_value[:3])
                    break
    return tex, rgb


def split_by_material(obj):
    """Separate obj into one object per material slot; returns them in slot order."""
    slots = [m.name if m else None for m in obj.data.materials]
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.mesh.separate(type="MATERIAL")
    prefix = obj.name
    pieces = [o for o in bpy.context.scene.objects if o.type == "MESH" and o.name.startswith(prefix)]
    pieces = [o for o in pieces if len(o.data.vertices)]
    return sorted(pieces, key=lambda o: slots.index(o.data.materials[0].name if o.data.materials else None))


# 2. the carton, one mesh per material like the room: six of its nine materials
#    are photographs of the real box, and MuJoCo binds one texture per geom.
#    Exported at the ORIGIN so the MJCF can position the body like any other
#    target asset.
box.location = (0, 0, 0)
bpy.context.view_layer.update()
carton = []
for i, o in enumerate(split_by_material(box)):
    mat = o.data.materials[0] if o.data.materials else None
    tex, rgb = material_record(mat, base_color=True)
    name = f"nor_cal_{i:02d}"
    export_obj([o], os.path.join(meshdir, f"{name}.obj"))
    carton.append(
        dict(
            name=name,
            material=(mat.name if mat else None),
            texture=tex,
            rgb=rgb,
            tris=sum(len(p.vertices) - 2 for p in o.data.polygons),
        )
    )
    print(f"[mj] {name:10s} {str(mat.name if mat else '-')[:30]:30s} tris={carton[-1]['tris']:>7} tex={tex}")
json.dump(carton, open(os.path.join(outdir, "carton_parts.json"), "w"), indent=1)
print(f"[mj] wrote {len(carton)} carton parts -> {outdir}")

# 3. the room, one mesh per material so MuJoCo can bind one texture per geom
room = [o for o in bpy.context.scene.objects if o.type == "MESH" and not o.name.startswith("nor_cal_box")]
print(f"[mj] joining {len(room)} room objects")
bpy.ops.object.select_all(action="DESELECT")
for o in room:
    o.select_set(True)
bpy.context.view_layer.objects.active = room[0]
bpy.ops.object.join()
joined = bpy.context.view_layer.objects.active
joined.name = "room"
bpy.ops.object.select_all(action="DESELECT")
joined.select_set(True)
bpy.ops.mesh.separate(type="MATERIAL")
parts = [o for o in bpy.context.scene.objects if o.name.startswith("room")]
print(f"[mj] split into {len(parts)} material parts")

manifest = []
for i, o in enumerate(sorted(parts, key=lambda x: x.name)):
    if not len(o.data.vertices):
        continue
    mat = o.data.materials[0] if o.data.materials else None
    tex, rgb = material_record(mat)
    name = f"room_{i:02d}"
    export_obj([o], os.path.join(meshdir, f"{name}.obj"))
    manifest.append(
        dict(
            name=name,
            material=(mat.name if mat else None),
            texture=tex,
            rgb=rgb,
            tris=sum(len(p.vertices) - 2 for p in o.data.polygons),
        )
    )
    print(f"[mj] {name:9s} {str(mat.name if mat else '-')[:26]:26s} tris={manifest[-1]['tris']:>7} tex={tex}")

json.dump(manifest, open(os.path.join(outdir, "room_parts.json"), "w"), indent=1)
print(f"[mj] wrote {len(manifest)} parts -> {outdir}")
