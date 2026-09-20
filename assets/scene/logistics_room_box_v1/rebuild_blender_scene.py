"""Prepare the logistics-room Blender session for a deterministic combined export.

Run this inside Blender before ``scripts/blender_to_usd.py``. It remaps source
image paths that refer to the original author's machine to the packaged texture
closure, then appends and places the tracked NOR-CAL box source.
"""

from pathlib import Path

import bpy

PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[2]
TEXTURES = PACKAGE / "isaac/textures"
BOX_BLEND = REPO / "assets/blender/NOR_CAL_Box_Refined_V2.blend"
BOX_OBJECT = "NOR-CAL | complete single carton mesh"
BOX_POSITION = (0.0, -0.39, 0.371)


remapped = 0
for image in bpy.data.images:
    if image.source != "FILE" or image.packed_file is not None:
        continue
    current = Path(bpy.path.abspath(image.filepath))
    if current.is_file():
        continue
    candidate = TEXTURES / current.name
    if not candidate.is_file():
        continue
    image.filepath = str(candidate)
    image.reload()
    remapped += 1

before = set(bpy.data.objects)
bpy.ops.wm.append(
    filepath=f"{BOX_BLEND}/Object/{BOX_OBJECT}",
    directory=f"{BOX_BLEND}/Object/",
    filename=BOX_OBJECT,
)
added = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
if len(added) != 1:
    raise RuntimeError(f"expected one appended box mesh, got {[obj.name for obj in added]}")
box = added[0]
if box.name not in bpy.context.scene.objects:
    bpy.context.scene.collection.objects.link(box)
box.name = "nor_cal_box"
box.rotation_euler = (0.0, 0.0, 0.0)
box.location = BOX_POSITION
bpy.context.view_layer.update()
dimensions = tuple(float(value) for value in box.dimensions)
if any(abs(actual - expected) > 1e-6 for actual, expected in zip(dimensions, (0.45, 0.32, 0.33))):
    raise RuntimeError(f"unexpected box dimensions: {dimensions}")
print(f"logistics prepare: remapped {remapped} images")
print(f"logistics prepare: appended {box.name} dimensions={dimensions} position={BOX_POSITION}")
