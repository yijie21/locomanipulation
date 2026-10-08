#!/usr/bin/env python3
"""Convert a .blend room into a USD that Isaac renders correctly.

    blender --background room.blend --python scripts/blender_to_usd.py -- outputs/scene/room

Blender's USD exporter carries geometry, materials, cameras and lights, but it
writes five things Isaac does not read the way Blender meant them. Each one was
found by rendering the same camera in both and looking at the result (2026-09-19
and 2026-09-28, docs/ledger/blender_scenes.md):

* camera focalLength/horizontalAperture in metres (0.185 / 0.36). The FOV is the
  correct 88.4 degrees either way, but RTX does not honour a sub-millimetre focal
  length and framed the shot somewhere else entirely.
* light intensity from watts, landing ~1000x too dark to render.
* BasisCurves widths of 2.0 on a stage whose unit IS the metre, so a decorative
  floor cable renders as a two-metre grey ribbon across the room.
* a DomeLight whose environment texture RTX rejects outright with
  "MirroredBall environment format not supported yet", which removes the ambient
  fill and leaves unlit corners black.
* a Bump node's grey height map wired straight into UsdPreviewSurface `normal`
  (scale 2, bias -1). The carton photos' heights sit at 0.50 +- 0.03, so the
  "normal" is about the zero vector and flips sign from texel to texel: every
  photo-textured carton renders under white speckle in RTX Real-Time.
  UsdPreviewSurface has no bump input, so the connection is dropped. The exporter
  already drops every procedural Bump without a word.

This runs inside Blender because Blender bundles the USD API, so exporting and
repairing are one invocation and the file on disk is already correct. Nothing
downstream has to know about any of it.
"""

from __future__ import annotations

import os
import sys

import bpy
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

# Blender writes camera dimensions in metres; USD convention is millimetres. Only
# the ratio sets the FOV, so scaling both by the same factor is lossless.
CAMERA_SCALE = 100.0
# Watts -> RTX intensity has no principled conversion: it depends on the
# renderer's exposure. Tune per scene by comparing mean luma against a Blender
# render of the same camera (scripts/compare_frames.py reports both).
DEFAULT_LIGHT_GAIN = 1.0  # see render_usd_frame.py --light-gain: scaling on disk does not take
DEFAULT_CURVE_WIDTH = 0.002  # a cable, not a two-metre ribbon
# Stands in for the environment texture RTX refused. A flat indoor ambient.
DOME_COLOUR = (0.757, 0.808, 0.878)


def export(destination: str) -> None:
    # No unpack_all here: the exporter copies packed images next to the USD by
    # itself. Unpacking first writes a second copy beside the .blend instead,
    # which on the first room was 71 MB of duplicate textures nothing reads.
    options = dict(
        filepath=destination,
        export_materials=True,
        generate_preview_surface=True,
        export_cameras=True,
        export_lights=True,
        export_uvmaps=True,
        export_normals=True,
        selected_objects_only=False,
        root_prim_path="/root",
    )
    accepted = bpy.ops.wm.usd_export.get_rna_type().properties.keys()
    bpy.ops.wm.usd_export(**{k: v for k, v in options.items() if k in accepted})


def bump_height_images() -> set[str]:
    """File names of the images that drive a Bump node's Height -- the textures the exporter writes to `normal`."""
    names = set()
    for material in bpy.data.materials:
        if material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.bl_idname != "ShaderNodeBump":
                continue
            for link in node.inputs["Height"].links:
                image = getattr(link.from_node, "image", None)
                if image is not None:
                    names.update(n for n in (image.name, bpy.path.basename(image.filepath)) if n)
    return names


def repair(stage, light_gain: float, curve_width: float, bump_heights: set[str] = frozenset()) -> list[str]:
    """Apply the five corrections; return one line per correction made."""
    done = []

    cameras = [p for p in stage.Traverse() if str(p.GetTypeName()) == "Camera"]
    for prim in cameras:
        camera = UsdGeom.Camera(prim)
        for attribute in ("FocalLength", "HorizontalAperture", "VerticalAperture"):
            handle = getattr(camera, f"Get{attribute}Attr")()
            value = handle.Get()
            if value is not None and value < 1.0:  # metres, not millimetres
                handle.Set(float(value) * CAMERA_SCALE)
    if cameras:
        done.append(f"camera: rescaled {len(cameras)} by {CAMERA_SCALE:g}x, field of view unchanged")

    lights = [p for p in stage.Traverse() if "Light" in str(p.GetTypeName())]
    scaled = 0
    for prim in lights:
        handle = prim.GetAttribute("inputs:intensity")
        if handle and handle.Get() is not None:
            handle.Set(float(handle.Get()) * light_gain)
            scaled += 1
    if scaled:
        done.append(f"light: scaled {scaled} intensities by {light_gain:g}")

    curves = 0
    for prim in stage.Traverse():
        if str(prim.GetTypeName()) != "BasisCurves":
            continue
        handle = UsdGeom.BasisCurves(prim).GetWidthsAttr()
        widths = handle.Get()
        if widths is not None and max(widths) > curve_width:
            handle.Set(Vt.FloatArray([curve_width] * len(widths)))
            curves += 1
    if curves:
        done.append(f"curve: narrowed {curves} prims to {curve_width:g} metres")

    domes = 0
    for prim in stage.Traverse():
        if "DomeLight" not in str(prim.GetTypeName()):
            continue
        handle = prim.GetAttribute("inputs:texture:file")
        if handle and handle.Get() is not None:
            handle.Clear()
            # Author the colour even when GetAttribute reports one: that is the
            # schema fallback, not an opinion, and a dome with the texture gone
            # and nothing authored contributes no light at all. Leaving it that
            # way renders the floor and every shadowed box face black.
            prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*DOME_COLOUR))
            domes += 1
    if domes:
        done.append(f"dome: dropped {domes} unsupported environment textures, kept a flat colour")

    heights = 0
    for prim in stage.Traverse():
        shader = UsdShade.Shader(prim)
        if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        normal = shader.GetInput("normal")
        if not normal or not normal.HasConnectedSource():
            continue
        texture = UsdShade.Shader(normal.GetConnectedSource()[0].GetPrim())
        file = texture.GetInput("file")
        if file and file.Get() is not None and os.path.basename(file.Get().path) in bump_heights:
            normal.GetAttr().ClearConnections()
            heights += 1
    if heights:
        done.append(f"normal: disconnected {heights} bump height maps the exporter wired in as normal maps")
    return done


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: blender --background <file.blend> --python scripts/blender_to_usd.py -- <out-dir> [gain]")
        return 2
    out_dir = argv[0]
    light_gain = float(argv[1]) if len(argv) > 1 else DEFAULT_LIGHT_GAIN
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(bpy.data.filepath))[0] or "scene"
    destination = os.path.join(out_dir, f"{name}.usdc")

    export(destination)
    from pxr import Usd

    stage = Usd.Stage.Open(destination)
    print(f"convert: exported {destination}")
    print(f"convert: upAxis {UsdGeom.GetStageUpAxis(stage)}, {UsdGeom.GetStageMetersPerUnit(stage)} metres per unit")
    for line in repair(stage, light_gain, DEFAULT_CURVE_WIDTH, bump_height_images()):
        print(f"convert: {line}")
    stage.GetRootLayer().Save()
    counts: dict[str, int] = {}
    for prim in stage.Traverse():
        kind = str(prim.GetTypeName())
        if kind:
            counts[kind] = counts.get(kind, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:5])
    print(f"convert: saved {destination} with {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
