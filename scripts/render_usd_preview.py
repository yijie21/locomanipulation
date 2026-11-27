#!/usr/bin/env python3
"""Render one USD asset from four views in headless Blender, with its Omniverse MDL materials rebuilt.

    blender --background --factory-startup --python scripts/render_usd_preview.py -- \\
        assets/logistics_manufacturing_samsung_assets/Cable007/Cable007.usd outputs/samsung_assets_preview Cable007

Writes `<out_dir>/<name>_{iso,back,front,top}.png` (1280x960, Cycles, GPU when one is found) and
`<out_dir>/<name>_info.json`: size, face count, which materials were rebuilt and which textures loaded.
Tile several assets' renders into one picture with scripts/usd_contact_sheet.py.

Why the material step: assets authored for Isaac Sim (the Samsung logistics / manufacturing set,
2026-11-27) bind OmniPBR, OmniGlass and OmniSurface MDL shaders. Blender's USD importer converts only
UsdPreviewSurface, so every such mesh rendered black. This script re-reads each material's MDL inputs
with the pxr bundled in Blender and rebuilds it as a Principled BSDF: OmniPBR base colour, normal map and
the ORM texture's roughness (G) and metallic (B); OmniGlass as full transmission; OmniSurface's main lobes.
The ORM occlusion channel and `flip_tangent_v` are not used, so shading is close to, not identical with,
Isaac RTX.

Physics is not simulated: the picture shows the pose the USD authors (the dishwasher's door is open in the
file). Blender imports only the visual meshes (22 of the dishwasher's 837 Mesh prims); no collision hull
shows in the renders.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import bpy
from mathutils import Vector
from pxr import Usd, UsdShade

# (azimuth from +X counter-clockwise, elevation above the horizon), degrees. The appliances (Cable007-009)
# face -Y, so "iso" and "front" show their fronts and "back" the side where the power cords run.
VIEWS = {"iso": (-55, 28), "back": (125, 28), "front": (-90, 8), "top": (-90, 88)}


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("usd", help="the USD file to render")
    parser.add_argument("out_dir", help="directory for the PNGs and the info JSON (created if missing)")
    parser.add_argument("name", help="file name prefix, e.g. the asset name")
    parser.add_argument("--size", type=int, nargs=2, default=(1280, 960), metavar=("W", "H"))
    parser.add_argument("--samples", type=int, default=96, help="Cycles samples per pixel (denoised)")
    return parser.parse_args(argv)


def read_mdl_materials(path: str) -> tuple[dict, list[str]]:
    """USD material prim name -> (MDL subIdentifier, {input: value}), texture paths resolved.

    Blender names its materials after the USD material prims, so the name is the join key. The second
    value lists names that occur more than once with different inputs (the first one wins)."""
    stage = Usd.Stage.Open(path)
    out, conflicts = {}, []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Material":
            continue
        for child in prim.GetChildren():
            if child.GetTypeName() != "Shader":
                continue
            sub = child.GetAttribute("info:mdl:sourceAsset:subIdentifier")
            kind = sub.Get() if sub and sub.HasAuthoredValue() else None
            values = {}
            for inp in UsdShade.Shader(child).GetInputs():
                v = inp.Get()
                if v is None:
                    continue
                if hasattr(v, "resolvedPath"):
                    v = v.resolvedPath or None
                elif hasattr(v, "__len__") and not isinstance(v, str):
                    v = tuple(v)
                values[inp.GetBaseName()] = v
            name = prim.GetName()
            if name in out and out[name] != (kind, values):
                conflicts.append(name)
            out.setdefault(name, (kind, values))
    return out, conflicts


def image_node(nodes, path: str, colorspace: str, y: float):
    node = nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(path, check_existing=True)
    node.image.colorspace_settings.name = colorspace
    node.location = (-700, y)
    return node


def rebuild_material(mat, kind: str | None, v: dict) -> bool:
    """Replace `mat`'s node tree with a Principled BSDF driven by the MDL inputs `v`; False if `kind` is unknown."""
    if kind not in ("OmniPBR", "OmniGlass", "OmniSurface"):
        return False
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    p = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(p.outputs["BSDF"], out.inputs["Surface"])
    if kind == "OmniPBR":
        tint = tuple(v.get("diffuse_tint", (1, 1, 1)))
        if v.get("diffuse_texture"):
            color = image_node(nodes, v["diffuse_texture"], "sRGB", 300).outputs["Color"]
            if tint != (1, 1, 1):
                mix = nodes.new("ShaderNodeMix")
                mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
                mix.inputs["Factor"].default_value = 1
                links.new(color, mix.inputs[6])
                mix.inputs[7].default_value = (*tint, 1)
                color = mix.outputs[2]
            links.new(color, p.inputs["Base Color"])
        else:
            c = v.get("diffuse_color_constant", (0.2, 0.2, 0.2))
            p.inputs["Base Color"].default_value = (*[a * b for a, b in zip(c, tint)], 1)
        p.inputs["Roughness"].default_value = v.get("reflection_roughness_constant", 0.5)
        p.inputs["Metallic"].default_value = v.get("metallic_constant", 0.0)
        if v.get("enable_ORM_texture") and v.get("ORM_texture"):
            orm = image_node(nodes, v["ORM_texture"], "Non-Color", 0)
            sep = nodes.new("ShaderNodeSeparateColor")
            sep.location = (-400, 0)
            links.new(orm.outputs["Color"], sep.inputs["Color"])
            if v.get("reflection_roughness_texture_influence", 0) > 0:
                links.new(sep.outputs["Green"], p.inputs["Roughness"])  # ORM: R occlusion, G roughness
            if v.get("metallic_texture_influence", 0) > 0:
                links.new(sep.outputs["Blue"], p.inputs["Metallic"])  # B metallic
        if v.get("normalmap_texture"):
            normal = image_node(nodes, v["normalmap_texture"], "Non-Color", -300)
            nm = nodes.new("ShaderNodeNormalMap")
            nm.location = (-400, -300)
            nm.inputs["Strength"].default_value = v.get("bump_factor", 1.0)
            links.new(normal.outputs["Color"], nm.inputs["Color"])
            links.new(nm.outputs["Normal"], p.inputs["Normal"])
        if v.get("enable_opacity") and v.get("opacity_texture"):
            opacity = image_node(nodes, v["opacity_texture"], "Non-Color", -600)
            links.new(opacity.outputs["Color"], p.inputs["Alpha"])
        if "specular_level" in v:
            p.inputs["Specular IOR Level"].default_value = v["specular_level"]
    elif kind == "OmniGlass":
        p.inputs["Base Color"].default_value = (*v.get("glass_color", (1, 1, 1)), 1)
        p.inputs["Transmission Weight"].default_value = 1.0
        p.inputs["IOR"].default_value = v.get("glass_ior", 1.491)
        p.inputs["Roughness"].default_value = v.get("frosting_roughness", 0.0)
    else:  # OmniSurface
        p.inputs["Base Color"].default_value = (*v.get("diffuse_reflection_color", (0.8, 0.8, 0.8)), 1)
        p.inputs["Roughness"].default_value = v.get("specular_reflection_roughness", 0.2)
        p.inputs["Metallic"].default_value = v.get("metalness", 0.0)
        if v.get("enable_specular_transmission"):
            p.inputs["Transmission Weight"].default_value = v.get("specular_transmission_weight", 0.0)
        p.inputs["Coat Weight"].default_value = v.get("coat_weight", 0.0)
        p.inputs["Coat Roughness"].default_value = v.get("coat_roughness", 0.1)
        p.inputs["Coat IOR"].default_value = v.get("coat_ior", 1.5)
        if v.get("enable_opacity"):
            p.inputs["Alpha"].default_value = v.get("geometry_opacity", 1.0)
    return True


def rebuild_materials(usd_path: str) -> dict:
    mdl, conflicts = read_mdl_materials(usd_path)
    rebuilt, unmatched = {}, []
    for mat in bpy.data.materials:
        key = re.sub(r"\.\d{3}$", "", mat.name)  # Blender's ".001" suffix on a repeated name
        if key in mdl and rebuild_material(mat, *mdl[key]):
            rebuilt[mat.name] = mdl[key][0]
        elif mat.users:
            unmatched.append(mat.name)
    return {"materials_rebuilt": rebuilt, "materials_unmatched": unmatched, "mdl_name_conflicts": conflicts}


def add_studio(scene, center: Vector, lo: Vector, radius: float) -> None:
    """Neutral ground just under the asset (for a contact shadow), a soft sky and one key sun."""
    bpy.ops.mesh.primitive_plane_add(size=radius * 40, location=(center.x, center.y, lo.z - radius * 0.002))
    ground = bpy.context.active_object
    mat = bpy.data.materials.new("ground")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.64, 1)
    bsdf.inputs["Roughness"].default_value = 0.9
    ground.data.materials.append(mat)

    world = bpy.data.worlds.new("world")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (0.82, 0.85, 0.9, 1)
    bg.inputs["Strength"].default_value = 0.6
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
    sun.data.energy = 3.0
    sun.data.angle = math.radians(8)
    sun.rotation_euler = (math.radians(40), math.radians(10), math.radians(-35))
    scene.collection.objects.link(sun)


def use_gpu(scene) -> str:
    """Point Cycles at the first OptiX or CUDA device; returns the backend used, or CPU."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for backend in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = backend
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == backend for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == backend
            scene.cycles.device = "GPU"
            return backend
    return "CPU"


def fit_distance(direction: Vector, center: Vector, corners: list[Vector], hfov: float, vfov: float) -> float:
    """Smallest camera distance along `direction` (unit, centre to camera) that keeps every bbox corner in frame."""
    fwd = -direction
    world_up = Vector((0, 0, 1)) if abs(direction.z) < 0.99 else Vector((0, 1, 0))
    right = fwd.cross(world_up).normalized()
    up = right.cross(fwd).normalized()
    margin = 0.88
    th, tv = math.tan(hfov / 2) * margin, math.tan(vfov / 2) * margin
    need = 0.0
    for c in corners:
        o = c - center
        x, y, z = o.dot(right), o.dot(up), o.dot(direction)
        need = max(need, z + abs(x) / th, z + abs(y) / tv)
    return need


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=args.usd)
    scene = bpy.context.scene
    materials = rebuild_materials(args.usd)

    meshes = [o for o in scene.objects if o.type == "MESH" and not o.hide_render]
    if not meshes:
        print(f"no renderable mesh in {args.usd}", file=sys.stderr)
        return 1
    pts = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)])
    hi = Vector([max(p[i] for p in pts) for i in range(3)])
    center = (lo + hi) / 2
    radius = max((hi - lo).length / 2, 1e-3)
    corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]

    images = [i for i in bpy.data.images if i.source == "FILE"]
    loaded = [i.name for i in images if i.size[0] > 0]  # reading .size loads the file
    missing = [i.filepath for i in images if i.size[0] == 0]

    add_studio(scene, center, lo, radius)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    cam_data.clip_start = radius * 0.01
    cam_data.clip_end = radius * 100
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    target = bpy.data.objects.new("target", None)
    target.location = center
    scene.collection.objects.link(target)
    track = cam.constraints.new("TRACK_TO")
    track.target = target
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = args.size
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    device = use_gpu(scene)

    # sensor fit AUTO: the sensor width spans the longer (horizontal) side of a landscape frame
    hfov = 2 * math.atan(cam_data.sensor_width / (2 * cam_data.lens))
    vfov = 2 * math.atan(math.tan(hfov / 2) * args.size[1] / args.size[0])
    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for view, (az, el) in VIEWS.items():
        a, e = math.radians(az), math.radians(el)
        direction = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
        cam.location = center + fit_distance(direction, center, corners, hfov, vfov) * direction
        scene.render.filepath = os.path.join(args.out_dir, f"{args.name}_{view}.png")
        bpy.ops.render.render(write_still=True)
        written.append(scene.render.filepath)

    info = {
        "usd": args.usd,
        "device": device,
        "mesh_objects": len(meshes),
        "faces": sum(len(o.data.polygons) for o in meshes),
        "bbox_min": [round(v, 4) for v in lo],
        "bbox_max": [round(v, 4) for v in hi],
        "size_m": [round(v, 3) for v in hi - lo],
        "images_total": len(images),
        "images_loaded": len(loaded),
        "images_missing": missing,
        **materials,
        "renders": written,
    }
    with open(os.path.join(args.out_dir, f"{args.name}_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("RENDER_INFO", json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []))
