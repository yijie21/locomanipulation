"""Bake logistics_room.blend's procedural materials to textures and export the room as one glTF for Unreal.

The MuJoCo export (scripts/blender_to_mujoco.py) keeps one colour or one photograph per material; Isaac gets the
same through Blender's USD exporter, which drops every procedural node. 28 of the room's 40 materials are Noise
textures on Generated coordinates driving base colour and bump. This bakes them (Cycles, EMIT pass for base colour
and roughness, NORMAL pass for the bumped normal) into one atlas per material on a new non-overlapping UV map,
rewires each material to the baked images, joins the room into one mesh per material in world coordinates -- the
frame MuJoCo uses, since blender_to_mujoco exports the same objects without an offset -- and writes:

    <out>/room.glb           meshes + materials + baked textures (curves and text converted to meshes too)
    <out>/room_scene.json    area lights, emissive materials, world background, render settings, cameras

Photograph materials keep their images; their height-map bump is unlinked before export, because the glTF
exporter (like the USD one) writes the Bump's height image as a normal map, which renders as speckle.

    CUDA_VISIBLE_DEVICES=1 taskset -c 24-31 /workspace/tools/blender/blender --background \
        assets/blender/logistics_room.blend --python scripts/unreal/warehouse_bake.py -- \
        outputs/unreal_warehouse_20260928/room [texel_mm=3] [samples=8]
"""

import collections
import json
import math
import sys
import time
from pathlib import Path

import bpy
import mathutils

PROCEDURAL = {
    "TEX_NOISE",
    "TEX_VORONOI",
    "TEX_WAVE",
    "TEX_MUSGRAVE",
    "TEX_GRADIENT",
    "TEX_BRICK",
    "TEX_CHECKER",
    "TEX_MAGIC",
    "TEX_WHITE_NOISE",
}
BAKE_UV = "bake"

argv = sys.argv[sys.argv.index("--") + 1 :]
OUT = Path(argv[0])
TEXEL_M = (float(argv[1]) if len(argv) > 1 else 3.0) / 1000.0
SAMPLES = int(argv[2]) if len(argv) > 2 else 8
TEXDIR = OUT / "textures"
TEXDIR.mkdir(parents=True, exist_ok=True)

sc = bpy.context.scene
view_layer = bpy.context.view_layer


def log(msg):
    print(f"[bake] {msg}", flush=True)


def setup_cycles():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
        except TypeError:
            continue
        prefs.get_devices()
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            log(f"cycles {kind}: {[d.name for d in gpus]}")
            break
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    sc.cycles.samples = SAMPLES
    sc.render.bake.margin = 8
    sc.render.bake.use_clear = True


def select_only(objs, active=None):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    view_layer.objects.active = active or objs[0]


def principled(mat):
    return next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)


def is_procedural(mat):
    return bool(mat and mat.use_nodes and any(n.type in PROCEDURAL for n in mat.node_tree.nodes))


def world_area(obj):
    s = obj.matrix_world.to_scale()
    k = abs(s.x * s.y * s.z) ** (2.0 / 3.0)
    return sum(p.area for p in obj.data.polygons) * k


def pow2_res(area_m2):
    side = math.sqrt(max(area_m2, 1e-6)) / TEXEL_M * 1.4
    return int(min(4096, max(256, 2 ** math.ceil(math.log2(side)))))


def unwrap(reps):
    """One shared non-overlapping UV layout for the unique meshes of a material."""
    for o in reps:
        uvs = o.data.uv_layers
        layer = uvs.get(BAKE_UV) or uvs.new(name=BAKE_UV)
        uvs.active = layer
    select_only(reps)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.003, scale_to_bounds=False)
    bpy.ops.uv.pack_islands(margin=0.003, rotate=True)
    bpy.ops.object.mode_set(mode="OBJECT")


def bake_input(mat, reps, socket_name, image, kind):
    """Bake what feeds `socket_name` of the material's Principled BSDF into `image`.

    kind "emit": route the socket's source through an Emission shader and bake EMIT (base colour, roughness).
    kind "normal": bake the NORMAL pass (tangent space) of the untouched material (the bumped shading normal).
    """
    nt = mat.node_tree
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output)
    surf_src = out.inputs["Surface"].links[0].from_socket if out.inputs["Surface"].is_linked else None
    img_node = nt.nodes.new("ShaderNodeTexImage")
    img_node.image = image
    uvmap = nt.nodes.new("ShaderNodeUVMap")
    uvmap.uv_map = BAKE_UV
    nt.links.new(uvmap.outputs["UV"], img_node.inputs["Vector"])
    nt.nodes.active = img_node
    emit = None
    if kind == "emit":
        src = principled(mat).inputs[socket_name].links[0].from_socket
        emit = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(src, emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    select_only(reps)
    if kind == "emit":
        bpy.ops.object.bake(type="EMIT", uv_layer=BAKE_UV)
    else:
        bpy.ops.object.bake(type="NORMAL", normal_space="TANGENT", uv_layer=BAKE_UV)
    if emit is not None:
        nt.nodes.remove(emit)
        if surf_src is not None:
            nt.links.new(surf_src, out.inputs["Surface"])
    nt.nodes.remove(img_node)
    nt.nodes.remove(uvmap)


def new_image(name, res, non_color):
    img = bpy.data.images.new(name, res, res, alpha=False, float_buffer=False)
    img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
    return img


def save_image(img):
    path = TEXDIR / f"{img.name}.png"
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    return path


def drop_height_bump(mat):
    """Unlink a Bump node from the Principled Normal. glTF (like USD) exports the Bump's height image as a normal
    map; a grey height map read as a normal is noise (the carton speckle, 2026-09-28)."""
    p = principled(mat) if mat and mat.use_nodes else None
    if not p or not p.inputs["Normal"].is_linked:
        return False
    link = p.inputs["Normal"].links[0]
    if link.from_node.type != "BUMP":
        return False
    mat.node_tree.links.remove(link)
    return True


def make_images_pot():
    """Scale every non-power-of-two image the materials use to the nearest power of two per side (log scale).
    Unreal imports a non-power-of-two texture with NoMipmaps; minified without mips, the carton prints and
    labels alias (jagged text, speckle). UVs are normalised, so a stretch leaves the mapping intact."""
    used = set()
    for m in bpy.data.materials:
        if m.use_nodes and m.users:
            used |= {n.image for n in m.node_tree.nodes if n.type == "TEX_IMAGE" and n.image}
    done = []
    for img in used:
        w, h = img.size
        if w == 0 or h == 0:
            continue
        w2, h2 = (min(4096, 2 ** round(math.log2(v))) for v in (w, h))
        if (w2, h2) != (w, h):
            img.scale(w2, h2)
            done.append((img.name, [w, h], [w2, h2]))
    return done


def rewire(mat, baked):
    """Point the Principled inputs at the baked images (UV map BAKE_UV); other nodes stay but are unused."""
    nt = mat.node_tree
    p = principled(mat)
    uvmap = nt.nodes.new("ShaderNodeUVMap")
    uvmap.uv_map = BAKE_UV
    for socket_name, img in baked.items():
        for link in list(p.inputs[socket_name].links):
            nt.links.remove(link)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        nt.links.new(uvmap.outputs["UV"], tex.inputs["Vector"])
        if socket_name == "Normal":
            nm = nt.nodes.new("ShaderNodeNormalMap")
            nm.uv_map = BAKE_UV
            nt.links.new(tex.outputs["Color"], nm.inputs["Color"])
            nt.links.new(nm.outputs["Normal"], p.inputs["Normal"])
        else:
            nt.links.new(tex.outputs["Color"], p.inputs[socket_name])


def scene_record():
    lights = []
    for o in sc.objects:
        if o.type != "LIGHT" or o.hide_render:
            continue
        d = o.data
        rec = {
            "name": o.name,
            "type": d.type,
            "energy_W": d.energy,
            "color": list(d.color),
            "matrix_world": [list(r) for r in o.matrix_world],
        }
        if d.type == "AREA":
            rec.update(shape=d.shape, size=d.size, size_y=d.size_y, spread=getattr(d, "spread", math.pi))
        else:
            rec.update(radius=getattr(d, "shadow_soft_size", 0.0))
        lights.append(rec)
    emissive = []
    for m in bpy.data.materials:
        p = principled(m) if m.use_nodes else None
        if p and p.inputs["Emission Strength"].default_value > 0:
            emissive.append(
                {
                    "material": m.name,
                    "color": list(p.inputs["Emission Color"].default_value[:3]),
                    "strength": p.inputs["Emission Strength"].default_value,
                }
            )
    w = sc.world
    bg = next((n for n in w.node_tree.nodes if n.type == "BACKGROUND"), None) if w and w.use_nodes else None
    cams = []
    for o in sc.objects:
        if o.type == "CAMERA":
            c = o.data
            cams.append(
                {
                    "name": o.name,
                    "matrix_world": [list(r) for r in o.matrix_world],
                    "lens_mm": c.lens,
                    "sensor_width_mm": c.sensor_width,
                    "sensor_fit": c.sensor_fit,
                    "angle_x_deg": math.degrees(c.angle_x),
                    "angle_y_deg": math.degrees(c.angle_y),
                }
            )
    return {
        "frame": "Blender world = MuJoCo world (Z up, metres); Unreal = (x, -y, z) * 100 cm",
        "lights": lights,
        "emissive_materials": emissive,
        "world": {
            "color": list(bg.inputs["Color"].default_value[:3]) if bg else None,
            "strength": bg.inputs["Strength"].default_value if bg else None,
        },
        "render": {
            "engine": sc.render.engine,
            "view_transform": sc.view_settings.view_transform,
            "look": sc.view_settings.look,
            "exposure": sc.view_settings.exposure,
            "resolution": [sc.render.resolution_x, sc.render.resolution_y],
        },
        "cameras": cams,
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "room_scene.json").write_text(json.dumps(scene_record(), indent=1))
    setup_cycles()

    for o in sc.objects:
        o.hide_set(False)
        o.hide_viewport = False
        o.hide_select = False
    curves = [o for o in sc.objects if o.type in ("CURVE", "FONT") and not o.hide_render]
    if curves:
        select_only(curves)
        bpy.ops.object.convert(target="MESH")
        log(f"converted {len(curves)} curve/text objects to meshes")

    meshes = [o for o in sc.objects if o.type == "MESH" and not o.hide_render and len(o.data.polygons)]
    by_mat = collections.defaultdict(list)
    for o in meshes:
        m = o.material_slots[0].material if o.material_slots else None
        by_mat[m].append(o)
    log(f"{len(meshes)} meshes, {len(by_mat)} materials")

    report = []
    for mat, objs in sorted(by_mat.items(), key=lambda kv: kv[0].name if kv[0] else ""):
        if not is_procedural(mat):
            dropped = drop_height_bump(mat)
            report.append(
                {"material": mat.name if mat else None, "objects": len(objs), "baked": [], "bump_dropped": dropped}
            )
            continue
        p = principled(mat)
        todo = [s for s in ("Base Color", "Roughness", "Normal") if p and p.inputs[s].is_linked]
        reps = list({o.data.name: o for o in objs}.values())
        area = sum(world_area(o) for o in reps)
        res = pow2_res(area)
        t = time.time()
        # EMIT and NORMAL bakes need no lighting: render only this material's meshes, or every bake re-syncs
        # the whole 490k-triangle room (111 s per material before, most of it scene sync)
        for o in meshes:
            o.hide_render = o not in reps
        unwrap(reps)
        safe = "".join(c if c.isalnum() else "_" for c in mat.name).strip("_")
        baked = {}
        for s in todo:
            kind = "normal" if s == "Normal" else "emit"
            img = new_image(f"{safe}_{s.replace(' ', '').lower()}", res, non_color=(s != "Base Color"))
            bake_input(mat, reps, s, img, kind)
            save_image(img)
            baked[s] = img
        rewire(mat, baked)
        # the other UV maps of these meshes are unused by the baked material; keep only BAKE_UV
        for o in reps:
            for layer in [layer for layer in o.data.uv_layers if layer.name != BAKE_UV]:
                o.data.uv_layers.remove(layer)
        report.append(
            {
                "material": mat.name,
                "objects": len(objs),
                "unique_meshes": len(reps),
                "area_m2": round(area, 2),
                "res": res,
                "baked": todo,
                "seconds": round(time.time() - t, 1),
            }
        )
        log(
            f"{mat.name[:34]:34s} objs={len(objs):4d} unique={len(reps):4d} area={area:7.1f} m2 res={res} "
            f"baked={todo} {time.time() - t:.0f}s"
        )

    for o in meshes:
        o.hide_render = False

    pot = make_images_pot()
    log(f"resized {len(pot)} non-power-of-two images: {pot}")

    # one mesh per material, world coordinates
    joined = []
    for mat, objs in by_mat.items():
        select_only(objs)
        if len(objs) > 1:
            bpy.ops.object.join()
        j = view_layer.objects.active
        select_only([j])
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        j.name = "room_" + ("".join(c if c.isalnum() else "_" for c in mat.name) if mat else "nomaterial")
        joined.append(j)
    select_only(joined)
    bpy.ops.export_scene.gltf(
        filepath=str(OUT / "room.glb"),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_lights=False,
        export_cameras=False,
    )
    (OUT / "bake_report.json").write_text(json.dumps(report, indent=1))
    tris = sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in joined)
    log(f"exported {len(joined)} meshes, {tris} triangles -> {OUT / 'room.glb'} in {time.time() - t0:.0f}s")


main()
