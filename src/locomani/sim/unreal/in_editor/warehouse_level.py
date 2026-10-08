"""Unreal editor Python: build the warehouse level from the Blender export (scripts/unreal/warehouse_bake.py).

Runs inside the editor (PythonScriptPlugin); inputs are module-level names the sender sets (editor.run_file), or
the environment:

    WAREHOUSE_DIR   the bake output: room.glb + room_scene.json
    WAREHOUSE_LEVEL level asset path (default /Game/Levels/Warehouse)
    WAREHOUSE_ASSETS where the room's meshes and materials are imported (default /Game/Warehouse/Room); another
                    room (a scene_discovery apartment, build_level --room-level) takes its own directory
    WAREHOUSE_EXPOSURE_ONLY=1  load the existing level and only (re)set the exposure volume
    WAREHOUSE_EXPOSURE_OFFSET  stops added to the computed exposure compensation (calibration; default 0)
    WAREHOUSE_EMISSIVE_ONLY=1  load the existing level and only (re)build its emissive materials (stop PIE first)

    bash scripts/start_unreal.sh --gpu 0
    WAREHOUSE_DIR=<abs bake dir> "$BRIDGE_PY" -m locomani.sim.unreal.build_level --room <abs bake dir>

What it builds (the MuJoCo robot and carton come later through URLab, see build_level.py):

- the room: room.glb imported through Interchange (one static mesh per Blender material, world coordinates) and
  placed at the origin; Blender world = MuJoCo world, and URLab maps MuJoCo (x, y, z) m to Unreal (x, -y, z) cm.
  Nanite is turned off on every mesh (Interchange builds it by default): Nanite shadows cost 13 ms per capture
  here, see no_nanite.py;
- the six Blender area lights as Rect Lights, photometric: 1 Blender W = K = 683 lm (radiometric watts at the
  555 nm efficacy), so an emissive strength s becomes s * K nits, on our own emissive material (the glTF master
  renders no emissive, see emissive_materials);
- an unbound post-process volume, manual exposure with the physical camera off, where Unreal writes L * 2^bias / 1.2
  for a luminance L (PostProcessEyeAdaptation.cpp: CalculateManualAutoExposure, EV100 0; the bias is an exposure
  COMPENSATION, positive brightens). Cycles writes r * 2^exposure for a radiance r = L / K, so
  bias = log2(1.2 * 2^exposure / K) = -9.40 here. Measured against the Cycles still at reverse_room (2026-09-29,
  TODO 12.3.3) it needs no offset: walls and floor land within 2 of Blender, +0.5 stop lifts the floor 79 -> 109
  (Blender 80). The volume also forces Lumen GI and reflections (URLab copies it into every camera, and scene
  captures do not inherit the project's Lumen default) and boosts Lumen's bounce, see place_exposure.

Writes <WAREHOUSE_DIR>/unreal_level_report.json: every placed mesh's bounds in Unreal cm, for the frame check.
"""

import json
import math
import os
import traceback
from pathlib import Path

import unreal

K_LM_PER_W = 683.0


def _opt(name, default=None):
    """An option: a module-level name set by the sender (editor.run_file), else the environment."""
    return globals().get(name, os.environ.get(name, default))


DIR = Path(_opt("WAREHOUSE_DIR"))
LEVEL = _opt("WAREHOUSE_LEVEL", "/Game/Levels/Warehouse")
ASSETS = _opt("WAREHOUSE_ASSETS", "/Game/Warehouse/Room")
EMISSIVE_MASTER = "/Game/Loco/M_LocoEmissive"
LUMEN_DIFFUSE_COLOR_BOOST = 4.0  # place_exposure


def log(msg):
    unreal.log(f"[warehouse] {msg}")
    print(f"[warehouse] {msg}", flush=True)


def to_ue_pos(p):
    return unreal.Vector(p[0] * 100.0, -p[1] * 100.0, p[2] * 100.0)


def to_ue_dir(v):
    return unreal.Vector(v[0], -v[1], v[2])


def new_level():
    les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if unreal.EditorAssetLibrary.does_asset_exist(LEVEL):
        # deleting the level the editor has open segfaults it (ULevelInstanceSubsystem::OnAssetsPreDelete, 2026-09-29),
        # and the host project opens the room level at startup: step onto the engine's empty map first
        les.load_level("/Engine/Maps/Entry")
        unreal.EditorAssetLibrary.delete_asset(LEVEL)
    if not les.new_level(LEVEL):
        raise RuntimeError(f"new_level({LEVEL}) failed")
    return les


def import_room():
    if unreal.EditorAssetLibrary.does_directory_exist(ASSETS):
        unreal.EditorAssetLibrary.delete_directory(ASSETS)
    task = unreal.AssetImportTask()
    task.filename = str(DIR / "room.glb")
    task.destination_path = ASSETS
    task.automated = True
    task.replace_existing = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = unreal.EditorAssetLibrary.list_assets(ASSETS, recursive=True, include_folder=False)
    meshes = []
    for p in paths:
        a = unreal.EditorAssetLibrary.load_asset(p)
        if isinstance(a, unreal.StaticMesh):
            meshes.append(a)
    sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    for m in meshes:
        settings = sub.get_nanite_settings(m)
        if settings.enabled:
            settings.enabled = False
            sub.set_nanite_settings(m, settings, apply_changes=True)
    log(f"imported {len(paths)} assets, {len(meshes)} static meshes, Nanite off")
    return paths, meshes


def emissive_master():
    """/Game/Loco/M_LocoEmissive: BaseColor, Roughness, EmissiveColor * EmissiveStrength; two-sided."""
    existing = unreal.load_asset(EMISSIVE_MASTER)
    if existing is not None:
        return existing
    mel = unreal.MaterialEditingLibrary
    folder, name = EMISSIVE_MASTER.rsplit("/", 1)
    m = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, folder, unreal.Material, unreal.MaterialFactoryNew()
    )
    m.set_editor_property("two_sided", True)
    base = mel.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -600, -200)
    base.set_editor_property("parameter_name", "BaseColor")
    mel.connect_material_property(base, "", unreal.MaterialProperty.MP_BASE_COLOR)
    rough = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -600, 0)
    rough.set_editor_property("parameter_name", "Roughness")
    rough.set_editor_property("default_value", 0.5)
    mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    colour = mel.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -600, 200)
    colour.set_editor_property("parameter_name", "EmissiveColor")
    strength = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -600, 400)
    strength.set_editor_property("parameter_name", "EmissiveStrength")
    mul = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -300, 300)
    mel.connect_material_expressions(colour, "", mul, "A")
    mel.connect_material_expressions(strength, "", mul, "B")
    mel.connect_material_property(mul, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    mel.recompile_material(m)
    unreal.EditorAssetLibrary.save_loaded_asset(m)
    return m


def emissive_materials(meshes, scene):
    """Every emissive Blender material (room_scene.json), rebuilt on M_LocoEmissive at strength * K nits, on its
    mesh's slot.

    Interchange's glTF master (/InterchangeAssets/gltf/M_Default) renders no emissive in these captures: the ceiling
    panels read 167/255 however the strength is set (0, 2049, 20490 nits give the same pixels) against Blender's 238,
    and a green EmissiveFactor leaves them their base colour (2026-09-29, TODO 12.3.2). The new instances take base
    colour and roughness from the glTF instance and the emission from the bake's record, so rerunning is idempotent.
    """
    mel = unreal.MaterialEditingLibrary
    master = emissive_master()
    blender = {e["material"].replace(" ", "_"): e for e in scene["emissive_materials"]}
    out = []
    for mesh in meshes:
        for i, slot in enumerate(mesh.static_materials):
            src = slot.material_interface
            if src is None:
                continue
            if src.get_name().startswith("MI_LocoEmissive_"):
                src = unreal.load_asset(src.get_path_name().replace("MI_LocoEmissive_", ""))
            if not isinstance(src, unreal.MaterialInstanceConstant) or src.get_name() not in blender:
                continue
            e = blender[src.get_name()]
            factor, s = unreal.LinearColor(*e["color"], 1.0), float(e["strength"])
            folder = src.get_path_name().rsplit("/", 1)[0]
            name = f"MI_LocoEmissive_{src.get_name()}"
            mi = unreal.load_asset(f"{folder}/{name}") or unreal.AssetToolsHelpers.get_asset_tools().create_asset(
                name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew()
            )
            mi.set_editor_property("parent", master)
            mel.set_material_instance_vector_parameter_value(
                mi, "BaseColor", mel.get_material_instance_vector_parameter_value(src, "BaseColorFactor")
            )
            mel.set_material_instance_scalar_parameter_value(
                mi, "Roughness", mel.get_material_instance_scalar_parameter_value(src, "RoughnessFactor")
            )
            mel.set_material_instance_vector_parameter_value(mi, "EmissiveColor", factor)
            mel.set_material_instance_scalar_parameter_value(mi, "EmissiveStrength", s * K_LM_PER_W)
            mel.update_material_instance(mi)
            unreal.EditorAssetLibrary.save_loaded_asset(mi)
            mesh.set_material(i, mi)
            unreal.EditorAssetLibrary.save_loaded_asset(mesh)
            out.append({"mesh": mesh.get_name(), "material": src.get_name(), "strength": s, "unreal": s * K_LM_PER_W})
    log(f"emissive materials on {EMISSIVE_MASTER} at {K_LM_PER_W} nits per Blender unit: {out}")
    return out


def place_meshes(meshes):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    placed = []
    for m in meshes:
        actor = eas.spawn_actor_from_object(m, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        actor.set_actor_label(m.get_name())
        origin, extent = actor.get_actor_bounds(False)
        placed.append(
            {
                "mesh": m.get_name(),
                "origin_cm": [origin.x, origin.y, origin.z],
                "extent_cm": [extent.x, extent.y, extent.z],
            }
        )
    return placed


def place_lights(scene):
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    out = []
    for light in scene["lights"]:
        if light["type"] != "AREA":
            log(f"skipping {light['type']} light {light['name']}")
            continue
        m = light["matrix_world"]
        pos = [m[0][3], m[1][3], m[2][3]]
        fwd = [-m[0][2], -m[1][2], -m[2][2]]  # area lights emit along local -Z
        width = [m[0][0], m[1][0], m[2][0]]  # local X carries `size`
        rot = unreal.MathLibrary.make_rot_from_xy(to_ue_dir(fwd), to_ue_dir(width))
        actor = eas.spawn_actor_from_class(unreal.RectLight, to_ue_pos(pos), rot)
        actor.set_actor_label(light["name"])
        comp = actor.rect_light_component
        sx = light["size"]
        sy = light["size"] if light.get("shape") in ("SQUARE", "DISK") else light.get("size_y", sx)
        comp.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
        comp.set_intensity(light["energy_W"] * K_LM_PER_W)
        comp.set_light_color(unreal.LinearColor(*light["color"], 1.0))
        comp.set_editor_property("source_width", sx * 100.0)
        comp.set_editor_property("source_height", sy * 100.0)
        comp.set_editor_property("barn_door_length", 0.0)
        comp.set_editor_property("attenuation_radius", 3000.0)
        comp.set_editor_property("mobility", unreal.ComponentMobility.MOVABLE)
        out.append({"name": light["name"], "lumens": light["energy_W"] * K_LM_PER_W, "size_m": [sx, sy]})
    log(f"placed {len(out)} rect lights")
    return out


def place_exposure(scene):
    exposure = scene["render"]["exposure"]
    offset = float(_opt("WAREHOUSE_EXPOSURE_OFFSET", "0"))
    bias = math.log2(1.2 * 2.0**exposure / K_LM_PER_W) + offset
    eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    for a in eas.get_all_level_actors():
        if isinstance(a, unreal.PostProcessVolume):
            eas.destroy_actor(a)
    ppv = eas.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    ppv.set_actor_label("ManualExposure")
    ppv.set_editor_property("unbound", True)
    s = ppv.get_editor_property("settings")
    s.set_editor_property("override_auto_exposure_method", True)
    s.set_editor_property("auto_exposure_method", unreal.AutoExposureMethod.AEM_MANUAL)
    s.set_editor_property("override_auto_exposure_bias", True)
    s.set_editor_property("auto_exposure_bias", bias)
    s.set_editor_property("override_auto_exposure_apply_physical_camera_exposure", True)
    s.set_editor_property("auto_exposure_apply_physical_camera_exposure", False)
    # scene captures do not inherit the project's Lumen default: without these overrides a capture renders no GI
    # at all, however many frames it accumulates (diag 2026-09-28: ceiling strip 40 -> 133 with the override)
    s.set_editor_property("override_dynamic_global_illumination_method", True)
    s.set_editor_property("dynamic_global_illumination_method", unreal.DynamicGlobalIlluminationMethod.LUMEN)
    s.set_editor_property("override_reflection_method", True)
    s.set_editor_property("reflection_method", unreal.ReflectionMethod.LUMEN)
    # Lumen bounces about a third of Cycles' light into the ceiling (reverse_room: no GI 40, Lumen 90, Blender 174);
    # the diffuse colour boost raises bounce albedo: 4 takes the ceiling to 114, the grey error to Blender 23.1 -> 17.4,
    # the shelf cartons 141 -> 154 (Blender 153), and moves the direct-lit walls and floor 2-4 (TODO 12.3.3). The
    # indirect-lighting intensity and higher Lumen quality do nothing in these captures; 6 gives only 120.
    s.set_editor_property("override_lumen_diffuse_color_boost", True)
    s.set_editor_property("lumen_diffuse_color_boost", LUMEN_DIFFUSE_COLOR_BOOST)
    ppv.set_editor_property("settings", s)
    log(f"manual exposure compensation {bias:.2f} (Blender exposure {exposure}, offset {offset})")
    return bias


def main():
    report = {}
    try:
        scene = json.loads((DIR / "room_scene.json").read_text())
        if str(_opt("WAREHOUSE_EMISSIVE_ONLY")) == "1":
            les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
            les.load_level(LEVEL)
            world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
            actors = unreal.GameplayStatics.get_all_actors_of_class(world, unreal.StaticMeshActor)
            meshes = {a.static_mesh_component.static_mesh for a in actors if a.static_mesh_component.static_mesh}
            report["emissive"] = emissive_materials(sorted(meshes, key=lambda m: m.get_name()), scene)
            report["ok"] = True
            (DIR / "unreal_emissive_report.json").write_text(json.dumps(report, indent=1))
            return
        if str(_opt("WAREHOUSE_EXPOSURE_ONLY")) == "1":
            les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
            les.load_level(LEVEL)
            report["exposure_bias"] = place_exposure(scene)
            les.save_current_level()
            report["ok"] = True
            (DIR / "unreal_exposure_report.json").write_text(json.dumps(report, indent=1))
            return
        les = new_level()
        _paths, meshes = import_room()
        report["emissive"] = emissive_materials(meshes, scene)
        report["meshes"] = place_meshes(meshes)
        report["lights"] = place_lights(scene)
        report["exposure_bias"] = place_exposure(scene)
        les.save_current_level()
        unreal.EditorAssetLibrary.save_directory(ASSETS.rsplit("/", 1)[0])
        report["level"] = LEVEL
        report["ok"] = True
        log(f"saved {LEVEL}")
    except Exception:
        report["ok"] = False
        report["error"] = traceback.format_exc()
        log(report["error"])
    (DIR / "unreal_level_report.json").write_text(json.dumps(report, indent=1))


main()
