"""Unreal editor Python: give the URLab-imported robot and carton physically plausible PBR materials.

URLab's M_MuJoCo_Master takes BaseColor from the MJCF rgba but, without textures, ends Metallic in a constant 0.0
and Roughness in a constant 0.5: its "Metallic"/"Roughness" scalar parameters are not connected, so no instance
can change them and every G1 part renders as matte grey plastic (white under the room lights). This builds
/Game/Loco/M_LocoPBR, a drop-in master with URLab's parameter names (BaseColor, BaseColorTexture, bUseTexture,
Metallic, Roughness) actually wired, re-parents every instance of one URLab import to it, and sets:

    robot light grey (>= 0.6)   SILVER  } the robot look, `locomani.sim.robot_look` (SHELL, DARK), which build_level
    robot dark (< 0.4)          DARK    } passes in: the values Isaac's stage.G1_LOOKS binds, {base, metallic, roughness}
    carton (MI_nor_cal_*)       metallic 0  roughness 0.85
    anything else               metallic 0  roughness 0.6   base colour sRGB -> linear
    The third-person video's ghost grippers (export_mjcf.GHOST_MATERIAL) get their own translucent master,
    /Game/Loco/M_LocoGhost: the ghost colour (GHOST_BASE, linear) lit, with 0.4 of it emissive as the Isaac ghost,
    at GHOST_OPACITY; build_level passes the three from locomani.sim.isaac.ghost.

Earlier robot looks, for the record (robot_look's docstring has the current one's history): "metal", rejected
2026-09-28 (a uniform silver: light grey metallic 0.85 roughness 0.35 base 0.62, dark metallic 0.35 roughness 0.45);
"mujoco", until 2026-09-29 (the MJCF colours as non-metallic paint, light grey roughness 0.35, dark 0.5: too grey,
the gripper not black); "unitree", 2026-09-29 to 2026-10-01 (Unitree's Isaac Lab asset: light grey metallic 1
roughness 0.75 base 0.675, dark metallic 0.42 roughness 0.5 base 0.0107).

MJCF rgba is a display (sRGB) value -- MuJoCo's renderer applies no gamma -- while Unreal's BaseColor parameter is
linear, so URLab's straight copy shows 0.2 as 0.48: the G1's black head and joints come out mid grey and the
whole robot reads white. The robot look sets the robot's colours directly (linear PBR values); the carton's colours
are Blender Principled values (blender_to_mujoco.py), already linear, and stay.

build_level runs this after every URLab import (the instances are regenerated); by hand, with the globals it passes:

    editor.run_file(<this file>, URLAB_ASSET="<level>", SILVER=robot_look.SHELL, DARK=robot_look.DARK)
"""

import collections

import unreal


def srgb_to_linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


URLAB_ASSET = globals().get("URLAB_ASSET", "dynamic")
MASTER = "/Game/Loco/M_LocoPBR"
SRC_TAG = "loco_imported_base_color"
SILVER, DARK = globals().get("SILVER"), globals().get("DARK")  # the robot look: the caller's, no default of its own
if SILVER is None or DARK is None:
    raise RuntimeError("robot_materials.py takes the robot look from its caller (SILVER, DARK): run it via build_level")
CARDBOARD = {"metallic": 0.0, "roughness": 0.85}
OTHER = {"metallic": 0.0, "roughness": 0.6}
GHOST_MATERIAL = globals().get("GHOST_MATERIAL", "loco_ghost")
GHOST_BASE = globals().get("GHOST_BASE", [1.0, 0.0144, 0.5225])
GHOST_OPACITY = globals().get("GHOST_OPACITY", 0.6)
GHOST_MASTER = "/Game/Loco/M_LocoGhost"

eal = unreal.EditorAssetLibrary
mel = unreal.MaterialEditingLibrary


def build_master():
    existing = unreal.load_asset(MASTER)
    if existing is not None:
        return existing
    folder, name = MASTER.rsplit("/", 1)
    m = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, folder, unreal.Material, unreal.MaterialFactoryNew()
    )
    color = mel.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -800, -200)
    color.set_editor_property("parameter_name", "BaseColor")
    color.set_editor_property("default_value", unreal.LinearColor(0.7, 0.7, 0.7, 1.0))
    tex = mel.create_material_expression(m, unreal.MaterialExpressionTextureSampleParameter2D, -800, 50)
    tex.set_editor_property("parameter_name", "BaseColorTexture")
    tex.set_editor_property("texture", unreal.load_asset("/Engine/EngineResources/WhiteSquareTexture"))
    mul = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -500, 0)
    mel.connect_material_expressions(tex, "RGB", mul, "A")
    mel.connect_material_expressions(color, "", mul, "B")
    switch = mel.create_material_expression(m, unreal.MaterialExpressionStaticSwitchParameter, -250, -100)
    switch.set_editor_property("parameter_name", "bUseTexture")
    mel.connect_material_expressions(mul, "", switch, "True")
    mel.connect_material_expressions(color, "", switch, "False")
    mel.connect_material_property(switch, "", unreal.MaterialProperty.MP_BASE_COLOR)
    for pname, default, prop, y in (
        ("Metallic", 0.0, unreal.MaterialProperty.MP_METALLIC, 250),
        ("Roughness", 0.5, unreal.MaterialProperty.MP_ROUGHNESS, 400),
    ):
        p = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -250, y)
        p.set_editor_property("parameter_name", pname)
        p.set_editor_property("default_value", default)
        mel.connect_material_property(p, "", prop)
    mel.recompile_material(m)
    eal.save_loaded_asset(m)
    return m


def build_ghost_master():
    existing = unreal.load_asset(GHOST_MASTER)
    if existing is not None:
        return existing
    folder, name = GHOST_MASTER.rsplit("/", 1)
    m = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        name, folder, unreal.Material, unreal.MaterialFactoryNew()
    )
    m.set_editor_property("blend_mode", unreal.BlendMode.BLEND_TRANSLUCENT)
    m.set_editor_property("translucency_lighting_mode", unreal.TranslucencyLightingMode.TLM_SURFACE_PER_PIXEL_LIGHTING)
    color = mel.create_material_expression(m, unreal.MaterialExpressionVectorParameter, -700, -200)
    color.set_editor_property("parameter_name", "BaseColor")
    mel.connect_material_property(color, "", unreal.MaterialProperty.MP_BASE_COLOR)
    glow = mel.create_material_expression(m, unreal.MaterialExpressionConstant, -700, 50)
    glow.set_editor_property("r", 0.4)
    mul = mel.create_material_expression(m, unreal.MaterialExpressionMultiply, -400, 0)
    mel.connect_material_expressions(color, "", mul, "A")
    mel.connect_material_expressions(glow, "", mul, "B")
    mel.connect_material_property(mul, "", unreal.MaterialProperty.MP_EMISSIVE_COLOR)
    opacity = mel.create_material_expression(m, unreal.MaterialExpressionScalarParameter, -400, 250)
    opacity.set_editor_property("parameter_name", "Opacity")
    mel.connect_material_property(opacity, "", unreal.MaterialProperty.MP_OPACITY)
    rough = mel.create_material_expression(m, unreal.MaterialExpressionConstant, -400, 400)
    rough.set_editor_property("r", 0.6)
    mel.connect_material_property(rough, "", unreal.MaterialProperty.MP_ROUGHNESS)
    mel.recompile_material(m)
    eal.save_loaded_asset(m)
    return m


master = build_master()
ghost_master = build_ghost_master()
counts = collections.Counter()
for path in eal.list_assets(f"/Game/MuJoCoImports/{URLAB_ASSET}_ue_Assets", recursive=False, include_folder=False):
    mi = eal.load_asset(path)
    if not isinstance(mi, unreal.MaterialInstanceConstant):
        continue
    name = mi.get_name()
    if GHOST_MATERIAL in name:
        mi.set_editor_property("parent", ghost_master)
        mel.set_material_instance_vector_parameter_value(mi, "BaseColor", unreal.LinearColor(*GHOST_BASE, 1.0))
        mel.set_material_instance_scalar_parameter_value(mi, "Opacity", GHOST_OPACITY)
        mel.update_material_instance(mi)
        eal.save_loaded_asset(mi)
        counts["ghost"] += 1
        continue
    # URLab reuses an existing instance on re-import, so read the imported colour once and keep it as metadata:
    # a second run must not convert an already converted colour (0.2 -> 0.033 -> 0.0026 happened)
    src = eal.get_metadata_tag(mi, SRC_TAG)
    if src:
        c = unreal.LinearColor(*(float(v) for v in src.split(",")))
    else:
        c = mel.get_material_instance_vector_parameter_value(mi, "BaseColor")
        eal.set_metadata_tag(mi, SRC_TAG, f"{c.r},{c.g},{c.b},{c.a}")
    textured = mel.get_material_instance_static_switch_parameter_value(mi, "bUseTexture")
    mi.set_editor_property("parent", master)
    lum = (c.r + c.g + c.b) / 3
    if "nor_cal" in name:
        rule, kind = CARDBOARD, "cardboard"
    elif lum >= 0.6 and abs(c.r - c.b) < 0.05 and not textured:
        rule, kind = SILVER, "silver"
    elif lum < 0.4:
        rule, kind = DARK, "dark"
    else:
        rule, kind = OTHER, "other"
    # the carton's colours come from Blender Principled values (blender_to_mujoco.py), already linear
    if rule.get("base") is not None:
        c = unreal.LinearColor(*rule["base"], c.a)
    elif kind in ("silver", "dark", "other") and not textured:
        c = unreal.LinearColor(srgb_to_linear(c.r), srgb_to_linear(c.g), srgb_to_linear(c.b), c.a)
    mel.set_material_instance_vector_parameter_value(mi, "BaseColor", c)
    mel.set_material_instance_static_switch_parameter_value(mi, "bUseTexture", bool(textured))
    mel.set_material_instance_scalar_parameter_value(mi, "Metallic", rule["metallic"])
    mel.set_material_instance_scalar_parameter_value(mi, "Roughness", rule["roughness"])
    mel.update_material_instance(mi)
    eal.save_loaded_asset(mi)
    counts[kind] += 1
print(f"[materials] master {MASTER}, robot silver {SILVER} dark {DARK}: {dict(counts)}")
