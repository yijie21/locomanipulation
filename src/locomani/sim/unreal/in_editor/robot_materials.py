"""Unreal editor Python: give the URLab-imported robot and carton physically plausible PBR materials.

URLab's M_MuJoCo_Master takes BaseColor from the MJCF rgba but, without textures, ends Metallic in a constant 0.0
and Roughness in a constant 0.5: its "Metallic"/"Roughness" scalar parameters are not connected, so no instance
can change them and every G1 part renders as matte grey plastic (white under the room lights). This builds
/Game/Loco/M_LocoPBR, a drop-in master with URLab's parameter names (BaseColor, BaseColorTexture, bUseTexture,
Metallic, Roughness) actually wired, re-parents every instance of one URLab import to it, and applies a preset:

    "unitree" (default, user 2026-09-29): Unitree's own Isaac Lab asset for the G1 with Dex1
    (huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds, g1_29dof_with_dex1_rev_1_0.usd), the same
    values as the Isaac renderer's stage.G1_LOOKS -- brushed-metal shells and a near-black head, joints and gripper.
    Base colours are OmniPBR's diffuse_color_constant x diffuse_tint, already linear; the dark roughness is
    OmniPBR's default, which that material leaves unset:
        robot light grey (>= 0.6)   metallic 1     roughness 0.75  base colour (0.675, 0.675, 0.692)
        robot dark (< 0.4)          metallic 0.42  roughness 0.5   base colour (0.0107, 0.0099, 0.0104)
    "mujoco": the look of the MJCF itself, the default until 2026-09-29 -- Unitree's two colours (0.7 shells, 0.2
    head/feet/joints/grippers) as painted, non-metallic surfaces with a specular sheen, which is what MuJoCo draws
    (g1_source/hero_vs_urlab_g1_meshes.png). The user found it too grey, the gripper not black enough:
        robot light grey (>= 0.6)   metallic 0     roughness 0.35  base colour sRGB -> linear
        robot dark (< 0.4)          metallic 0     roughness 0.5   base colour sRGB -> linear
    "metal": the first attempt, rejected 2026-09-28 (a uniform reflective silver, not the real G1):
        robot light grey            metallic 0.85  roughness 0.35  base colour 0.62
        robot dark                  metallic 0.35  roughness 0.45  base colour sRGB -> linear
    all: carton (MI_nor_cal_*) metallic 0 roughness 0.85; anything else metallic 0 roughness 0.6, sRGB -> linear.
    The third-person video's ghost grippers (export_mjcf.GHOST_MATERIAL) get their own translucent master,
    /Game/Loco/M_LocoGhost: the ghost colour (GHOST_BASE, linear) lit, with 0.4 of it emissive as the Isaac ghost,
    at GHOST_OPACITY; build_level passes the three from locomani.sim.isaac.ghost.

MJCF rgba is a display (sRGB) value -- MuJoCo's renderer applies no gamma -- while Unreal's BaseColor parameter is
linear, so URLab's straight copy shows 0.2 as 0.48: the G1's black head and joints come out mid grey and the
whole robot reads white. A preset either sets a robot colour directly (a linear PBR value) or, with base None,
converts the imported one; the carton's colours are Blender Principled values (blender_to_mujoco.py), already
linear, and stay.

Re-run after every URLab import (the instances are regenerated):

    "$BRIDGE_PY" -m locomani.sim.unreal.editor -f src/locomani/sim/unreal/in_editor/robot_materials.py
    (URLAB_ASSET below names the import; the command server passes no environment, so another preset is chosen by
    prefixing the source: editor.run_file(<this file>, PRESET="metal", URLAB_ASSET="<level>"))
"""

import collections

import unreal


def srgb_to_linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


URLAB_ASSET = globals().get("URLAB_ASSET", "dynamic")
MASTER = "/Game/Loco/M_LocoPBR"
SRC_TAG = "loco_imported_base_color"
PRESETS = {
    "unitree": {
        "silver": {"metallic": 1.0, "roughness": 0.75, "base": (0.675, 0.675, 0.6916)},
        "dark": {"metallic": 0.42, "roughness": 0.5, "base": (0.01074, 0.00991, 0.01043)},
    },
    "mujoco": {
        "silver": {"metallic": 0.0, "roughness": 0.35, "base": None},
        "dark": {"metallic": 0.0, "roughness": 0.5, "base": None},
    },
    "metal": {
        "silver": {"metallic": 0.85, "roughness": 0.35, "base": (0.62, 0.62, 0.62)},
        "dark": {"metallic": 0.35, "roughness": 0.45, "base": None},
    },
}
PRESET = globals().get("PRESET", "unitree")
SILVER, DARK = PRESETS[PRESET]["silver"], PRESETS[PRESET]["dark"]
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
print(f"[materials] master {MASTER}, preset {PRESET}: {dict(counts)}")
