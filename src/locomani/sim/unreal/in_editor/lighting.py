"""Unreal editor Python: shift the running PIE world's lighting, never the saved level (TODO 21.3.1 pre-check).

Inputs are module-level names the sender sets (editor.run_file):

    LIGHT_SCALE        multiplies every light's intensity and every M_LocoEmissive panel's EmissiveStrength
    LIGHT_TEMPERATURE  kelvin for every light, or None to keep each light's own colour

PIE plays a copy of the level, so the shift ends with PIE and the level asset is untouched. Every value is taken
from the editor world's actor of the same label, so a second call replaces the first rather than compounding.
The emissive panels are given a dynamic instance of their own material; their colour stays as built (only the
lights take LIGHT_TEMPERATURE). Prints one JSON line: what was changed.
"""

import json

import unreal

EMISSIVE_MASTER = "/Game/Loco/M_LocoEmissive"  # in_editor/warehouse_level.py
LIGHT_SCALE = globals().get("LIGHT_SCALE", 1.0)
LIGHT_TEMPERATURE = globals().get("LIGHT_TEMPERATURE")

game = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_game_world()
if game is None:
    raise RuntimeError("no PIE world: the lighting shift applies to a running PIE only")
# get_editor_world() is None while PIE runs; the level's own world object is still loaded under its asset path
# (PIE's is the same path with a UEDPIE_<n>_ prefix on the name)
folder, name = game.get_path_name().rsplit("/", 1)
edit = unreal.load_object(None, f"{folder}/{name.split('_', 2)[2]}")


def actors(world):
    return {a.get_actor_label(): a for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)}


def emissive_source(material):
    """The M_LocoEmissive instance behind a slot (its constant, under a dynamic instance an earlier call made)."""
    if isinstance(material, unreal.MaterialInstanceDynamic):
        material = material.get_editor_property("parent")
    if not isinstance(material, unreal.MaterialInstanceConstant):
        return None
    parent = material.get_editor_property("parent")
    return material if parent is not None and parent.get_path_name().startswith(EMISSIVE_MASTER) else None


built = actors(edit)
lights, panels = [], []
for label, actor in actors(game).items():
    source = built.get(label)
    if source is None:
        continue
    for c, c0 in zip(
        actor.get_components_by_class(unreal.LightComponent), source.get_components_by_class(unreal.LightComponent)
    ):
        c.set_intensity(c0.get_editor_property("intensity") * LIGHT_SCALE)
        use = LIGHT_TEMPERATURE is not None
        c.set_use_temperature(use or c0.get_editor_property("use_temperature"))
        c.set_temperature(LIGHT_TEMPERATURE if use else c0.get_editor_property("temperature"))
        lights.append(label)
    for c in actor.get_components_by_class(unreal.StaticMeshComponent):
        for i, m in enumerate(c.get_materials()):
            mic = emissive_source(m)
            if mic is None:
                continue
            strength = unreal.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(
                mic, "EmissiveStrength"
            )
            mid = m if isinstance(m, unreal.MaterialInstanceDynamic) else c.create_dynamic_material_instance(i, mic)
            mid.set_scalar_parameter_value("EmissiveStrength", strength * LIGHT_SCALE)
            panels.append(f"{label}[{i}]")
print(json.dumps({"scale": LIGHT_SCALE, "temperature": LIGHT_TEMPERATURE, "lights": lights, "panels": panels}))
