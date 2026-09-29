"""Unreal editor Python: remove one URLab import (level actors, Blueprint, generated meshes and material instances)
so the next import_xml rebuilds every asset from the MJCF. URLab's force_reimport replaces the Blueprint but reuses
existing material instances, so edits made to them (unreal_robot_materials_20260928.py) survive a re-import.

    python3 /workspace/ue_57/uepy.py -f scripts/experiments/unreal_clear_import_20260928.py
"""

import unreal

URLAB_ASSET = "dynamic"

eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
gone = [a.get_name() for a in eas.get_all_level_actors() if a.get_class().get_name().startswith(f"{URLAB_ASSET}_C")]
for a in eas.get_all_level_actors():
    if a.get_class().get_name().startswith(f"{URLAB_ASSET}_C"):
        eas.destroy_actor(a)
unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()
eal = unreal.EditorAssetLibrary
deleted = []
for path in (f"/Game/MuJoCoImports/{URLAB_ASSET}", f"/Game/MuJoCoImports/{URLAB_ASSET}_ue_Assets"):
    if eal.does_asset_exist(path):
        deleted.append(path) if eal.delete_asset(path) else None
    elif eal.does_directory_exist(path):
        deleted.append(path) if eal.delete_directory(path) else None
print(f"[clear] actors {gone}, deleted {deleted}")
