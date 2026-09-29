"""Unreal editor Python: turn Nanite off on every static mesh in the level, and save the meshes that changed.

The render default since 2026-09-28 (TODO 10.7.4): no Nanite mesh, and r.Shadow.Nanite 0 in the host project's
DefaultEngine.ini. Nanite shadows run a fixed per-face pass for every shadowed light, 13 ms per capture in the
warehouse however little moves; without them a capture costs 3.9 ms instead of 23.9, and the images differ only
on edges (outputs/unreal_render_speed_20260928/i_nanite_shadows).

Run after every URLab import (unreal_smoke_20260928.py setup), before the first bridge request starts PIE:
Interchange's default pipeline builds Nanite, and the import is rebuilt every session. The room meshes are
saved non-Nanite already (the level script turns Nanite off at import), so for them this is a no-op.
The mirror and the trace video refuse to render until it has run (unreal_smoke_20260928.require_render_defaults).

    python3 /workspace/ue_57/uepy.py -f scripts/experiments/unreal_no_nanite_20260928.py

SCOPE "robot" (prepend SCOPE = "robot") reproduces the 10.7.4.1 diagnostic: the robot only, the carton keeps
Nanite, nothing saved.
"""

import unreal

SCOPE = globals().get("SCOPE", "all")
ROOT = "/Game/MuJoCoImports"
KEEP_NANITE = ("mesh_nor_cal_",)  # the carton, in SCOPE "robot"

sub = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
if SCOPE == "all":
    world = unreal.EditorLevelLibrary.get_editor_world()
    meshes = {
        c.static_mesh.get_path_name(): c.static_mesh
        for a in unreal.GameplayStatics.get_all_actors_of_class(world, unreal.Actor)
        for c in a.get_components_by_class(unreal.StaticMeshComponent)
        if c.static_mesh
    }
else:
    meshes = {}
    for path in unreal.EditorAssetLibrary.list_assets(ROOT, recursive=True):
        mesh = unreal.EditorAssetLibrary.load_asset(path)
        if isinstance(mesh, unreal.StaticMesh) and not mesh.get_name().startswith(KEEP_NANITE):
            meshes[mesh.get_path_name()] = mesh
changed = []
for mesh in meshes.values():
    settings = sub.get_nanite_settings(mesh)
    if settings.enabled:
        settings.enabled = False
        sub.set_nanite_settings(mesh, settings, apply_changes=True)
        changed.append(mesh)
if SCOPE == "all":
    for mesh in changed:
        unreal.EditorAssetLibrary.save_loaded_asset(mesh, only_if_is_dirty=False)
left = [m for m in meshes.values() if sub.get_nanite_settings(m).enabled]
print(f"[no-nanite] scope {SCOPE}: {len(meshes) - len(left)}/{len(meshes)} meshes non-Nanite ({len(changed)} changed)")
