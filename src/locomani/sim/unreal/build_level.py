"""Build a scene's Unreal level in the running editor: the room level plus the scene's robot and carton.

    bash scripts/start_unreal.sh --gpu 0                                 # the editor, once
    bash scripts/build_scene.sh warehouse_textured --unreal              # = the scene's `unreal.build`:
    "$BRIDGE_PY" -m locomani.sim.unreal.build_level warehouse_textured
    "$BRIDGE_PY" -m locomani.sim.unreal.build_level --room <bake dir>    # the room level itself, once per install

A scene's level (`configs/scenes.yaml`, `unreal.level`) is a copy of the room level ROOM_LEVEL -- the Blender-baked
logistics room with its six area lights, emissive panels and the post-process volume that fixes exposure, Lumen
and its bounce boost (`in_editor/warehouse_level.py`, docs/ledger/blender_scenes.md section 7) -- with the scene's
moving part imported through URLab: the export of `export_mjcf` (robot, carton, the three dataset cameras), named
after the level, so the Blueprint is /Game/MuJoCoImports/<level>. Then the G1 materials (`in_editor/robot_materials`,
preset unitree, the Unitree Isaac asset's values, as Isaac's), Nanite off on every mesh (`in_editor/no_nanite`), the
level and the Blueprint saved: the next editor session opens it ready to render, without importing again.
`$UNREAL_ASSETS/<level>/source.json` records the scene XML's hash; `run_context.unreal_scene` refuses a level
built from another scene. Rebuilding replaces the level and its import.

The room level is built once per install from the Blender bake (scripts/unreal/warehouse_bake.py, in Blender):
`--room <bake dir>` runs `in_editor/warehouse_level.py` on it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.sim.unreal import editor, export_mjcf

ROOM_LEVEL = "/Game/Levels/Warehouse"  # in_editor/warehouse_level.py's default WAREHOUSE_LEVEL

_COPY_ROOM = """
# Open the room level, save it under the scene's name (save_map) and open that copy. Copying the asset instead
# (EditorAssetLibrary.duplicate_asset) and opening the copy kills the editor: the duplicate World stays loaded,
# standalone, and Map_Load's leak check is fatal (EditorServer.cpp:2524, 2026-09-29).
eal = unreal.EditorAssetLibrary
les = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if not eal.does_asset_exist({src!r}):
    raise SystemExit("no room level {src}: build it once, python -m locomani.sim.unreal.build_level --room <bake dir>")
if not les.load_level({src!r}):
    raise SystemExit("could not open {src}")
if eal.does_asset_exist({dst!r}) and not eal.delete_asset({dst!r}):
    raise SystemExit("could not delete the old {dst}")
world = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()
if not unreal.EditorLoadingAndSavingUtils.save_map(world, {dst!r}):
    raise SystemExit("could not save {src} as {dst}")
if not les.load_level({dst!r}):  # save_map writes the copy; the editor stays on the room until the copy is opened
    raise SystemExit("could not open {dst}")
eas = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
imports = [a for a in eas.get_all_level_actors() if a.get_class().get_path_name().startswith("/Game/MuJoCoImports/")]
gone = [a.get_name() for a in imports]
for a in imports:
    eas.destroy_actor(a)
les.save_current_level()
print("level", unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world().get_path_name(), "removed", gone)
"""


def build_room(bake_dir: Path) -> None:
    # Checked here, not in the editor: there the missing file is a traceback from the script and a second one from
    # the report it cannot write into the missing directory (a bake step skipped, 2026-09-29).
    missing = [f for f in ("room.glb", "room_scene.json") if not (bake_dir / f).is_file()]
    if missing:
        raise SystemExit(
            f"{bake_dir} is not a Blender bake ({', '.join(missing)} missing): bake the room first, blender"
            " --background assets/blender/logistics_room.blend --python scripts/unreal/warehouse_bake.py --"
            f" {bake_dir} (docs/setup.md, the Unreal step)"
        )
    editor.stop_pie()
    report_path = bake_dir / "unreal_level_report.json"
    report_path.unlink(missing_ok=True)  # the script writes it last: an old one must not read as this run's
    out = editor.checked(
        editor.run_file(editor.IN_EDITOR / "warehouse_level.py", WAREHOUSE_DIR=str(bake_dir.resolve()))
    )
    print(out.strip())
    if not report_path.is_file():
        raise SystemExit(f"the room level script wrote no {report_path}; see the editor log")
    report = json.loads(report_path.read_text())
    if not report.get("ok"):
        raise SystemExit(f"the room level failed: {report.get('error')}")


def build(name: str) -> dict:
    entry = run_context.unreal_entry(name)
    level = entry["level"]
    level_path = f"/Game/Levels/{level}"
    scene_dir = run_context.ROOT / run_context.resolve_scene({"scene": {"name": name}})["scene"]["dir"]
    out = Path(paths.UNREAL_ASSETS) / level
    t0 = time.time()
    exported = export_mjcf.export(scene_dir, out / "mjcf", level)
    print(f"[build_level] exported {exported}", flush=True)

    editor.stop_pie()
    print(
        "[build_level]",
        editor.checked(editor.uecmd(_COPY_ROOM.format(src=ROOM_LEVEL, dst=level_path))).strip(),
        flush=True,
    )
    print(
        "[build_level]",
        editor.checked(editor.run_file(editor.IN_EDITOR / "clear_import.py", URLAB_ASSET=level)).strip(),
        flush=True,
    )
    client = editor.connect(local_model=False)
    client.scene.load_level(level_path)
    client.scene.ensure_manager()
    bp = client.scene.import_xml(exported["xml"], force_reimport=True)
    client.scene.spawn_actor(bp, actor_id="g1")
    client.scene.save_level()
    client.close()  # PIE is not running: close() reverting to live is safe here
    asset = bp.class_path.split(".")[0]  # /Game/MuJoCoImports/<level>.<level>_C -> /Game/MuJoCoImports/<level>
    # URLab's import leaves the Blueprint in memory only (TODO 12.11): saved, the next session opens the level ready
    saved = editor.checked(
        editor.uecmd(f"print(unreal.EditorAssetLibrary.save_asset({asset!r}, only_if_is_dirty=False))")
    )
    if saved.strip() != "True":
        raise SystemExit(f"could not save {asset}: {saved.strip()}")
    for script, extra in (("robot_materials.py", {"URLAB_ASSET": level}), ("no_nanite.py", {})):
        print("[build_level]", editor.checked(editor.run_file(editor.IN_EDITOR / script, **extra)).strip(), flush=True)
    editor.checked(editor.uecmd("unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).save_current_level()"))
    editor.require_render_defaults()
    xml = scene_dir / "scene_locomani.xml"
    source = {
        "scene": name,
        "scene_xml": str(xml.resolve()),
        "scene_xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(),
        "level": level_path,
        "room_level": ROOM_LEVEL,
        "blueprint": asset,
        "builder": "locomani.sim.unreal.build_level",
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
    }
    (out / "source.json").write_text(json.dumps(source, indent=1) + "\n")
    return source


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", help="a scene name with an `unreal` entry in configs/scenes.yaml")
    ap.add_argument("--room", type=Path, help="build the room level from this Blender bake directory instead")
    args = ap.parse_args(argv)
    if args.room:
        build_room(args.room)
        return 0
    if not args.scene:
        ap.error("give a scene name, or --room <bake dir>")
    print(json.dumps(build(args.scene), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
