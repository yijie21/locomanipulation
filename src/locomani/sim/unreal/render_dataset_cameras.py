"""Fill a backend.render offline run's dataset cameras with Unreal pictures, before convert.

    "$BRIDGE_PY" -m locomani.sim.unreal.render_dataset_cameras outputs/runs/<run>
    "$BRIDGE_PY" -m locomani.sim.unreal.render_dataset_cameras outputs/runs/<run> --dry-run

`scripts/render_dataset_cameras_unreal.sh` is one line over this; the editor must be running
(`scripts/start_unreal.sh`, which chooses the GPU) with the scene's level built. The pictures come from
`view.UnrealView` on the scene's level and capture count (`run_context.unreal_scene`) -- the object and settings
the eval mirror serves the policy from. Which rows, which sessions and the marker (`renderer: unreal` plus the
level and the capture count) are `locomani.sim.view_render`, shared with the Isaac re-render. Every rendered row
is a fresh capture of its own pose (camera cut, `repeats` captures), as a mirror request is, so a picture does
not depend on the rows before it; about 130 ms a row, and only the rows convert keeps are rendered.
Then the same view films every filled episode's third-person video that is missing (`view_render.film`,
`<run>/videos_unreal/`); the level must carry the camera rig and ghosts (`export_mjcf`), or it asks for a rebuild.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.sim import view_render

ROOT = Path(paths.REPO)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory collection wrote")
    ap.add_argument("--dry-run", action="store_true", help="say what would be rendered, render nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    unreal = run_context.unreal_scene(cfg) if cfg.get("backend", {}).get("render") == "offline" else None
    look = {"unreal_level": unreal["level_path"], "repeats": unreal["repeats"]} if unreal else {}
    planned = view_render.plan(run_dir, cfg, "unreal", look)
    if planned is None:
        return 0
    traces, todo = planned
    total = sum(len(v) for v in todo.values())
    films = len(view_render.film_todo(traces, list(todo), "unreal")) + total
    print(
        f"{run_dir.name}: {total} kept episode(s) to render in {len(todo)} finished session(s) "
        f"of {len(traces.sessions)}, then {films} third-person video(s), Unreal level {unreal['level_path']}",
        flush=True,
    )
    if args.dry_run or not films:
        return 0

    from locomani.sim.unreal.view import UnrealView

    scene_xml = str(traces.scene_dir / "scene_locomani.xml")
    view = UnrealView(scene_xml, unreal["level_path"], unreal["repeats"], log=lambda m: print(m, flush=True))
    try:
        view_render.fill(view, traces, todo, "unreal", look, log=lambda m: print(m, flush=True))
        filming = view_render.film_todo(traces, list(todo), "unreal")
        view_render.film(view, traces, filming, "unreal", log=lambda m: print(m, flush=True))
    finally:
        view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
