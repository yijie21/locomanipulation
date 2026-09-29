"""Film a closed-loop evaluation's rounds with Unreal, for a run whose `images.source` is unreal.

    "$BRIDGE_PY" -m locomani.sim.unreal.render_eval_videos outputs/runs/<run> [--all]
    "$BRIDGE_PY" -m locomani.sim.unreal.render_eval_videos outputs/runs/<run> --dry-run

`scripts/render_eval_videos_unreal.sh` is one line over this; the editor must be running (`scripts/start_unreal.sh`)
with the scene's level built, camera rig and ghosts included (`export_mjcf`). The video is the collection's
(`view_render.film_eval`): the round's third-person shot, and as insets the dataset cameras as the Unreal mirror
rendered them for the policy -- one `view.UnrealView` on the scene's level and capture count
(`run_context.unreal_scene`), in `<run>/eval/videos_unreal/`, with previews in `<run>/eval/previews_unreal/`.
Successful rounds only, as `closed_loop.manifest.jsonl` judged them, unless --all; a filmed round is skipped.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.pipeline.lines import line_of
from locomani.pipeline.render_dataset_cameras import expected_renderer
from locomani.sim import view_render

ROOT = Path(paths.REPO)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory whose eval/ the closed loop wrote")
    ap.add_argument("--all", action="store_true", help="every traced round, not only the successful ones")
    ap.add_argument("--dry-run", action="store_true", help="say what would be filmed, film nothing")
    args = ap.parse_args(argv)

    run_dir = (args.run if args.run.is_absolute() else ROOT / args.run).resolve()
    cfg = run_context.load_run_config(run_dir)
    if expected_renderer(cfg) != "unreal":
        raise SystemExit(
            f"{run_dir.name}: images.source is {expected_renderer(cfg)}, not unreal; see docs/rendering.md"
        )
    unreal = run_context.unreal_scene(cfg)
    scene_dir = Path(line_of(cfg).scene_name(cfg, run_dir))
    labels = view_render.eval_rounds(run_dir, "unreal", args.all)
    print(f"{run_dir.name}: {len(labels)} evaluation round(s) to film, Unreal level {unreal['level_path']}", flush=True)
    if args.dry_run or not labels:
        return 0

    from locomani.sim.unreal.view import UnrealView

    log = lambda m: print(m, flush=True)  # noqa: E731
    view = UnrealView(str(scene_dir / "scene_locomani.xml"), unreal["level_path"], unreal["repeats"], log=log)
    try:
        view_render.film_eval(view, run_dir, scene_dir, labels, "unreal", log=log)
    finally:
        view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
