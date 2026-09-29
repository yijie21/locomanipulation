"""Film a closed-loop evaluation's rounds with Isaac, for a run whose `images.source` is isaac.

    "$RENDER_PY" -m locomani.sim.isaac.render_eval_videos outputs/runs/<run> --gpu 1 [--all]
    "$RENDER_PY" -m locomani.sim.isaac.render_eval_videos outputs/runs/<run> --dry-run

`scripts/render_eval_videos_isaac.sh` is one line over this. The video is the collection's (`view_render.film_eval`):
the round's third-person shot from `third_person_shot`, and as insets the dataset cameras as the Isaac mirror
rendered them for the policy -- one `view.IsaacView` on the run's stage and look (`run_context.isaac_scene`), in
`<run>/eval/videos_isaac/`, with previews in `<run>/eval/previews_isaac/`. Successful rounds only, as
`closed_loop.manifest.jsonl` judged them, unless --all; a filmed round is skipped, so a rerun resumes.
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
    ap.add_argument("--gpu", default="1", help="the card Isaac renders on")
    ap.add_argument("--all", action="store_true", help="every traced round, not only the successful ones")
    ap.add_argument("--dry-run", action="store_true", help="say what would be filmed, film nothing")
    args = ap.parse_args(argv)

    run_dir = (args.run if args.run.is_absolute() else ROOT / args.run).resolve()
    cfg = run_context.load_run_config(run_dir)
    if expected_renderer(cfg) != "isaac":
        raise SystemExit(f"{run_dir.name}: images.source is {expected_renderer(cfg)}, not isaac; see docs/rendering.md")
    isaac = run_context.isaac_scene(cfg)
    scene_dir = Path(line_of(cfg).scene_name(cfg, run_dir))
    labels = view_render.eval_rounds(run_dir, "isaac", args.all)
    print(f"{run_dir.name}: {len(labels)} evaluation round(s) to film, Isaac stage {isaac['dir']}", flush=True)
    if args.dry_run or not labels:
        return 0

    from locomani.sim.isaac.view import run_in_kit

    def work(view):
        view_render.film_eval(view, run_dir, scene_dir, labels, "isaac", log=lambda m: print(m, flush=True))

    run_in_kit(str(scene_dir / "scene_locomani.xml"), isaac, args.gpu, work, log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
