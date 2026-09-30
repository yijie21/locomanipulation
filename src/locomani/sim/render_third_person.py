"""A run's third-person videos, drawn by the renderer of its `images.source`: one command, run whenever wanted.

    bash scripts/render_third_person.sh outputs/runs/<run> [--gpu 1] [--eval] [--all] [--dry-run] [labels...]

`scripts/render_third_person.sh` runs this in the renderer's venv (MuJoCo: VENV_SIM_PY, Isaac: RENDER_PY, Unreal:
BRIDGE_PY). No stage runs it: collection, the dataset fill and convert work without it (user, 2026-09-30). Each round
is filmed from the shot `third_person_shot.round_camera` chooses -- the camera `render_trace_videos.py` films in
MuJoCo, the edge camera of a four-edge scene -- with the expert's ghost grippers (`<label>_ghost.npz`) and the
dataset cameras as insets (`view_render.film_rounds`).

Rounds: the collection's (`<run>/trace`, judged by the batch manifests) into `<run>/videos_<renderer>/`, or with
--eval the closed loop's (`<run>/eval/state_trace`, judged by `closed_loop.manifest.jsonl`) into
`<run>/eval/videos_<renderer>/` -- MuJoCo's directory is plain `videos/`, where its live sim records. Successful
rounds by default, every traced round with --all, or the labels given; a round with a video is skipped, so a rerun
resumes. Previews <= 500 KB go to `previews_<renderer>/` beside. Convert links the collection's videos into the
dataset's `third_person/` when they exist, so run this before convert to ship them in the dataset.

Insets: a collection round whose dataset cameras this renderer filled shows those frames (`view_render.filled_insets`);
any other round renders them first. The renderer is the view the dataset fill and the eval mirror use: Isaac's stage
and look on --gpu (Kit), Unreal's level in the running editor (its card: `start_unreal.sh --gpu`), MuJoCo on --gpu.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.pipeline.lines import line_of
from locomani.pipeline.render_dataset_cameras import expected_renderer
from locomani.pipeline.third_person import video_dir

ROOT = Path(paths.REPO)


def traces_dir(run_dir: Path, evaluation: bool) -> Path:
    return run_dir / "eval" / "state_trace" if evaluation else run_dir / "trace"


def chosen_rounds(run_dir: Path, evaluation: bool, every: bool, labels: list[str]) -> list[str]:
    """The labels to film: those given, every traced round, or the ones their manifest judged successful."""
    traces = traces_dir(run_dir, evaluation)
    traced = sorted(p.name.removesuffix("_state_trace.npz") for p in traces.glob("*_state_trace.npz"))
    if labels:
        missing = [label for label in labels if label not in traced]
        if missing:
            raise SystemExit(f"no state trace for {missing} in {traces}")
        return list(labels)
    return traced if every else run_context.successful_traces(run_dir, traces)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory")
    ap.add_argument("labels", nargs="*", help="only these rounds (their trace names without _state_trace.npz)")
    ap.add_argument("--gpu", default="1", help="the card Isaac or MuJoCo renders on (Unreal: the editor's card)")
    ap.add_argument("--eval", action="store_true", help="the closed-loop evaluation's rounds, not the collection's")
    ap.add_argument("--all", action="store_true", help="every traced round, not only the successful ones")
    ap.add_argument("--dry-run", action="store_true", help="say what would be filmed, film nothing")
    args = ap.parse_intermixed_args(argv)

    run_dir = (args.run if args.run.is_absolute() else ROOT / args.run).resolve()
    cfg = run_context.load_run_config(run_dir)
    renderer = expected_renderer(cfg)
    scene_dir = Path(line_of(cfg).scene_name(cfg, run_dir))
    base = run_dir / "eval" if args.eval else run_dir
    out_dir, previews = video_dir(base, renderer), base / f"previews_{renderer}"
    labels = chosen_rounds(run_dir, args.eval, args.all, args.labels)
    todo = [label for label in labels if not (out_dir / f"{label}.mp4").exists()]
    where = {"isaac": lambda: f"Isaac stage {run_context.isaac_scene(cfg)['dir']} on GPU {args.gpu}",
             "unreal": lambda: f"Unreal level {run_context.unreal_scene(cfg)['level_path']} in the running editor",
             "mujoco": lambda: f"MuJoCo on GPU {args.gpu}"}[renderer]()  # fmt: skip
    print(
        f"{run_dir.name}: {len(todo)} {'evaluation' if args.eval else 'collection'} round(s) to film with {where} "
        f"into {out_dir.relative_to(run_dir.parent)} ({len(labels) - len(todo)} of {len(labels)} already filmed)",
        flush=True,
    )
    if args.dry_run or not todo:
        return 0
    if renderer == "mujoco":  # before mujoco is imported: it picks the GL backend at import, the card at context
        os.environ.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu))
    from locomani.sim import view_render

    filled = {} if args.eval else view_render.filled_insets(run_dir, renderer, todo)
    traces = traces_dir(run_dir, args.eval)
    rounds = [(label, traces / f"{label}_state_trace.npz", filled.get(label)) for label in todo]
    print(f"  insets: {len(filled)} round(s) from the filled dataset, {len(todo) - len(filled)} rendered", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = str(scene_dir / "scene_locomani.xml")
    log = lambda m: print(m, flush=True)  # noqa: E731

    def work(view):
        view_render.film_rounds(view, scene_dir, rounds, renderer, out_dir, previews, log=log)

    if renderer == "isaac":
        from locomani.sim.isaac.view import run_in_kit

        run_in_kit(scene_xml, run_context.isaac_scene(cfg), args.gpu, work, log=log)  # Kit's close ends the process
    elif renderer == "unreal":
        from locomani.sim.unreal.view import UnrealView

        unreal = run_context.unreal_scene(cfg)
        view = UnrealView(scene_xml, unreal["level_path"], unreal["repeats"], log=log)
        try:
            work(view)
        finally:
            view.close()
    else:
        from locomani.sim.mujoco_view import MujocoView

        view = MujocoView(scene_xml, log=log)
        try:
            work(view)
        finally:
            view.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
