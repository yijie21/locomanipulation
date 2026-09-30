"""A run's third-person videos, drawn by the renderer of its `images.source`: one command, run whenever wanted.

    bash scripts/render_third_person.sh outputs/runs/<run> [--gpu 1] [--eval] [--all] [--side-by-side] [--dry-run]
        [labels...]

`scripts/render_third_person.sh` runs this in the renderer's venv (MuJoCo: VENV_SIM_PY, Isaac: RENDER_PY, Unreal:
BRIDGE_PY). No stage runs it: collection, the dataset fill and convert work without it (user, 2026-09-30). Each round
is filmed from the shot `third_person_shot.round_camera` chooses -- the camera `render_trace_videos.py` films in
MuJoCo, the edge camera of a four-edge scene -- with the expert's ghost grippers (`<label>_ghost.npz`) and the
dataset cameras as insets (`view_render.film_rounds`).

Rounds: the collection's (`<run>/trace`, judged by the batch manifests) into `<run>/videos_<renderer>/`, or with
--eval the closed loop's (`<run>/eval/state_trace`, judged by `closed_loop.manifest.jsonl`) into
`<run>/eval/videos_<renderer>/` -- MuJoCo's directory is plain `videos/`, where its live sim records. Successful
rounds by default, every traced round with --all, or the labels given; a round with a video is skipped, so a rerun
resumes -- unless the video is not `sim_video.WIDTH` x `HEIGHT` (filmed before the size changed): it is filmed again. Previews <= 500 KB go to `previews_<renderer>/` beside. Convert links the collection's videos into the
dataset's `third_person/` when they exist, so run this before convert to ship them in the dataset.

Insets: a collection round whose dataset cameras this renderer filled shows those frames (`view_render.filled_insets`);
any other round renders them first. The renderer is the view the dataset fill and the eval mirror use: Isaac's stage
and look on --gpu (Kit), Unreal's level in the running editor (its card: `start_unreal.sh --gpu`), MuJoCo on --gpu.

--side-by-side (Isaac or Unreal) also writes `<label>_vs_mujoco.mp4` beside each video: the MuJoCo replay of the
same trace on the left, this renderer's video on the right. Both film the same shot at the same rows
(`render_trace_videos.py` and `view_render.film_round` share `third_person_shot`), so the halves agree frame for frame
and any difference is the renderer's; a pair whose frame counts differ is refused. A missing MuJoCo half is rendered
first by `render_trace_videos.py` into MuJoCo's directory (`videos/`, `eval/videos/`), on --gpu. Previews go to
`previews_<renderer>/<label>_vs_mujoco*.mp4` (not `_side_by_side`: `render_isaac_videos.sh`'s tracking-shot previews
have that name in `previews_isaac/`, and a preview replaces every earlier file of its name).
"""

from __future__ import annotations

import argparse
import os
import subprocess
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


def replay_in_mujoco(run_dir: Path, evaluation: bool, labels: list[str], gpu: str) -> None:
    """MuJoCo's replay of `labels` into MuJoCo's video directory: `render_trace_videos.py`, the replay the runbook's
    MuJoCo command writes. Its venv, not this one: this process is the renderer's (Kit, or URLab's bridge)."""
    base = run_dir / "eval" if evaluation else run_dir
    env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(gpu), CUDA_VISIBLE_DEVICES=str(gpu))
    argv = [paths.VENV_SIM_PY, str(ROOT / "scripts/render_trace_videos.py"), str(run_dir)]
    argv += ["--traces", str(traces_dir(run_dir, evaluation)), "--out", str(video_dir(base, "mujoco")), *labels]
    subprocess.run(argv, env=env, check=True)


def frame_count(video: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=nb_read_packets", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return int(out.strip())


def video_size(video: Path) -> tuple[int, int] | None:
    """(width, height) of a video's first stream; None when ffprobe cannot read the file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
         str(video)],
        capture_output=True, text=True,
    )  # fmt: skip
    try:
        width, height = out.stdout.strip().split(",")
        return int(width), int(height)
    except ValueError:
        return None


def is_filmed(video: Path, size: tuple[int, int]) -> bool:
    """A video is there at the current picture size (one ffprobe cannot read counts as filmed)."""
    return video.exists() and video_size(video) in (None, size)


def side_by_side(left: Path, right: Path, dest: Path, previews: Path, log=print) -> None:
    """`left` beside `right` into `dest`, and its previews; the two must be the same shot, frame for frame."""
    from locomani.pipeline.video_preview import make_previews

    sizes = video_size(left), video_size(right)
    if sizes[0] is None or sizes[0] != sizes[1]:
        raise SystemExit(
            f"{left} is {sizes[0]} and {right} {sizes[1]}: not the same picture size; film the stale one again"
        )
    counts = frame_count(left), frame_count(right)
    if counts[0] != counts[1]:
        raise SystemExit(f"{left} has {counts[0]} frames and {right} {counts[1]}: not the same rows of one trace")
    tmp = dest.with_suffix(".stacking.mp4")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(left), "-i", str(right), "-filter_complex",
         "[0:v][1:v]hstack=inputs=2", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(tmp)],
        check=True,
    )  # fmt: skip
    os.replace(tmp, dest)
    written = make_previews(dest, previews / dest.stem)
    log(
        f"  {dest.name}: {counts[0]} frames; previews {', '.join(f'{p.name} {p.stat().st_size} B' for p, _s, _t in written)}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory")
    ap.add_argument("labels", nargs="*", help="only these rounds (their trace names without _state_trace.npz)")
    ap.add_argument("--gpu", default="1", help="the card Isaac or MuJoCo renders on (Unreal: the editor's card)")
    ap.add_argument("--eval", action="store_true", help="the closed-loop evaluation's rounds, not the collection's")
    ap.add_argument("--all", action="store_true", help="every traced round, not only the successful ones")
    ap.add_argument(
        "--side-by-side",
        action="store_true",
        help="also write <round>_vs_mujoco.mp4: MuJoCo's replay (left) beside this renderer's video (right)",
    )
    ap.add_argument("--dry-run", action="store_true", help="say what would be filmed, film nothing")
    args = ap.parse_intermixed_args(argv)

    run_dir = (args.run if args.run.is_absolute() else ROOT / args.run).resolve()
    cfg = run_context.load_run_config(run_dir)
    renderer = expected_renderer(cfg)
    if renderer == "mujoco":  # before mujoco is imported: it picks the GL backend at import, the card at context
        os.environ.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(args.gpu), CUDA_VISIBLE_DEVICES=str(args.gpu))
    from locomani.pipeline import sim_video

    size = (sim_video.WIDTH, sim_video.HEIGHT)
    scene_dir = Path(line_of(cfg).scene_name(cfg, run_dir))
    base = run_dir / "eval" if args.eval else run_dir
    out_dir, previews = video_dir(base, renderer), base / f"previews_{renderer}"
    labels = chosen_rounds(run_dir, args.eval, args.all, args.labels)
    todo = [label for label in labels if not is_filmed(out_dir / f"{label}.mp4", size)]
    stale = [label for label in todo if (out_dir / f"{label}.mp4").exists()]
    where = {"isaac": lambda: f"Isaac stage {run_context.isaac_scene(cfg)['dir']} on GPU {args.gpu}",
             "unreal": lambda: f"Unreal level {run_context.unreal_scene(cfg)['level_path']} in the running editor",
             "mujoco": lambda: f"MuJoCo on GPU {args.gpu}"}[renderer]()  # fmt: skip
    print(
        f"{run_dir.name}: {len(todo)} {'evaluation' if args.eval else 'collection'} round(s) to film with {where} "
        f"into {out_dir.relative_to(run_dir.parent)} ({len(labels) - len(todo)} of {len(labels)} already filmed)",
        flush=True,
    )
    pairs, replay = [], []
    if args.side_by_side:
        if renderer == "mujoco":
            raise SystemExit("--side-by-side puts MuJoCo beside another renderer; this run's images.source is mujoco")
        mujoco_dir = video_dir(base, "mujoco")
        pairs = [label for label in labels if not is_filmed(out_dir / f"{label}_vs_mujoco.mp4", (2 * size[0], size[1]))]
        replay = [label for label in pairs if not is_filmed(mujoco_dir / f"{label}.mp4", size)]
        stale += [label for label in replay if (mujoco_dir / f"{label}.mp4").exists()]
        print(
            f"  side by side: {len(pairs)} to stack, {len(replay)} MuJoCo replay(s) to render first into "
            f"{mujoco_dir.relative_to(run_dir.parent)}",
            flush=True,
        )
    if stale:
        print(f"  {len(stale)} earlier video(s) are not {size[0]}x{size[1]} and are filmed again", flush=True)
    if args.dry_run or not (todo or pairs):
        return 0
    if replay:
        replay_in_mujoco(run_dir, args.eval, replay, args.gpu)
    log = lambda m: print(m, flush=True)  # noqa: E731

    def stack_pairs():
        for label in pairs:
            left, right = video_dir(base, "mujoco") / f"{label}.mp4", out_dir / f"{label}.mp4"
            side_by_side(left, right, out_dir / f"{label}_vs_mujoco.mp4", previews, log=log)

    if not todo:
        stack_pairs()
        return 0
    from locomani.sim import view_render

    filled = {} if args.eval else view_render.filled_insets(run_dir, renderer, todo)
    traces = traces_dir(run_dir, args.eval)
    rounds = [(label, traces / f"{label}_state_trace.npz", filled.get(label)) for label in todo]
    print(f"  insets: {len(filled)} round(s) from the filled dataset, {len(todo) - len(filled)} rendered", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_xml = str(scene_dir / "scene_locomani.xml")

    def work(view):
        view_render.film_rounds(view, scene_dir, rounds, renderer, out_dir, previews, log=log)
        stack_pairs()  # inside: Isaac's Kit ends the process when it closes

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
