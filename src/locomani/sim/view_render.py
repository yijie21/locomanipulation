"""Fill an offline run's dataset cameras from a renderer's view, before convert: the part Isaac and Unreal share.

`locomani.sim.isaac.render_dataset_cameras` (Isaac, `view.IsaacView`) and `locomani.sim.unreal.render_dataset_cameras`
(Unreal, `unreal.view.UnrealView`) each own their renderer's lifetime and call `plan` then `fill`. A view is any
object with `warm_up(qpos)` and `render(qpos) -> {stream: HxWx3 uint8}` -- the same object the renderer's eval
mirror serves the policy from (`locomani.sim.mirror`), so training and closed-loop images are one function of the
state (DECISIONS 2026-09-24).

The join is `locomani.pipeline.render_dataset_cameras`'s: each parquet row's priv_seq -> the trace qpos at that
seq, written over the same placeholder videos in the session and recorded in the same marker, with the renderer's
name and its `look` (the settings that decide the pixels: Isaac's stage, RTX settings and supersampling; Unreal's
level and capture passes). A session is never filled by two renderers or two looks. Convert takes a run only when
its sessions' renderer is the run's `images.source`.

Failed rounds (a session's `discarded_episode_indices`) keep their placeholders, as in the MuJoCo re-render.
A renderer's frame of the three cameras costs 70-220 ms against MuJoCo's ~3, so one more thing convert never reads
is cut: only the rows the converter's 50 -> 25 Hz pick keeps (`world_lift.resample_selection`) are rendered; every
other row repeats the rendered row before it, 20 ms older (the live camera trails 1-2 rows anyway).
Only finished batches (`collect/inputs.json` `batches`) are touched, so this can run beside a collection that
is still recording, and be rerun until the run is done.

`film` then draws each filled episode's third-person video with the same view (user, 2026-09-29: the video that
ships with the dataset is drawn by the renderer that drew its cameras): the round's shot
(`locomani.sim.third_person_shot`, the camera MuJoCo's `render_trace_videos.py` films) at every `sim_video.EVERY_NTH`-th
trace row, the ghost grippers of `<label>_ghost.npz` in it, and as insets the dataset's own filled frames of the
same rows -- what the policy trains on, decoded, never a second render. A view films with `film_setup(width, height,
fovy)`, `film_warm_up` and `film(qpos, (cam_pos, cam_quat), (ghost_pos, ghost_quat)) -> HxWx3 uint8`; the ghosts
are in that picture only. The videos land in `<run>/videos_<renderer>/<label>.mp4` (`third_person.video_dir`),
which convert links into the dataset's `third_person/`; an episode that has one is not filmed again.

`film_eval` films a closed-loop evaluation's rounds the same way (user, 2026-09-29: the evaluation's video is drawn
by the renderer that drew the policy's pictures), into `<run>/eval/videos_<renderer>/`. An evaluation records no
dataset, so its insets are the view's own `render` of each filmed row -- what the eval mirror serves the policy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from locomani.pipeline import sim_video
from locomani.pipeline.render_dataset_cameras import (
    MARKER,
    RENDER_SCRIPT,
    encoder,
    episode_length,
    expected_renderer,
    pending,
    unfilmed,
    video_path,
)
from locomani.pipeline.third_person import video_dir
from locomani.pipeline.world_lift import resample_selection
from locomani.sim.dataset_cameras import CAMERA_CONFIGS

STREAMS = tuple(CAMERA_CONFIGS)


def rendered_rows(length: int) -> set[int]:
    """The rows convert will read from an episode of `length` rows, whatever prefix its verification trim keeps:
    the pick of a prefix is a prefix of the whole episode's pick."""
    return set(resample_selection(length).tolist())


def finished_sessions(run_dir: Path, sessions: list[str]) -> list[str]:
    """The sessions of batches the collection has finished; the one still recording is left alone."""
    inputs = run_dir / "collect" / "inputs.json"
    return sessions[: json.loads(inputs.read_text())["batches"]] if inputs.exists() else sessions


def render_episode(view, traces, session: Path, ep: int, info: dict, streams) -> dict:
    """Render one episode's cameras over its placeholders; returns its receipt."""
    t0 = time.time()
    priv = traces.priv_seq(str(session), ep)
    length = episode_length(session, ep)
    if len(priv) != length:
        raise SystemExit(f"{session.name} episode {ep}: {len(priv)} priv_seq rows but {length} parquet rows")
    trace, idx = traces.trace_rows(str(session), ep, priv, hold_tail=True)
    qpos = np.asarray(trace["qpos"])
    wanted = rendered_rows(length)
    view.warm_up(qpos[idx[0]])
    fps = info["fps"]
    tmp = {k: video_path(session, info, f"observation.images.{k}", ep).with_suffix(".rendering.mp4") for k in streams}
    writers = {k: encoder(tmp[k], CAMERA_CONFIGS[k]["width"], CAMERA_CONFIGS[k]["height"], fps) for k in streams}
    frames, last, renders = None, None, []
    for row, i in enumerate(idx):
        if frames is None or (row in wanted and i != last):
            r0 = time.perf_counter()
            frames = view.render(qpos[i])
            renders.append((time.perf_counter() - r0) * 1e3)
            last = i
        for k, w in writers.items():
            w.stdin.write(frames[k].tobytes())
    for k, w in writers.items():
        w.stdin.close()
        if w.wait() != 0:
            raise SystemExit(f"ffmpeg failed on {tmp[k]}")
        os.replace(tmp[k], video_path(session, info, f"observation.images.{k}", ep))
    return {
        "rows": length,
        "renders": len(renders),
        "render_ms_p50": round(float(np.percentile(renders, 50)), 1),
        "held_tail_rows": int(np.sum(priv > np.asarray(trace["seq"])[-1])),
        "seconds": round(time.time() - t0, 1),
    }


def plan(run_dir: Path, cfg: dict, renderer: str, look: dict):
    """What `renderer` has to render in this run: (traces, {session: [episodes]}), or None when nothing applies.

    Refuses a run whose `images.source` is another renderer's, and a session another renderer or another look
    has started filling -- convert would then read a dataset drawn by two functions of the state.
    """
    if cfg.get("backend", {}).get("render") != "offline":
        print(f"{run_dir.name}: backend.render is live; the cameras were recorded, nothing to render")
        return None
    source = expected_renderer(cfg)
    if source != renderer:
        raise SystemExit(
            f"{run_dir.name}: images.source is {source}, so convert would refuse {renderer}-rendered sessions; "
            f"its dataset cameras are {source}'s: scripts/{RENDER_SCRIPT[source]}"
        )
    from locomani.pipeline.world_lift import RunTraces

    traces = RunTraces(run_dir)
    sessions = [Path(s) for s in finished_sessions(run_dir, traces.sessions)]
    for session in sessions:
        marker = session / MARKER
        if not marker.exists():
            continue
        done = json.loads(marker.read_text())
        if done.get("renderer", "mujoco") != renderer:
            raise SystemExit(f"{session.name}: its placeholders were filled by another renderer; not mixing")
        if any(done.get(k, 1 if k == "supersample" else None) != v for k, v in look.items()):
            raise SystemExit(f"{session.name}: its episodes were rendered with another {renderer} look; not mixing")
    return traces, {s: pending(s) for s in sessions}


def fill(view, traces, todo: dict, renderer: str, look: dict, log=print) -> None:
    """Render every pending episode in `todo` through `view`, recording each in its session's marker."""
    for session, eps in todo.items():
        if not eps:  # nothing to render, so nothing to open: a batch of rounds that failed before recording leaves a
            continue  # session that may be gone by now (warehouse_smoke batch 01, its instance root cleaned, 2026-09-29)
        info = json.loads((session / "meta" / "info.json").read_text())
        keys = [f.split(".")[-1] for f in info["features"] if f.startswith("observation.images.")]
        if sorted(keys) != sorted(STREAMS):
            raise SystemExit(f"{session.name}: image streams {keys}, {renderer} renders {list(STREAMS)}")
        marker = session / MARKER
        state = (
            json.loads(marker.read_text())
            if marker.exists()
            else {
                "episodes": [],
                "held_tail_rows": {},
                "scene": str(traces.scene_dir),
                "renderer": renderer,
                **look,
                "rows": "world_lift.resample_selection; the rows between repeat the one before",
                "receipts": {},
            }
        )
        for ep in eps:
            receipt = render_episode(view, traces, session, ep, info, STREAMS)
            state["episodes"].append(ep)
            state["held_tail_rows"][str(ep)] = receipt["held_tail_rows"]
            state["receipts"][str(ep)] = receipt
            marker.write_text(json.dumps(state, indent=1))
            log(
                f"  {session.name} episode {ep}: {receipt['renders']} renders for {receipt['rows']} rows in "
                f"{receipt['seconds']:.0f} s (p50 {receipt['render_ms_p50']} ms)"
            )


def film_todo(traces, sessions: list[Path], renderer: str) -> list[tuple[Path, int, str]]:
    """(session, episode, label) of the filled episodes of `sessions` that have no `renderer` third-person video yet;
    an episode not filled yet waits, since its insets are its filled frames."""
    todo = []
    for s, ep, label in unfilmed(traces, [str(s) for s in sessions], renderer):
        marker = Path(s) / MARKER
        if not marker.exists() or ep not in json.loads(marker.read_text())["episodes"]:
            continue
        if not label:
            raise SystemExit(f"{Path(s).name} episode {ep}: its manifest row names no label, so no trace to film")
        todo.append((Path(s), ep, label))
    return todo


def inset_frames(traces, session: Path, ep: int, info: dict, rows) -> dict:
    """`{mjcf camera: [HxWx3 RGB at its inset size, one per filmed row]}` from the episode's filled dataset videos:
    each filmed trace row shows the first parquet row that used it or a later seq (the exporter trails the sim)."""
    import cv2

    priv = traces.priv_seq(str(session), ep)
    _trace, idx = traces.trace_rows(str(session), ep, priv, hold_tail=True)
    pick = np.minimum(np.searchsorted(np.asarray(idx), np.asarray(list(rows))), len(idx) - 1)
    sizes = {name: (w, h) for name, _label, w, h, _x, _y in sim_video.INSETS}
    wanted = set(pick.tolist())
    out = {}
    for key in STREAMS:
        name = CAMERA_CONFIGS[key]["mjcf_name"]
        if name not in sizes:
            continue
        path = video_path(session, info, f"observation.images.{key}", ep)
        cap, frames, row = cv2.VideoCapture(str(path)), {}, 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if row in wanted:
                frames[row] = cv2.resize(bgr[..., ::-1], sizes[name], interpolation=cv2.INTER_AREA)
            row += 1
        cap.release()
        if len(frames) != len(wanted):
            raise SystemExit(f"{path}: {row} frames decoded, rows up to {max(wanted)} wanted for the insets")
        out[name] = [frames[p] for p in pick.tolist()]
    return out


def film_round(view, model, meta: dict, trace: Path, insets, dest: Path, renderer: str, log=print) -> None:
    """One round's third-person video into `dest`: the view films every recorded row of `trace` (`film_setup` done)
    with the ghosts of the `_ghost.npz` beside it, and `insets` gives each filmed row's `{mjcf camera: image}`."""
    from locomani.sim import third_person_shot as shot
    from locomani.tasks.crate.ghost_trace import ghost_poses, load_ghost_trace

    t0 = time.time()
    npz = np.load(trace)
    qpos, seq = np.asarray(npz["qpos"]), np.asarray(npz["seq"])
    rows = list(shot.film_rows(len(qpos)))
    cam_pos, cam_quat = shot.camera_poses(model, qpos[rows], shot.round_camera(model, meta, qpos[0]))
    ghost_file = trace.with_name(trace.name.replace("_state_trace.npz", "_ghost.npz"))
    table = load_ghost_trace(str(ghost_file)) if ghost_file.exists() else {}  # none (an eval round): parked
    size = (sim_video.HEIGHT, sim_video.WIDTH, 3)
    tmp = dest.with_suffix(".rendering.mp4")
    writer = encoder(tmp, sim_video.WIDTH, sim_video.HEIGHT, sim_video.FPS)
    renders = []
    for k, (t, row_insets) in enumerate(zip(rows, insets(rows), strict=True)):
        ghosts = ghost_poses(table, int(seq[t]))
        if k == 0:
            view.film_warm_up(qpos[t], (cam_pos[k], cam_quat[k]), ghosts)
        r0 = time.perf_counter()
        frame = np.array(view.film(qpos[t], (cam_pos[k], cam_quat[k]), ghosts), dtype=np.uint8)
        renders.append((time.perf_counter() - r0) * 1e3)
        if frame.shape != size:
            raise SystemExit(f"{renderer} filmed a {frame.shape} frame, the video is {size}")
        sim_video.paste_insets(frame, row_insets)
        writer.stdin.write(frame.tobytes())
    writer.stdin.close()
    if writer.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {tmp}")
    os.replace(tmp, dest)
    log(
        f"  third-person {dest.name}, {len(rows)} frames in {time.time() - t0:.0f} s "
        f"(p50 {np.percentile(renders, 50):.0f} ms)"
    )


def film(view, traces, todo: list, renderer: str, log=print) -> None:
    """Film every (session, episode, label) of `todo` through `view` into `third_person.video_dir(run, renderer)`."""
    if not todo:
        return
    import mujoco

    from locomani.sim import third_person_shot as shot

    out_dir = video_dir(traces.run, renderer)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(traces.scene_dir / "scene_locomani.xml"))
    meta = json.loads((traces.scene_dir / "scene_meta.json").read_text())
    view.film_setup(sim_video.WIDTH, sim_video.HEIGHT, shot.scene_fovy(model, meta))
    for session, ep, label in todo:
        info = json.loads((session / "meta" / "info.json").read_text())

        def insets(rows, session=session, ep=ep, info=info):
            frames = inset_frames(traces, session, ep, info, rows)
            return ({name: images[k] for name, images in frames.items()} for k in range(len(rows)))

        log(f"  {session.name} episode {ep}:")
        film_round(view, model, meta, traces.trace_dir / f"{label}_state_trace.npz", insets, out_dir / f"{label}.mp4",
                   renderer, log)  # fmt: skip


# --- a closed-loop evaluation's rounds ------------------------------------------------------------------------------


def eval_rounds(run_dir: Path, renderer: str, every: bool) -> list[str]:
    """The evaluation rounds to film: those `closed_loop.manifest.jsonl` judged successful, or every traced round
    with `every`; a round `renderer` has filmed already is left out."""
    from locomani.pipeline.run_context import successful_traces

    traces = Path(run_dir) / "eval" / "state_trace"
    if every:
        labels = sorted(p.name.removesuffix("_state_trace.npz") for p in traces.glob("*_state_trace.npz"))
    else:
        labels = successful_traces(Path(run_dir), traces)
    done = video_dir(Path(run_dir) / "eval", renderer)
    return [label for label in labels if not (done / f"{label}.mp4").exists()]


def _render_insets(view, trace: Path, dest: Path) -> None:
    """Pass 1 of `film_eval`: the dataset cameras at every filmed row of `trace`, at their inset sizes, one video each
    in `dest` -- the pictures the eval mirror would have served the policy for those states."""
    import cv2

    from locomani.sim import third_person_shot as shot

    qpos = np.asarray(np.load(trace)["qpos"])
    rows = list(shot.film_rows(len(qpos)))
    sizes = {name: (w, h) for name, _label, w, h, _x, _y in sim_video.INSETS}
    cams = {k: CAMERA_CONFIGS[k]["mjcf_name"] for k in STREAMS if CAMERA_CONFIGS[k]["mjcf_name"] in sizes}
    tmp = dest.with_name(dest.name + ".rendering")
    tmp.mkdir(parents=True, exist_ok=True)
    writers = {k: encoder(tmp / f"{name}.mp4", *sizes[name], sim_video.FPS) for k, name in cams.items()}
    view.warm_up(qpos[rows[0]])
    for t in rows:
        images = view.render(qpos[t])
        for k, name in cams.items():
            writers[k].stdin.write(cv2.resize(images[k], sizes[name], interpolation=cv2.INTER_AREA).tobytes())
    for k, w in writers.items():
        w.stdin.close()
        if w.wait() != 0:
            raise SystemExit(f"ffmpeg failed on {tmp / cams[k]}.mp4")
    os.replace(tmp, dest)


def _read_insets(directory: Path, n_rows: int):
    """The per-row `{mjcf camera: image}` of `_render_insets`' videos, streamed."""
    import cv2

    caps = {p.stem: cv2.VideoCapture(str(p)) for p in sorted(directory.glob("*.mp4"))}
    for _ in range(n_rows):
        row = {}
        for name, cap in caps.items():
            ok, bgr = cap.read()
            if not ok:
                raise SystemExit(f"{directory / name}.mp4 ends before the {n_rows} filmed rows")
            row[name] = bgr[..., ::-1]
        yield row
    for cap in caps.values():
        cap.release()


def film_eval(view, run_dir: Path, scene_dir: Path, labels: list[str], renderer: str, log=print) -> None:
    """Film evaluation rounds into `<run>/eval/videos_<renderer>/<label>.mp4`, with ≤ 500 KB previews in
    `<run>/eval/previews_<renderer>/`. An evaluation records no dataset, so the insets are rendered here: pass 1
    renders every round's dataset cameras (`view.render`, before `film_setup`, which Isaac cannot undo) into
    `videos_<renderer>/insets/<label>/`, pass 2 films the rounds and deletes them."""
    if not labels:
        return
    import shutil

    import mujoco

    from locomani.pipeline.video_preview import make_previews
    from locomani.sim import third_person_shot as shot

    run_dir = Path(run_dir)
    traces = run_dir / "eval" / "state_trace"
    out_dir = video_dir(run_dir / "eval", renderer)
    for label in labels:
        dest = out_dir / "insets" / label
        if not dest.is_dir():
            t0 = time.time()
            _render_insets(view, traces / f"{label}_state_trace.npz", dest)
            log(f"  {label}: dataset cameras for the insets in {time.time() - t0:.0f} s")
    model = mujoco.MjModel.from_xml_path(str(scene_dir / "scene_locomani.xml"))
    meta = json.loads((scene_dir / "scene_meta.json").read_text())
    view.film_setup(sim_video.WIDTH, sim_video.HEIGHT, shot.scene_fovy(model, meta))
    for label in labels:
        dest = out_dir / f"{label}.mp4"
        pending = out_dir / "insets" / label
        film_round(view, model, meta, traces / f"{label}_state_trace.npz",
                   lambda rows, pending=pending: _read_insets(pending, len(rows)), dest, renderer, log)  # fmt: skip
        shutil.rmtree(pending)
        if not any(pending.parent.iterdir()):
            pending.parent.rmdir()
        written = make_previews(dest, run_dir / "eval" / f"previews_{renderer}" / label)
        log(f"  {label}: previews {', '.join(f'{p.name} {p.stat().st_size} B' for p, _s, _t in written)}")
