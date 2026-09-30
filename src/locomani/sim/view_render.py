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
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from locomani.pipeline.render_dataset_cameras import (
    MARKER,
    RENDER_SCRIPT,
    encoder,
    episode_length,
    expected_renderer,
    pending,
    video_path,
)
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
