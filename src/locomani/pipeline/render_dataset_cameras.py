#!/usr/bin/env python3
"""Fill a backend.render offline run's dataset cameras from its state traces, before convert.

    python -m locomani.pipeline.render_dataset_cameras outputs/runs/<run> --gpu 0
    python -m locomani.pipeline.render_dataset_cameras outputs/runs/<run> --dry-run

`scripts/render_dataset_cameras.sh` is one line over this. Offline, the sim renders nothing and the exporter
records black placeholder frames (`locomani.sim.dataset_cameras`); every other column is real. Each parquet
row names the sim seq it used (`priv_seq/episode_N.json`) and the run's `trace/` holds the qpos at every seq,
so each row's three camera images are rendered exactly as the live sim does -- same scene copy, same
`CAMERA_CONFIGS`, default visual options -- and written over the placeholder videos in the session.

Checked against a live-recorded crate episode (2026-09-24): identical content; the live frame trails by 1-2
rows (the live cameras publish at ~30 Hz and the exporter takes the latest), so a rendered row is that much
fresher than a live one. Mean absolute difference 4-8 of 255 at the matching row, compression included.

Episodes done are recorded in `<session>/meta/offline_render.json`, so a re-run resumes, and convert refuses an
offline run while any episode still has its placeholder frames -- black frames would otherwise train silently
(the converter reuses the last decoded frame when one is missing, and black frames decode fine).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from locomani import paths
from locomani.pipeline import run_context
from locomani.sim.dataset_cameras import CAMERA_CONFIGS

ROOT = Path(paths.REPO)
MARKER = Path("meta") / "offline_render.json"


def recorded_episodes(session: Path) -> list[int]:
    """The session's saved episodes, from its own episode index. Not the priv_seq `discarded` flag: a failed
    round is flagged discarded and its parquet is written all the same (2 of 10 in a live crate session,
    2026-09-24), and `--include-discarded` converts it."""
    index = session / "meta" / "episodes.jsonl"
    if not index.exists():
        return []
    return [json.loads(line)["episode_index"] for line in index.read_text().splitlines() if line.strip()]


def pending(session: Path) -> list[int]:
    marker = session / MARKER
    done = set(json.loads(marker.read_text())["episodes"]) if marker.exists() else set()
    return [e for e in recorded_episodes(session) if e not in done]


def require_rendered(cfg: dict, sessions: list[str]) -> None:
    """Convert's guard: an offline run's sessions hold placeholder frames until this stage has run."""
    if cfg.get("backend", {}).get("render") != "offline":
        return
    left = {s: pending(Path(s)) for s in sessions}
    if any(left.values()):
        n = sum(len(v) for v in left.values())
        raise SystemExit(
            f"backend.render offline: {n} episode(s) still hold black placeholder frames; "
            "run scripts/render_dataset_cameras.sh <run> --gpu N first"
        )


def _video_path(session: Path, info: dict, key: str, ep: int) -> Path:
    chunk = ep // info["chunks_size"]
    return session / info["video_path"].format(episode_chunk=chunk, video_key=key, episode_index=ep)


def _encoder(dst: Path, width: int, height: int, fps: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst),
        ],
        stdin=subprocess.PIPE,
    )  # fmt: skip


def render_episode(traces, session: Path, ep: int, info: dict, model, data, renderers) -> int:
    """Render one episode's cameras over its placeholders; returns how many tail rows held the last trace sample."""
    import mujoco

    priv = traces.priv_seq(str(session), ep)
    length = next(
        json.loads(line)["length"]
        for line in (session / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip() and json.loads(line)["episode_index"] == ep
    )
    if len(priv) != length:
        raise SystemExit(f"{session.name} episode {ep}: {len(priv)} priv_seq rows but {length} parquet rows")
    trace, idx = traces.trace_rows(str(session), ep, priv, hold_tail=True)
    held = int(np.sum(priv > np.asarray(trace["seq"])[-1]))
    qpos = np.asarray(trace["qpos"])
    fps = info["fps"]
    tmp = {
        k: _video_path(session, info, f"observation.images.{k}", ep).with_suffix(".rendering.mp4") for k in renderers
    }
    writers = {k: _encoder(tmp[k], CAMERA_CONFIGS[k]["width"], CAMERA_CONFIGS[k]["height"], fps) for k in renderers}
    frames, last = {}, None
    for i in idx:
        if i != last:  # the exporter outpaces the sim's publishes now and then: repeated seqs render once
            data.qpos[:] = qpos[i]
            mujoco.mj_forward(model, data)
            for k, r in renderers.items():
                r.update_scene(data, camera=CAMERA_CONFIGS[k]["mjcf_name"])
                frames[k] = r.render()
            last = i
        for k, w in writers.items():
            w.stdin.write(frames[k].tobytes())
    for k, w in writers.items():
        w.stdin.close()
        if w.wait() != 0:
            raise SystemExit(f"ffmpeg failed on {tmp[k]}")
        os.replace(tmp[k], _video_path(session, info, f"observation.images.{k}", ep))
    return held


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory collection wrote")
    ap.add_argument("--gpu", default="0", help="the card EGL renders on")
    ap.add_argument("--dry-run", action="store_true", help="say what would be rendered, render nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    if cfg.get("backend", {}).get("render") != "offline":
        print(f"{run_dir.name}: backend.render is live; the cameras were recorded, nothing to render")
        return 0
    if not cfg.get("collect", {}).get("record", True):
        print(f"{run_dir.name}: collect.record is false; no exporter session was recorded, nothing to render")
        return 0
    from locomani.pipeline.world_lift import RunTraces

    traces = RunTraces(run_dir)
    todo = {Path(s): pending(Path(s)) for s in traces.sessions}
    total = sum(len(v) for v in todo.values())
    print(f"{run_dir.name}: {total} episode(s) to render in {len(todo)} session(s), scene {traces.scene_dir.name}")
    if args.dry_run or not total:
        return 0

    # before mujoco is imported: it picks the GL backend at import and the EGL device at context creation
    os.environ.update(MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=args.gpu, CUDA_VISIBLE_DEVICES=args.gpu)
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(traces.scene_dir / "scene_locomani.xml"))
    data = mujoco.MjData(model)
    for session, eps in todo.items():
        info = json.loads((session / "meta" / "info.json").read_text())
        keys = [f.split(".")[-1] for f in info["features"] if f.startswith("observation.images.")]
        missing = [k for k in keys if k not in CAMERA_CONFIGS]
        if missing:
            raise SystemExit(f"{session.name}: no camera config for {missing}")
        renderers = {k: mujoco.Renderer(model, CAMERA_CONFIGS[k]["height"], CAMERA_CONFIGS[k]["width"]) for k in keys}
        marker = session / MARKER
        state = (
            json.loads(marker.read_text())
            if marker.exists()
            else {"episodes": [], "held_tail_rows": {}, "scene": str(traces.scene_dir)}
        )
        for ep in eps:
            t0 = time.time()
            held = render_episode(traces, session, ep, info, model, data, renderers)
            state["episodes"].append(ep)
            state["held_tail_rows"][str(ep)] = held
            marker.write_text(json.dumps(state, indent=1))
            print(
                f"  {session.name} episode {ep}: {len(keys)} cameras in {time.time() - t0:.0f} s, "
                f"{held} tail rows past the trace hold its last sample",
                flush=True,
            )
        for r in renderers.values():
            r.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
