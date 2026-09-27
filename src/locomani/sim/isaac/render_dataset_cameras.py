"""Fill a backend.render offline run's dataset cameras with Isaac pictures, before convert.

    "$RENDER_PY" -m locomani.sim.isaac.render_dataset_cameras outputs/runs/<run> --gpu 1
    "$RENDER_PY" -m locomani.sim.isaac.render_dataset_cameras outputs/runs/<run> --dry-run

`scripts/render_dataset_cameras_isaac.sh` is one line over this. The Isaac counterpart of
`locomani.pipeline.render_dataset_cameras`: the same join (each parquet row's priv_seq -> the trace qpos at that
seq), written over the same placeholder videos in the session, recorded in the same marker with
`renderer: isaac`. The pictures come from `view.IsaacView` on the scene's Isaac stage and look
(`run_context.isaac_scene`) -- the object and settings the eval mirror serves the policy from, so training and
closed-loop images are one function of the state (DECISIONS 2026-09-24). Convert takes an Isaac-rendered run
only when its `images.source` is isaac.

An Isaac frame of the three cameras costs ~70 ms against MuJoCo's ~3, so two things are cut, neither of which
convert reads:
- failed rounds (a session's `discarded_episode_indices`) keep their placeholders;
- only the rows the converter's 50 -> 25 Hz pick keeps (`world_lift.resample_selection`) are rendered; every
  other row repeats the rendered row before it, 20 ms older (the live camera trails 1-2 rows anyway).
Only finished batches (`collect/inputs.json` `batches`) are touched, so this can run beside a collection that
is still recording, and be rerun until the run is done.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from locomani import paths
from locomani.pipeline import run_context
from locomani.pipeline.render_dataset_cameras import (
    MARKER,
    encoder,
    episode_length,
    expected_renderer,
    pending,
    video_path,
)
from locomani.pipeline.world_lift import resample_selection
from locomani.sim.dataset_cameras import CAMERA_CONFIGS

ROOT = Path(paths.REPO)


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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory collection wrote")
    ap.add_argument("--gpu", default="1", help="the card Isaac renders on")
    ap.add_argument("--dry-run", action="store_true", help="say what would be rendered, render nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    if cfg.get("backend", {}).get("render") != "offline":
        print(f"{run_dir.name}: backend.render is live; the cameras were recorded, nothing to render")
        return 0
    if expected_renderer(cfg) != "isaac":
        raise SystemExit(
            f"{run_dir.name}: images.source is not isaac, so convert would refuse Isaac-rendered sessions; "
            "its dataset cameras are MuJoCo's: scripts/render_dataset_cameras.sh"
        )
    isaac = run_context.isaac_scene(cfg)
    from locomani.pipeline.world_lift import RunTraces

    traces = RunTraces(run_dir)
    sessions = [Path(s) for s in finished_sessions(run_dir, traces.sessions)]
    for session in sessions:
        marker = session / MARKER
        if marker.exists() and json.loads(marker.read_text()).get("renderer", "mujoco") != "isaac":
            raise SystemExit(f"{session.name}: its placeholders were filled by another renderer; not mixing")
    todo = {s: pending(s, kept_only=True) for s in sessions}
    total = sum(len(v) for v in todo.values())
    print(
        f"{run_dir.name}: {total} kept episode(s) to render in {len(sessions)} finished session(s) "
        f"of {len(traces.sessions)}, Isaac stage {isaac['dir']}",
        flush=True,
    )
    if args.dry_run or not total:
        return 0

    # before Kit starts: it takes the card from CUDA_VISIBLE_DEVICES
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES")
    from locomani.sim.isaac import stage as S
    from locomani.sim.isaac.view import IsaacView

    scene_xml = str(traces.scene_dir / "scene_locomani.xml")
    stage_cfg = S.StageCfg(scene_dir=isaac["dir"], robot_mjcf=scene_xml, absolute_camera=True)
    app = S.start_app(stage_cfg)
    try:
        streams = tuple(CAMERA_CONFIGS)
        view = IsaacView(stage_cfg, scene_xml, streams, isaac["rtx_settings"], log=lambda m: print(m, flush=True))
        for session, eps in todo.items():
            info = json.loads((session / "meta" / "info.json").read_text())
            keys = [f.split(".")[-1] for f in info["features"] if f.startswith("observation.images.")]
            if sorted(keys) != sorted(streams):
                raise SystemExit(f"{session.name}: image streams {keys}, Isaac renders {list(streams)}")
            marker = session / MARKER
            state = (
                json.loads(marker.read_text())
                if marker.exists()
                else {
                    "episodes": [],
                    "held_tail_rows": {},
                    "scene": str(traces.scene_dir),
                    "renderer": "isaac",
                    "isaac_stage": isaac["dir"],
                    "rtx_settings": isaac["rtx_settings"],
                    "rows": "world_lift.resample_selection; the rows between repeat the one before",
                    "receipts": {},
                }
            )
            for ep in eps:
                receipt = render_episode(view, traces, session, ep, info, streams)
                state["episodes"].append(ep)
                state["held_tail_rows"][str(ep)] = receipt["held_tail_rows"]
                state["receipts"][str(ep)] = receipt
                marker.write_text(json.dumps(state, indent=1))
                print(
                    f"  {session.name} episode {ep}: {receipt['renders']} renders for {receipt['rows']} rows in "
                    f"{receipt['seconds']:.0f} s (p50 {receipt['render_ms_p50']} ms)",
                    flush=True,
                )
    except BaseException:
        import sys
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)  # NOT app.close() first: it ends the process with status 0
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
