"""Render a run's videos from its state traces -- the backend.render offline counterpart of SIM_VIDEO_DIR.

Each <run>/trace/<label>_state_trace.npz holds one qpos row per sim publish (the cadence the live
VideoRecorder.frame is called at), so replaying it through the same VideoRecorder, scene camera and insets
gives the video the sim would have written, without the sim paying for the rendering.

    MUJOCO_GL=egl third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/render_trace_videos.py \
        outputs/runs/<run> [--scene <dir>] [--out <run>/videos] [--traces <run>/trace]

A closed-loop evaluation's stack writes its traces to <run>/eval/state_trace: pass --traces and --out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locomani.pipeline.sim_video import VideoRecorder, recording_camera  # noqa: E402
from locomani.sim.scene_edges import round_edge  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--scene", type=Path, help="scene directory; default: the run's scene copy")
    ap.add_argument("--out", type=Path, help="default: <run>/videos")
    ap.add_argument("--traces", type=Path, help="state-trace directory; default: <run>/trace")
    a = ap.parse_args(argv)
    scene = a.scene or next(p for p in a.run.iterdir() if p.is_dir() and (p / "scene_locomani.xml").exists())
    out = a.out or a.run / "videos"
    traces = a.traces or a.run / "trace"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((scene / "scene_meta.json").read_text())
    # Ghost-hands rule (2026-09-24): a run whose expert wrote <label>_ghost.npz (the commanded grip targets, by
    # state seq) gets translucent ghost grippers at those targets in the third-person view. Geom group 4 is on
    # for the tracking camera only; the ego/wrist insets render with default options and never show them.
    ghost_files = {p.name.removesuffix("_ghost.npz"): p for p in traces.glob("*_ghost.npz")}
    ghost_ids = None
    if ghost_files:
        from locomani.sim.isaac import ghost as G
        from locomani.tasks.crate.ghost_trace import ghost_poses, load_ghost_trace

        model, ghost_ids = G.inject_mujoco(str(scene / "scene_locomani.xml"))
    else:
        model = mujoco.MjModel.from_xml_path(str(scene / "scene_locomani.xml"))
    data = mujoco.MjData(model)
    recorder = VideoRecorder(model, str(out), camera_name=meta.get("video_camera"))
    if ghost_ids is not None:
        recorder.vopt.geomgroup[4] = 1
    for trace in sorted(traces.glob("*_state_trace.npz")):
        npz = np.load(trace)
        qpos, seqs = npz["qpos"], npz["seq"]
        if qpos.shape[1] != model.nq:
            raise SystemExit(f"{trace.name}: qpos width {qpos.shape[1]} != scene nq {model.nq}")
        label = trace.name.removesuffix("_state_trace.npz")
        table = load_ghost_trace(str(ghost_files[label])) if ghost_ids is not None and label in ghost_files else None
        # A four-edge scene films each round from its edge's camera (the robot's first yaw names the edge).
        edge = round_edge(meta, qpos[0])
        if edge is not None:
            recorder.cam = recording_camera(model, camera_name=edge["video_camera"])
        recorder.start(label)
        for q, seq in zip(qpos, seqs):
            data.qpos[:] = q
            if ghost_ids is not None:
                gpos, gquat = ghost_poses(table, int(seq)) if table else ghost_poses({}, 0)
                G.set_mujoco(data, ghost_ids, gpos, gquat)
            mujoco.mj_forward(model, data)
            recorder.frame(data)
        recorder.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
