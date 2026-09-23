"""Render a run's videos from its state traces -- the backend.render offline counterpart of SIM_VIDEO_DIR.

Each <run>/trace/<label>_state_trace.npz holds one qpos row per sim publish (the cadence the live
VideoRecorder.frame is called at), so replaying it through the same VideoRecorder, scene camera and insets
gives the video the sim would have written, without the sim paying for the rendering.

    MUJOCO_GL=egl third_party/GR00T-WholeBodyControl/.venv_sim/bin/python scripts/render_trace_videos.py \
        outputs/runs/<run> [--scene <dir>] [--out <run>/videos]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locomani.pipeline.sim_video import VideoRecorder  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("--scene", type=Path, help="scene directory; default: the run's scene copy")
    ap.add_argument("--out", type=Path, help="default: <run>/videos")
    a = ap.parse_args(argv)
    scene = a.scene or next(p for p in a.run.iterdir() if p.is_dir() and (p / "scene_locomani.xml").exists())
    out = a.out or a.run / "videos"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((scene / "scene_meta.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(scene / "scene_locomani.xml"))
    data = mujoco.MjData(model)
    recorder = VideoRecorder(model, str(out), camera_name=meta.get("video_camera"))
    for trace in sorted((a.run / "trace").glob("*_state_trace.npz")):
        qpos = np.load(trace)["qpos"]
        if qpos.shape[1] != model.nq:
            raise SystemExit(f"{trace.name}: qpos width {qpos.shape[1]} != scene nq {model.nq}")
        recorder.start(trace.name.removesuffix("_state_trace.npz"))
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            recorder.frame(data)
        recorder.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
