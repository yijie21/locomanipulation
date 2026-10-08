"""Render one episode's views in Unreal (URLab) from its state trace, next to MuJoCo's.

The Unreal counterpart of render_isaac_videos.sh's main pass: every --stride-th state-trace row, pushed to a
running URLab editor as a puppet qpos, captured from the --camera list with the patched plugin; MuJoCo renders the
same row from the same cameras. Writes <out>/frames/NNNNN.jpg and <out>/source.mp4 at the trace rate /
stride (12.5 fps for 50 Hz, 4). One camera (the scene's video_camera, reverse_room for the warehouse) lays out as
Unreal | MuJoCo; several (head_camera,left_wrist,right_wrist) as an Unreal row over a MuJoCo row.

The trace is sim time, so the video is smooth where the lockstep eval's own frames are not: the mirror renders
only when the policy asks (~1.4 fps) and sim waits for it, which the trace does not record.

--mode stream (default) renders the rows as a game renders frames: one capture each that keeps TSR / Lumen history
from the row before (the first row primes it with a "fresh" capture). --mode fresh treats every row as a jump, as
the closed-loop mirror must: camera cut plus urlab.FreshCaptureRepeats captures (8 in the host project), ~5x slower.

The cameras must be in the URLab import (unreal_export_mjcf_20260928.py dynamic.xml plus the camera, at the
resolution to render); every other URLab camera is captured once first so none keeps rendering each frame.
Like the mirror, it refuses to start while any mesh is Nanite (run unreal_no_nanite_20260928.py after the import).

    MUJOCO_GL=egl "$BRIDGE_PY" scripts/experiments/unreal_trace_video_20260928.py --xml <scene.xml> \
        --trace <state_trace.npz> --trace-joints <scene_joints.json> --camera reverse_room --out <dir>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unreal_smoke_20260928 import apply_row, connect, push, qpos_map, require_render_defaults  # noqa: E402


def label(a: np.ndarray, text: str) -> np.ndarray:
    im = Image.fromarray(np.ascontiguousarray(a))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 8 * len(text) + 12, 24], fill=(0, 0, 0))
    d.text((6, 6), text, fill=(255, 255, 255))
    return np.asarray(im)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", required=True, help="the MuJoCo scene with the camera (reference renders)")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--trace-joints", required=True)
    ap.add_argument("--camera", default="reverse_room", help="comma-separated MJCF camera names")
    ap.add_argument("--mode", choices=["stream", "fresh"], default="stream")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--rate-hz", type=float, default=50.0, help="state-trace row rate")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    args = ap.parse_args()
    (args.out / "frames").mkdir(parents=True, exist_ok=True)

    require_render_defaults()
    trace = np.load(args.trace)
    src = json.loads(Path(args.trace_joints).read_text())
    client = connect(args.host, local_model=False)
    if not client.manager_present:
        res = client.sim.start(raise_on_failure=False, timeout_s=300)
        print(f"[video] PIE {res.state} {res.compile_error[:200]!r}", flush=True)
    client.local_model = True
    client.connect()
    client.runtime.set_mode("puppet")
    names = client.camera_names()
    cams = args.camera.split(",")
    ue_cam = {c: next(n for n in names if n.split("/")[-1].endswith(c)) for c in cams}
    mapping = qpos_map(src, client.model)

    ref = mujoco.MjModel.from_xml_path(args.xml)
    ref_map = qpos_map(src, ref)
    size = {}
    for c in cams:
        w, h = (int(v) for v in ref.cam_resolution[ref.camera(c).id])
        size[c] = (w, h) if w > 1 else (640, 480)
    ref.vis.global_.offwidth = max(640, *(w for w, _ in size.values()))
    ref.vis.global_.offheight = max(480, *(h for _, h in size.values()))
    data = mujoco.MjData(ref)

    rows = list(range(0, len(trace["qpos"]), args.stride))
    apply_row(client, mapping, trace["qpos"][rows[0]])
    push(client)
    push(client, {n: "fresh" for n in names})  # every camera once: all switch to on-demand
    times = []
    with ExitStack() as stack:
        renderers = {c: stack.enter_context(mujoco.Renderer(ref, size[c][1], size[c][0])) for c in cams}
        for k, i in enumerate(rows):
            t0 = time.perf_counter()
            apply_row(client, mapping, trace["qpos"][i])
            push(client)
            mode = "fresh" if k == 0 else args.mode
            reply = push(client, {ue_cam[c]: mode for c in cams})
            ue = {}
            for c in cams:
                blk = reply["cameras"][ue_cam[c]]
                px = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)
                ue[c] = px[..., [2, 1, 0]]
            times.append((time.perf_counter() - t0) * 1e3)
            for s, d, wd in ref_map:
                data.qpos[d : d + wd] = trace["qpos"][i][s : s + wd]
            mujoco.mj_forward(ref, data)
            tile = ue[cams[0]].shape[:2]
            top, bottom = [], []
            for c in cams:
                renderers[c].update_scene(data, camera=c)
                mj = renderers[c].render()
                u = ue[c]
                if u.shape[:2] != tile:
                    u = np.asarray(Image.fromarray(u).resize((tile[1], tile[0]), Image.LANCZOS))
                if mj.shape[:2] != tile:
                    mj = np.asarray(Image.fromarray(mj).resize((tile[1], tile[0]), Image.LANCZOS))
                name = "" if len(cams) == 1 else f"  {c}"
                clock = f"  t={i / args.rate_hz:5.1f}s" if c == cams[0] else ""
                top.append(label(u, f"Unreal{name}{clock}"))
                bottom.append(label(mj, f"MuJoCo{name}"))
            if len(cams) == 1:
                frame = np.concatenate(top + bottom, axis=1)
            else:
                frame = np.concatenate([np.concatenate(top, axis=1), np.concatenate(bottom, axis=1)], axis=0)
            Image.fromarray(frame).save(args.out / "frames" / f"{k:05d}.jpg", quality=92)
            if k % 100 == 0:
                print(f"[video] {k}/{len(rows)} capture p50 {np.median(times):.0f} ms", flush=True)
    fps = args.rate_hz / args.stride
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-framerate",
            f"{fps:g}",
            "-i",
            str(args.out / "frames" / "%05d.jpg"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(args.out / "source.mp4"),
        ],
        check=True,
    )
    print(
        f"[video] {len(rows)} frames at {fps:g} fps -> {args.out / 'source.mp4'}; capture p50 {np.median(times):.0f} ms"
    )
    client.close()


if __name__ == "__main__":
    main()
