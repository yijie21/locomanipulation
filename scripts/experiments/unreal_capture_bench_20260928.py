"""Where a URLab camera request's time goes: latency per request by mode, cameras and captures per request.

Drives a running URLab editor (warehouse level, robot imported) in puppet mode through consecutive state-trace rows,
the way the trace video does. Per configuration: request latency (push + camera request), GPU utilisation sampled by
nvidia-smi during it, and the editor's Slate tick interval (the game-thread frame the request waits for). Sets
urlab.FreshCaptureRepeats per configuration through the editor's command server (/workspace/ue_57/ue_cmd_server.py);
restores 8 at the end. --cvar applies console commands first (a diagnostic: shadows off, viewport off); the last
image of each configuration's first camera goes to <out>/<config>.jpg so a toggle's effect on the picture is visible.

    . /workspace/ue_57/ue_env.sh
    "$BRIDGE_PY" scripts/experiments/unreal_capture_bench_20260928.py --trace <state_trace.npz> \
        --trace-joints <scene_joints.json> --out <dir> --configs fresh:head_camera:1 fresh:all:8 stream:all:1

A configuration is mode:cameras:repeats; cameras is one MJCF camera name or "all" (the three dataset cameras).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unreal_smoke_20260928 import CAMERAS, apply_row, connect, push, qpos_map, uecmd  # noqa: E402


def set_repeats(n: int) -> None:
    uecmd(f"unreal.SystemLibrary.execute_console_command(None, 'urlab.FreshCaptureRepeats {n}')")


TICK_START = """
import time
unreal._loco_ticks = []
unreal._loco_tick_handle = unreal.register_slate_post_tick_callback(lambda dt: unreal._loco_ticks.append(time.perf_counter()))
"""
TICK_STOP = """
unreal.unregister_slate_post_tick_callback(unreal._loco_tick_handle)
t = unreal._loco_ticks
print(','.join(f'{b - a:.5f}' for a, b in zip(t, t[1:])))
"""


def tick_intervals_ms() -> np.ndarray:
    raw = uecmd(TICK_STOP).strip()
    return np.array([float(v) * 1e3 for v in raw.split(",") if v]) if raw else np.zeros(0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--trace-joints", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--configs", nargs="+", required=True, help="mode:cameras:repeats")
    ap.add_argument("--requests", type=int, default=30)
    ap.add_argument("--start-row", type=int, default=1000)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--host", default="tcp://127.0.0.1")
    ap.add_argument("--cvar", nargs="*", default=[], help="console commands to run first, e.g. 'r.ShadowQuality 0'")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for c in args.cvar:
        uecmd(f"unreal.SystemLibrary.execute_console_command(None, {c!r})")

    trace = np.load(args.trace)
    client = connect(args.host, local_model=False)
    if not client.manager_present:
        res = client.sim.start(raise_on_failure=False, timeout_s=300)
        print(f"[bench] PIE {res.state} {res.compile_error[:200]!r}", flush=True)
    client.local_model = True
    client.connect()
    client.runtime.set_mode("puppet")
    names = client.camera_names()
    ue = {c: next(n for n in names if n.split("/")[-1].endswith(c)) for c in CAMERAS}
    mapping = qpos_map(json.loads(Path(args.trace_joints).read_text()), client.model)
    apply_row(client, mapping, trace["qpos"][args.start_row])
    push(client)
    push(client, {n: "fresh" for n in names})  # every camera once: all switch to on-demand

    uecmd(TICK_START)
    time.sleep(3.0)
    idle = tick_intervals_ms()
    print(f"[bench] idle editor tick p50 {np.median(idle):.1f} ms ({len(idle)} ticks in 3 s)", flush=True)
    results = {"cvars": args.cvar, "idle_tick_ms_p50": float(np.median(idle)), "configs": []}

    for spec in args.configs:
        mode, cams, repeats = spec.split(":")
        want = list(CAMERAS) if cams == "all" else [cams]
        set_repeats(int(repeats))
        rows = [args.start_row + k * args.stride for k in range(args.requests + 3)]
        for i in rows[:3]:  # settle history at the new setting
            apply_row(client, mapping, trace["qpos"][i])
            push(client)
            push(client, {ue[c]: mode for c in want})
        smi = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
                "-lms",
                "100",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        uecmd(TICK_START)
        push_ms, cam_ms = [], []
        for i in rows[3:]:
            t0 = time.perf_counter()
            apply_row(client, mapping, trace["qpos"][i])
            push(client)
            t1 = time.perf_counter()
            reply = push(client, {ue[c]: mode for c in want})
            t2 = time.perf_counter()
            missing = [c for c in want if ue[c] not in reply.get("cameras", {})]
            if missing:
                raise SystemExit(f"[bench] {spec}: no frame for {missing}")
            push_ms.append((t1 - t0) * 1e3)
            cam_ms.append((t2 - t1) * 1e3)
        ticks = tick_intervals_ms()
        smi.terminate()
        util = np.array([float(v) for v in smi.communicate()[0].split() if v.strip().isdigit()])
        r = dict(
            spec=spec,
            requests=args.requests,
            push_ms_p50=float(np.median(push_ms)),
            camera_request_ms_p50=float(np.median(cam_ms)),
            camera_request_ms_p90=float(np.percentile(cam_ms, 90)),
            ticks_per_request=len(ticks) / args.requests,
            tick_ms_p50=float(np.median(ticks)) if len(ticks) else None,
            gpu_util_mean=float(util.mean()) if len(util) else None,
        )
        results["configs"].append(r)
        blk = reply["cameras"][ue[want[0]]]
        px = np.frombuffer(blk["data"], np.uint8).reshape(int(blk["height"]), int(blk["width"]), 4)[..., [2, 1, 0]]
        Image.fromarray(np.ascontiguousarray(px)).save(args.out / f"{spec.replace(':', '_')}.jpg", quality=92)
        print(
            f"[bench] {spec:28s} push {r['push_ms_p50']:6.1f} ms  camera request p50 {r['camera_request_ms_p50']:6.1f}"
            f" p90 {r['camera_request_ms_p90']:6.1f} ms  ticks/request {r['ticks_per_request']:.1f}"
            f" (tick p50 {r['tick_ms_p50'] or 0:.1f} ms)  GPU {r['gpu_util_mean'] or 0:.0f}%",
            flush=True,
        )
    set_repeats(8)
    (args.out / "bench.json").write_text(json.dumps(results, indent=1))
    client.close()


if __name__ == "__main__":
    main()
