"""Measure T_render: how long one Isaac ego frame costs, at dataset resolution.

This is the number that decides whether closed-loop evaluation with Isaac
observations needs a hybrid clock. It is a timing measurement only -- whether
the pixels are CORRECT is a different question, answered by the offline
re-render's side-by-side against MuJoCo.

Two cadences, because the offline and online users differ:

  continuous : frames back to back, which is what the offline re-renderer does.
  cadenced   : one frame, then a gap, repeated -- which is what a live mirror
               does, one render per VLA observation. RTX reuses temporal state
               between consecutive frames, so a paused renderer can be slower;
               measuring only the continuous case would understate the online
               cost, which is the one that matters for the verdict. --gap takes
               several values: a mirror rendering at the action rate (0.04 s)
               and one rendering on demand per replan (0.4 s) differ here.

Resolution defaults to the dataset's 640x480 and quality is NOT reduced to
improve the number -- the whole point of the measurement is the cost of the
quality we intend to ship.

--state-trace times view.IsaacView -- the object the live mirror and the
offline dataset re-render both call -- over --streams (the dataset camera
streams; the bottle policy reads ego_view and right_wrist). push() is the
"sync" time, grab() (SETTLE_STEPS renders) the "render" time. The parquet +
sidecar path parents one ego camera to the torso instead, so it times a camera
mount that neither the mirror nor the offline re-render uses.

    "$RENDER_PY" -m locomani.sim.isaac.bench_render \
        --state-trace <trace.npz> --scene-xml <snapshot>/scene_locomani.xml \
        --scene-dir assets/isaac/scene/<label> --streams ego_view right_wrist \
        --gap 0.04 0.4 --out bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)  # reach src/ for `python src/locomani/sim/isaac/<mod>.py`
from locomani import paths
from locomani.sim.isaac import frames as F
from locomani.sim.isaac import stage as S

# The action stream runs at TF_DATASET_FPS, which run_eval_arm.sh pins to 25 --
# NOT 50. A chunk of 40 steps is 1.6 s at this rate, which is what the 0.4 s
# replan interval and the "10 frames of old plan per cycle" budget are measured
# against. Getting this wrong halves or doubles the verdict, so it is read from
# the same env var the eval uses rather than written as a literal.
CONTROL_HZ = float(os.environ.get("TF_DATASET_FPS", "25"))


def pct(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=float), q))


def machine_snapshot() -> dict:
    """Proof that nothing else was competing. A number measured next to an eval
    arm is a number about GPU contention, not about rendering."""

    def run(cmd):
        try:
            return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception as e:  # noqa: BLE001
            return f"<{e}>"

    procs = [
        l
        for l in run("ps -eo pid,pcpu,args --sort=-pcpu | head -25").splitlines()
        if any(
            k in l for k in ("eval_closed_loop", "grasp_gym", "run_data_exporter", "serve_policy", "g1_deploy", "isaac")
        )
    ]
    return {
        "gpus": run("nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv,noheader"),
        "competing_processes": procs,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "loadavg": run("uptime"),
    }


def series(one, frames, gap: float) -> dict:
    """Time one(t) over `frames`, sleeping `gap` seconds before each."""
    sync_ms, render_ms, empty = [], [], 0
    for t in frames:
        if gap:
            time.sleep(gap)
        s, r, ok = one(t)
        if not ok:
            empty += 1
            continue
        sync_ms.append(s * 1e3)
        render_ms.append(r * 1e3)
    total = [a + b for a, b in zip(sync_ms, render_ms)]
    return {
        "n": len(total),
        "empty_frames": empty,
        "sync_ms": {"p50": pct(sync_ms, 50), "p90": pct(sync_ms, 90)},
        "render_ms": {"p50": pct(render_ms, 50), "p90": pct(render_ms, 90)},
        "total_ms": {
            "p50": pct(total, 50),
            "p90": pct(total, 90),
            "p99": pct(total, 99),
            "mean": float(np.mean(total)),
            "max": float(np.max(total)),
        },
        "stale_frames_at_action_rate": {
            "p50": pct(total, 50) / 1e3 * CONTROL_HZ,
            "p90": pct(total, 90) / 1e3 * CONTROL_HZ,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-trace", default="", help="time IsaacView, the path the live mirror runs (preferred)")
    ap.add_argument("--streams", nargs="+", default=["ego_view"], help="dataset camera streams (--state-trace)")
    ap.add_argument(
        "--rtx-setting",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="the scene's carb settings (--state-trace), as isaac_render takes them; time the look that ships",
    )
    ap.add_argument("--session", default="")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--sidecar", default="")
    ap.add_argument("--assets", default=paths.ISAAC_ASSETS)
    ap.add_argument("--scene-dir", default="")
    ap.add_argument("--robot-usd", default="")
    ap.add_argument("--scene-xml", default=paths.SCENE_XML)
    ap.add_argument("--width", type=int, default=640)  # dataset resolution
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--n", type=int, default=300, help="timed frames per cadence")
    ap.add_argument(
        "--gap", type=float, nargs="+", default=[0.20], help="cadenced gaps in seconds, one timed series per value"
    )
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if not (args.state_trace or (args.session and args.sidecar)):
        raise SystemExit("give --state-trace (preferred) or --session with --sidecar")

    before = machine_snapshot()
    if args.state_trace:
        _seq, qpos, qvel = F.load_state_trace(args.state_trace, "", 0)
    else:
        joints, pel, objs = F.load_frames(args.session, args.episode, args.sidecar)
    cam_spec = F.head_camera_spec(args.scene_xml)
    joint_map = F.build_joint_map()
    cfg = S.StageCfg(
        assets=args.assets,
        scene_dir=args.scene_dir,
        robot_usd=args.robot_usd,
        robot_mjcf=args.scene_xml,
        width=args.width,
        height=args.height,
        absolute_camera=bool(args.state_trace),
    )

    app = S.start_app(cfg)
    try:
        if args.state_trace:
            from locomani.sim.isaac.view import IsaacView

            view = IsaacView(cfg, args.scene_xml, args.streams, args.rtx_setting, log=lambda *_: None)
            T = len(qpos)

            def push(t):
                view.push(qpos[t], qvel[t])

            grab = view.grab
        else:
            stg = S.assemble_stage(cfg, log=lambda *_: None)
            world = S.make_world()
            robot, jidx, jsrc = S.bind_robot(joint_map, log=lambda *_: None)
            sensor, _ = S.mount_camera(stg, cfg, cam_spec, log=lambda *_: None)
            T = len(joints)

            def push(t):
                S.sync_frame(stg, robot, jidx, jsrc, joints[t], pel[t], {k: v[t] for k, v in objs.items()})

            def grab():
                return S.step_and_grab(world, sensor)

        def one(t):
            t0 = time.perf_counter()
            push(t)
            t1 = time.perf_counter()
            img = grab()
            t2 = time.perf_counter()
            return (t1 - t0), (t2 - t1), img is not None

        for i in range(args.warmup):
            one(i % T)

        result = {}
        for gap in [0.0, *args.gap]:
            cadence = "continuous" if gap == 0 else f"cadenced_{gap:g}s"
            result[cadence] = series(one, [(args.warmup + i) % T for i in range(args.n)], gap)
        out = {
            "path": "isaac_view" if args.state_trace else "parquet_sidecar",
            "streams": args.streams if args.state_trace else ["ego_view"],
            "state_trace": args.state_trace,
            "settle_steps": S.SETTLE_STEPS,
            "rtx_settings": args.rtx_setting,
            "resolution": [args.width, args.height],
            "action_rate_hz": CONTROL_HZ,
            "scene_dir": S.scene_dir(cfg),
            "robot_usd": S.robot_usd(cfg),
            "warmup": args.warmup,
            "gap_s": args.gap,
            "machine_before": before,
            "machine_after": machine_snapshot(),
            "results": result,
        }
        # written BEFORE app.close(): close() terminates the process itself.
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        sys.stderr.write(f"wrote {args.out}\n")
    except BaseException:
        import traceback

        traceback.print_exc()
        os._exit(1)
    app.close()


if __name__ == "__main__":
    main()
