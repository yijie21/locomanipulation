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
               cost, which is the one that matters for the verdict.

Resolution defaults to the dataset's 640x480 and quality is NOT reduced to
improve the number -- the whole point of the measurement is the cost of the
quality we intend to ship.

    "$RENDER_PY" -m locomani.sim.isaac.bench_render \
        --session <dir> --episode 0 --sidecar '<glob>' \
        --scene-dir assets/isaac/scene/<seed> --out bench.json
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


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--scene-dir", default="")
    ap.add_argument("--robot-usd", default="")
    ap.add_argument("--scene-xml", default=paths.SCENE_XML)
    ap.add_argument("--width", type=int, default=640)  # dataset resolution
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--n", type=int, default=300, help="timed frames per cadence")
    ap.add_argument("--gap", type=float, default=0.20, help="cadenced gap in seconds, ~ one VLA inference")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    before = machine_snapshot()
    joints, pel, objs = F.load_frames(args.session, args.episode, args.sidecar)
    cam_spec = F.head_camera_spec(args.scene_xml)
    joint_map = F.build_joint_map()
    cfg = S.StageCfg(
        scene_dir=args.scene_dir, robot_usd=args.robot_usd, width=args.width, height=args.height, absolute_camera=False
    )

    app = S.start_app(cfg)
    try:
        stg = S.assemble_stage(cfg, log=lambda *_: None)
        world = S.make_world()
        robot, jidx, jsrc = S.bind_robot(joint_map, log=lambda *_: None)
        sensor, _ = S.mount_camera(stg, cfg, cam_spec, log=lambda *_: None)

        T = len(joints)

        def one(t):
            t0 = time.perf_counter()
            S.sync_frame(stg, robot, jidx, jsrc, joints[t], pel[t], {k: v[t] for k, v in objs.items()})
            t1 = time.perf_counter()
            rgba = S.step_and_grab(world, sensor)
            t2 = time.perf_counter()
            return (t1 - t0), (t2 - t1), rgba is not None

        for i in range(args.warmup):
            one(i % T)

        result = {}
        for cadence in ("continuous", "cadenced"):
            sync_ms, render_ms, empty = [], [], 0
            for i in range(args.n):
                if cadence == "cadenced":
                    time.sleep(args.gap)
                s, r, ok = one((args.warmup + i) % T)
                if not ok:
                    empty += 1
                    continue
                sync_ms.append(s * 1e3)
                render_ms.append(r * 1e3)
            total = [a + b for a, b in zip(sync_ms, render_ms)]
            result[cadence] = {
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
        out = {
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
