#!/usr/bin/env python3
"""Box-rim stance sweep: one request per round, one SONIC stack for the whole sweep.

    python scripts/box_rim_sweep.py plan  <run_dir> --template <request.json> [--gpu 1 --instance 0]
    python scripts/box_rim_sweep.py run   <run_dir>            # round-robin over levels; STOP file halts
    python scripts/box_rim_sweep.py table <run_dir>            # per-level successes and reached phases

`plan` clones the template request once per round: `runtime.robot_dxy[1]` steps through the
depth levels (the robot's start along its facing axis, metres; the box does not move), and each
round adds a seeded uniform jitter of +-lateral_jitter_m across and +-yaw_jitter_deg about the
template yaw. Rounds are ordered round-robin over the levels so a drift over the session is not
confounded with depth. `run` mirrors scripts/probe_box_grasp.py but brings the stack up once and
runs the box-rim CLI child once per request; a round whose result.json exists is skipped, so a
killed sweep resumes. `table` reads each result's `ok`, `why`, `execution.phase` and
`execution.travel_m`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_LEVELS = [0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375]  # robot_dxy[1]; 0.225 is the 09-20 near stance


def plan(args) -> int:
    run_dir = args.run_dir.resolve()
    template = json.loads(args.template.read_text())
    rng = np.random.default_rng(args.seed)
    levels = [float(v) for v in args.levels]
    rounds = []
    for k in range(args.per_level):
        for level in levels:
            lateral = float(rng.uniform(-args.lateral_jitter_m, args.lateral_jitter_m))
            yaw = float(rng.uniform(-args.yaw_jitter_deg, args.yaw_jitter_deg))
            name = f"dy{level:.3f}_r{k:02d}"
            request = json.loads(json.dumps(template))
            runtime = request["runtime"]
            runtime["robot_dxy"] = [round(runtime["robot_dxy"][0] + lateral, 4), round(level, 4)]
            runtime["robot_yaw"] = round(float(runtime["robot_yaw"]) + math.radians(yaw), 6)
            runtime["gpu"] = str(args.gpu)
            runtime["instance"] = int(args.instance)
            runtime["scene_dir"] = str(Path(runtime["scene_dir"]).resolve())
            runtime["video_label"] = name
            (run_dir / "requests").mkdir(parents=True, exist_ok=True)
            (run_dir / "requests" / f"{name}.request.json").write_text(json.dumps(request, indent=1))
            rounds.append(dict(name=name, level=level, repeat=k, lateral_m=lateral, yaw_jitter_deg=yaw))
    sweep = dict(
        schema_version=1,
        template=str(args.template.resolve()),
        levels=levels,
        per_level=args.per_level,
        seed=args.seed,
        lateral_jitter_m=args.lateral_jitter_m,
        yaw_jitter_deg=args.yaw_jitter_deg,
        rounds=rounds,
    )
    (run_dir / "sweep.json").write_text(json.dumps(sweep, indent=1))
    print(f"{len(rounds)} requests under {run_dir / 'requests'}")
    return 0


def run(args) -> int:
    from locomani import paths
    from locomani.backends.sonic.config import SonicBackendConfig
    from locomani.stack import Stack, StackSpec
    from locomani.tasks.crate.box_rim_execution_cli import load_execution_request

    run_dir = args.run_dir.resolve()
    sweep = json.loads((run_dir / "sweep.json").read_text())
    pending = [r for r in sweep["rounds"] if not (run_dir / "rounds" / r["name"] / "result.json").exists()]
    if not pending:
        print("nothing pending")
        return 0
    first = load_execution_request(run_dir / "requests" / f"{pending[0]['name']}.request.json")
    runtime = first["runtime"]
    backend = SonicBackendConfig.from_dict(runtime["backend"])
    spec = StackSpec(
        instance=runtime["instance"],
        gpu=str(runtime["gpu"]),
        scene_dir=str(runtime["scene_dir"]),
        prompt="grasp the box rim with both hands",
        record=False,
        policy=os.environ.get("SONIC_POLICY", paths.SONIC_POLICY),
        deploy_root=os.environ.get("SONIC_DEPLOY_ROOT", paths.SONIC_DEPLOY),
        deploy_bin=os.environ.get("SONIC_DEPLOY_BIN") or None,
        encoder=not backend.no_encoder,
        lockstep=bool(backend.lockstep.enabled),
        auto_start=bool(backend.clean_deploy or backend.lockstep.enabled),
        wrist_cams=bool(backend.wrist_cams),
        video_dir=str(run_dir / "videos"),
        sim_vis=backend.sim_vis,
        scene_random=False,
        dds_domain=backend.sonic_dds_domain,
        env={"LOCOMANI_BACKEND_JSON": json.dumps(backend.to_dict(), sort_keys=True)},
    )
    stack = Stack(spec, run_dir=run_dir)
    log = open(run_dir / "sweep.log", "a")
    try:
        env = stack.up()
        env["PYTHONPATH"] = str(ROOT / "src")
        for r in pending:
            if (run_dir / "STOP").exists():
                print("STOP", file=log, flush=True)
                break
            out = run_dir / "rounds" / r["name"]
            out.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            command = [
                sys.executable,
                "-m",
                "locomani.tasks.crate.box_rim_execution_cli",
                str(run_dir / "requests" / f"{r['name']}.request.json"),
                "--output",
                str(out / "result.json"),
            ]
            with open(out / "child.log", "w") as child_log:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=child_log, stderr=subprocess.STDOUT)
            verdict = "?"
            if (out / "result.json").exists():
                d = json.loads((out / "result.json").read_text())
                verdict = f"ok={d.get('ok')} why={d.get('why')} phase={(d.get('execution') or {}).get('phase')}"
            print(f"{r['name']} rc={completed.returncode} {verdict} {time.time() - t0:.0f}s", file=log, flush=True)
    finally:
        stack.down()
    return 0


def table(args) -> int:
    run_dir = args.run_dir.resolve()
    sweep = json.loads((run_dir / "sweep.json").read_text())
    per_level: dict[float, list] = {lv: [] for lv in sweep["levels"]}
    for r in sweep["rounds"]:
        p = run_dir / "rounds" / r["name"] / "result.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        e = d.get("execution") or {}
        per_level[r["level"]].append(
            dict(
                name=r["name"], ok=bool(d.get("ok")), why=d.get("why"), phase=e.get("phase"), travel_m=e.get("travel_m")
            )
        )
    rows = []
    print("| dy (m) | n | ok | phases at exit | why | travel m (median) |")
    print("|---|---|---|---|---|---|")
    for lv, rs in per_level.items():
        phases: dict[str, int] = {}
        whys: dict[str, int] = {}
        for x in rs:
            phases[str(x["phase"])] = phases.get(str(x["phase"]), 0) + 1
            if not x["ok"]:
                whys[str(x["why"])] = whys.get(str(x["why"]), 0) + 1
        travel = [x["travel_m"] for x in rs if isinstance(x["travel_m"], (int, float))]
        ok = sum(x["ok"] for x in rs)
        rows.append(
            dict(
                level=lv,
                n=len(rs),
                ok=ok,
                phases=phases,
                why=whys,
                travel_median_m=float(np.median(travel)) if travel else None,
                rounds=rs,
            )
        )
        fmt = lambda m: ", ".join(f"{k} {v}" for k, v in sorted(m.items(), key=lambda kv: -kv[1]))  # noqa: E731
        med = f"{np.median(travel):.2f}" if travel else "-"
        print(f"| {lv:.3f} | {len(rs)} | {ok} | {fmt(phases)} | {fmt(whys)} | {med} |")
    (run_dir / "table.json").write_text(json.dumps(rows, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--levels", type=float, nargs="+", default=DEFAULT_LEVELS)
    p.add_argument("--per-level", type=int, default=10)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--lateral-jitter-m", type=float, default=0.02)
    p.add_argument("--yaw-jitter-deg", type=float, default=3.0)
    p.add_argument("--gpu", default="1")
    p.add_argument("--instance", type=int, default=0)
    p.set_defaults(func=plan)
    p = sub.add_parser("run")
    p.add_argument("run_dir", type=Path)
    p.set_defaults(func=run)
    p = sub.add_parser("table")
    p.add_argument("run_dir", type=Path)
    p.set_defaults(func=table)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
