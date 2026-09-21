#!/usr/bin/env python3
"""Freeze a run and bring up the stack it asks for, then return and leave it running.

    python -m locomani.pipeline.run_stack configs/experiments/bottle_simple_200.yaml --mode collect --gpu 1
    python -m locomani.pipeline.run_stack outputs/runs/bottle_simple_200 --mode eval --gpu 1

`scripts/collect_data.sh` and `scripts/start_stack.sh` are one line each over this; the round
driver is a separate process that attaches (`Stack.attach`). Which of the two backend blocks is
used is the whole difference between the modes and is spelled out here rather than defaulted:
collection runs the deploy WITH its encoder (without it the planner's arm and crouch references
are dropped) and evaluation runs it WITHOUT (the token stream will not hold otherwise).

The run-specific variables the children need go into the spec's own `env`, not into a base
environment that only this process has: the spec is what `stack.json` carries, so the driver that
attaches later gets byte-identical values instead of rebuilding them from the config a second time.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.pipeline.eval_config import EvalConfig
from locomani.pipeline.lines import line_of
from locomani.pipeline.stages import Stage
from locomani.stack import Stack, spec_from_config

ROOT = Path(paths.REPO)
EXPORTER = ROOT / "src/locomani/pipeline/run_instance_exporter.py"


def frame_audit_path(collect_root: Path, batch: int) -> Path:
    """One receipt file per batch, because `audit_bottle_actions` reads them by batch number."""
    return collect_root / f"frame_audit_{batch:02d}.jsonl"


def collect_spec(cfg: dict, run_dir: Path, scene: Path, collect_root: Path, trace_root: Path, gpu: str):
    """The recording stack: the exporter's first session pairs with batch 00's manifest."""
    extra = {
        "SIM_STATE_TRACE_DIR": str(trace_root),
        "TF_FRAME_AUDIT_PATH": str(frame_audit_path(collect_root, 0)),
        "TF_EXPORTER_SCRIPT": str(EXPORTER),
        **dict(kv.split("=", 1) for kv in cfg["collect"].get("gym_env", [])),
        **dict(kv.split("=", 1) for kv in cfg["collect"].get("stack_env", [])),
    }
    if bool(cfg["collect"]["record"]):
        extra["RECORD_INSTANCE"] = "1"  # the exporter runs on this instance, whichever one it is
    spec = spec_from_config(
        cfg,
        run_dir=run_dir,
        gpu=gpu,
        scene_dir=scene,
        video_dir=collect_root / "videos",
        record=bool(cfg["collect"]["record"]),
    )
    return dataclasses.replace(spec, env={**spec.env, **{k: str(v) for k, v in extra.items()}})


def eval_spec(cfg: dict, run_dir: Path, scene: Path, gpu: str):
    """The evaluation stack, built from the VALIDATED eval block.

    `EvalConfig` is where the client/clean_deploy invariant is applied; handing `spec_from_config`
    the file's raw dict instead walks straight past it, and the deploy then launches without
    SONIC_AUTO_START while the async commander suppresses the start_control that would have
    engaged it (2026-09-17).
    """
    evaluation = EvalConfig.from_dict(cfg["eval"])
    spec = spec_from_config(
        cfg,
        run_dir=run_dir,
        gpu=gpu,
        backend=evaluation.backend.to_dict(),
        scene_dir=scene,
        video_dir=run_dir / "eval" / "videos",
        record=False,
    )
    return spec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path, help="the experiment YAML, or a run directory already opened")
    ap.add_argument("--mode", choices=("collect", "eval"), required=True)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="run directory name; the experiment's own name by default")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    if args.experiment.is_dir():
        run_dir = args.experiment if args.experiment.is_absolute() else ROOT / args.experiment
        cfg = run_context.load_run_config(run_dir)
    else:
        cfg, run_dir = run_context.open_run(args.experiment, args.name, write=not args.dry_run)

    line = line_of(cfg)
    collect_root = line.collect_root(run_dir)

    if args.dry_run:
        scene = line.scene_name(cfg, run_dir)  # a dry run names the scene without freezing it
        spec = (
            collect_spec(cfg, run_dir, scene, collect_root, line.trace_root(run_dir), args.gpu)
            if args.mode == "collect"
            else eval_spec(cfg, run_dir, scene, args.gpu)
        )
        run_context.show(f"stack_up_{args.mode}", ["locomani.stack", "up", str(spec.instance)], spec.settings())
        return 0

    recording = args.mode == "collect" and bool(cfg["collect"]["record"])
    run_context.require_headroom("collect" if args.mode == "collect" else "eval", gpu=args.gpu, record=recording)
    run_context.require_deploy_patches(cfg)
    scene = line.prepare(cfg, run_dir)
    collect_root.mkdir(parents=True, exist_ok=True)
    line.trace_root(run_dir).mkdir(parents=True, exist_ok=True)
    spec = (
        collect_spec(cfg, run_dir, scene, collect_root, line.trace_root(run_dir), args.gpu)
        if args.mode == "collect"
        else eval_spec(cfg, run_dir, scene, args.gpu)
    )
    # The children's argv belongs in the run's commands.jsonl, as it did when one process owned
    # the whole stage: `stack_sim` / `stack_deploy` / `stack_exporter` are how a finished run
    # answers "what was the deploy actually launched with", and the answer stops existing the
    # moment the starter forgets to hand the recorder over (found by the 2026-09-21 smoke, which
    # produced a one-line commands.jsonl next to the old path's stack rows).
    stage = Stage(run_dir)
    stack = Stack(spec, base_env=spec.settings(), record_command=stage.record, run_dir=run_dir)
    stack.up()
    print(json.dumps(dict(instance=spec.instance, run=str(run_dir), pids=stack.pids, ports=spec.ports())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
