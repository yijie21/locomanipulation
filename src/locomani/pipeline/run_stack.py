#!/usr/bin/env python3
"""Freeze a run and bring up the stack it asks for, then return and leave it running.

    python -m locomani.pipeline.run_stack configs/experiments/bottle_simple_200.yaml --mode collect --gpu 1
    python -m locomani.pipeline.run_stack outputs/runs/bottle_simple_200 --mode eval --gpu 1

`scripts/start_stack.sh` is one line over this and passes `--mode` straight through; the round
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
    """The evaluation stack, built from the VALIDATED eval block, as the run's line runs it.

    `EvalConfig` is where the client/clean_deploy invariant is applied; handing `spec_from_config`
    the file's raw dict instead walks straight past it, and the deploy then launches without
    SONIC_AUTO_START while the async commander suppresses the start_control that would have
    engaged it (2026-09-17). The line then has the last word, because the bimanual evaluator runs
    its rounds clean under a pinned blocking client, which that invariant reads as "not clean"
    (`lines.Line.eval_backend`, 2026-09-23).
    """
    evaluation = EvalConfig.from_dict(cfg["eval"])
    spec = spec_from_config(
        cfg,
        run_dir=run_dir,
        gpu=gpu,
        backend=line_of(cfg).eval_backend(evaluation),
        scene_dir=scene,
        video_dir=run_dir / "eval" / "videos",
        record=False,
    )
    # Per-round qpos traces, so a round can be rendered afterwards (backend.render offline writes no video)
    # and an Isaac-mirror image re-rendered from the seq it was taken at (TODO 4.7.9.8).
    return dataclasses.replace(spec, env={**spec.env, "SIM_STATE_TRACE_DIR": str(run_dir / "eval" / "state_trace")})


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
        # Checked before open_run claims the directory: the copy a refused run left behind would make the
        # corrected experiment file "differ" from it and force a new name.
        cfg, run_dir = run_context.open_run(args.experiment, args.name, write=False)
        if args.mode == "collect":
            run_context.require_train_split(cfg)
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
    # The children inherit this process's environment under the spec's settings, as they did before
    # the stack starter was its own process (`pipeline.stack.resolve_env`: "host bootstrap settings
    # are inherited"). Handing them `spec.settings()` ALONE left sim, deploy and the first exporter
    # without PATH, HOME or TMPDIR, while every child the attached driver restarts later inherits
    # the driver's -- two environments inside one run (2026-09-23).
    stack = Stack(spec, record_command=stage.record, run_dir=run_dir, mode=args.mode)
    stack.up()
    print(json.dumps(dict(instance=spec.instance, run=str(run_dir), pids=stack.pids, ports=spec.ports())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
