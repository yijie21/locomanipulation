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
from locomani.pipeline import lanes, run_context
from locomani.pipeline.lines import line_of
from locomani.pipeline.stages import Stage
from locomani.stack import Stack, spec_from_config

ROOT = Path(paths.REPO)
EXPORTER = ROOT / "src/locomani/pipeline/run_instance_exporter.py"


def frame_audit_path(collect_root: Path, batch: int) -> Path:
    """One receipt file per batch, because `audit_bottle_actions` reads them by batch number."""
    return collect_root / f"frame_audit_{batch:02d}.jsonl"


def collect_spec(
    cfg: dict,
    run_dir: Path,
    scene: Path,
    collect_root: Path,
    trace_root: Path,
    gpu: str,
    lane: lanes.Lane | None = None,
):
    """The recording stack: the exporter's first session pairs with batch 00's manifest -- or, for lane k of a run
    collected in lanes, batch k's, on the lane's own instance and card."""
    extra = {
        "SIM_STATE_TRACE_DIR": str(trace_root),
        "TF_FRAME_AUDIT_PATH": str(frame_audit_path(collect_root, 0 if lane is None else lane.index)),
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
        video_dir=collect_root / "videos" if run_context.third_person(cfg) else None,
        record=bool(cfg["collect"]["record"]),
    )
    env = {**spec.env, **{k: str(v) for k, v in extra.items()}}
    if lane is None:
        return dataclasses.replace(spec, env=env)
    return dataclasses.replace(spec, env=env, instance=lane.instance, gpu=str(lane.gpu))


def start_lanes(cfg: dict, run_dir: Path, line, *, gpus: str | None, want: int | None, dry_run: bool) -> int:
    """Collection on a line that collects in lanes: plan them on this machine, say so, start one stack per lane.

    The plan is the headroom check -- it is sized from the same measurement the single-stack gate reads -- and
    `collect/lanes.json` is what the driver attaches to. A stack that fails to start takes the ones already
    started down with it: a run on fewer lanes than it printed is the half-applied state this repository keeps
    paying for."""
    plan = lanes.plan_run(cfg, gpus=gpus, want=want)
    print("\n".join(plan.describe()), flush=True)
    collect_root = line.collect_root(run_dir)
    if dry_run:
        scene = line.scene_name(cfg, run_dir)  # a dry run names the scene without freezing it
        # With nothing free now, still say what the first lane's stack would be told: a dry run answers "what is
        # delivered", which does not depend on this minute's load.
        shown = plan.lanes or [lanes.Lane(0, int(cfg["collect"].get("instance", 0)), lanes.gpu_list(gpus)[0])]
        for lane in shown:
            spec = collect_spec(cfg, run_dir, scene, collect_root, line.trace_root(run_dir), lane.gpu, lane)
            run_context.show(
                f"stack_up_collect_lane{lane.index:02d}", ["locomani.stack", "up", str(spec.instance)], spec.settings()
            )
        return 0
    run_context.require_deploy_patches(cfg)  # static, so before the refusal that depends on this minute's load
    if not plan.lanes:
        raise SystemExit("collect: not enough headroom for one lane -- see 'what fits' above")
    scene = line.prepare(cfg, run_dir)
    collect_root.mkdir(parents=True, exist_ok=True)
    line.trace_root(run_dir).mkdir(parents=True, exist_ok=True)
    stage, started = Stage(run_dir), []
    try:
        for lane in plan.lanes:
            spec = collect_spec(cfg, run_dir, scene, collect_root, line.trace_root(run_dir), lane.gpu, lane)
            name = "stack_env.json" if lane.index == 0 else f"stack_env_lane{lane.index:02d}.json"
            stack = Stack(spec, record_command=stage.record, run_dir=run_dir, mode="collect", settings_file=name)
            stack.up()
            started.append(stack)
            print(json.dumps(dict(lane=lane.index, instance=spec.instance, gpu=spec.gpu, pids=stack.pids)), flush=True)
    except BaseException:
        for stack in started:
            stack.down()
        raise
    lanes.write(run_dir, plan)
    return 0


def eval_spec(cfg: dict, run_dir: Path, scene: Path, gpu: str):
    """The evaluation stack, built from the VALIDATED eval block, as the run's line runs it.

    `EvalConfig` is where the client/clean_deploy invariant is applied; handing `spec_from_config`
    the file's raw dict instead walks straight past it, and the deploy then launches without
    SONIC_AUTO_START while the async commander suppresses the start_control that would have
    engaged it (2026-09-17). The line then has the last word, because the bimanual evaluator runs
    its rounds clean under a pinned blocking client, which that invariant reads as "not clean"
    (`lines.Line.eval_backend`, 2026-09-23).
    """
    evaluation = run_context.evaluation_of(cfg)  # the eval sim renders what images.source needs
    spec = spec_from_config(
        cfg,
        run_dir=run_dir,
        gpu=gpu,
        backend=line_of(cfg).eval_backend(evaluation),
        scene_dir=scene,
        video_dir=run_dir / "eval" / "videos" if run_context.third_person(cfg) else None,
        record=False,
    )
    # Per-round qpos traces, so a round can be rendered afterwards (backend.render offline writes no video)
    # and an Isaac-mirror image re-rendered from the seq it was taken at (TODO 4.7.9.8).
    return dataclasses.replace(spec, env={**spec.env, "SIM_STATE_TRACE_DIR": str(run_dir / "eval" / "state_trace")})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path, help="the experiment YAML, or a run directory already opened")
    ap.add_argument("--mode", choices=("collect", "eval"), required=True)
    ap.add_argument(
        "--gpu",
        default=None,
        help="card(s): N, N,M or all. Collection on a line that collects in lanes spreads its lanes over them "
        "(default all); every other stack runs on the first (default 0)",
    )
    ap.add_argument(
        "--lanes",
        default=None,
        help="collection lanes: a number (at most that many) or auto (as many as fit; the default, or collect.lanes)",
    )
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
    run_context.require_images(cfg)  # an Isaac run this machine cannot render is refused before anything starts
    if args.mode == "collect" and line.parallel:
        return start_lanes(cfg, run_dir, line, gpus=args.gpu, want=lanes.wanted(args.lanes, cfg), dry_run=args.dry_run)
    if args.lanes not in (None, "1"):
        print(f"collect: the {line.name} line collects in one lane; --lanes {args.lanes} is ignored", flush=True)
    args.gpu = lanes.gpu_list(args.gpu or "0")[0]

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
