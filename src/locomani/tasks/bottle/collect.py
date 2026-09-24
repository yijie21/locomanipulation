#!/usr/bin/env python3
"""Collect single-arm episodes on a stack someone else started: batch after batch, until enough.

    python -m locomani.tasks.bottle.collect configs/experiments/bottle_simple_200.yaml --gpu 1

`scripts/start_stack.sh collect <same yaml>` brought the stack up in the other terminal. This
process attaches to it, so every value the children were told comes back from `stack.json` rather
than being rebuilt from the config a second time.

This runs the gym IN THIS PROCESS. There is no argv to assemble and no subprocess per batch: the
gym already lives in this interpreter, and the hop through a command line bought nothing but a
second place for the environment to differ (docs/handoff/flat-functional-scripts-20260922.md).
Two consequences follow, and both are load-bearing:

  * the stack's environment has to be in `os.environ` before the gym opens any client, because
    that is where the instance's port block lives. `privileged_pub` resolves its ports when they
    are used rather than at import, so this is an ordinary assignment and not import-order setup;
  * `grasp_gym.run` closes the robot and the episode client it opened, because this process does
    not exit between batches.

A batch is one exporter session. The exporter restarts between batches for two reasons, and only
one is about resources: `convert` pairs exporter session k with manifest k BY POSITION, and the
exporter's threads grow with every episode against a container ceiling of 3840 pids. The sim and
the deploy live across the whole run.

Interrupted runs continue: `collect/inputs.json` records which batches finished and what they
kept, and rerunning the same name reruns the unfinished batch from its start, its partial manifest
moved aside (`batches.resume_state`). It is not episode-level resume.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from locomani.pipeline import batches, run_context
from locomani.stack import Stack


def profile_of(cfg: dict, run_dir: Path, *, frozen: bool):
    """The task profile this batch runs, and the reach calibration inside it.

    `frozen=True` reads the run's own copies, which is the rule for anything downstream of
    `prepare`: a later stage reads the frozen file, never `configs/`. A dry run has not frozen
    anything and must not, so it builds the same objects from the experiment in memory --
    `prepare` dumps exactly these, so the two agree by construction.
    """
    from locomani.tasks.bottle.config import BottleTaskConfig
    from locomani.tasks.bottle.reach_config import ReachConfig

    if frozen:
        profile = BottleTaskConfig.load(run_dir / "task_config.json")
        return profile, ReachConfig.load(run_dir / "reach_config.json")
    profile = BottleTaskConfig.from_dict(cfg["task"])
    return profile, profile.reach


def gym_options(
    cfg: dict, run_dir: Path, collect_root: Path, batch: int, kept: int, manifest: Path, *, profile, reach, write: bool
):
    """The gym options for one batch, resolved through the gym's own merge.

    `write=False` is what a dry run passes: it names the case-bank slice this batch would read
    without creating it. Resolving through `cli_args.resolve_options` -- the same function
    `grasp_gym.main` uses for a command line -- is what keeps an option set here and an option
    set on the CLI from diverging.
    """
    from locomani.tasks.bottle import cli_args

    collect, spawn = cfg["collect"], cfg["task"]["spawn"]
    explicit = dict(
        rounds=collect["batch_rounds"],
        seed=collect["seed0"] + 100 * batch,
        targets=cfg["task"]["target"],
        stand_retries=cfg["task"].get("stand_retries", 0),
        manifest=str(manifest),
        spawn_edge=[float(v) for v in spawn["edge_range"]],
        spawn_yaw_err=float(spawn["yaw_err_deg"]),
        spawn_lateral=float(spawn["lateral"]),
        edge_jitter=float(spawn["edge_jitter"]),
    )
    if spawn["face_object"]:
        explicit["face_object"] = True
    if collect["record"]:
        explicit["record"] = True
    if cfg["task"].get("video_include_init"):
        explicit["video_include_init"] = True
    cases = case_slice(cfg, run_dir, collect_root, batch, write=write)
    if cases is not None:
        explicit["cases"] = str(cases)
    else:
        explicit["until_ok"] = collect["successes"] - kept
    # The run's own copies are the provenance answer to "where did this value come from".
    # The scene is named rather than inherited: `locomani.paths` froze LOCOMANI_SIM_DIR when it
    # was imported, which for this process was before it knew which run it was serving.
    explicit["scene_meta"] = str(prepare_scene_meta(cfg, run_dir))
    explicit["task_config"] = str(run_dir / "task_config.json")
    explicit["reach_config"] = str(run_dir / "reach_config.json")
    return cli_args.resolve_options(explicit, profile=profile, reach=reach, vla=False)


def prepare_scene_meta(cfg: dict, run_dir: Path) -> Path:
    """The run's own scene copy, named without freezing it -- a dry run must name it too."""
    from locomani.tasks.bottle import prepare

    return prepare.scene_name(cfg, run_dir) / "scene_meta.json"


def case_slice(cfg: dict, run_dir: Path, collect_root: Path, batch: int, *, write: bool) -> Path | None:
    """This batch's slice of the fixed sample bank, or None when every round is drawn fresh."""
    from locomani.pipeline.experiment_cases import load_cases, save_cases

    if not cfg["scene"].get("cases"):
        return None
    path = collect_root / f"cases_{batch:02d}.json"
    if not write:
        return path
    meta = prepare_scene_meta(cfg, run_dir)  # the same scene the gym is told about
    every = load_cases(Path(run_context.ROOT) / cfg["scene"]["cases"], meta)
    rounds, budget = cfg["collect"]["batch_rounds"], cfg["collect"]["max_batches"]
    if len(every) < rounds * budget:
        raise SystemExit(f"{cfg['scene']['cases']} holds {len(every)} cases, the budget needs {rounds * budget}")
    mine = every[batch * rounds : (batch + 1) * rounds]
    if path.exists():
        if load_cases(path, meta) != mine:
            raise SystemExit(f"{path} is not the slice this batch had; inspect before continuing")
    else:
        save_cases(path, mine, meta)
    return path


def run_batch(cfg: dict, run_dir: Path, collect_root: Path, stack: Stack, state: dict) -> Path:
    from locomani.tasks.bottle import grasp_gym
    from locomani.tasks.bottle.runtime import GymRuntime

    batch = state["batches"]
    manifest = collect_root / f"manifest_{batch:02d}.jsonl"
    if manifest.exists() and str(manifest) not in state["manifests"]:
        raise SystemExit(f"{manifest} exists but no batch claimed it; inspect before continuing")
    profile, reach = profile_of(cfg, run_dir, frozen=True)
    options = gym_options(
        cfg, run_dir, collect_root, batch, state["kept"], manifest, profile=profile, reach=reach, write=True
    )
    state["sessions"].append(batches.open_exporter_session(stack, collect_root, batch))
    state["manifests"].append(str(manifest))
    (collect_root / "inputs.json").write_text(json.dumps(state, indent=1))
    stage = _stage(run_dir)
    stage.record_config(f"collect_{batch:02d}", options.report(), env=dict(stack.env))
    with stage.timed(f"collect_{batch:02d}"):
        grasp_gym.run(options, runtime=GymRuntime())
    return manifest


def _stage(run_dir: Path):
    from locomani.pipeline.stages import Stage

    return Stage(run_dir)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="run directory name; the experiment's own name by default")
    ap.add_argument("--keep-stack", action="store_true", help="leave the stack running when the run ends")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    from locomani.tasks.bottle import prepare

    cfg, run_dir = run_context.open_run(args.experiment, args.name, write=not args.dry_run)
    run_context.require_train_split(cfg)
    collect = cfg["collect"]
    collect_root = prepare.collect_root(run_dir)

    if args.dry_run:
        profile, reach = profile_of(cfg, run_dir, frozen=False)
        options = gym_options(
            cfg,
            run_dir,
            collect_root,
            0,
            0,
            collect_root / "manifest_00.jsonl",
            profile=profile,
            reach=reach,
            write=False,
        )
        run_context.show_config("collect_00", options.report())
        print(f"\n  attaches to instance {collect['instance']}; {collect['max_batches']} batches at most")
        return 0

    try:
        stack = Stack.attach(int(collect["instance"]))
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {collect['instance']}; start one in another terminal:\n"
            f"  bash scripts/start_stack.sh collect {args.experiment} --gpu {args.gpu}"
        )
    run_context.require_stack_of(stack, run_dir, "collect")
    # The stack is already up and was admitted by whoever started it; ask only for the rest.
    run_context.require_headroom(
        "collect", gpu=args.gpu, record=bool(collect["record"]), already=run_context.group_threads(stack.pids.values())
    )
    run_context.require_deploy_patches(cfg)
    # The instance's port block and the scene the sim is showing. The subprocess this used to be
    # was handed exactly this as its whole environment; in-process it has to be assigned, before
    # any client is opened and after `privileged_pub` stopped freezing ports at import.
    os.environ.update({k: str(v) for k, v in stack.env.items()})
    prepare.prepare(cfg, run_dir)  # idempotent: the stack starter normally got here first
    collect_root.mkdir(parents=True, exist_ok=True)
    prepare.trace_root(run_dir).mkdir(parents=True, exist_ok=True)
    inputs = collect_root / "inputs.json"
    state = batches.resume_state(inputs)  # only finished batches count; an unfinished one reruns

    try:
        while state["kept"] < collect["successes"] and state["batches"] < collect["max_batches"]:
            manifest = run_batch(cfg, run_dir, collect_root, stack, state)
            state["kept"] += batches.kept_in(manifest, bool(collect["record"]))
            state["batches"] += 1
            inputs.write_text(json.dumps(state, indent=1))
            print(
                f"collect: batch {state['batches'] - 1:02d} -> {state['kept']}/{collect['successes']} kept", flush=True
            )
    finally:
        if not args.keep_stack:
            stack.down()  # a recording stack left up holds ~1800 of the container's 3840 pids

    rows = [json.loads(row) for m in state["manifests"] for row in Path(m).read_text().splitlines() if row.strip()]
    run_context.summarise(
        run_dir,
        "collect",
        attempts=len(rows),
        kept=state["kept"],
        batches=state["batches"],
        yield_rate=round(state["kept"] / len(rows), 3) if rows else 0.0,
        failures=sorted({str(r.get("why") or "")[:70] for r in rows if not r.get("ok")}),
        sessions=state["sessions"],
        manifests=state["manifests"],
    )
    if state["kept"] < collect["successes"]:
        raise SystemExit(f"stopped at {state['kept']}/{collect['successes']} kept after {state['batches']} batches")
    print(f"collect: {state['kept']} kept in {state['batches']} batches -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
