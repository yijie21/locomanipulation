#!/usr/bin/env python3
"""Collect bimanual episodes on a stack someone else started: batch after batch, until enough.

    python -m locomani.tasks.crate.collect configs/experiments/crate_table30_record.yaml --gpu 1

`scripts/start_stack.sh collect <same yaml>` brought the stack up in the other terminal. This
process attaches to it and runs the expert IN THIS PROCESS -- no argv, no subprocess per batch
(docs/handoff/flat-functional-scripts-20260922.md). The single-arm counterpart is
`locomani.tasks.bottle.collect`; what they share lives in `locomani.pipeline.batches`.

Two things this line does differently from single-arm, both long-standing:

  * its artifacts sit BESIDE the run rather than under `collect/`, because the scene copy keeps
    the source directory's name and the expert's own lookups depend on that;
  * a batch may end on a failed round without the run being over, so the expert's exit status is
    recorded rather than raised on. In-process that means catching, not a non-zero exit.

All three experts run here, in this process. `visual` -- the recording line -- is called through
`expert.run(config, ...)`, which takes values rather than a command line. The two diagnostics
(`wall`, `carry`) are still called through their own `main(argv)`: their post-parse bodies use
module-level state and neither has a gate or a checkpoint to verify a deeper refactor against, so
their CLI stays the seam for now. That argv never crosses a process boundary, which is the part
that mattered -- but converting them to a `run(...)` like `expert.py`'s is the cleanup left.

Interrupted runs continue: `collect/inputs.json` records which batches finished and what they
kept, and rerunning the same name reruns the unfinished batch from its start
(`batches.resume_state`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from locomani.pipeline import batches, run_context
from locomani.stack import Stack


def profile_of(cfg: dict, run_dir: Path, *, frozen: bool):
    """The task profile this run executes.

    `frozen=True` reads the run's own copy, which is the rule downstream of `prepare`. A dry run
    has frozen nothing and must not, so it builds the same object from the experiment in memory --
    `prepare` dumps exactly this.
    """
    from locomani.tasks.crate.config import CrateTaskConfig

    if frozen:
        return CrateTaskConfig.load(run_dir / "task_config.json")
    return CrateTaskConfig.from_dict(cfg["task"])


def batch_plan(cfg: dict, run_dir: Path, batch: int) -> dict:
    """Everything one batch of the expert is told. One definition, so a dry run names what a run does."""
    from locomani.tasks.crate import prepare

    collect = cfg["collect"]
    seed = collect["seed0"] + 100 * batch
    return dict(
        rounds=collect["batch_rounds"],
        seed=seed,
        manifest=prepare.collect_root(run_dir) / f"manifest_{batch:02d}.jsonl",
        label=f"{prepare.EXPERTS[cfg['task']['expert']][1]}_s{seed}",
        record=bool(collect["record"]),
        scene_dir=prepare.scene_name(cfg, run_dir),
    )


def run_batch(cfg: dict, run_dir: Path, stack: Stack, state: dict) -> Path:
    from locomani.pipeline.stages import Stage
    from locomani.tasks.crate import expert, prepare

    batch = state["batches"]
    plan = batch_plan(cfg, run_dir, batch)
    manifest = Path(plan["manifest"])
    if manifest.exists() and str(manifest) not in state["manifests"]:
        raise SystemExit(f"{manifest} exists but no batch claimed it; inspect before continuing")
    config = profile_of(cfg, run_dir, frozen=True)
    prepare.trace_root(run_dir).mkdir(parents=True, exist_ok=True)
    state["sessions"].append(batches.open_exporter_session(stack, prepare.collect_root(run_dir), batch))
    state["manifests"].append(str(manifest))
    inputs_path(run_dir).parent.mkdir(parents=True, exist_ok=True)
    inputs_path(run_dir).write_text(json.dumps(state, indent=1))

    chosen = cfg["task"]["expert"]
    name = f"{chosen}_{batch:02d}"
    stage = Stage(run_dir)
    stage.record_config(name, dict(plan, config=config.to_dict()), env=dict(stack.env))
    try:
        with stage.timed(name):
            if chosen == "visual":
                expert.run(
                    config,
                    **{k: v for k, v in plan.items() if k != "manifest"},
                    manifest=manifest,
                    # this process attaches to the stack, so the stack's LOCOMANI_BACKEND_JSON never reached it
                    render=cfg.get("backend", {}).get("render", "live"),
                )
            else:
                rc = diagnostic(chosen).main(diagnostic_argv(cfg, run_dir, batch))
                if rc:
                    raise SystemExit(rc)
    except (Exception, SystemExit) as error:
        # This line has always recorded a failing batch rather than ending the run: a batch can
        # end on a failed round with the run still live. The subprocess form of this was a
        # non-zero exit that `Stage.run` recorded and did not raise on -- whatever the expert
        # exited with, `SystemExit` and a non-zero `main()` return included. Caught OUTSIDE
        # `timed`, so timings.jsonl says rc 1 for this batch rather than 0 (2026-09-23).
        print(f"{name}: expert ended with {type(error).__name__}: {error}", flush=True)
    return manifest


def diagnostic(expert_name: str):
    """The module for a diagnostic expert. In this process -- only its CLI is still the seam."""
    from locomani.tasks.crate import bimanual_carry, bimanual_wall

    return {"wall": bimanual_wall, "carry": bimanual_carry}[expert_name]


def diagnostic_argv(cfg: dict, run_dir: Path, batch: int) -> list:
    """The diagnostic's own options, built by the same function the old driver used."""
    from locomani.tasks.crate import prepare

    collect = cfg["collect"]
    argv = prepare.expert_argv(
        cfg["task"]["expert"],
        run_dir / "task_config.json",
        prepare.collect_root(run_dir) / f"manifest_{batch:02d}.jsonl",
        collect["batch_rounds"],
        collect["seed0"] + 100 * batch,
        record=bool(collect["record"]),
        trace_dir=prepare.trace_root(run_dir),
    )
    # Drop the interpreter and the script: this process IS the interpreter.
    return [str(a) for a in argv[3:]] + ["--scene-dir", str(prepare.scene_name(cfg, run_dir))]


# A round that measures the stack, not the layout: the reset never produced a task, or the process or its clock broke.
NOT_A_MEASUREMENT = frozenset(
    {
        "initial_stance_unstable",
        "invalid_initial_distance",
        "runtime_error",
        "camera_unavailable",
        "no_camera_frames",
        "simulation_stalled",
        "simulation_clock_reversed",
        "stale_camera_abort",
    }
)


def stop_tally(manifests: list[str], rule: dict) -> dict:
    """What `collect.stop_rule` counts over the finished batches: successes and rounds, leaving out the rounds that
    measured the stack (NOT_A_MEASUREMENT, or a real-time factor under `min_rtf` -- a starved sim fails timed phases
    as control errors). `excluded` says how many were left out."""
    rows = [json.loads(line) for m in manifests for line in Path(m).read_text().splitlines() if line.strip()]
    floor = float(rule.get("min_rtf", 0.0))

    def measured(row):
        rtf = row.get("real_time_factor")
        return not ({row.get("why"), row.get("control_reason")} & NOT_A_MEASUREMENT) and (rtf is None or rtf >= floor)

    counted = [row for row in rows if measured(row)]
    return dict(
        successes=sum(bool(row.get("ok")) for row in counted), rounds=len(counted), excluded=len(rows) - len(counted)
    )


def inputs_path(run_dir: Path) -> Path:
    """The batch bookkeeping, under `collect/` even though this line keeps everything else beside the run."""
    return Path(run_dir) / "collect" / "inputs.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="run directory name; the experiment's own name by default")
    ap.add_argument("--keep-stack", action="store_true", help="leave the stack running when the run ends")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    from locomani.tasks.crate import prepare

    cfg, run_dir = run_context.open_run(args.experiment, args.name, write=not args.dry_run)
    run_context.require_train_split(cfg)
    collect, chosen = cfg["collect"], cfg["task"]["expert"]

    if args.dry_run:
        # The real path meets this inside prepare_scene, after admission; a dry run touches no
        # resource, so it is all that stands between a bad combination and a confusing failure.
        prepare.check_profile(profile_of(cfg, run_dir, frozen=False), chosen, bool(collect["record"]))
        plan = batch_plan(cfg, run_dir, 0)
        run_context.show_config(f"{chosen}_00", {k: str(v) for k, v in plan.items()})
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
    run_context.require_headroom(
        "collect", gpu=args.gpu, record=bool(collect["record"]), already=run_context.group_threads(stack.pids.values())
    )
    run_context.require_deploy_patches(cfg)
    # The instance's port block and the scene the sim is showing. The subprocess this used to be
    # was handed exactly this as its whole environment.
    run_context.adopt_stack_env(stack)
    prepare.prepare(cfg, run_dir)  # idempotent: the stack starter normally got here first
    inputs = inputs_path(run_dir)
    state = batches.resume_state(inputs)  # only finished batches count; an unfinished one reruns

    rule = collect.get(
        "stop_rule"
    )  # a run that measures a success rate: stop once it is decided (batches.stop_verdict)
    try:
        while (
            state["kept"] < collect["successes"] and state["batches"] < collect["max_batches"] and not state.get("stop")
        ):
            manifest = run_batch(cfg, run_dir, stack, state)
            batches.wait_saved(state["sessions"][-1], manifest)  # before the next batch or stack.down() kills it
            state["kept"] += batches.kept_in(manifest, bool(collect["record"]))
            state["batches"] += 1
            inputs.write_text(json.dumps(state, indent=1))
            print(
                f"collect: batch {state['batches'] - 1:02d} -> {state['kept']}/{collect['successes']} kept", flush=True
            )
            if rule:
                tally = stop_tally(state["manifests"], rule)
                verdict = batches.stop_verdict(tally["successes"], tally["rounds"], rule)
                print(f"collect: {tally} -> {verdict or 'another batch'}", flush=True)
                if verdict:
                    state["stop"] = dict(tally, verdict=verdict)
                    inputs.write_text(json.dumps(state, indent=1))
    finally:
        if not args.keep_stack:
            stack.down()  # a recording stack left up holds ~1800 of the container's 3840 pids

    rows = [json.loads(row) for m in state["manifests"] for row in Path(m).read_text().splitlines() if row.strip()]
    run_context.summarise(
        run_dir,
        "collect",
        expert=chosen,
        attempts=len(rows),
        kept=state["kept"],
        batches=state["batches"],
        yield_rate=round(state["kept"] / len(rows), 3) if rows else 0.0,
        failures=sorted({str(r.get("why") or "")[:70] for r in rows if not r.get("ok")}),
        sessions=state["sessions"],
        manifests=state["manifests"],
        **({"stop": state.get("stop")} if rule else {}),
    )
    if state["kept"] < collect["successes"] and not state.get("stop"):
        raise SystemExit(f"stopped at {state['kept']}/{collect['successes']} kept after {state['batches']} batches")
    print(f"collect: {state['kept']} kept in {state['batches']} batches -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
