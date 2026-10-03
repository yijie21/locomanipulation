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
(`batches.resume_state`). `--until-kept N` pauses a run on purpose at the first batch end with N kept (exit 0);
the same command without it carries on.

Before the first round it prints the layout the rounds will draw (`layout_reminder`) and how many lanes run it;
from a terminal it then waits REMINDER_SECONDS, so a run started on the wrong layout can be stopped before it
records anything.

Lanes (`locomani.pipeline.lanes`): the stack starter planned them from the machine's free CPU, pids, card memory and
instances, and started one stack per lane (`collect/lanes.json`). Lane 0 runs here; every other lane runs in its own
process (`lane_process`), because a stack's ports and scene are process-wide state. The lanes share the run's
batches through `batches.claim_batch` / `finish_batch`: batch k keeps its seed and manifest whichever lane runs it,
and an interrupted run reruns every batch that was running.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

from locomani.pipeline import batches, lanes, run_context
from locomani.stack import Stack

REMINDER_SECONDS = 10  # from a terminal: the wait between the layout reminder and the first round


def collect_target(successes: int, until_kept: int | None) -> int:
    """How many kept episodes this invocation collects to: collect.successes, or fewer under --until-kept."""
    if until_kept is not None and until_kept < 1:
        raise SystemExit("--until-kept must be at least 1")
    return successes if until_kept is None else min(successes, until_kept)


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


def layout_reminder(cfg: dict, config, scene_dir: Path) -> list[str]:
    """What the collection says before its first round: the scene, how much it keeps, what each round's placement
    draws (expert.layout_lines) and, on a scene with measured edges, where the other warehouse layouts are. Only the
    visual expert places anything; a diagnostic gets no reminder."""
    if cfg["task"]["expert"] != "visual":
        return []
    from locomani.tasks.crate.expert import layout_lines
    from locomani.tasks.crate.visual_approach import target_surface

    meta_path = Path(scene_dir) / "scene_meta.json"
    if not meta_path.is_file():  # only a dry run gets here: the real one copied the scene first
        return [f"collect: {meta_path} is missing; build the scene first (bash scripts/build_scene.sh <name>)"]
    meta = json.loads(meta_path.read_text())
    collect, scene = cfg["collect"], cfg["scene"]
    lines = [
        f"collect: scene {scene.get('name') or scene['dir']}, {collect['successes']} kept wanted in batches of "
        f"{collect['batch_rounds']} rounds (at most {collect['max_batches']}). Each round draws (task.visual):",
        *(f"  {line}" for line in layout_lines(meta, config.visual)),
    ]
    # the warehouse's three layouts: one edge (the default, user 2026-10-03), all four (the stock line's), one place
    if len(config.visual.start_edges) > 1:
        other = "from the ym edge only: the default warehouse_mars_* experiments; from one place: *_fixed_start"
    elif config.visual.start_edges:
        other = "from all four edges: the stock warehouse_smoke / warehouse_200; from one place: *_fixed_start"
    elif target_surface(meta).get("edges"):
        other = "a sampled start on the ym edge: the default warehouse_mars_* experiments; on all four: warehouse_smoke"
    else:
        return lines
    return [*lines, f"  {other} (docs/warehouse.md §2)"]


def run_batch(cfg: dict, run_dir: Path, stack: Stack, batch: int) -> Path:
    """One claimed batch on this process's stack; returns its manifest. The exporter session was opened for it."""
    from locomani.pipeline.stages import Stage
    from locomani.tasks.crate import expert

    plan = batch_plan(cfg, run_dir, batch)
    manifest = Path(plan["manifest"])
    if manifest.exists():
        raise SystemExit(f"{manifest} exists but no batch claimed it; inspect before continuing")
    config = profile_of(cfg, run_dir, frozen=True)
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


def run_lane(cfg: dict, run_dir: Path, stack: Stack, lane: int, batch: int | None, target: int) -> None:
    """Batch after batch on one lane's stack: run the claimed one, wait for its episodes, record it, claim the next.

    The lane's first batch takes over the session its stack's exporter opened at start only when it is the batch
    that exporter was opened for (lane k, batch k); every other batch restarts the exporter into a fresh one."""
    from locomani.tasks.crate import prepare

    collect, record = cfg["collect"], bool(cfg["collect"]["record"])
    rule = collect.get(
        "stop_rule"
    )  # a run that measures a success rate: stop once it is decided (batches.stop_verdict)

    def decide(state):
        tally = stop_tally(state["manifests"], rule)
        verdict = batches.stop_verdict(tally["successes"], tally["rounds"], rule)
        print(f"collect: {tally} -> {verdict or 'another batch'}", flush=True)
        return dict(tally, verdict=verdict) if verdict else None

    starter = lane
    while batch is not None:
        session = batches.open_exporter_session(stack, prepare.collect_root(run_dir), batch, starter_batch=starter)
        starter = -1
        manifest = run_batch(cfg, run_dir, stack, batch)
        batches.wait_saved(session, manifest)  # before the next batch or stack.down() kills it
        state = batches.finish_batch(
            inputs_path(run_dir), batch, session, manifest, batches.kept_in(manifest, record), decide if rule else None
        )
        print(f"collect: batch {batch:02d} (lane {lane}) -> {state['kept']}/{collect['successes']} kept", flush=True)
        batch = batches.claim_batch(
            inputs_path(run_dir),
            lane,
            target=target,
            max_batches=collect["max_batches"],
            manifest_for=lambda b: batch_plan(cfg, run_dir, b)["manifest"],
        )


def lane_process(run_dir: str, lane: int, instance: int, batch: int, target: int) -> None:
    """A lane other than the first, in its own process: the stack's ports and scene are process-wide state
    (`run_context.adopt_stack_env`), so one process drives one stack. Reads the run's frozen config."""
    cfg = run_context.load_run_config(Path(run_dir))
    stack = Stack.attach(instance)
    run_context.adopt_stack_env(stack)
    run_lane(cfg, Path(run_dir), stack, lane, batch, target)


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
    ap.add_argument(
        "--until-kept",
        type=int,
        default=None,
        help="pause once this many episodes are kept, at the end of that batch; the run stays resumable: the same "
        "command without it continues toward collect.successes (a first look before the long collection)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    from locomani.tasks.crate import prepare

    cfg, run_dir = run_context.open_run(args.experiment, args.name, write=not args.dry_run)
    run_context.require_train_split(cfg)
    collect, chosen = cfg["collect"], cfg["task"]["expert"]

    if args.dry_run:
        # The real path meets this inside prepare_scene, after admission; a dry run touches no
        # resource, so it is all that stands between a bad combination and a confusing failure.
        profile = profile_of(cfg, run_dir, frozen=False)
        prepare.check_profile(profile, chosen, bool(collect["record"]))
        plan = batch_plan(cfg, run_dir, 0)
        run_context.show_config(f"{chosen}_00", {k: str(v) for k, v in plan.items()})
        print(
            f"\n  attaches to the stacks the stack starter planned (collect/lanes.json; `start_stack.sh collect --dry-run` "
            f"prints the plan), else instance {collect['instance']}; {collect['max_batches']} batches at most"
        )
        # the run's scene copy does not exist yet; the source directory it is copied from has the same meta
        print("\n".join(["", *layout_reminder(cfg, profile, run_context.ROOT / cfg["scene"]["dir"])]))
        return 0

    # The lanes the stack starter planned and started (collect/lanes.json); a stack started before lanes existed
    # is one lane on collect.instance.
    plan = lanes.read(run_dir)
    lane_list = plan.lanes if plan else [lanes.Lane(0, int(collect["instance"]), str(args.gpu))]
    stacks = []
    for lane in lane_list:
        try:
            stacks.append(Stack.attach(lane.instance))
        except FileNotFoundError:
            raise SystemExit(
                f"no stack on instance {lane.instance} (lane {lane.index}); start the run's stacks in another terminal:\n"
                f"  bash scripts/start_stack.sh collect {args.experiment}"
            )
        run_context.require_stack_of(stacks[-1], run_dir, "collect")
    already = run_context.group_threads(pid for stack in stacks for pid in stack.pids.values())
    if plan:
        lanes.require_drivers(len(stacks), already, plan.cost)
    else:
        run_context.require_headroom("collect", gpu=args.gpu, record=bool(collect["record"]), already=already)
    run_context.require_deploy_patches(cfg)
    # The instance's port block and the scene the sim is showing. The subprocess this used to be
    # was handed exactly this as its whole environment. Lane 0 runs in this process; each other lane
    # adopts its own stack in its own process (lane_process).
    run_context.adopt_stack_env(stacks[0])
    prepare.prepare(cfg, run_dir)  # idempotent: the stack starter normally got here first
    inputs = inputs_path(run_dir)
    batches.resume_state(inputs)  # only finished batches count; an unfinished one reruns

    target = collect_target(collect["successes"], args.until_kept)
    children = []
    try:
        reminder = layout_reminder(cfg, profile_of(cfg, run_dir, frozen=True), prepare.scene_name(cfg, run_dir))
        reminder += (
            plan.describe()
            if plan
            else [f"collect: 1 lane -- the stack on instance {lane_list[0].instance} and this process"]
        )
        print("\n".join(reminder), flush=True)
        if reminder and sys.stdin.isatty():
            print(f"collect: first round in {REMINDER_SECONDS} s; Ctrl-C stops here, stacks and all", flush=True)
            time.sleep(REMINDER_SECONDS)
        # Each lane's first batch is claimed here, in lane order, so lane k starts with batch k on a fresh run --
        # the batch its stack's exporter session was opened for.
        first = [
            batches.claim_batch(
                inputs,
                lane.index,
                target=target,
                max_batches=collect["max_batches"],
                manifest_for=lambda b: batch_plan(cfg, run_dir, b)["manifest"],
            )
            for lane in lane_list
        ]
        spawn = multiprocessing.get_context("spawn")
        for lane, batch in zip(lane_list[1:], first[1:]):
            if batch is not None:
                child = spawn.Process(
                    target=lane_process,
                    args=(str(run_dir), lane.index, lane.instance, batch, target),
                    name=f"collect-lane{lane.index:02d}",
                )
                child.start()
                children.append(child)
        run_lane(cfg, run_dir, stacks[0], lane_list[0].index, first[0], target)
        for child in children:
            child.join()
        failed = [child.name for child in children if child.exitcode]
        if failed:
            print(
                f"collect: {', '.join(failed)} ended with an error; their unfinished batches rerun next time",
                flush=True,
            )
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()  # before its stack goes down under it
                child.join(10)
        if not args.keep_stack:
            for stack in stacks:
                stack.down()  # a recording stack left up holds hundreds of the container's 3840 pids
    rule = collect.get("stop_rule")
    state = batches.resume_state(inputs)  # a lane that died mid-batch leaves a claim; the next run reruns it

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
        if state["kept"] >= target:  # --until-kept: summary.json says kept < successes, so nothing treats it as done
            print(
                f"collect: paused at {state['kept']}/{collect['successes']} kept (--until-kept {args.until_kept}); "
                f"the same command without it continues -> {run_dir}"
            )
            return 0
        raise SystemExit(f"stopped at {state['kept']}/{collect['successes']} kept after {state['batches']} batches")
    print(f"collect: {state['kept']} kept in {state['batches']} batches -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
