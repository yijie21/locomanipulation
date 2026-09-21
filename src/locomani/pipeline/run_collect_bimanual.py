#!/usr/bin/env python3
"""Drive bimanual collection on a stack someone else started: the crate experts, batch by batch.

    python -m locomani.pipeline.run_collect_bimanual configs/experiments/crate_table30_record.yaml --gpu 1

`scripts/bimanual/run_agent_server.sh` is one line over this, and `scripts/collect_data.sh` is
what brought the stack up in the other terminal.

`task.expert` chooses which expert runs -- `visual` is the recording line, `wall` and `carry` are
diagnostics -- and the combination is checked against the task profile before any resource is
touched. This line's artifacts sit beside the run rather than under `collect/`, because the scene
copy keeps the source's name and the expert's own lookups depend on that.

A batch may end on a failed round without the run being over, so the expert's exit status is
recorded rather than raised on; the single-arm line treats a non-zero exit as fatal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.pipeline.lines import line_of
from locomani.pipeline.run_collect_single_arm import kept_in, open_exporter_session
from locomani.pipeline.stages import Stage
from locomani.stack import Stack
from locomani.tasks.crate.config import CrateTaskConfig
from locomani.tasks.crate.prepare import check_profile, expert_argv

ROOT = Path(paths.REPO)


def run_batch(cfg: dict, run_dir: Path, collect_root: Path, stack: Stack, state: dict) -> Path:
    batch = state["batches"]
    tag = f"{batch:02d}"
    manifest = collect_root / f"manifest_{tag}.jsonl"
    if manifest.exists() and str(manifest) not in state["manifests"]:
        raise SystemExit(f"{manifest} exists but no batch claimed it; inspect before continuing")
    collect = cfg["collect"]
    argv = expert_argv(
        cfg["task"]["expert"],
        run_dir / "task_config.json",
        manifest,
        collect["batch_rounds"],
        collect["seed0"] + 100 * batch,
        record=bool(collect["record"]),
        trace_dir=run_dir / "trace",
    )
    (run_dir / "trace").mkdir(exist_ok=True)
    state["sessions"].append(open_exporter_session(stack, collect_root, batch))
    state["manifests"].append(str(manifest))
    (run_dir / "collect").mkdir(exist_ok=True)
    (run_dir / "collect" / "inputs.json").write_text(json.dumps(state, indent=1))
    Stage(run_dir).run(f"{cfg['task']['expert']}_{tag}", argv, env=dict(stack.env))
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="run directory name; the experiment's own name by default")
    ap.add_argument("--keep-stack", action="store_true", help="leave the stack running when the run ends")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    cfg, run_dir = run_context.open_run(args.experiment, args.name, write=not args.dry_run)
    collect, line = cfg["collect"], line_of(cfg)
    collect_root = line.collect_root(run_dir)

    if args.dry_run:
        # The real path reaches this check inside prepare_scene, after admission; a dry run
        # touches no resource, so it is the only thing standing between a bad expert/profile
        # combination and a confusing failure later.
        check_profile(CrateTaskConfig.from_dict(cfg["task"]), cfg["task"]["expert"], bool(collect["record"]))
        argv_expert = expert_argv(
            cfg["task"]["expert"],
            run_dir / "task_config.json",
            collect_root / "manifest_00.jsonl",
            collect["batch_rounds"],
            collect["seed0"],
            record=bool(collect["record"]),
            trace_dir=run_dir / "trace",
        )
        run_context.show(f"{cfg['task']['expert']}_00", argv_expert)
        print(f"\n  attaches to instance {collect['instance']}; {collect['max_batches']} batches at most")
        return 0

    run_context.require_headroom("collect", gpu=args.gpu)
    run_context.require_deploy_patches(cfg)
    line.prepare(cfg, run_dir)  # idempotent: the stack starter normally got here first
    inputs = run_dir / "collect" / "inputs.json"
    state = json.loads(inputs.read_text()) if inputs.exists() else dict(sessions=[], manifests=[], kept=0, batches=0)

    try:
        stack = Stack.attach(int(collect["instance"]))
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {collect['instance']}; start one in another terminal:\n"
            f"  bash scripts/collect_data.sh {args.experiment} --gpu {args.gpu}"
        )
    try:
        while state["kept"] < collect["successes"] and state["batches"] < collect["max_batches"]:
            manifest = run_batch(cfg, run_dir, collect_root, stack, state)
            state["kept"] += kept_in(manifest, bool(collect["record"]))
            state["batches"] += 1
            inputs.write_text(json.dumps(state, indent=1))
            print(
                f"collect: batch {state['batches'] - 1:02d} -> {state['kept']}/{collect['successes']} kept", flush=True
            )
    finally:
        if not args.keep_stack:
            stack.down()  # a recording stack left up holds ~1800 of the container's 3840 pids

    rows = [json.loads(line) for m in state["manifests"] for line in Path(m).read_text().splitlines() if line.strip()]
    run_context.summarise(
        run_dir,
        "collect",
        expert=cfg["task"]["expert"],
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
