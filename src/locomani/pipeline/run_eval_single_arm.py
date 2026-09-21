#!/usr/bin/env python3
"""Drive single-arm closed-loop rounds against a running stack and a running policy server.

    python -m locomani.pipeline.run_eval_single_arm outputs/runs/bottle_simple_200 --gpu 1

`scripts/single_arm/run_vla_inference.sh` is one line over this. It is terminal 3 of three:

    bash scripts/start_stack.sh    <run> --gpu 1     # terminal 1
    bash scripts/run_vla_server.sh <run> --gpu 1     # terminal 2
    bash scripts/single_arm/run_vla_inference.sh <run> --gpu 1

The whole `eval` block is handed to the evaluator as one file -- copying fields by hand is how a
hundred rounds once spawned at 0.7 m while the config said 1.1-1.3 m (2026-09-17).

Both of the other two are checked before a round starts: no stack on the instance, or nothing
listening on the policy port, and this refuses with the command that fixes it rather than failing
a round at a time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.pipeline.eval_config import EvalConfig
from locomani.pipeline.run_vla_server import wait_for_server
from locomani.pipeline.stages import Stage
from locomani.stack import Stack

ROOT = Path(paths.REPO)
SIM_PY = Path(paths.VENV_SIM_PY)
EVALUATOR = ROOT / "src/locomani/pipeline/eval_closed_loop.py"


def checkpoint_of(run_dir: Path) -> Path:
    summary = run_dir / "summary.json"
    entry = json.loads(summary.read_text()).get("train") if summary.exists() else None
    if not entry or not entry.get("checkpoint"):
        raise SystemExit(f"{summary} has no checkpoint; run the train stage first")
    return Path(entry["checkpoint"])


def rounds_argv(cfg: dict, run_dir: Path, ev: Path) -> list:
    return [
        SIM_PY, EVALUATOR,
        "--config", ev / "eval_config.json",   # the whole block, not a hand-picked few fields
        "--set", "backend.dataset_fps=25",     # the stack is pinned to the dataset's rate
        "--reach-config", run_dir / "reach_config.json",
        "--initial-token-file", ROOT / cfg["eval"]["initial_token"],
        "--out", ev / "closed_loop.json",
        "--trace-dir", ev / "traces",
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory train wrote")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--keep-stack", action="store_true", help="leave the stack running when the rounds end")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    ev = run_dir / "eval"
    evaluation = EvalConfig.from_dict(cfg["eval"])  # validated here, so a bad field fails before a round

    if args.dry_run:
        checkpoint_of(run_dir)
        run_context.show("eval", rounds_argv(cfg, run_dir, ev))
        print(f"\n  attaches to instance {cfg['collect']['instance']}, policy port {evaluation.policy_port}")
        return 0

    if (ev / "closed_loop.json").exists():
        raise SystemExit(f"{ev / 'closed_loop.json'} exists; move the finished evaluation aside before rerunning")
    checkpoint = checkpoint_of(run_dir)
    run_context.require_headroom("eval", gpu=args.gpu)
    run_context.require_deploy_patches(cfg)
    ev.mkdir(parents=True, exist_ok=True)
    evaluation.dump(ev / "eval_config.json")

    instance = int(cfg["collect"]["instance"])
    try:
        stack = Stack.attach(instance)
    except FileNotFoundError:
        raise SystemExit(
            f"no stack on instance {instance}; start one in another terminal:\n"
            f"  bash scripts/start_stack.sh {args.run} --gpu {args.gpu}"
        )
    wait_for_server(evaluation.policy_port)
    try:
        Stage(run_dir).check("eval", rounds_argv(cfg, run_dir, ev), env=dict(stack.env))
    finally:
        if not args.keep_stack:
            stack.down()

    result = json.loads((ev / "closed_loop.json").read_text())
    run_context.summarise(
        run_dir,
        "eval",
        checkpoint=str(checkpoint),
        **{k: result[k] for k in ("rounds", "successes", "success_rate") if k in result},
    )
    print(f"eval: {result.get('successes')}/{result.get('rounds')} -> {ev / 'closed_loop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
