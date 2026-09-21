#!/usr/bin/env python3
"""Drive bimanual closed-loop rounds against a running stack and a running policy server.

    python -m locomani.pipeline.run_eval_bimanual outputs/runs/crate_table30_record --gpu 1

`scripts/bimanual/run_vla_inference.sh` is one line over this. It is terminal 3 of three:

    bash scripts/start_stack.sh    <run> --gpu 1   # terminal 1
    bash scripts/run_vla_server.sh <run> --gpu 1   # terminal 2

This line's evaluator (`eval_crate_closed_loop`) used to start the stack and serve the checkpoint
itself, one stack per batch; `--attach` is what makes it use the two that are already running. The
per-batch restart is kept as a DEPLOY restart, which is what it was for -- an evaluation runs no
exporter, so there are no exporter threads to reclaim.

The whole `eval` block is handed down as one file, as on the single-arm line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from locomani import paths
from locomani.pipeline import run_context
from locomani.pipeline.eval_config import EvalConfig
from locomani.pipeline.stages import Stage

ROOT = Path(paths.REPO)
SIM_PY = Path(paths.VENV_SIM_PY)


def checkpoint_of(run_dir: Path) -> Path:
    summary = run_dir / "summary.json"
    entry = json.loads(summary.read_text()).get("train") if summary.exists() else None
    if not entry or not entry.get("checkpoint"):
        raise SystemExit(f"{summary} has no checkpoint; run the train stage first")
    return Path(entry["checkpoint"])


def scene_of(run_dir: Path, cfg: dict) -> Path:
    """The run's own scene copy, which keeps the source directory's name for the expert's lookups."""
    return run_dir / Path(cfg["scene"]["dir"]).name


def rounds_argv(cfg: dict, run_dir: Path, ev: Path, checkpoint: Path, gpu: str) -> list:
    collect = cfg["collect"]
    return [
        SIM_PY, "-m", "locomani.pipeline.eval_crate_closed_loop",
        "--attach",                                       # stack and policy server are already up
        "--checkpoint", checkpoint,
        "--scene", scene_of(run_dir, cfg),
        "--out", ev,
        "--config", run_dir / "task_config.json",
        "--eval-config", ev / "eval_config.json",         # whole block, not hand-picked fields
        "--batch-rounds", str(collect["batch_rounds"]),   # deploy restart cadence, as the crate pilots
        "--instance", str(collect["instance"]),
        "--gpu", str(gpu),
        "--serve-gpu", str(gpu),
    ]  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory train wrote")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_context.load_run_config(run_dir)
    ev = run_dir / "eval"
    evaluation = EvalConfig.from_dict(cfg["eval"])
    checkpoint = checkpoint_of(run_dir)

    if args.dry_run:
        run_context.show("eval", rounds_argv(cfg, run_dir, ev, checkpoint, args.gpu))
        print(f"\n  attaches to instance {cfg['collect']['instance']}, policy port {evaluation.policy_port}")
        return 0

    run_context.require_headroom("eval", gpu=args.gpu, record=False)  # an evaluation runs no exporter
    ev.mkdir(parents=True, exist_ok=True)
    evaluation.dump(ev / "eval_config.json")
    Stage(run_dir).check("eval", rounds_argv(cfg, run_dir, ev, checkpoint, args.gpu))

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
