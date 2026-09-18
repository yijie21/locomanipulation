#!/usr/bin/env python3
"""Run the bimanual policy in closed loop.

    python scripts/eval_crate.py outputs/runs/crate_table30_record --gpu 1
    python scripts/eval_crate.py outputs/runs/crate_table30_record --dry-run

Unlike the single-arm line, the bimanual evaluator owns its own stack and policy server, so this
script's job is to hand it the run: the checkpoint, the scene copy, the task config the experts
read, and the eval fields it accepts. Those fields were previously left at the evaluator's defaults
-- the port, the budget, the execution horizon and the opening token were all declared in the
config and never delivered, which is the defect this refactor exists to remove.

The opening token is the one that matters most here: this line opens in the crate pose, and the
single-arm token would put the robot somewhere its training data never starts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_common  # noqa: E402

from locomani import paths  # noqa: E402
from locomani.pipeline.eval_config import EvalConfig  # noqa: E402
from locomani.pipeline.stages import Stage  # noqa: E402

SIM_PY = Path(paths.VENV_SIM_PY)
CHILD_ENV = dict(os.environ, PYTHONPATH=str(ROOT / "src"))


def checkpoint_of(run_dir: Path) -> Path:
    summary = run_dir / "summary.json"
    entry = json.loads(summary.read_text()).get("train") if summary.exists() else None
    if not entry or not entry.get("checkpoint"):
        raise SystemExit(f"{summary} has no checkpoint; run the train stage first")
    return Path(entry["checkpoint"])


def scene_of(run_dir: Path, cfg: dict) -> Path:
    """The run's own scene copy, which keeps the source directory's name for the expert's lookups."""
    scene = run_dir / Path(cfg["scene"]["dir"]).name
    if not (scene / "scene_meta.json").exists() and not scene.name:
        raise SystemExit(f"no scene copy at {scene}; run the collection stage in this directory first")
    return scene


def rounds_argv(cfg: dict, run_dir: Path, ev: Path, checkpoint: Path, gpu: str) -> list:
    e, collect = cfg["eval"], cfg["collect"]
    return [
        SIM_PY, "-m", "locomani.pipeline.eval_crate_closed_loop",
        "--checkpoint", checkpoint,
        "--scene", scene_of(run_dir, cfg),
        "--out", ev,
        "--config", run_dir / "task_config.json",
        "--eval-config", ev / "eval_config.json",         # whole block, not hand-picked fields
        "--batch-rounds", str(collect["batch_rounds"]),   # stack restart cadence, as the crate pilots
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
    cfg = run_common.load_run_config(run_dir)
    checkpoint = checkpoint_of(run_dir)
    ev = run_dir / "eval"
    evaluation = EvalConfig.from_dict(cfg["eval"])  # validated before anything starts
    argv_rounds = rounds_argv(cfg, run_dir, ev, checkpoint, args.gpu)

    if args.dry_run:
        run_common.show("eval", argv_rounds, {"PYTHONPATH": CHILD_ENV["PYTHONPATH"]})
        return 0
    if (ev / "closed_loop.json").exists():
        raise SystemExit(f"{ev / 'closed_loop.json'} exists; move the finished evaluation aside before rerunning")
    if int(cfg["collect"]["instance"]) == 0:
        raise SystemExit("this line does not run on instance 0; give it its own port block")
    run_common.require_headroom("eval", gpu=args.gpu)
    run_common.require_deploy_patches(cfg)
    ev.mkdir(parents=True, exist_ok=True)
    evaluation.dump(ev / "eval_config.json")  # provenance: what this evaluation was asked for
    Stage(run_dir).check("eval", argv_rounds, env=CHILD_ENV)

    result = json.loads((ev / "closed_loop.json").read_text())
    run_common.summarise(
        run_dir,
        "eval",
        checkpoint=str(checkpoint),
        **{k: result[k] for k in ("rounds", "successes", "success_rate") if k in result},
    )
    print(f"eval: {result.get('successes')}/{result.get('rounds')} -> {ev / 'closed_loop.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
