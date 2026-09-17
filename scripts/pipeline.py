#!/usr/bin/env python3
"""Run one experiment through every stage, in order, skipping what is already there.

    python scripts/pipeline.py configs/experiments/bottle_simple_200.yaml --gpu 1
    python scripts/pipeline.py configs/experiments/bottle_simple_200.yaml --gpu 1 --from eval
    python scripts/pipeline.py configs/experiments/bottle_simple_smoke.yaml --dry-run

Each stage is the same script you would run by hand, in its own process; this only chooses which
ones the task needs and stops at the first failure. Parallelism is one terminal per card, one
experiment each -- there is no scheduler here and nothing to configure.

A stage whose artifact is already in the run directory is skipped. To redo one, delete what it
produced: the dataset directory, the model directory, `eval/closed_loop.json`. That rule is the
whole resume story, and it is visible in `ls`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_common  # noqa: E402

STAGES = ("collect", "convert", "train", "eval")
SCRIPTS = {  # by task plugin: the lines differ where their experts and evaluators differ
    "bottle": dict(collect="collect_bottle.py", eval="eval_bottle.py"),
    "pick_place": dict(collect="collect_bottle.py", eval="eval_bottle.py"),
    "bottle_visual": dict(collect="collect_bottle.py", eval="eval_bottle.py"),
    "crate": dict(collect="collect_crate.py", eval="eval_crate.py"),
}


def done(stage: str, run_dir: Path) -> bool:
    """Is this stage's artifact already here? Delete it to run the stage again."""
    if stage == "collect":
        summary = run_dir / "summary.json"
        return summary.exists() and "collect" in json.loads(summary.read_text())
    return {
        "convert": run_dir / "train_dataset",
        "train": run_dir / "model",
        "eval": run_dir / "eval" / "closed_loop.json",
    }[stage].exists()


def command(stage: str, plugin: str, experiment: Path, run_dir: Path, gpu: str, name: str | None) -> list[str]:
    scripts = SCRIPTS[plugin]
    if stage == "collect":  # the first stage opens the run from the experiment file
        argv = [sys.executable, str(ROOT / "scripts" / scripts["collect"]), str(experiment), "--gpu", gpu]
        return argv + (["--name", name] if name else [])
    script = scripts["eval"] if stage == "eval" else f"{stage}.py"
    argv = [sys.executable, str(ROOT / "scripts" / script), str(run_dir)]
    return argv + ([] if stage == "convert" else ["--gpu", gpu])  # convert needs no card


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", type=Path)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--name", default=None, help="run directory name; the experiment's own name by default")
    ap.add_argument("--from", dest="first", choices=STAGES, default="collect")
    ap.add_argument("--to", dest="last", choices=STAGES, default="eval")
    ap.add_argument("--dry-run", action="store_true", help="print what each stage would run, start nothing")
    args = ap.parse_args(argv)

    cfg, run_dir = run_common.open_run(args.experiment, args.name, write=False)
    plugin = cfg["task"]["plugin"]
    if plugin not in SCRIPTS:
        raise SystemExit(f"task.plugin {plugin!r} has no scripts; known plugins: {sorted(SCRIPTS)}")
    wanted = STAGES[STAGES.index(args.first) : STAGES.index(args.last) + 1]

    for stage in wanted:
        argv_stage = command(stage, plugin, args.experiment, run_dir, args.gpu, args.name)
        if args.dry_run:
            run_common.show(stage, argv_stage + ["--dry-run"])
            continue
        if done(stage, run_dir):
            print(f"pipeline: {stage} already done in {run_dir}; delete its artifact to redo it", flush=True)
            continue
        print(f"pipeline: {stage} -> {' '.join(argv_stage)}", flush=True)
        result = subprocess.run(argv_stage, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(f"pipeline: {stage} failed ({result.returncode}); the run is at {run_dir}")
    if not args.dry_run:
        print(f"pipeline: {' -> '.join(wanted)} done -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
