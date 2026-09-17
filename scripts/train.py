#!/usr/bin/env python3
"""Finetune GR00T on the dataset this run converted.

    python scripts/train.py outputs/runs/bottle_simple_200 --gpu 1
    python scripts/train.py outputs/runs/bottle_simple_200 --gpu 1 --dry-run

The recipe -- batch composition, the optimizer choice by VRAM, the single-card path -- stays in
`scripts/train_bottle.sh`, which reads it all from the environment. This script's whole job is to
turn the run's `train` and `embodiment` blocks into that environment, so the same file that says
what was collected also says what was trained on it.
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

from locomani.pipeline.stages import Stage  # noqa: E402

RECIPE = ROOT / "scripts" / "train_bottle.sh"


def dataset_of(run_dir: Path) -> Path:
    entry = (
        json.loads((run_dir / "summary.json").read_text()).get("convert")
        if (run_dir / "summary.json").exists()
        else None
    )
    if not entry or not entry.get("train_dataset"):
        raise SystemExit(f"{run_dir}/summary.json has no converted dataset; run the convert stage first")
    return Path(entry["train_dataset"])


def checkpoint_of(run_dir: Path, train: dict) -> Path:
    """Where the recipe leaves the finished weights: a step directory when the last step saves."""
    steps, every = train["steps"], min(train["save_steps"], train["steps"])
    return run_dir / "model" / f"checkpoint-{steps}" if steps % every == 0 else run_dir / "model"


def recipe_env(cfg: dict, run_dir: Path, dataset: Path, gpu: str) -> dict[str, str]:
    train = cfg["train"]
    env = dict(
        DATA=str(dataset),
        RUN="model",
        OUT_ROOT=str(run_dir),
        MAX_STEPS=str(train["steps"]),
        SAVE_STEPS=str(min(train["save_steps"], train["steps"])),
        SAVE_LIMIT="1",
        WANDB="0",
        WORKERS="2",
        GPUS=str(train["gpus"]),
        SINGLE_GPU="1" if train["gpus"] == 1 else "0",
        GLOBAL_BATCH=str(train["global_batch"]),
        SQUAT_MAX=str(train["squat_max"]),  # how much foreign VRAM the recipe tolerates on the card
        CUDA_VISIBLE_DEVICES=str(gpu),
        MODALITY=str(ROOT / cfg["embodiment"]["gr00t_modality_config"]),
    )
    if train.get("base_model"):
        env["BASE_MODEL"] = str(train["base_model"])
    return env


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="the run directory convert wrote")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, start nothing")
    args = ap.parse_args(argv)

    run_dir = args.run if args.run.is_absolute() else ROOT / args.run
    cfg = run_common.load_run_config(run_dir)
    dataset = dataset_of(run_dir)
    env = recipe_env(cfg, run_dir, dataset, args.gpu)
    checkpoint = checkpoint_of(run_dir, cfg["train"])

    if args.dry_run:
        run_common.show("train", ["bash", RECIPE], env)
        print(f"\n  checkpoint would be {checkpoint}")
        return 0
    if (run_dir / "model").exists():
        raise SystemExit(
            f"{run_dir / 'model'} exists; a half-written checkpoint is worse than none -- inspect it first"
        )
    run_common.require_headroom("train", gpu=args.gpu)
    Stage(run_dir).check("train", ["bash", RECIPE], env=dict(os.environ, **env))
    if not checkpoint.exists():
        raise SystemExit(f"the recipe finished but {checkpoint} is not there")
    run_common.summarise(
        run_dir, "train", checkpoint=str(checkpoint), steps=cfg["train"]["steps"], dataset=str(dataset)
    )
    print(f"train: {cfg['train']['steps']} steps -> {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
