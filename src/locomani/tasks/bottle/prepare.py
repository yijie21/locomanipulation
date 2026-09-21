"""Freeze what the single-arm line's processes read before any of them starts.

The sim reads the scene at launch and the expert reads the two config dumps at launch, so this has
to happen before the stack comes up -- which, since 2026-09-21, is a different process from the one
that drives the rounds (`scripts/collect_data.sh` and `scripts/single_arm/run_agent_server.sh`).
Both call this, so it is idempotent: whoever gets there first freezes the run, the other finds it
already frozen. The crate line's equivalent is `locomani.tasks.crate.prepare.prepare`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from locomani import paths
from locomani.tasks.bottle.config import BottleTaskConfig

ROOT = Path(paths.REPO)


def prepare(cfg: dict, run_dir: Path) -> Path:
    """Copy the scene into the run and dump the two files the expert process reads. Returns the scene."""
    run_dir = Path(run_dir)
    scene = run_dir / "scene"
    if not scene.exists():
        source = Path(cfg["scene"]["dir"])
        shutil.copytree(source if source.is_absolute() else ROOT / source, scene)
    task = BottleTaskConfig.from_dict(cfg["task"])
    task.dump(run_dir / "task_config.json")
    task.reach.dump(run_dir / "reach_config.json")
    return scene


def collect_root(run_dir: Path) -> Path:
    """Where this line keeps manifests, videos, traces and the batch bookkeeping."""
    return Path(run_dir) / "collect"
