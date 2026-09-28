"""Freeze what the single-arm line's processes read before any of them starts.

The sim reads the scene at launch and the expert reads the two config dumps at launch, so this has
to happen before the stack comes up -- which, since 2026-09-21, is a different process from the one
that drives the rounds (`scripts/start_stack.sh collect` and `scripts/single_arm/run_agent_server.sh`).
Both call this, so it is idempotent: whoever gets there first freezes the run, the other finds it
already frozen. The crate line's equivalent is `locomani.tasks.crate.prepare.prepare`.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from locomani import paths
from locomani.tasks.bottle.config import BottleTaskConfig

ROOT = Path(paths.REPO)


def scene_name(cfg: dict, run_dir: Path) -> Path:
    """What the frozen scene is called, without freezing it -- a dry run must name it too."""
    return Path(run_dir) / "scene"


def prepare(cfg: dict, run_dir: Path) -> Path:
    """Copy the scene into the run and dump the two files the expert process reads. Returns the scene."""
    run_dir = Path(run_dir)
    scene = scene_name(cfg, run_dir)
    if not scene.exists():
        source = Path(cfg["scene"]["dir"])
        source = source if source.is_absolute() else ROOT / source
        if not (source / "scene_meta.json").is_file():
            raise SystemExit(f"no bottle scene at {source}; build it once per machine (docs/single_arm.md, section 1)")
        shutil.copytree(source, scene)
    task = BottleTaskConfig.from_dict(cfg["task"])
    task.dump(run_dir / "task_config.json")
    task.reach.dump(run_dir / "reach_config.json")
    return scene


def collect_root(run_dir: Path) -> Path:
    """Where this line keeps manifests, videos, traces and the batch bookkeeping."""
    return Path(run_dir) / "collect"


def trace_root(run_dir: Path) -> Path:
    """One directory per run since 2026-09-21; the file names carry the episode's timestamp."""
    return Path(run_dir) / "collect" / "state_trace"


def eval_backend(evaluation) -> dict:
    """The eval stack's backend block, as `EvalConfig` validated it, lockstep when the client is.

    `EvalConfig` already ties `clean_deploy` to `client == "async"`, which is exactly the pairing
    `eval_closed_loop` installs its commander by. The lockstep client needs the sim's step server
    and the deploy's `--sim-lockstep`, which `backend.lockstep.enabled` starts (stack/manager.py);
    `eval_closed_loop` derives the same flag from the client, so the stack is started to match it
    rather than trusting a second copy of the decision in the file (as the crate line does).
    """
    block = evaluation.backend.to_dict()
    return {**block, "lockstep": {**block["lockstep"], "enabled": evaluation.client == "lockstep"}}
