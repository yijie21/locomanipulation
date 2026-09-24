"""Each experiment file is complete, and the blocks it declares are ones this code accepts.

`configs/experiments/*.yaml` is one fully expanded file per experiment: no inheritance, no
composition, and therefore nothing to resolve before reading it. What that buys is only real if the
file is also correct, so the blocks whose consumers validate them -- task, the collection backend,
and the evaluator's own config -- are parsed here with the same strict `from_dict` the pipeline
uses, where an unknown field is an error rather than a silently ignored line.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from locomani.backends.sonic.config import SonicBackendConfig
from locomani.pipeline.eval_config import EvalConfig
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.crate.config import CrateTaskConfig

EXPERIMENTS = Path(__file__).resolve().parents[1] / "configs" / "experiments"
FILES = sorted(EXPERIMENTS.glob("*.yaml"))
TASK_SCHEMA = {"bottle": BottleTaskConfig, "pick_place": BottleTaskConfig, "crate": CrateTaskConfig}
BLOCKS = {"name", "task", "embodiment", "scene", "backend", "collect", "train", "eval"}


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def test_there_are_experiments():
    assert FILES, "configs/experiments/ is the entry point; an empty directory means no run is reproducible"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_file_is_self_contained(path):
    cfg = load(path)
    assert cfg["name"] == path.stem, "the run name is the file name, so a run directory names its recipe"
    assert set(cfg) == BLOCKS, f"unexpected or missing blocks: {set(cfg) ^ BLOCKS}"
    assert "defaults" not in cfg, "an expanded file composes nothing"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_task_block_is_accepted_by_its_expert(path):
    cfg = load(path)
    TASK_SCHEMA[cfg["task"]["plugin"]].from_dict(cfg["task"])


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_backend_and_eval_blocks_are_accepted_by_their_consumers(path):
    cfg = load(path)
    SonicBackendConfig.from_dict(cfg["backend"])
    EvalConfig.from_dict(cfg["eval"])  # the file the evaluator is handed, nested backend included


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_paths_the_file_names_exist(path):
    cfg = load(path)
    root = EXPERIMENTS.parents[1]
    # `scene: {name: ...}` resolves through configs/scenes.yaml (run_context.resolve_scene); the registry promises
    # the name exists, so here only the registry entry is required, not the (gitignored) outputs/ directory.
    if "name" in cfg["scene"]:
        registry = yaml.safe_load((root / "configs/scenes.yaml").read_text())
        assert cfg["scene"]["name"] in registry, f"scene name {cfg['scene']['name']!r} not in configs/scenes.yaml"
        scene_dir = registry[cfg["scene"]["name"]]["dir"]
    else:
        scene_dir = cfg["scene"]["dir"]
    for value in (cfg["embodiment"]["gr00t_modality_config"], scene_dir, cfg["eval"]["initial_token"]):
        assert (root / value).exists(), f"{value} does not exist"
    cases = cfg["scene"].get("cases")
    assert cases is None or (root / cases).exists()
