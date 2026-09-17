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
COMPOSED = {  # expanded file -> the configs/experiment/ composition it was taken from (transition only)
    "bottle_simple_200": "bottle_simple_learn200",
    "bottle_simple_gate8": "bottle_simple_lowcom",
    "pick_place": "pick_place_default",
    "crate_table30_record": "crate_table30_record",
}


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


@pytest.mark.parametrize("new,old", sorted(COMPOSED.items()))
def test_expanded_file_equals_what_the_composed_one_resolved_to(new, old):
    """Transition test: the expanded file is the old composition, not a retyping of it.

    Delete it together with `conf.py` in the commit that removes the composed path -- until then it
    is what makes the two paths interchangeable, so a gate run on one is evidence about the other.
    """
    from locomani import conf  # the path being replaced; not imported at module level for that reason

    run, _ = conf.load_run([f"+experiment={old}", f"name={new}"])
    cfg = load(EXPERIMENTS / f"{new}.yaml")
    assert TASK_SCHEMA[cfg["task"]["plugin"]].from_dict(cfg["task"]).to_dict() == run.task.to_dict()
    assert SonicBackendConfig.from_dict(cfg["backend"]).to_dict() == run.backend.to_dict()
    assert EvalConfig.from_dict(cfg["eval"]).to_dict() == run.eval.to_dict()
    for block, resolved in (("collect", run.collect), ("train", run.train)):
        assert cfg[block] == {k: v for k, v in resolved.to_dict().items() if k != "schema_version"}


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
def test_paths_the_file_names_exist(path):
    cfg = load(path)
    root = EXPERIMENTS.parents[1]
    for value in (cfg["embodiment"]["gr00t_modality_config"], cfg["scene"]["dir"], cfg["eval"]["initial_token"]):
        assert (root / value).exists(), f"{value} does not exist"
    cases = cfg["scene"].get("cases")
    assert cases is None or (root / cases).exists()
