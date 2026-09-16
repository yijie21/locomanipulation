"""The YAML groups are the truth; the JSON twins still read by tests and experts may not drift from them."""

from __future__ import annotations

import glob
import os

import pytest

from locomani import conf
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.pipeline.eval_config import EvalConfig
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle_visual.config import BottleVisualTaskConfig
from locomani.tasks.crate.config import CrateTaskConfig

PAIRS = [(os.path.basename(p)[:-5], p) for p in sorted(glob.glob(os.path.join(conf.CONFIGS, "tasks", "*.json")))]
CLASSES = {"_bottle": BottleTaskConfig, "_crate": CrateTaskConfig, "_bottle_visual": BottleVisualTaskConfig}


def _schema_of(yaml_path: str) -> str:
    for line in open(yaml_path):
        if line.strip().startswith("- _"):
            return line.strip()[2:]
    raise AssertionError(f"{yaml_path} has no schema default")


@pytest.mark.parametrize("name,json_path", PAIRS)
def test_task_yaml_equals_its_json_twin(name, json_path):
    yaml_path = os.path.join(conf.CONFIGS, "task", f"{name}.yaml")
    if not os.path.exists(yaml_path):
        pytest.skip(f"{name} has no YAML group file (legacy JSON only)")
    cls = CLASSES[_schema_of(yaml_path)]
    from_json = cls.load(json_path)
    plugin = "pick_place" if name == "pick_place" else from_json.plugin
    embodiment = "g1_bimanual" if cls is CrateTaskConfig else "g1_single_arm"
    scene = "crate_table30_depth04" if cls is CrateTaskConfig else "sim"
    run, _ = conf.load_run([f"task={name}", f"embodiment={embodiment}", f"scene={scene}"])
    assert run.task.plugin == plugin
    assert run.task.to_dict() == from_json.to_dict(), f"configs/task/{name}.yaml drifted from configs/tasks/{name}.json"


@pytest.mark.parametrize(
    "group,name,cls",
    [
        ("backend", "sonic", SonicBackendConfig),
        ("backend", "scalebfm", ScaleBfmBackendConfig),
        ("eval", "closed_loop_100", EvalConfig),
    ],
)
def test_backend_and_eval_yaml_match_their_json_twins(group, name, cls):
    json_dir = {"backend": "backends", "eval": "eval"}[group]
    from_json = cls.load(os.path.join(conf.CONFIGS, json_dir, f"{name}.json"))
    run, _ = conf.load_run([f"{group}={name}"])
    composed = getattr(run, group)
    if group == "backend" and name == "sonic":
        # collection composes with the encoder on (the JSON carries the eval-era default)
        from_json = SonicBackendConfig.from_dict({**from_json.to_dict(), "no_encoder": False})
    assert composed.to_dict() == from_json.to_dict()
