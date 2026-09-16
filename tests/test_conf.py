"""Hydra composition contract (docs/handoff/infra-plan-20260916.md decisions 2 and 5)."""

from __future__ import annotations

import glob
import os

import pytest
from omegaconf import OmegaConf

from locomani import conf
from locomani.embodiment_config import EmbodimentConfig
from locomani.pipeline.collect_config import CollectConfig
from locomani.pipeline.train_config import TrainConfig
from locomani.scene_config import SceneConfig
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.crate.config import CrateTaskConfig

EXPERIMENTS = sorted(
    os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(conf.CONFIGS, "experiment", "*.yaml"))
)


def test_default_run_composes_into_typed_dataclasses():
    run, cfg = conf.load_run([])
    assert isinstance(run, conf.RunConfig)
    assert isinstance(run.task, BottleTaskConfig) and run.task.plugin == "bottle"
    assert isinstance(run.embodiment, EmbodimentConfig) and run.embodiment.arms == "single"
    assert isinstance(run.scene, SceneConfig) and isinstance(run.collect, CollectConfig)
    assert isinstance(run.train, TrainConfig)
    assert OmegaConf.to_container(cfg, resolve=False)["out"] == "outputs/runs/${name}" and run.out == "outputs/runs/run"


@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_file_composes_and_validates(name):
    run, _ = conf.load_run([f"+experiment={name}"])
    assert run.name == name, "an experiment file names the run after itself"
    assert run.embodiment.name in conf.TASK_EMBODIMENTS[run.task.plugin]


def test_group_yaml_cannot_carry_a_key_the_dataclass_does_not_declare():
    with pytest.raises(Exception, match="not_a_field|Key"):
        conf.load_run(["+task.not_a_field=1"])


def test_dataclass_validation_still_fires_after_composition():
    with pytest.raises(ValueError, match="closure"):
        conf.load_run(["task.reach.simple.close_seconds=99"])


def test_task_and_embodiment_must_be_compatible():
    with pytest.raises(ValueError, match="runs on"):
        conf.load_run(["task=crate_table30", "embodiment=g1_single_arm", "scene=crate_table30_depth04"])
    run, _ = conf.load_run(["task=crate_table30", "embodiment=g1_bimanual", "scene=crate_table30_depth04"])
    assert isinstance(run.task, CrateTaskConfig) and run.embodiment.arms == "bimanual"


def test_cli_override_reaches_a_nested_field_and_the_stage_is_recorded():
    run, cfg = conf.load_run(["collect.successes=8", "stage=eval", "gpu=1"])
    assert run.collect.successes == 8 and run.stage == "eval"
    pilot = conf.pilot_from(run, through="eval")
    assert pilot.successes == 8 and pilot.train_gpu == "1"
    assert "CUDA_VISIBLE_DEVICES=1" in pilot.stack_env and "MUJOCO_EGL_DEVICE_ID=1" in pilot.stack_env


def test_snapshot_writes_the_resolved_tree(tmp_path):
    _, cfg = conf.load_run(["name=snap"])
    path = conf.snapshot(cfg, tmp_path)
    saved = OmegaConf.load(path)
    assert saved.out == "outputs/runs/snap" and saved.task.plugin == "bottle"
