"""Hydra composition over the config dataclasses (decision 2 of docs/handoff/infra-plan-20260916.md).

A run is task x embodiment x scene x backend x collect x train x eval, each a config group under
`configs/` whose YAML files are typed by the matching dataclass: a key the dataclass does not
declare is rejected at composition, and `__post_init__` validation runs when the composed tree is
turned back into dataclass instances (`objects`). `configs/experiment/<name>.yaml` files fix a
version of a whole run and are selected with `+experiment=<name>`.

Only the compose API is used (no `@hydra.main`): the launcher and the CLI both call `compose_run`,
and no working-directory or logging behaviour is taken over.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import MISSING, DictConfig, OmegaConf

from locomani import paths
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.embodiment_config import EmbodimentConfig
from locomani.pipeline.collect_config import CollectConfig
from locomani.pipeline.eval_config import EvalConfig
from locomani.pipeline.train_config import TrainConfig
from locomani.scene_config import SceneConfig
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle_visual.config import BottleVisualTaskConfig
from locomani.tasks.crate.config import CrateTaskConfig

CONFIGS = os.path.join(paths.REPO, "configs")
STAGES = ("collect", "convert", "train", "eval", "stats")

# Schema node name -> dataclass, per group. A group YAML starts with `defaults: [_<schema>, _self_]`.
TASK_SCHEMAS = {"_bottle": BottleTaskConfig, "_crate": CrateTaskConfig, "_bottle_visual": BottleVisualTaskConfig}
BACKEND_SCHEMAS = {"_sonic": SonicBackendConfig, "_scalebfm": ScaleBfmBackendConfig}
SINGLE_SCHEMAS = {
    "embodiment": EmbodimentConfig,
    "scene": SceneConfig,
    "collect": CollectConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}
# Task plugin -> embodiments it can run on (decision 5: an incompatible pair fails at composition).
TASK_EMBODIMENTS = {
    "bottle": ("g1_single_arm",),
    "pick_place": ("g1_single_arm",),
    "bottle_visual": ("g1_single_arm",),
    "crate": ("g1_bimanual",),
}


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """The composed run: root fields plus one node per group (typed by that group's schema)."""

    name: str = "run"  # run name; the default output directory is outputs/runs/<name>
    stage: str = "collect"  # which stage the CLI executes: collect | convert | train | eval | stats
    out: str = "outputs/runs/${name}"  # run directory (stages read and write their manifests here)
    gpu: str = "0"  # CUDA_VISIBLE_DEVICES / MUJOCO_EGL_DEVICE_ID for every process of the run
    seed: int = 0  # run-level seed offset recorded in the snapshot (collect.seed0 and eval.seed stay authoritative)
    task: Any = MISSING
    embodiment: Any = MISSING
    scene: Any = MISSING
    backend: Any = MISSING
    collect: Any = MISSING
    train: Any = MISSING
    eval: Any = MISSING


def writable(node: DictConfig) -> DictConfig:
    """Clear the read-only flag OmegaConf sets on frozen dataclass nodes so YAML/CLI can override them."""
    OmegaConf.set_readonly(node, False)
    for key in node:
        child = node._get_node(key)
        if isinstance(child, DictConfig):
            writable(child)
    return node


def _schema(cls) -> DictConfig:
    return writable(OmegaConf.structured(cls))


_REGISTERED = False


def register() -> None:
    """Register every schema node in Hydra's ConfigStore (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(name="run_schema", node=_schema(RunConfig))
    for name, cls in TASK_SCHEMAS.items():
        cs.store(group="task", name=name, node=_schema(cls))
    for name, cls in BACKEND_SCHEMAS.items():
        cs.store(group="backend", name=name, node=_schema(cls))
    for group, cls in SINGLE_SCHEMAS.items():
        cs.store(group=group, name="_schema", node=_schema(cls))
    _REGISTERED = True


def compose_run(overrides: list[str] | None = None, config_name: str = "config") -> DictConfig:
    """Compose the run tree from `configs/` with Hydra overrides (`task=crate_table30`, `+experiment=x`, `collect.successes=8`)."""
    from hydra import compose, initialize_config_dir

    register()
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        return compose(config_name=config_name, overrides=list(overrides or []))


def objects(cfg: DictConfig) -> RunConfig:
    """The composed tree as dataclass instances (validation runs here), with the task/embodiment pairing checked."""
    run = OmegaConf.to_object(cfg)
    if not isinstance(run, RunConfig):
        raise TypeError("root config must be typed by run_schema (configs/config.yaml defaults)")
    if run.stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {run.stage!r}")
    plugin = getattr(run.task, "plugin", None)
    allowed = TASK_EMBODIMENTS.get(plugin)
    if allowed is None:
        raise ValueError(f"task.plugin {plugin!r} is not a registered task plugin ({sorted(TASK_EMBODIMENTS)})")
    if run.embodiment.name not in allowed:
        raise ValueError(f"task {plugin!r} runs on {allowed}, not on embodiment {run.embodiment.name!r}")
    return run


def load_run(overrides: list[str] | None = None) -> tuple[RunConfig, DictConfig]:
    cfg = compose_run(overrides)
    return objects(cfg), cfg


def snapshot(cfg: DictConfig, out: Path) -> Path:
    """Write the fully resolved run config next to the run's artifacts (this is the run's provenance)."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / "run_config.yaml"
    OmegaConf.save(cfg, path, resolve=True)
    (out / "run_config.json").write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=1))
    return path


def resolve_repo_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(paths.REPO) / p
