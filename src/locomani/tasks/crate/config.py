"""Crate task configuration, resolved at the entry and passed to its consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict
from locomani.tasks.crate.carry_config import CrateCarryConfig
from locomani.tasks.crate.visual_config import CrateVisualConfig

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CrateTaskConfig:
    plugin: str = "crate"  # task plugin name (locomani.tasks.registry): decides which pipeline runs this config
    carry: CrateCarryConfig = field(default_factory=CrateCarryConfig)
    visual: CrateVisualConfig = field(
        default_factory=CrateVisualConfig
    )  # Camera-guided expert controls; legacy experts ignore these
    wall_feature: str = "rail"  # WALL_FEATURE: rail (rim bars) or wall (full-height long walls) grasp geometry
    wall_walk: bool = False  # WALL_WALK=1: bimanual_carry's walk-in approach instead of a ready stand
    gripper_gear: float = 250.0  # GRIPPER_GEAR: dex1 finger actuator gear (env override for A/B runs)
    couple_fingers: bool = True  # COUPLE_FINGERS=0: independent dex1 fingers instead of a coupled parallel jaw
    crate_setback: float = 0.0  # CRATE_SETBACK: moves the near wall this far in from the table edge
    crate_setback_levels_m: tuple[float, ...] = ()  # per-episode crate setback levels (m) from the edge; () = 0
    crate_yaw_levels_deg: tuple[
        float, ...
    ] = ()  # per-episode crate yaw levels (deg) about its centre; () = 0. With both set, rounds walk the levels' product in a fixed shuffled order
    crate_dy_range_m: tuple[float, float] = (
        -0.08,
        0.08,
    )  # per-episode crate lateral offset (m) along the table edge, sampled uniformly each round
    scene_preset: str = "crate_bimanual"  # SCENE_PRESET: build_sim_scene.py --preset for the crate scene
    # -- the fields BottleTaskConfig has and this one was missing (2026-09-11) --
    # Task.config_cls is meant to be the one object a pipeline needs to drive a
    # task, and `pipeline/stack.py` already reads `task_cfg.prompt`; without it
    # the crate stack fell back to start_stack.sh's "walk forward". Defaults keep
    # today's behaviour: the prompt reaches only the data exporter
    # (start_stack.sh:327 --task-prompt), which the crate line does not run yet.
    prompt: str = "pick up the crate"  # the episode instruction, and the stack's --task-prompt
    deploy_self_heal_rounds: int = 2  # consecutive '[stand] did not stabilise' rounds before restarting this
    # instance's deploy (0 = off). Was a bare `dead_rounds >= 2` in bimanual_wall.main and absent from
    # bimanual_carry.py; backends/sonic/deploy.SelfHeal is the shared trigger.

    def __post_init__(self):
        if self.visual.pose_source == "assisted" and self.carry.reach_control != "whole_body":
            raise ValueError("visual.pose_source=assisted requires carry.reach_control=whole_body")
        bounds = self.crate_dy_range_m
        if (
            not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) for v in bounds)
            or bounds[0] > bounds[1]
        ):
            raise ValueError("crate_dy_range_m must be a finite numeric (low, high) pair with low <= high")

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "CrateTaskConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> "CrateTaskConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "CrateTaskConfig":
        return apply_overrides(self, overrides)


DEFAULT_CRATE_TASK = CrateTaskConfig()
