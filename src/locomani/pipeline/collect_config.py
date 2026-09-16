"""Collect stage knobs: how many episodes, in what batches, with which extra process environment.

Split out of `PilotConfig` (decision 4 of docs/handoff/infra-plan-20260916.md: one config per
stage). `PilotConfig` is kept only for the crate driver's legacy fields until that line moves to stages.
"""

from __future__ import annotations

from dataclasses import dataclass

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CollectConfig:
    successes: int = 60  # kept episodes to collect before the stage is done
    batch_rounds: int = 10  # rounds attempted per collection batch (one stack lifetime)
    max_batches: int = 12  # give up collecting after this many batches
    seed0: int = 30000  # first batch seed; batch i uses seed0 + 100 * i
    instance: int = 0  # stack instance (port block); a launcher assigns it per job
    record: bool = True  # run the exporter and keep LeRobot sessions (false = manifests and traces only)
    gym_env: tuple[str, ...] = ()  # KEY=VAL pairs for the expert/gym process only
    stack_env: tuple[str, ...] = ()  # KEY=VAL pairs for the stack processes (sim, deploy, exporter)

    def __post_init__(self):
        if not (self.successes > 0 and self.batch_rounds > 0 and self.max_batches > 0):
            raise ValueError("collect.successes, batch_rounds and max_batches must be positive")
        if not 0 <= self.instance <= 9:
            raise ValueError("collect.instance must be 0-9 (one port block each)")
        for kv in (*self.gym_env, *self.stack_env):
            if "=" not in kv:
                raise ValueError(f"collect env entries are KEY=VAL, got {kv!r}")

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> CollectConfig:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> CollectConfig:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, pairs) -> CollectConfig:
        return apply_overrides(self, pairs)
