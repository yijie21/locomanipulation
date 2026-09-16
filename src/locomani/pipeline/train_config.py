"""Train stage knobs for the GR00T finetune (scripts/train_bottle.sh reads these through the bridge).

Split out of `PilotConfig` (decision 4 of docs/handoff/infra-plan-20260916.md). The recipe itself
(batch composition, optimizer choice by VRAM) stays in the training script; this file only carries
what an experiment varies.
"""

from __future__ import annotations

from dataclasses import dataclass

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 3000  # final training step
    save_steps: int = 1500  # periodic checkpoint interval, capped at the final step
    holdout: int = 6  # episodes held out for the validation split at conversion time
    gpus: int = 1  # torchrun --nproc_per_node; 1 = the single-GPU recipe (8-bit optimizer by VRAM)
    global_batch: int = 64  # effective batch (per-device batch x grad accumulation x gpus)
    squat_max: int = 4096  # MuJoCo-EGL VRAM guard (MiB) for the serve/eval processes sharing the train card
    base_model: str | None = None  # checkpoint to start from; null = the recipe's default GR00T base

    def __post_init__(self):
        if not (self.steps > 0 and self.save_steps > 0 and self.holdout >= 0):
            raise ValueError("train.steps and save_steps must be positive, holdout non-negative")
        if not (1 <= self.gpus <= 8 and self.global_batch > 0 and self.squat_max > 0):
            raise ValueError("train.gpus must be 1-8, global_batch and squat_max positive")

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> TrainConfig:
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> TrainConfig:
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, pairs) -> TrainConfig:
        return apply_overrides(self, pairs)
