"""Learn-pilot knobs (decision 8, refactor step 3 item 4).

Every field's default equals `run_learn_pilot.py`'s old argparse default, so a
run with no config file behaves identically to before this module existed.
See `configs/pilot/learn_200.json` for the canonical ready-visible protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PilotConfig:
    successes: int = 60  # --successes: kept episodes to collect before training
    batch_rounds: int = 10  # --batch-rounds: rounds attempted per collection batch
    max_batches: int = 12  # --max-batches: give up collecting after this many batches
    holdout: int = 6  # --holdout: episodes held out for the validation split
    steps: int = 3000  # --steps: final training step
    save_steps: int = 1500  # --save-steps: periodic checkpoint interval, capped at the final step
    train_gpu: str = "1"  # --train-gpu: CUDA_VISIBLE_DEVICES for the train/serve stages
    eval_rounds: int = 100  # --eval-rounds: closed-loop eval rounds
    eval_seed: int = 6000  # --eval-seed: closed-loop eval seed
    seed0: int = 27000  # --seed0: first collection batch's scene/round seed (batch b uses seed0 + 100*b)
    through: str = field(
        default="eval", metadata={"choices": ("collect", "convert", "train", "eval")}
    )  # --through: stop after this stage
    squat_max: int = 4096  # SQUAT_MAX: train stage's MuJoCo-EGL VRAM guard (MiB); shared with another instance
    gym_env: tuple[str, ...] = ()  # --gym-env KEY=VAL: extra environment for the collection gym only
    stack_env: tuple[str, ...] = ()  # --stack-env KEY=VAL: extra environment for both stacks (collection and eval)

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "PilotConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> "PilotConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "PilotConfig":
        return apply_overrides(self, overrides)
