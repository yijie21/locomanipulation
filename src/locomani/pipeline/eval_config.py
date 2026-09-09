"""Closed-loop eval knobs (decision 8, refactor step 3 item 3).

Every field's default equals `eval_closed_loop.py`'s old argparse default, so
a run with no config file behaves identically to before this module existed.
`backend` nests a `SonicBackendConfig`; eval_closed_loop.py builds the actual
per-client backend config (no_encoder/clean_deploy/lockstep.enabled) from it
and hands it to `pipeline/stack.export_env` -- see the module docstring there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locomani.backends.sonic.config import SonicBackendConfig
from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvalConfig:
    backend: SonicBackendConfig = field(default_factory=SonicBackendConfig)
    rounds: int = 20  # --fresh: rounds to run, sampled fresh from the collection distribution
    seed: int = 1000  # --seed: round-sampling seed, deliberately disjoint from the collection's seeds
    client: str = "async"  # --client: async | lockstep | blocking
    budget_s: float = 90.0  # --budget: per-round VLA time budget (the ladder's setting)
    replan_interval: float = 0.4  # --replan-interval: async client seconds between chunk requests
    execution_horizon: int = 16  # --execution-horizon: blocking client chunk steps per replan
    ensemble_n: int = 1  # --ensemble N: chunks to blend (1 = off, the ladder's setting)
    ensemble_m: float = 0.3  # --ensemble-m: exponential weight decay per chunk of age
    rtc: bool = False  # --rtc: async/lockstep real-time chunking (test-time inpainting)
    transition_commit: bool = False  # --transition-commit: experimental async RTC crouch-prefix commitment
    transition_decoder: str | None = None  # --transition-decoder: ONNX decoder matching the live deploy
    policy_port: int = 5590  # --policy-port: serve_policy.py's port
    stand_retries: int = 1  # --stand-retries: 0 or 1, reset retries after a stand-stage failure
    latency_dist: str | None = None  # --latency-dist: empirical latency distribution JSON (lockstep client)
    latency_frames: int | None = None  # --latency-frames: fixed simulated inference latency, frames (lockstep client)
    grip_gain: float = 1.0  # --grip-gain K: amplify the predicted gripper's deviation from fully-open
    snap: bool = True  # not --no-snap: send the predicted token snapped to the FSQ grid
    stall_abort_s: float = 0.0  # --stall-abort: end a round early after this many stalled seconds (0 = off)
    resume: bool = False  # --resume: top up the existing manifest to `rounds` instead of starting over
    force_resume: bool = False  # --force-resume: allow --resume across a change of seed
    spawn_edge: tuple[float, ...] = (0.7,)  # --spawn-edge: metres from the table, one or two values
    task: str = "pick up the water bottle"  # --task: the episode instruction
    target: str = "water_bottle"  # --target: which scene object grasp_gym draws from

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalConfig":
        return config_from_dict(cls, data)

    @classmethod
    def load(cls, path) -> "EvalConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "EvalConfig":
        return apply_overrides(self, overrides)
