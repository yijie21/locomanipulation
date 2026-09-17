"""Closed-loop eval knobs (decision 8, refactor step 3 item 3).

Every field's default equals `eval_closed_loop.py`'s old argparse default, so
a run with no config file behaves identically to before this module existed.
`backend` nests a `SonicBackendConfig`; eval_closed_loop.py builds the actual
per-client backend config (no_encoder/clean_deploy/lockstep.enabled) from it
and hands it to `pipeline/stack.export_env` -- see the module docstring there.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locomani.backends.sonic.vla_config import VlaConfig
from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EvalConfig(VlaConfig):
    rounds: int = 20  # --fresh: rounds to run, sampled fresh from the collection distribution
    seed: int = 1000  # --seed: round-sampling seed, deliberately disjoint from the collection's seeds
    client: str = field(default="async", metadata={"choices": ("async", "lockstep", "blocking")})  # --client
    replan_interval: float = 0.4  # --replan-interval: async client seconds between chunk requests
    execution_horizon: int = 16  # --execution-horizon: blocking client chunk steps per replan
    policy_port: int = 5590  # --policy-port: serve_policy.py's port
    stand_retries: int = 1  # --stand-retries: 0 or 1, reset retries after a stand-stage failure
    latency_dist: str | None = None  # --latency-dist: empirical latency distribution JSON (lockstep client)
    latency_frames: int | None = None  # --latency-frames: fixed simulated inference latency, frames (lockstep client)
    restart_deploy: bool = True  # --set restart_deploy=false: skip the per-round deploy restart, as collection
    # does since 2026-09-10 (ledger bottle §11). Default True keeps every historical eval's protocol; the two
    # paths establish the deploy heading anchor differently (fresh start_control vs hold+reanchor), which is
    # the suspect in ledger bottle §18's 2/100.
    resume: bool = False  # --resume: top up the existing manifest to `rounds` instead of starting over
    force_resume: bool = False  # --force-resume: allow --resume across a change of seed
    spawn_edge: tuple[float, ...] = (0.7,)  # --spawn-edge: metres from the table, one or two values
    task: str = "pick up the water bottle"  # --task: the episode instruction
    target: str = "water_bottle"  # --target: which scene object grasp_gym draws from
    video_include_init: bool = False  # --video-include-init: start the round video at the reset, so a prelude
    # failure ([stand]/[view]) is on film too; the default starts it after the visibility gate
    initial_token: str = "configs/deployment/ready_grasp_initial_token.json"  # --initial-token-file: the token the
    # deploy holds while the first chunk is computed. It MUST match the opening pose of the training data -- the
    # READY token on arms-down data stalls every round in the prelude (docs/evaluation.md, "初始 token 文件").

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> "EvalConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "EvalConfig":
        return apply_overrides(self, overrides)
