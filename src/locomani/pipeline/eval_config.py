"""Closed-loop eval knobs (decision 8, refactor step 3 item 3).

Every field's default equals `eval_closed_loop.py`'s old argparse default, so
a run with no config file behaves identically to before this module existed.
`backend` nests a `SonicBackendConfig`; eval_closed_loop.py builds the actual
per-client backend config (no_encoder/clean_deploy/lockstep.enabled) from it
and hands it to `pipeline/stack.export_env` -- see the module docstring there.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

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
    # 0 since 2026-09-17: a stand-stage failure IS a failed round. Collection has
    # always scored it that way (task.stand_retries is 0 everywhere); evaluation,
    # the side that publishes a number, was the lenient one. Retrying also cannot
    # tell an initialisation stumble from a genuinely bad approach -- both arrive
    # as `[stand]` -- so it quietly credits the policy for the ones it caused.
    # Set it to 1 only to diagnose a struggling stack, never to produce a result.
    stand_retries: int = 0  # --stand-retries: 0 or 1, reset retries after a stand-stage failure
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
    # READY token on arms-down data stalls every round in the prelude (docs/single_arm_design.md, "初始 token 文件").

    def __post_init__(self) -> None:
        """The async client and `backend.clean_deploy` are one decision, not two.

        `vla_round.CleanClock` exists for a deploy engaged by SONIC_AUTO_START: its
        commander drops every planner command, which is how "no planner command in
        the deploy's lifetime" is achieved. `eval_closed_loop` installs that
        commander for `client == "async"` -- the default -- while the stack only
        exports SONIC_AUTO_START when `backend.clean_deploy` is set. With the two
        apart the suppressed `start_control` has nothing standing in for it: the
        deploy takes the mode switch, receives the standing token and never starts
        its control loop, so the robot simply falls where the reset dropped it and
        the round dies as `[stand] robot did not stabilise after reset`.

        The pre-refactor runner kept them together in one line
        (`wall) export CLEAN_DEPLOY=1; CL="--client async"`); config groups split
        them across two dataclasses and nothing held the pair. Measured 2026-09-17:
        with clean_deploy false, the deploy's log for such a round holds 245
        received tokens and not one `Loop timing` line.
        """
        object.__setattr__(self, "backend", replace(self.backend, clean_deploy=self.client == "async"))
        if self.image_source == "isaac" and self.client == "lockstep":
            # The lockstep step reply carries MuJoCo images rendered at the frozen boundary; the mirror
            # renders the live sim on request, which a stepped sim does not have between steps.
            raise ValueError("EvalConfig: image_source 'isaac' needs a wall-clock client, not client 'lockstep'")

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
