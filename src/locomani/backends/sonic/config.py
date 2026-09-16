"""SONIC backend knobs (decision 8, refactor step 3 item 1).

Every field's default equals today's `os.environ.get(NAME, default)` fallback,
so a run with no config file behaves identically to before this module
existed. `locomani/pipeline/stack.py` is the only module that turns a
`SonicBackendConfig` into environment variables for the shell stack
(`scripts/stack/start_stack.sh` and everything it launches); code elsewhere
takes the config object as a plain argument or module-level default instead of
reading `os.environ`.

A few fields (`sim_vis`, `scene_random`, `sonic_dds_domain`, and the bare
lockstep toggle) are still read from `os.environ` at their one remaining call
site: that site runs inside the MuJoCo sim, a separate OS process launched by
`start_stack.sh` with no channel to receive config other than its inherited
environment (upstream's `SimLoopConfig`/tyro CLI is not ours to extend). Those
names are whitelisted in `scripts/lint_env.py` with the same reasoning as the
pre-existing `SONIC_ROOT`/`SIM_VIDEO_DIR` entries; `stack.py` remains the sole
place that decides their values.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LockstepConfig:
    enabled: bool = False  # LOCKSTEP=1: sim-side lockstep mode instead of wall-clock (lockstep_sim.enabled)
    ack_timeout: float = 5.0  # LOCKSTEP_ACK_TIMEOUT: seconds the barrier waits for the deploy's ack
    warmup_ticks: int = 60  # LOCKSTEP_WARMUP_TICKS: ticks held at the seeded pose after a session begins
    cmd_delay: int = 0  # LOCKSTEP_CMD_DELAY: apply the LowCmd acked N ticks ago (diagnostic inner-loop delay)
    jitter: bool = False  # LOCKSTEP_JITTER=1: reproduce the wall-clock stack's timing noise, deterministically
    jitter_seed: int = 0  # LOCKSTEP_JITTER_SEED: seed for the jitter RNG, so a session replays exactly
    no_seed: bool = False  # LOCKSTEP_NO_SEED=1: leave the server's torch RNG alone (A/B vs the per-inference seed)

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "LockstepConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)


@dataclass(frozen=True)
class SonicBackendConfig:
    no_encoder: bool = True  # NO_ENCODER: deploy launches/restarts without its encoder, token-only streaming
    hand: str = field(
        default="dex1", metadata={"choices": ("dex1", "dex3")}
    )  # HAND: end-effector model -- dex1 (2-finger) vs dex3 (3-finger); selects grip poses
    upper_slew_deg: float = 2.0  # GYM_UPPER_SLEW_DEG: max degrees/tick the upper-body reference may move
    grip_delay: int = 0  # TF_GRIP_DELAY: frames (at dataset_fps) to delay the gripper command by
    obs_dump: str | None = None  # TF_OBS_DUMP: diagnostic dump path for every policy observation + action
    replace: bool = False  # LOCOMANI_REPLACE=1: legacy sim-only re-teleport, off by default
    wrist_cams: bool = True  # WRIST_CAMS=0 publishes the ego view alone, the old behaviour
    sim_vis: str = "none"  # SIM_VIS: debug visuals the simulator draws (none/full/bbox,readout/...); none since 2026-09-10 (18dd6dd made the bare default none, this is the pipeline's side of the same ruling)
    scene_random: bool = False  # SCENE_RANDOM=1 turns texture/lighting randomisation back on (off since 2026-08-28)
    dataset_fps: float = 30.0  # TF_DATASET_FPS: frame rate the dataset/replan/grip-delay math is pinned to
    clean_deploy: bool = False  # CLEAN_DEPLOY=1: the clean deploy binary, no planner warmup between rounds
    eval_parallel: bool = False  # EVAL_PARALLEL: this run is one arm of a parallel multi-instance measurement
    sonic_dds_domain: int | None = None  # SONIC_DDS_DOMAIN: DDS domain id so parallel sim+deploy pairs don't cross
    lockstep: LockstepConfig = field(default_factory=LockstepConfig)

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "SonicBackendConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> "SonicBackendConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "SonicBackendConfig":
        return apply_overrides(self, overrides)


# The process entry is the only reader. Consumers share the resolved immutable
# object, including settings that have no individual shell variable.
_serialized = os.environ.get("LOCOMANI_BACKEND_JSON")
DEFAULT_BACKEND = SonicBackendConfig.from_dict(json.loads(_serialized)) if _serialized else SonicBackendConfig()
