"""Bottle task knobs (decision 8, refactor step 3 item 2).

Every field's default equals today's effective default: the SPAWN constant
run_learn_pilot.py already used for the official ready-visible protocol, or
the `os.environ.get(NAME, default)` fallback for the gym knobs. A run with no
config file behaves identically to before this module existed.

`reach` nests the existing `ReachConfig` (its own to_dict/load stay as they
are -- `ReachConfig.load` keeps the `RETIRED_KEYS` handling from step 2 so
historical `reach_config.json` files still load unmodified).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict
from locomani.tasks.bottle.reach_config import ReachConfig

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpawnConfig:
    edge_range: tuple[float, float] = (1.1, 1.3)  # --spawn-edge M [M]: metres from the table, one or two values
    face_object: bool = True  # --face-object: wake up pointing at the object rather than the approach line
    yaw_err_deg: float = 20.0  # --spawn-yaw-err DEG: heading jitter around the wake-up facing
    lateral: float = 0.20  # --spawn-lateral M: how far to either side of the approach line
    edge_jitter: float = 0.20  # --edge-jitter M: how far the target may slide along its edge

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "SpawnConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)


@dataclass(frozen=True)
class BottleTaskConfig:
    reach: ReachConfig = field(default_factory=ReachConfig)
    target: str = "water_bottle"  # --targets: which scene object(s) grasp_gym draws from
    prompt: str = "pick up the water bottle"  # the episode instruction / stack.py's start_stack task prompt
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    stand_retries: int = 0  # --stand-retries: reset retries after a stand-stage failure (0 for fixed-case acceptance)
    restart_deploy: bool = True  # GYM_RESTART_DEPLOY: restart the deploy every round (buys back the wander/fall class)
    arm_pin: bool = False  # GYM_ARM_PIN=1: hold the transport pin's arm pose through the round
    pose_sidecar: bool = True  # LOCOMANI_POSE_SIDECAR=0 opts out of the 18-D pose sidecar during --record
    video_include_init: bool = False  # SIM_VIDEO_INCLUDE_INIT=1: include the ready-posture prelude in diagnostic video
    init_token_file: str | None = None  # TF_INIT_TOKEN_FILE: explicit, validated initial token JSON (opt-in)
    init_token_official: bool = False  # TF_INIT_TOKEN=official: NVIDIA's shipped token instead of our measured medoid
    arm_channel: str = "planner"  # GYM_ARM_CHANNEL: planner (frozen-upper-body) or streamed arm control
    arm_vel: bool = True  # GYM_ARM_VEL=0: omit the streamed arm's velocity command
    left_guard: bool = True  # GYM_LEFT_GUARD=0: disable the right-arm guard latch derived from the left snapshot
    one_shot: bool = False  # --one-shot: discard a round whose first approach parks out of reach

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "BottleTaskConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> "BottleTaskConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "BottleTaskConfig":
        return apply_overrides(self, overrides)


DEFAULT_BOTTLE_TASK = BottleTaskConfig()
