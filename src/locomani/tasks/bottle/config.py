"""Bottle task configuration. ReachConfig owns the versioned grasp strategy.

Profiles retain the existing numerical defaults. Nested version-1 reach JSON
migrates at load time; new reach snapshots serialize the single-selector format.
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
class PlaceConfig:
    """The place phase after the lift (tasks/bottle/place_phase.py; docs/handoff/pick_place_plan.md §2).
    Only read when BottleTaskConfig.goal names a container."""

    hover_clearance: float = (
        0.03  # object bottom this far above the container rim while carrying over it (lower = more forward reach)
    )
    release_clearance: float = -0.10  # object bottom this far above the rim when the hand opens; negative = inside the container. +0.03 (17 cm drop) tipped the can every time and -0.06 (8 cm) still did half the time; -0.10 leaves a 4 cm drop onto the floor of the 15 cm basket, the jaw closing along its long axis
    band_x: tuple[float, float] = (
        0.26,
        0.44,
    )  # container centre, pelvis-frame x, where the right hand can place: the pelvis cannot get nearer than ~0.25 m to the edge, so a container 0.31 m inboard sits at 0.38-0.43; IK with the waist reaches 0.44 at hover height
    band_y: tuple[float, float] = (
        -0.16,
        -0.06,
    )  # container centre, pelvis-frame y (right of centre for the right hand)
    turn_timeout: float = 8.0  # s to bring the container into the band by turning/shuffling
    carry_timeout: float = 6.0  # s for the raise and the carry-over servo each
    descend_timeout: float = 4.0  # s of descent before the hand opens anyway (no hover-and-wait)
    place_tol: float = 0.03  # m: object-to-container-centre error that releases at once
    place_tol_late: float = 0.05  # m: the largest error the timeout release still accepts
    v_travel: float = 0.15  # m/s reference speed for raise / carry-over
    v_descend: float = 0.10  # m/s reference speed for the descent and the retreat lift
    release_s: float = 0.6  # s over which the fingers open
    retreat_up: float = 0.14  # m the palm rises straight up after the release, before backing off (clears the rim from a release 10 cm inside)
    back_off: float = 0.35  # m walked back from the table before the terminal hold
    goal_inner_half: tuple[float, float] = (
        0.12,
        0.07,
    )  # container inner half-extents, its own frame (assets/simple/basket)
    goal_gap: float = 0.21  # m along the table edge between the target and the container centre at spawn (container half-length 0.15 + can radius 0.03 + 3 cm): a target drawn inside the container's wall band is pushed by the reset
    goal_move_max: float = 0.05  # m the container may move during the round before it counts as knocked

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "PlaceConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)


@dataclass(frozen=True)
class BottleTaskConfig:
    plugin: str = "bottle"  # task plugin name (locomani.tasks.registry): decides which pipeline runs this config
    reach: ReachConfig = field(default_factory=ReachConfig)
    target: str = "water_bottle"  # --targets: which scene object(s) grasp_gym draws from
    prompt: str = "pick up the water bottle"  # the episode instruction / stack.py's start_stack task prompt
    goal: str = ""  # scene object the target is placed into after the lift ("" = the bottle task: lift, back off, hold)
    place: PlaceConfig = field(default_factory=PlaceConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    stand_retries: int = 0  # --stand-retries: reset retries after a stand-stage failure (0 for fixed-case acceptance)
    restart_deploy: bool = False  # restart the deploy every round (was the default until 2026-09-10; the no-restart path re-anchors the heading instead, ledger bottle §11)
    arm_pin: bool = False  # GYM_ARM_PIN=1: hold the transport pin's arm pose through the round
    pose_sidecar: bool = True  # LOCOMANI_POSE_SIDECAR=0 opts out of the 18-D pose sidecar during --record
    video_include_init: bool = False  # SIM_VIDEO_INCLUDE_INIT=1: include the ready-posture prelude in diagnostic video
    init_token_file: str | None = None  # TF_INIT_TOKEN_FILE: explicit, validated initial token JSON (opt-in)
    init_token_official: bool = False  # TF_INIT_TOKEN=official: NVIDIA's shipped token instead of our measured medoid
    arm_channel: str = field(
        default="planner", metadata={"choices": ("planner", "streamed")}
    )  # GYM_ARM_CHANNEL: planner (frozen-upper-body) or streamed arm control
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
