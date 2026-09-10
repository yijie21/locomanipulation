"""bottle_visual task knobs (decision 8): the visual bottle expert, its reset geometry and run options.

`BottleVisualTaskConfig()` is the ruled yplus200 expert (docs/ledger/scalebfm.md, 2026-09-09):
`round.py`'s old argparse defaults were the exploratory values, the collection drivers overrode
them to these. `GraspConfig` is the planner's geometry (planner.py); its own defaults stay the
planner-test values, the task config's `grasp` field carries the ruled overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GraspConfig:
    stand_distance: float = 0.375
    stand_backstep: float = 0.06
    stand_left: float = 0.15
    reach_start: float = 0.5
    pre_back: float = 0.10
    carry_back: float = 0.07
    pre_tolerance: float = 0.035
    close_along: float = 0.02
    close_across: float = 0.018
    close_speed: float = 0.10
    pitch_feedforward_deg: float = -13.0
    lift_height: float = 0.10
    carry_clearance: float | None = None
    edge_clearance: float | None = None
    high_table_front: float | None = None
    facing: str = field(default="bottle", metadata={"choices": ("bottle", "edge")})


def ruled_grasp() -> GraspConfig:
    """The yplus200 collection expert (collect.py's expert_config before the 2026-09-10 reorg)."""
    return GraspConfig(
        stand_distance=0.375,
        pre_tolerance=0.035,
        lift_height=0.12,
        carry_clearance=0.12,
        edge_clearance=0.22,
        high_table_front=0.20,
        pitch_feedforward_deg=0.0,
        facing="edge",
    )


@dataclass(frozen=True)
class BottleVisualTaskConfig:
    grasp: GraspConfig = field(default_factory=ruled_grasp)  # planner geometry; see ruled_grasp()
    scene_profile: str = field(
        default="yplus_center_v1", metadata={"choices": ("yplus_center_v1", "legacy_edges_v1")}
    )  # reset geometry version (sampling.py); legacy_edges_v1 = the 2026-09-08 three-edge scenes
    strata: tuple[str, ...] = (
        "high_near",
        "low_near",
        "low_deep",
    )  # table-height / inboard strata cycled per case (scene.STRATA)
    recoveries: tuple[str, ...] = ("none",)  # recovery interventions cycled per case (recovery.KINDS)
    max_s: float = 24.0  # episode cap in sim seconds
    native_codec: bool = (
        True  # 29-D native action codec (backends/scalebfm/action.py); False = the pre-codec pose stream
    )
    action_hz: int = 25  # ActionExecutor rate (25 -> the 50 Hz sim); 0 = no executor, per-tick targets (requires native_codec when set)
    integral_gain: float = 1.0  # GraspMotion integral gain on the measured pad error (0 = pure feed-forward)
    reference_style: str = field(
        default="direct", metadata={"choices": ("legacy", "direct")}
    )  # direct = no hand bias; legacy = the v3 +x-measured BIAS (motion.py)
    policy_device: str = field(
        default="cpu", metadata={"choices": ("cpu", "cuda")}
    )  # torch device for the ScaleBFM policy; collection ran cuda, replays/checks cpu

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "BottleVisualTaskConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> "BottleVisualTaskConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "BottleVisualTaskConfig":
        return apply_overrides(self, overrides)
