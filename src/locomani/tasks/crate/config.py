"""Crate (bimanual) task knobs (decision 8, refactor step 3 item 5).

Every field's default equals today's `os.environ.get(NAME, default)` fallback
(none of them is ever actually overridden in any committed script -- checked
by grepping scripts/, configs/ and src/ for `NAME=`), so a run with no config
file behaves identically to before this module existed. `rounds`/`seed`/
`label` stay plain argparse flags on bimanual_carry.py's own CLI (a per-run
case, not a task default).
"""

from __future__ import annotations

from dataclasses import dataclass

from locomani.config_base import apply_overrides, config_dump, config_from_dict, config_load, config_to_dict

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CrateTaskConfig:
    wall_feature: str = "rail"  # WALL_FEATURE: rail (rim bars) or wall (full-height long walls) grasp geometry
    wall_walk: bool = False  # WALL_WALK=1: bimanual_carry's walk-in approach instead of a ready stand
    gripper_gear: float = 250.0  # GRIPPER_GEAR: dex1 finger actuator gear (env override for A/B runs)
    couple_fingers: bool = True  # COUPLE_FINGERS=0: independent dex1 fingers instead of a coupled parallel jaw
    crate_setback: float = 0.0  # CRATE_SETBACK: moves the near wall this far in from the table edge
    scene_preset: str = "crate_bimanual"  # SCENE_PRESET: build_sim_scene.py --preset for the crate scene

    def to_dict(self) -> dict:
        return config_to_dict(self, SCHEMA_VERSION)

    @classmethod
    def from_dict(cls, data: dict) -> "CrateTaskConfig":
        return config_from_dict(cls, data, SCHEMA_VERSION)

    @classmethod
    def load(cls, path) -> "CrateTaskConfig":
        return config_load(cls, path)

    def dump(self, path) -> None:
        config_dump(self, path)

    def with_overrides(self, overrides: list[str]) -> "CrateTaskConfig":
        return apply_overrides(self, overrides)


DEFAULT_CRATE_TASK = CrateTaskConfig()
