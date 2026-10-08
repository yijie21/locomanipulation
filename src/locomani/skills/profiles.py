"""Grasp profiles: the approach/ready/grasp numbers a primitive needs, by name (design doc §6).

A profile is not a new config format -- it is a read-only, three-part view over a real
task config (`CrateTaskConfig` or `BottleTaskConfig`), the same objects `configs/task/*.yaml`
and `configs/experiments/*.yaml` already resolve to. `skills/catalogue/*` primitives take a
`GraspProfile` instead of poking `session.scene`/`session.stack` fields directly, so a new
scene only needs a new entry in `PROFILE_SOURCES`, not new primitive code.

Two source shapes exist today (see `locomani/profiles.py` and the experiment files under
`configs/experiments/`): a profile-style YAML with a top-level `defaults:` list, loaded
through the existing `locomani.profiles.load`; and a full experiment YAML with a top-level
`task:` block and no `defaults` key, loaded by picking the dataclass off `task.plugin`.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from locomani import paths
from locomani import profiles as named_profiles
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.crate.config import CrateTaskConfig

# Field split for a crate task config: which of CrateVisualConfig's (and CrateCarryConfig's)
# fields belong to the walk-in, the stationary arm-readying, and the close/servo phase.
_CRATE_APPROACH_FIELDS = (
    "approach_stop",
    "stand_off",
    "walk_speed",
    "walk_speed_min",
    "walk_brake_margin",
    "walk_steer",
    "guard_extra",
    "guard_margin",
    "guard_stride0",
    "guard_stride_mode",
    "guard_brake",
    "guard_phase_band",
    "guard_rewalk_depth",
    "guard_rewalk_back",
    "stand_lateral_tolerance",
    "stand_depth_tolerance",
    "face_after_brake",
    "start_edge_min",
    "start_edge_max",
)
_CRATE_READY_FIELDS = (
    "ready_pose",
    "ready_tool",
    "ready_pitch_deg",
    "elbow_min_deg",
    "ready_after_brake",
    "ready_blend_seconds",
    "ready_hold_seconds",
)
_CRATE_GRASP_FIELDS = (
    "reach_path",
    "close_trigger",
    "close_target",
    "hold_frame",
    "align_tolerance",
    "close_seconds",
    "contact_across",
    "contact_along",
    "contact_below_rim",
    "capture_across",
    "box_grasp_below_rim",
    "pregrasp_above",
)

# Same split for the bottle simple expert's fields (SimpleGraspConfig, under reach.simple).
_BOTTLE_APPROACH_FIELDS = (
    "approach_stop",
    "walk_speed",
    "guard_extra_m",
    "guard_margin_m",
    "guard_stride0_m",
    "guard_band_below_m",
    "guard_band_above_m",
)
_BOTTLE_READY_FIELDS = (
    "ready_tool_forward_m",
    "ready_tool_lateral_m",
    "ready_tool_up_m",
    "ready_pitch_deg",
    "ready_elbow_min_deg",
    "ready_seconds",
    "ready_hold_s",
)
_BOTTLE_GRASP_FIELDS = (
    "dash_speed_m_s",
    "entry_capture_depth_m",
    "entry_lateral_tolerance_m",
    "entry_height_tolerance_m",
    "close_seconds",
    "lift_height_m",
)


class GraspProfile:
    """A read-only, three-part view over one task config: `approach`, `ready`, `grasp`.

    Not a dataclass: the payload is three `MappingProxyType` dicts, and construction is
    always through `_build`, never field-by-field, so there is no `__init__` worth
    generating. `name`/`source` identify where it came from, for `session.trace` and
    error messages.
    """

    __slots__ = ("name", "source", "hands", "config", "approach", "ready", "grasp", "stance")

    def __init__(
        self,
        name: str,
        source: str,
        hands: str,
        config: Any,
        approach: Mapping[str, Any],
        ready: Mapping[str, Any],
        grasp: Mapping[str, Any],
    ):
        self.name = name
        self.source = source
        self.hands = hands  # "single" | "bimanual"
        self.config = config  # the resolved task dataclass; primitives that wrap an expert hand it over whole
        self.approach = MappingProxyType(dict(approach))
        self.ready = MappingProxyType(dict(ready))
        self.grasp = MappingProxyType(dict(grasp))
        self.stance = MappingProxyType(dict(STANCE_BANDS.get(name) or {}))  # grasp admission band over the Stance

    def __repr__(self) -> str:
        return f"GraspProfile({self.name!r}, source={self.source!r})"


def _split(cfg: Any, approach_fields: tuple, ready_fields: tuple, grasp_fields: tuple) -> tuple[dict, dict, dict]:
    approach = {f: getattr(cfg, f) for f in approach_fields}
    ready = {f: getattr(cfg, f) for f in ready_fields}
    grasp = {f: getattr(cfg, f) for f in grasp_fields}
    return approach, ready, grasp


def _crate_profile(name: str, source: str, cfg: CrateTaskConfig) -> GraspProfile:
    approach, ready, grasp = _split(cfg.visual, _CRATE_APPROACH_FIELDS, _CRATE_READY_FIELDS, _CRATE_GRASP_FIELDS)
    grasp["carry"] = dataclasses.asdict(cfg.carry)
    return GraspProfile(name, source, "bimanual", cfg, approach, ready, grasp)


def _bottle_profile(name: str, source: str, cfg: BottleTaskConfig) -> GraspProfile:
    simple = cfg.reach.simple
    approach, ready, grasp = _split(simple, _BOTTLE_APPROACH_FIELDS, _BOTTLE_READY_FIELDS, _BOTTLE_GRASP_FIELDS)
    return GraspProfile(name, source, "single", cfg, approach, ready, grasp)


def _load_task_profile(name: str, profile_name: str) -> GraspProfile:
    cfg = named_profiles.load("task", profile_name)
    source = f"task/{profile_name}.yaml"
    if isinstance(cfg, CrateTaskConfig):
        return _crate_profile(name, source, cfg)
    if isinstance(cfg, BottleTaskConfig):
        return _bottle_profile(name, source, cfg)
    raise ValueError(f"task profile {profile_name!r} resolved to {type(cfg).__name__}, not a crate or bottle task")


def _load_experiment_profile(name: str, relpath: str) -> GraspProfile:
    path = Path(paths.REPO) / relpath
    data = yaml.safe_load(path.read_text())
    task = data["task"]
    plugin = task.get("plugin")
    if plugin == "crate":
        return _crate_profile(name, relpath, CrateTaskConfig.from_dict(task))
    if plugin in ("bottle", "pick_place"):  # pick_place is the bottle line's task with a goal (pipeline/lines.py)
        return _bottle_profile(name, relpath, BottleTaskConfig.from_dict(task))
    raise ValueError(f"{relpath}: unsupported task.plugin {plugin!r} for a grasp profile")


# Grasp admission bands over the approach's `Stance` (skills.session.stance_violations), by profile: the
# stance depth / lateral numbers a `grasp_*` call is admitted at. An absent key or None is open on that side.
# Set from `scripts/replay_preconditions.py` over recorded rounds (2026-09-28, docs/ledger/*): a band only
# earns a place here when it rejects failures without rejecting successes of the same line.
STANCE_BANDS: dict[str, dict] = {
    "bottle_simple": {},
    "crate_low_table": {},
    "warehouse_box": {},
}

# name -> how to build it. "task_profile" locators are names under configs/task/ (loaded
# through locomani.profiles.load); "experiment" locators are paths under the repo root to a
# full experiment YAML (loaded for its task: block alone).
PROFILE_SOURCES: dict[str, tuple[str, str]] = {
    "crate_low_table": ("task_profile", "crate_visual_low_reach"),
    # box_centre_base_20260927 with stand_forward_min 0.50: the stand no longer pulls the held box back over the
    # platform edge, so put_down_bimanual rests it in place (8/8, crate ledger 2026-09-28). The collection line's
    # own file keeps 0.35; this is the skills-side choice (TODO 17.8.1.1 route B).
    "warehouse_box": ("experiment", "configs/experiments/skills_box_fwd50_20260928.yaml"),
    "bottle_simple": ("experiment", "configs/experiments/bottle_simple_200.yaml"),
    # 7.7.1 calibration variants for the soup can (docs/skills.md §5): one knob each over bottle_simple
    "can_entry15": ("experiment", "configs/experiments/skills_can_entry15_20260928.yaml"),
    "can_descend05": ("experiment", "configs/experiments/skills_can_descend05_20260928.yaml"),
    "can_overshoot0": ("experiment", "configs/experiments/skills_can_overshoot0_20260928.yaml"),
    # the pick_place line's own task block (coupled strategy, ledger §4 10/10) as a profile for the same can
    "pick_place_coupled": ("experiment", "configs/experiments/skills_pick_place_20260928.yaml"),
    # 7.8.1.1 route B: the warehouse box held out over the platform through the stand (stand_forward_min 0.50)
    "warehouse_box_fwd50": ("experiment", "configs/experiments/skills_box_fwd50_20260928.yaml"),
}


@lru_cache(maxsize=None)
def load_profile(name: str) -> GraspProfile:
    """Resolve one named profile. Raises `KeyError` for a name not in `PROFILE_SOURCES`."""
    kind, locator = PROFILE_SOURCES[name]
    if kind == "task_profile":
        return _load_task_profile(name, locator)
    if kind == "experiment":
        return _load_experiment_profile(name, locator)
    raise ValueError(f"unknown profile source kind {kind!r} for {name!r}")  # pragma: no cover -- table is closed


def load_profiles() -> dict[str, GraspProfile]:
    """All profiles in `PROFILE_SOURCES`, by name."""
    return {name: load_profile(name) for name in PROFILE_SOURCES}
