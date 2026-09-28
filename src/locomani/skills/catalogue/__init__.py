"""The primitive catalogue: name -> Primitive, built once per process.

`build_catalogue(profiles)` returns the entries of the design's §5 table. Runnable adapters wrap existing
code (locomotion, arm-channel transitions, the bottle line's approach / grasp_single / place, the crate
line's approach / grasp_bimanual, put_down_bimanual); `approach` dispatches on the profile's hand count and,
with something held, becomes a carried walk to the target's stand. `pending.py` holds any entry declared
but not built (none at the moment); the executor rejects those with an `unimplemented` violation.
"""

from __future__ import annotations

from locomani.skills.catalogue.arms import HandBack, Ready, Release, TakeOver
from locomani.skills.catalogue.bottle import BottleApproach, GraspSingle, Place
from locomani.skills.catalogue.crate import CrateApproach, GraspBimanual
from locomani.skills.catalogue.crate_put_down import PutDownBimanual
from locomani.skills.catalogue.locomotion import CarryTo, NavigateTo, Nudge, Retreat, TurnTo, WalkTo
from locomani.skills.catalogue.pending import PENDING, Unimplemented, pending_entries
from locomani.skills.contract import Violation
from locomani.skills.session import ArmOwner

RUNNABLE = (
    "turn_to",
    "walk_to",
    "retreat",
    "ready",
    "take_over",
    "hand_back",
    "release",
    "approach",
    "grasp_single",
    "grasp_bimanual",
    "place",
    "carry_to",
    "navigate_to",
    "put_down_bimanual",
    "nudge",
)


class Approach:
    """`approach(target, profile)`: the single-arm guard walk or the bimanual walk-in, by the profile's hands."""

    name = "approach"
    requires = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})
    yields = frozenset({ArmOwner.PLANNER_SWING, ArmOwner.PLANNER_FROZEN})

    def __init__(self, profiles, single, bimanual):
        self._profiles, self._single, self._bimanual = profiles, single, bimanual

    def _pick(self, profile: str):
        prof = self._profiles.get(profile)
        if prof is None:
            return None
        return self._single if prof.hands == "single" else self._bimanual

    def preconditions(self, session, *, target: str, profile: str, drift_max: float = 0.03):
        impl = self._pick(profile)
        if impl is None:
            return [Violation("profile", profile, note=f"known: {sorted(self._profiles)}")]
        extra = {"drift_max": drift_max} if _accepts(impl, "drift_max") else {}
        return impl.preconditions(session, target=target, profile=profile, **extra)

    def run(self, session, *, target: str, profile: str, drift_max: float = 0.03):
        impl = self._pick(profile)
        extra = {"drift_max": drift_max} if _accepts(impl, "drift_max") else {}
        return impl.run(session, target=target, profile=profile, **extra)


def _accepts(prim, arg: str) -> bool:
    import inspect

    return arg in inspect.signature(prim.run).parameters


def build_catalogue(profiles: dict | None = None, **overrides) -> dict:
    """`overrides` replace entries by name (tests inject adapters built on fakes)."""
    if profiles is None:
        from locomani.skills.profiles import load_profiles

        profiles = load_profiles()
    pending = pending_entries()
    entries = {
        "turn_to": TurnTo(),
        "walk_to": WalkTo(),
        "retreat": Retreat(),
        "ready": Ready(profiles),
        "take_over": TakeOver(),
        "hand_back": HandBack(),
        "release": Release(),
        "approach": Approach(
            profiles,
            overrides.pop("approach_single", None) or BottleApproach(profiles),
            overrides.pop("approach_bimanual", None) or CrateApproach(profiles),
        ),
        "grasp_single": GraspSingle(profiles),
        "grasp_bimanual": GraspBimanual(profiles),
        "place": Place(profiles),
        "carry_to": CarryTo(),
        "navigate_to": NavigateTo(),
        "put_down_bimanual": PutDownBimanual(profiles),
        "nudge": Nudge(),
    }
    entries.update(pending)
    entries.update(overrides)
    return entries


__all__ = [
    "PENDING",
    "RUNNABLE",
    "Approach",
    "BottleApproach",
    "CarryTo",
    "CrateApproach",
    "GraspBimanual",
    "GraspSingle",
    "HandBack",
    "NavigateTo",
    "Nudge",
    "Place",
    "PutDownBimanual",
    "Ready",
    "Release",
    "Retreat",
    "TakeOver",
    "TurnTo",
    "Unimplemented",
    "WalkTo",
    "build_catalogue",
]
