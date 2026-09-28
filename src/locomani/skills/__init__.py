"""Primitive contracts: the shared session, contract and scene types `skills/catalogue/*`
primitives are written against (design doc §9). This package intentionally does not import
`skills.catalogue` -- the catalogue depends on this package, never the reverse, so importing
`locomani.skills` never pulls in a specific robot's primitive implementations.
"""

from locomani.skills.contract import Outcome, Primitive, Violation, tag_of
from locomani.skills.executor import Executor
from locomani.skills.profiles import PROFILE_SOURCES, GraspProfile, load_profile, load_profiles
from locomani.skills.scene_model import (
    DEX1_OPENING_M,
    Affordance,
    ObjectInfo,
    SceneModel,
    SurfaceInfo,
    TaskInstance,
)
from locomani.skills.session import (
    ArmOwner,
    HeldObject,
    Session,
    Stance,
    invariant_violations,
    needs_heading,
    snapshot,
    stance_violations,
)

__all__ = [
    "ArmOwner",
    "HeldObject",
    "Session",
    "Stance",
    "invariant_violations",
    "needs_heading",
    "snapshot",
    "stance_violations",
    "Executor",
    "Outcome",
    "Primitive",
    "Violation",
    "tag_of",
    "GraspProfile",
    "PROFILE_SOURCES",
    "load_profile",
    "load_profiles",
    "DEX1_OPENING_M",
    "Affordance",
    "ObjectInfo",
    "SceneModel",
    "SurfaceInfo",
    "TaskInstance",
]
