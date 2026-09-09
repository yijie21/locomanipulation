"""Per-target stand planning and per-round sampling helpers, split out of
grasp_gym.py (refactor step 4b, pure extraction; decision 9's file-size cap).

`obj_top` stays in grasp_gym.py (it is also imported directly by
diag_crouched_handover.py, and `grasp_gym.py` monkeypatches nothing here that
would make moving it risky) -- `plan_for_target` reaches it with a deferred
import to avoid grasp_gym.py <-> plan.py becoming a load-order-sensitive
circular import (grasp_gym.py imports this module's names at its own top
level; this module must not need anything of grasp_gym.py's at ITS top level
in return).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from locomani.backends.sonic.primitives import STANDING_PELVIS_Z
from locomani.tasks.bottle import grasping, reach_map
from locomani.tasks.bottle.reach_config import DEFAULT_REACH

FACING_MAX = np.deg2rad(30.0)


class TargetPlan(NamedTuple):
    """Everything about one candidate target that does not change per round."""

    key: str
    phrase: str  # how the instruction names it
    nom: np.ndarray  # nominal xy on the surface
    grip_z: float  # world z of the grip centre
    rel_cands: np.ndarray  # pelvis-frame (x, y) stands that put it in the core
    squat_hint: float | None
    ax: int  # 0 = x edge, 1 = y edge -- the edge it hugs
    sgn: float  # which side of that axis


def resolve_pool(meta: dict, spec: str) -> list[str]:
    """--targets -> a list of object keys, checked against the scene."""
    if spec == "pool":
        want = list(meta.get("target_pool") or [meta["target_object"]])
    elif spec == "default":
        want = [meta["target_object"]]
    else:
        want = [t.strip() for t in spec.split(",") if t.strip()]
    missing = [t for t in want if t not in meta["objects"]]
    if missing:
        raise SystemExit(
            f"[gym] targets not in the scene: {missing}. "
            f"Rebuild it with build_sim_scene.py, or pick from "
            f"{sorted(meta['objects'])}"
        )
    return want


def build_rel_candidates(rm: reach_map.ReachMap, z_pf: float) -> np.ndarray:
    """Pelvis-frame (x, y) offsets of a graspable can, facing within 30 deg.

    No |y| >= 0.12 filter on purpose: dead-ahead placements are exactly what
    the side-grasp azimuth is being tested on."""
    cands = []
    # x capped at 0.32: stage-2 navigation delivers stands at depth
    # 0.22-0.30, and beyond ~0.32 the OUTBOARD staging point (pregrasp =
    # grip - 0.13*entry) leaves the waist-clamped arm's reach even where
    # the grip itself is in the map (batch: 12x 'arm not tracking',
    # IK res 0.06-0.09).
    for x in np.arange(0.16, 0.30, 0.025):
        for y in np.arange(-0.40, 0.401, 0.025):
            if abs(np.arctan2(y, x)) > FACING_MAX:
                continue
            # RIGHT arm only for now: GRASP_CENTRE_OFFSET / hand_points
            # are right-hand palm-frame constants that were never mirrored
            # — the all-left batch produced targets on the wrong side of the
            # body (9x arm-not-tracking). Left support is its own work item.
            if not rm.graspable([x, y, z_pf], "right"):
                continue
            cands.append((x, y))
    return np.asarray(cands)


def plan_for_target(meta: dict, key: str, rm: reach_map.ReachMap, config=DEFAULT_REACH) -> TargetPlan:
    """Solve the stand plan for one target.

    Split out of main() when the target stopped being fixed. The edge the
    object hugs is derived here too: the sampler slides the target along
    ITS OWN nearest edge, so a pool spread around the table is also a pool
    of approach directions.
    """
    from locomani.tasks.bottle.grasp_gym import obj_top

    surf = meta["surfaces"][meta["objects"][key].get("surface", "A")]
    cxy = np.asarray(surf["xy"], dtype=float)
    half = np.asarray(surf.get("half_extent", [0.6, 0.6])[:2], dtype=float)
    nom = np.asarray(meta["objects"][key]["init_pos"][:2], dtype=float)
    grip_z = obj_top(meta, key) - grasping.GRIP_BELOW_RIM
    if config.ready_grasp_recovery:
        obj = meta["objects"][key]
        grip_z = obj["init_pos"][2] + obj["box"].get("center", [0.0, 0.0, 0.0])[2] - config.grip_below_center

    e0 = nom - cxy
    ax = 0 if abs(abs(e0[0]) - half[0]) < abs(abs(e0[1]) - half[1]) else 1
    sgn = float(np.sign(e0[ax]))

    plan_cs = rm.plan_core_stand(float(grip_z - STANDING_PELVIS_Z))
    if plan_cs is not None:
        rel_cands, want_drop, _ = plan_cs
        # plan_core_stand's drop is a DELIVERED pelvis drop; arm.squat takes
        # a command whose delivery runs ~1.5x (measured: cmd 5/9/13 cm ->
        # 4.4/12/19.5 cm) — invert that curve for the hint.
        squat_hint = float(np.interp(want_drop, [0.0, 0.044, 0.12, 0.195], [0.0, 0.05, 0.09, 0.13]))
    else:
        rel_cands = build_rel_candidates(rm, float(grip_z - STANDING_PELVIS_Z))
        squat_hint = None
    phrase = meta.get("object_phrase", {}).get(key, f"the {key}")
    return TargetPlan(key, phrase, nom, grip_z, rel_cands, squat_hint, ax, sgn)


def sample_object_yaw(fixed=None):
    """Optional exact-case yaw; historical callers retain random orientation."""
    if fixed is not None:
        if type(fixed) not in (int, float) or not np.isfinite(fixed):
            raise ValueError("object_yaw must be finite")
        return float(fixed)
    return float(np.random.uniform(-np.pi, np.pi))
