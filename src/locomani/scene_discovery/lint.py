"""Scene checks that need no simulator, run before any loop spends on a scene (plan docs/handoff/scene-rsi-plan-20261016.md
§12, user 2026-10-19). Each one is a defect an inner loop found the expensive way in round 1 and 2:

- `upright_mismatch`: free objects whose `upright_axis` is more than adt.UPRIGHT_OFF_DEG off the up they rest with
  (28 of 64 on the ADT apartment: upright(X) failed where X never moved, and r2_on_single parked a task on it).
- `shadowed`: surfaces another one less than surfaces.SHADOW_GAP higher covers (the kitchen's two cabinet tops:
  on(X, medium counter) held at the start, the seed scored 7/8 on a task it never did).
- `flat_objects`: free objects lying with half their height at most a Dex1 finger pad's half height: a side grasp
  has no pad margin left (the wallet 2.76 cm, the car key 1.5 cm; r2_in_single parked both after 80 USD).
- `goal_at_start`: a task whose goal already holds at the scene pose (`scene_state`) or at drawn starts.

The rest test (objects left alone in the simulator) and the start draws are in scripts/scene_lint.py.
"""

from __future__ import annotations

import copy

import numpy as np

from locomani.scene_discovery import adt
from locomani.scene_discovery.predicates import ROBOT, UPRIGHT_MAX_DEG, compile_goal, parse

DEX1_PAD_HALF_M = 0.014  # a Dex1 finger pad's half height (the hardware; the catalog's grasp_program.PAD_Z)
PELVIS_Z_M = 0.75  # the standing pelvis height a start's robot is put at (only near / in_room read the pelvis xy)


def upright_mismatch(meta: dict) -> dict[str, tuple[list, list]]:
    """{key: (axis now, the resting one)} -- what adt.resting_upright would change, on a copy."""
    return adt.resting_upright(copy.deepcopy(meta["objects"]))


def tilted_at_rest(meta: dict, max_deg: float = UPRIGHT_MAX_DEG) -> dict[str, float]:
    """{key: tilt deg} for free objects whose upright axis is more than upright()'s limit off the vertical at the
    scene pose, with no body axis nearer (upright_mismatch has those): upright(X) fails before anything moves."""
    out = {}
    for key, o in sorted(meta["objects"].items()):
        sem = o.get("semantic") or {}
        if not o.get("free") or not sem:
            continue
        ax = np.asarray(sem.get("upright_axis") or (0.0, 0.0, 1.0), dtype=float)
        up = adt.quat_to_matrix(o["init_quat"]) @ (ax / np.linalg.norm(ax))
        tilt = float(np.degrees(np.arccos(np.clip(up[2], -1.0, 1.0))))
        if tilt >= max_deg:
            out[key] = round(tilt, 1)
    return out


def shadowed(meta: dict) -> dict[str, str]:
    from locomani.scene_discovery import surfaces  # cv2: kept out of the checker's imports

    return surfaces.shadowed(meta["surfaces"])


def resting_half_height(o: dict) -> float:
    """The object's box half extent along its body axis nearest the world vertical at the scene pose (as the pick
    plans it: the wallet tilted 2 deg on its shelf is 1.38 cm, its world-z extent 1.49)."""
    half = np.asarray((o.get("box") or {}).get("half_extent") or (0.0, 0.0, 0.0), dtype=float)
    r = adt.quat_to_matrix(o["init_quat"])
    return float(half[int(np.argmax(np.abs(r[2])))])


def resting_footprint(o: dict) -> np.ndarray:
    """The object's footprint as it rests, sorted (m): its box's full extents along the two body axes other than the
    one nearest the world vertical (`resting_half_height`'s), at the yaw that aligns them with a box."""
    half = np.asarray((o.get("box") or {}).get("half_extent") or (0.0, 0.0, 0.0), dtype=float)
    up = int(np.argmax(np.abs(adt.quat_to_matrix(o["init_quat"])[2])))
    return np.sort(2 * np.delete(half, up))


def flat_objects(meta: dict, pad_half: float = DEX1_PAD_HALF_M) -> dict[str, dict]:
    """{key: {half_height, pad_half, margin, support_top}} for the free objects whose resting half height is at most
    `pad_half`: closing a side grasp at their mid height puts the pads' lower edge at or under the surface."""
    out = {}
    for key, o in sorted(meta["objects"].items()):
        if not o.get("free") or not (o.get("box") or {}).get("half_extent"):
            continue
        h = resting_half_height(o)
        if h <= pad_half:
            s = meta["surfaces"].get(o.get("surface") or "") or {}
            out[key] = dict(half_height=round(h, 4), pad_half=pad_half, margin=round(h - pad_half, 4),
                            support=o.get("surface"), support_top=s.get("top"))  # fmt: skip
    return out


def scene_state(meta: dict, start: dict | None = None) -> dict:
    """The predicates' state at the scene pose, or at a start (start.Start.to_dict(): `placements` {key: [pos,
    quat]}, `robot_xy`): nothing touched, the robot standing at its spawn (the scene file's has none: the origin)."""
    objects = {k: dict(pos=np.asarray(o["init_pos"], float), quat=np.asarray(o["init_quat"], float), touching=[],
                       gripped=[]) for k, o in meta["objects"].items()}  # fmt: skip
    for k, (pos, quat) in ((start or {}).get("placements") or {}).items():
        objects[k] = dict(pos=np.asarray(pos, float), quat=np.asarray(quat, float), touching=[], gripped=[])
    xy = (start or {}).get("robot_xy") or (0.0, 0.0)
    return dict(objects=objects, pelvis=dict(pos=np.array([xy[0], xy[1], PELVIS_Z_M]), quat=np.array([1.0, 0, 0, 0])),
                fallen=False)  # fmt: skip


def goal_holds(goal: list[str], meta: dict, start: dict | None = None) -> bool:
    """The goal already true at `start`, or at the scene pose (start None; the robot has none there, so a goal on
    the robot alone does not hold and the robot's predicates are left out of the rest)."""
    if start is None:
        goal = [g for g in goal if ROBOT not in parse(g).args]
        if not goal:
            return False
    return bool(compile_goal(list(goal), meta)(scene_state(meta, start))[0])
