"""Where the robot's gripper and a carried object fit (r1_on_single loop iteration 4): the carried object's footprint at
the yaw it rides in the hand, whether it lies inside a surface's flat top, and the gripper's least clearance to the
scene's static objects along a stretch of the motor trace. place.py's spot planner refuses on the footprint; the gripper
clearance is only logged so far (the carried hand passed 1-4 cm from a thermostat beside the bathroom sink, and one
soap carry stalled against it).

The release's geometry (iteration 5): the carried object's measured up axis (`up_axis`, `lean_deg`), the wrist turn that
levels it (`level_rotation`) and the open jaw's move onto its centre (`centre_move`)."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import SIDES, expmap
from locomani.scene_discovery.scene_file import to_surface_frame

from . import grasp_program, pick

GRIPPER_HALF_M = grasp_program.PAD_Y + 0.012  # the open fingers' half width across the jaw (pads + finger thickness)
SWEEP_EVERY = 5  # `sweep` checks every this many trace rows
_STATIC: dict = {}  # id(meta) -> (meta, keys, box centres, rotations, half extents) of `static_boxes`


def carried_footprint(xy, half_xy, face: float, yaw_rel: float) -> tuple:
    """The footprint (start.footprint's form) of an object carried at yaw `yaw_rel` on a robot facing `face`, centred
    at `xy`."""
    return np.asarray(xy, float)[:2], np.asarray(half_xy, float)[:2], float(face + yaw_rel)


def inside(fp, surface: dict, margin: float) -> bool:
    """The footprint `fp` lies inside the surface's flat top by `margin` on every side."""
    xy, half, yaw = fp
    a = yaw - float(surface.get("yaw", 0.0))
    reach = np.abs([[np.cos(a), np.sin(a)], [np.sin(a), np.cos(a)]]) @ half
    flat = np.asarray(surface.get("flat_half_extent") or surface["half_extent"], float)
    return bool(np.all(np.abs(to_surface_frame(xy, surface)) + reach <= flat - margin))


def static_boxes(meta: dict) -> tuple:
    """(keys, centres (n, 3), rotations (n, 3, 3), half extents (n, 3)) of the boxes of the static objects other than a
    surface's furniture and the apartment's shell: what a carried gripper can strike (the thermostat on the wall beside
    the bathroom sink). Built once per scene."""
    hit = _STATIC.get(id(meta))
    if hit is None or hit[0] is not meta:
        furniture = {s.get("of_object") for s in meta["surfaces"].values()}
        keys, centres, rots, halves = [], [], [], []
        for k, o in meta["objects"].items():
            if o.get("free") or k in furniture or k.startswith("Apartment"):
                continue
            rot = R.mat(np.asarray(o["init_quat"], float))
            keys.append(k)
            centres.append(np.asarray(o["init_pos"], float) + rot @ np.asarray(o["box"].get("centre") or (0, 0, 0)))
            rots.append(rot)
            halves.append(np.asarray(o["box"]["half_extent"], float))
        hit = _STATIC[id(meta)] = (meta, keys, np.array(centres), np.array(rots), np.array(halves))
    return hit[1:]


def gripper_points(wrist, rot) -> np.ndarray:
    """(n, 3) points sampling the gripper: the wrist to the fingertips (R.OFF + pick.FINGERTIP_M along the wrist x), on
    the jaw axis and GRIPPER_HALF_M either side of it along the wrist y."""
    tip = R.OFF + np.array([pick.FINGERTIP_M, 0.0, 0.0])
    local = [u * tip + np.array([0.0, w, 0.0]) for u in np.linspace(0.0, 1.0, 5) for w in (-GRIPPER_HALF_M, 0.0,
                                                                                            GRIPPER_HALF_M)]  # fmt: skip
    return np.asarray(wrist, float) + np.asarray(local) @ np.asarray(rot).T


def sweep(world, m, side: str, k0: int) -> dict | None:
    """The gripper's (`gripper_points`) least distance to `static_boxes` over the motor trace from row `k0` (every
    SWEEP_EVERY-th row and the last): dict(cm, near, t). The trace keeps wrist positions only, so the wrist attitude is
    the one measured now, turned with the pelvis yaw. Logged, never refused on. None for an empty stretch."""
    rows = m.trace[k0:]
    if not rows:
        return None
    keys, centres, rots, halves = static_boxes(world.meta)
    i = SIDES.index(side)
    _, rot_now = m.jaw_centre(side)
    on_body = R.rot_z(-m.yaw()) @ rot_now
    best = (np.inf, "", 0.0)
    for r in rows[::SWEEP_EVERY] + rows[-1:]:
        pts = gripper_points(r["wrists"][i], R.rot_z(R.yaw_of(r["robot"][3:7])) @ on_body)
        q = np.einsum("bji,pbj->pbi", rots, pts[:, None, :] - centres[None])  # each point in each box's frame
        d = np.linalg.norm(np.maximum(np.abs(q) - halves[None], 0.0), axis=2).min(axis=0)
        j = int(np.argmin(d))
        if d[j] < best[0]:
            best = (float(d[j]), keys[j], float(r["t"]))
    return dict(cm=round(100 * best[0], 1), near=best[1], t=round(best[2], 2))


def up_axis(quat, rest_quat) -> np.ndarray:
    """The object's measured up axis (world): its axis that is vertical in the rest pose `rest_quat` (the scene file's
    init_quat; a start only turns it about z), turned by its pose now `quat`."""
    return R.mat(quat) @ (R.mat(rest_quat).T @ np.array([0.0, 0.0, 1.0]))


def lean_deg(quat, rest_quat) -> float:
    """The angle (deg) between the object's `up_axis` and the world vertical."""
    return float(np.degrees(np.arccos(np.clip(up_axis(quat, rest_quat)[2], -1.0, 1.0))))


def level_rotation(u, max_deg: float) -> tuple[np.ndarray, float]:
    """(corr, deg): the least rotation (world) taking the axis `u` toward the world z, about u x z, by at most
    `max_deg`; the angle it turns. Identity when `u` is vertical (or upside down: no unique axis)."""
    u = np.asarray(u, float) / float(np.linalg.norm(u))
    axis = np.cross(u, [0.0, 0.0, 1.0])
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return np.eye(3), 0.0
    ang = min(float(np.arctan2(s, u[2])), float(np.radians(max_deg)))
    return expmap(axis / s * ang), float(np.degrees(ang))


def centre_move(d: float, min_m: float, max_m: float) -> float:
    """How far (m) the open jaw moves along the wrist y onto an object whose centre is `d` along it from the jaw
    centre: none within `min_m`, else `d` clamped to +-`max_m`."""
    return 0.0 if abs(d) <= min_m else float(np.clip(d, -max_m, max_m))
