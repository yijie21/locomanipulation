"""Seed skill `pick` (v0, TODO 27.2.3): a single-hand side grasp of an upright object on a surface, the motor check's
canister sequence made general -- walk around the furniture to a stance at the approachable edge nearest the object,
reach the jaw to a pre-grasp short of the object, into it, close, lift. The grasp yaw closes the jaw across a side that
fits it (the object's own box), turned at most FACE_TOLERANCE_DEG from facing the surface; none: the skill refuses.
The ported 65-object hand program (entry gate, lateral correction, retreat retry; plan D4) replaces the reach-close
part next."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene.navigate import walk_path

JAW_SPAN_M = 0.085  # widest side the open Dex1 jaw (9.5 cm) closes across with a margin
STAND_AHEAD_M = 0.40  # object this far ahead of the pelvis ...
STAND_SIDE_M = 0.12  # ... and this far toward the grasping hand
PREGRASP_BACK_M = 0.12  # jaw centre short of the object centre before the entry
ENTRY_PAST_M = 0.01
ENTRY_SPEED = 0.08
LIFT_M = 0.10
FACE_TOLERANCE_DEG = 40.0
MAX_DEPTH_M = 0.30  # object centre at most this far in from the edge the robot stands at
NORMALS = {"-x": (-1.0, 0.0), "+x": (1.0, 0.0), "-y": (0.0, -1.0), "+y": (0.0, 1.0)}


def facings(surface: dict, xy) -> list[tuple[float, float]]:
    """(robot yaw into the surface, that edge's distance to `xy`) per approachable edge, nearest first."""
    yaw_s = float(surface.get("yaw", 0.0))
    Rs = R.rot_z(yaw_s)[:2, :2]
    local = Rs.T @ (np.asarray(xy, float)[:2] - np.asarray(surface["xy"], float))
    half = np.asarray(surface["half_extent"], float)
    out = []
    for e in surface.get("approach_edges") or NORMALS:
        n = np.asarray(NORMALS[e])
        w = Rs @ n
        out.append((float(np.arctan2(-w[1], -w[0])), float(half[0 if "x" in e else 1] - n @ local)))
    return sorted(out, key=lambda f: f[1])


def grasp_yaw(quat, half, face: float) -> float | None:
    """The approach yaw (wrist x) closing the jaw across a side of the object's box that fits it, nearest `face`."""
    yaw_o = R.yaw_of(quat)
    fit_y, fit_x = 2 * half[1] <= JAW_SPAN_M, 2 * half[0] <= JAW_SPAN_M  # closing across local y / local x
    if fit_x and fit_y:
        return face
    cands = []
    if fit_y:
        cands += [yaw_o, yaw_o + np.pi]
    if fit_x:
        cands += [yaw_o + np.pi / 2, yaw_o - np.pi / 2]
    if not cands:
        return None
    best = min(cands, key=lambda a: abs(R.wrap(a - face)))
    return float(best) if abs(R.wrap(best - face)) <= np.radians(FACE_TOLERANCE_DEG) else None


def pick(ctx, key: str, side: str = "right") -> Outcome:
    world, m = ctx.world, ctx.motor
    obj = world.meta["objects"][key]
    surf = world.meta["surfaces"][obj["surface"]]
    pos, quat = world.object_pose(key)
    # the nearest approachable edge from which a side of the object fits the jaw (the utility cart's crowded top
    # leaves the canister lying along one edge: only the edge at its end faces its narrow side)
    face = depth = psi = None
    for f, dep in facings(surf, pos):
        y = grasp_yaw(quat, obj["box"]["half_extent"], f) if dep <= MAX_DEPTH_M else None
        if y is not None:
            face, depth, psi = f, dep, y
            break
    if psi is None:
        return ctx.note("pick", Outcome(False, "no side fits the jaw within reach of an approachable edge", m.t, m.t))
    lateral = -STAND_SIDE_M if side == "right" else STAND_SIDE_M
    stance = pos[:2] - (R.rot_z(face) @ np.array([STAND_AHEAD_M, lateral, 0.0]))[:2]
    out = ctx.note("walk", walk_path(m, world.occ, world.nav, stance, face), edge_depth_m=round(depth, 3))
    if not out.ok:
        return out
    pos, quat = world.object_pose(key)  # the object as it is now
    Rg = R.rot_z(psi)
    out = ctx.note("pregrasp", m.reach(side, pos - Rg @ np.array([PREGRASP_BACK_M, 0.0, 0.0]), Rg))
    if m.fell_at is not None:
        return out
    out = ctx.note("entry", m.reach(side, pos + Rg @ np.array([ENTRY_PAST_M, 0.0, 0.0]), Rg, speed=ENTRY_SPEED))
    if m.fell_at is not None:
        return out
    out = ctx.note("grip", m.grip(side))
    if not out.ok:
        return out
    out = ctx.note("lift", m.reach(side, m.jaw_centre(side)[0] + np.array([0.0, 0.0, LIFT_M]), Rg, speed=ENTRY_SPEED))
    held = world.free[key] in m.between_fingers(side)
    return Outcome(held, "" if held else "not between the fingers after the lift", out.t0, out.t1)
