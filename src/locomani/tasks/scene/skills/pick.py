"""Seed skill `pick` (TODO 27.2.2-27.2.3): a single-hand side grasp of an object on a surface. Walk around the
furniture to a stance at an approachable edge from which a side of the object fits the jaw (its own box, `grasp_yaw`),
reach the jaw to the pre-grasp with the motor layer (whole-body IK + servo), then the 65-object line's closed-loop hand
program (grasp_program.py: settle, gated entry with lateral correction and retry, close, lift, hold; plan D4, user
2026-10-16). v0 (motor reaches + grip, no closed loop) stays as `pick_simple` for comparison."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome
from locomani.tasks.scene.navigate import walk_path

from . import grasp_program

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


def pick_simple(ctx, key: str, side: str = "right") -> Outcome:
    world, m = ctx.world, ctx.motor
    plan = approach(ctx, key, side)
    if not plan.ok:
        return plan
    pos, quat = world.object_pose(key)  # the object as it is now
    Rg = R.rot_z(plan.info["psi"])
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


def pick(ctx, key: str, side: str = "right") -> Outcome:
    """Walk, reach the pre-grasp, then the ported closed-loop hand program (grasp_program.run)."""
    world, m = ctx.world, ctx.motor
    plan = approach(ctx, key, side)
    if not plan.ok:
        return plan
    pos, quat = world.object_pose(key)
    psi = plan.info["psi"]
    Rg = R.rot_z(psi)
    out = ctx.note("pregrasp", m.reach(side, pos - Rg @ np.array([grasp_program.PREGRASP_BACK, 0.0, 0.0]), Rg))
    if m.fell_at is not None:
        return out
    pos, _ = world.object_pose(key)
    return grasp_program.run(
        ctx, key, side, Rg, pos.copy(), span(quat, world.meta["objects"][key]["box"]["half_extent"], psi)
    )


def span(quat, half, psi: float) -> float:
    """The object's width across the jaw (closing along the wrist y) for approach yaw `psi`."""
    a = R.wrap(psi - R.yaw_of(quat))
    return float(2 * (abs(np.sin(a)) * half[0] + abs(np.cos(a)) * half[1]))


def approach(ctx, key: str, side: str) -> Outcome:
    """Choose the edge and grasp yaw, walk to the stance. info: psi (approach yaw), edge_depth_m."""
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
    out.info.update(psi=psi, edge_depth_m=depth)
    return out
