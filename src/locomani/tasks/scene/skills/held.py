"""held(X), alone or with upright(X), after the pick (r1_held_single loop iteration 0).

The goal's held(X) reads X as still resting on a shelf above it (cover.py: a wolf figurine lifted 10 cm on the corner
display shelf's middle shelf failed as "resting on" the shelf 0.40 m above it). A pick that ends with X under such a
shelf pulls the jaw WITHDRAW_PULL_M straight back, level, and backs off WITHDRAW_BACK_M when X is still under it.

upright(X) after a side grasp: the mug hung 18-20 deg tilted at the end (limit 15), 9-11 deg right after the lift. The
wrist turns about X's centre by clearance.level_rotation of its measured up axis, at most LEVEL_MAX_DEG.

A pick that ends with X under no other surface, for a goal without upright(X), is returned as it is (no step added)."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm.motor import Outcome

from . import clearance, cover, pick

WITHDRAW_PULL_M = 0.20  # X under a surface other than its own after the pick: the jaw pulls this far back, level ...
WITHDRAW_BACK_M = 0.30  # ... then, X still under it, the robot backs off this far
LEVEL_MAX_DEG = 25.0  # upright(X) in the goal, side grasp: the wrist levels X's measured up axis by at most this


def pick_held(ctx, key: str, skill, upright: bool = False) -> Outcome:
    """`skill(ctx, key)` (the pick); after a successful one, with X gripped: level X when `upright` and the grasp is a
    side grasp ("level"), pull it out from under the surfaces above it ("pull_out", then "back_off" if still under).
    ok when X is still gripped and rests on no surface."""
    out = skill(ctx, key)
    world, m = ctx.world, ctx.motor
    if not out.ok or key not in world.free:
        return out
    body = world.free[key]
    sides = sorted(m.gripping(body))
    if not sides or not (upright or cover.overhead(world, key)):
        return out
    side = sides[0]
    if upright and not pick.is_top(world, key):
        pos, quat = world.object_pose(key)
        rest = world.meta["objects"][key]["init_quat"]
        corr, deg = clearance.level_rotation(clearance.up_axis(quat, rest), LEVEL_MAX_DEG)
        jaw, rot = m.jaw_centre(side)
        step = ctx.note("level", m.reach(side, pos - corr @ (pos - jaw), corr @ rot), deg=round(deg, 1),
                        lean_before=round(clearance.lean_deg(quat, rest), 1))  # fmt: skip
        if m.fell_at is not None:
            return step
    if cover.overhead(world, key):
        jaw, rot = m.jaw_centre(side)
        back = jaw - WITHDRAW_PULL_M * np.array([np.cos(m.yaw()), np.sin(m.yaw()), 0.0])
        step = ctx.note("pull_out", m.reach(side, back, rot), under=cover.overhead(world, key))
        if m.fell_at is not None:
            return step
        if cover.overhead(world, key):
            step = ctx.note("back_off", m.sidestep(forward=-WITHDRAW_BACK_M), under=cover.overhead(world, key))
            if m.fell_at is not None:
                return step
    gripping, under = sorted(m.gripping(body)), cover.resting_on(world, key)
    ok = side in gripping and not under
    why = "" if ok else f"after the pick: resting on {under}, gripping {gripping}"
    return Outcome(ok, why, out.t0, m.t)
