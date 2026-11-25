"""held(X), alone or with upright(X), after the pick (r1_held_single loop iteration 0).

The goal's held(X) reads X as still resting on a shelf above it (cover.py: a wolf figurine lifted 10 cm on the corner
display shelf's middle shelf failed as "resting on" the shelf 0.40 m above it). A pick that ends with X under such a
shelf pulls the jaw WITHDRAW_PULL_M straight back, level, and backs off WITHDRAW_BACK_M when X is still under it.
A pick whose outcome carries its own pull (info "withdraw", pick.NARROW: the thick novel, r1_slide_then_grasp loop
iteration 1) pulls its pull_m instead, and on to max_m from the same start ("pull_on") when X is still under the
shelves above: one start's 0.30 m pull left the book there 0.7 cm inside the edge and the back-off left it resting on
its own shelf; pulled on, it ended held. Not for an X already out: 3 cm past the edge as the bar, the second pull
also ran for 5 books out from under at 0.30 m, and dropped one of them. The pull's record keeps how far X's box centre
is outside its shelves' front edge (cover.past_edge).

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
    side grasp ("level"), pull it out from under the surfaces above it ("pull_out", for a pick with its own pull
    "pull_on" if still under, then "back_off" if still under).
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
    under = cover.overhead(world, key)
    if under:
        pull = out.info.get("withdraw") or {}
        jaw, rot = m.jaw_centre(side)
        yaw = m.yaw()
        facing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        shelves = [world.meta["objects"][key]["surface"], *under]

        def pull_to(name, dist):
            res = m.reach(side, jaw - dist * facing, rot)
            own = {}
            if pull:  # the pick's own pull: the jaw's travel back, X's centre outside the shelves' front edge
                back, past = float((jaw - m.jaw_centre(side)[0]) @ facing), cover.past_edge(world, key, shelves, yaw)
                own = dict(jaw_back_cm=round(100 * back, 1), past_edge_cm=round(100 * past, 1))
            return ctx.note(name, res, under=cover.overhead(world, key), **own)

        step = pull_to("pull_out", pull.get("pull_m", WITHDRAW_PULL_M))
        if m.fell_at is not None:
            return step
        if pull and cover.overhead(world, key):
            step = pull_to("pull_on", pull["max_m"])
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
