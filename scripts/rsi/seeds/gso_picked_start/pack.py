"""Skill `pack_into` (the pack line, user 2026-11-05): put a free object X into the container C, then lift C with both
hands -- the plan for a goal of one `in(X, C)` with `held(C)` (and `lifted(C, ...)` / `upright(C)`).

The chain: the put-in by X's kind (`bimanual.place_into` when the scene file says X takes two hands, else
`into.place_into`, the one-hand put-in), then `bimanual.pick` of C, the hug. A put-in that fails ends the plan (C is
not lifted with nothing in it); after the hug the goal is read once more, since a lift may tip X out. Both steps log
their own stages; this module adds `pack_put_in` and `pack_lift` summaries to the round record."""

from __future__ import annotations

from locomani.backends.scalebfm.motor import Outcome

from . import bimanual, into


def put_in(ctx, key: str, container: str) -> Outcome:
    """The put-in by X's features: two hands (bimanual.place_into) or one (into.place_into)."""
    if bimanual.two_handed(ctx.world, key):
        return bimanual.place_into(ctx, key, container)
    return into.place_into(ctx, key, container)


def pack_into(ctx, key: str, container: str) -> Outcome:
    world, m = ctx.world, ctx.motor
    two = bimanual.two_handed(world, key)
    out = ctx.note("pack_put_in", put_in(ctx, key, container), object=key, hands="both" if two else "one")
    if not out.ok:
        return Outcome(False, f"put-in of {key}: {out.why}", out.t0, out.t1, dict(out.info))
    out = ctx.note("pack_lift", bimanual.pick(ctx, container), container=container)
    if not out.ok:
        return Outcome(False, f"lift of {container} with {key} inside: {out.why}", out.t0, out.t1, dict(out.info))
    ok, why = world.check(list(ctx.task.goal))(world.state([key, container]))
    return Outcome(ok, why, out.t0, out.t1, dict(out.info))
