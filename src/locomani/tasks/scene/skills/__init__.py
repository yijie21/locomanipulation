"""The scene RSI loop's skill library (TODO 27, plan docs/handoff/scene-rsi-plan-20261016.md D4, D6): task-level
functions `skill(ctx, ...) -> Outcome` over the motor layer and the privileged scene state, and `plan_for(task)`, the
plan for a goal shape. The loop's Engineer adds and changes skills here; the motor layer below only takes default-off
switches.

The package is loaded both as `locomani.tasks.scene.skills` and, by the loop, from a candidate's snapshot directory
under another name (scripts/rsi/scene_round.py), so modules here import each other
relatively. `apply_set` is how a probe or a candidate overrides a parameter without editing code."""

from __future__ import annotations

from locomani.backends.scalebfm.motor import Outcome
from locomani.scene_discovery.predicates import parse

from . import bimanual, clearance, cover, grasp_program, held, into, near, pack, pick, place  # noqa: F401

PLAN_HELD = "pick"  # held(X): "pick" (the ported closed-loop program, plan D4) or "pick_simple"
MODULES = ("pick", "grasp_program", "place", "clearance", "held", "cover", "into", "near", "bimanual", "pack")


def plan_for(task):
    """The plan for a task's goal, or None when no skill covers it. Covered: held(X), alone or with upright(X) and
    lifted(X, ...) of the same X (`pick_held`: the two-hand hug for an object the scene file says takes two hands, else
    `held.pick_held` over the PLAN_HELD pick); held(X) and exactly one near(X, T, d), with upright(X) or not, all of the
    same X (`near.bring_near` over the PLAN_HELD pick); exactly one in(X, C), alone or with upright(X) of the same X
    (`put_into`: `bimanual.place_into` for a two-hand X, else `into.place_into`); one in(X, C) with held(C) and
    lifted(C, ...) / upright(C) of the container (`pack.pack_into`: the put-in, then the hug of C); a conjunction of
    on(X, S) and upright(X) only (`place_all`: `place.place_on` for each on(X, S) in goal order, stopping at the first
    failure; nothing when the goal already holds)."""
    preds = [parse(g) for g in task.goal]
    names, keys = {p.name for p in preds}, {p.args[0] for p in preds}
    if "held" in names and names <= {"held", "upright", "lifted"} and len(keys) == 1:
        key = preds[0].args[0]
        upright = "upright" in names
        return lambda ctx: pick_held(ctx, key, upright)
    nears = [p for p in preds if p.name == "near"]
    if "held" in names and names <= {"held", "near", "upright"} and len(nears) == 1 and len(keys) == 1:
        key, target, d = nears[0].args
        skill = getattr(pick, PLAN_HELD)
        upright = "upright" in names
        return lambda ctx: bring_near(ctx, key, target, d, skill, upright)
    ins = [tuple(p.args) for p in preds if p.name == "in"]
    if len(ins) == 1 and all(p.name == "in" or (p.name == "upright" and p.args[0] == ins[0][0]) for p in preds):
        key, container = ins[0]
        return lambda ctx: put_into(ctx, key, container)
    if (len(ins) == 1 and "held" in names
            and all(p.name == "in" or (p.name in ("held", "lifted", "upright") and p.args[0] == ins[0][1]) for p in preds)):  # fmt: skip
        key, container = ins[0]
        return lambda ctx: pack.pack_into(ctx, key, container)
    ons = [tuple(p.args) for p in preds if p.name == "on"]
    if ons and all(p.name in ("on", "upright") for p in preds):
        return lambda ctx: place_all(ctx, ons)
    return None


def pick_held(ctx, key, upright):
    """held(X): the two-hand hug (bimanual.pick) when the scene file's features say X takes two hands (never its name,
    rules.md 30), else held.pick_held over the PLAN_HELD pick."""
    if bimanual.two_handed(ctx.world, key):
        return bimanual.pick(ctx, key)
    return held.pick_held(ctx, key, getattr(pick, PLAN_HELD), upright)


def put_into(ctx, key, container):
    """in(X, C): the two-hand put-in for a two-hand X, else the one-hand one (pack.put_in)."""
    return pack.put_in(ctx, key, container)


def bring_near(ctx, key, target, d, skill, upright):
    """`near.bring_near` with pick.SIDE_AHEAD_M at near.PICK_SIDE_AHEAD_M throughout: the loop accepted the carry-to
    with that switch set for every round of its family (all of them carry-tos); held(X) and the put-in keep theirs."""
    saved = pick.SIDE_AHEAD_M
    pick.SIDE_AHEAD_M = near.PICK_SIDE_AHEAD_M
    try:
        return near.bring_near(ctx, key, target, d, skill, upright)
    finally:
        pick.SIDE_AHEAD_M = saved


def place_all(ctx, ons):
    """`place.place_on` for each (object, surface) in order; the first failure ends the plan. Nothing is done when the
    task's whole goal already holds (`goal_holds`): plate holder starts where it does succeeded only because the pick
    was refused before contact, a margin 2% of the creep rate wide (r2_on_single iteration 0)."""
    done = goal_holds(ctx)
    if done is not None:
        return done
    out = None
    for key, surface in ons:
        out = place.place_on(ctx, key, surface)
        if not out.ok:
            return out
    return out


def goal_holds(ctx):
    """A success Outcome, logged (step "goal_holds"), when the task's goal predicates hold on the state now, as the
    episode judges them at the end; else None. Reads the state only."""
    world, m = ctx.world, ctx.motor
    keys = list(dict.fromkeys(a for g in ctx.task.goal for a in parse(g).args
                              if isinstance(a, str) and a in world.meta["objects"]))  # fmt: skip
    ok, _ = world.check(ctx.task.goal)(world.state(keys))
    if not ok:
        return None
    return ctx.note("goal_holds", Outcome(True, "goal already holds", m.t, m.t))


def apply_set(package, overrides: dict) -> list[str]:
    """Set parameters on a loaded skills package: "NAME" (the package's own), "module.NAME", or "module.NAME.key"
    (an entry of a dict parameter). Unknown names raise (a typo must not run silently). Returns the keys set."""
    done = []
    for key, value in (overrides or {}).items():
        parts = key.split(".")
        target = package
        if len(parts) > 1 and hasattr(package, parts[0]) and not hasattr(package, key):
            target, parts = getattr(package, parts[0]), parts[1:]
        if not hasattr(target, parts[0]):
            raise KeyError(f"{key}: no parameter {parts[0]!r} in {getattr(target, '__name__', target)}")
        if len(parts) == 1:
            setattr(target, parts[0], value)
        elif len(parts) == 2 and isinstance(getattr(target, parts[0]), dict):
            d = getattr(target, parts[0])
            if parts[1] not in d:
                raise KeyError(f"{key}: no entry {parts[1]!r} in {parts[0]}")
            d[parts[1]] = value
        else:
            raise KeyError(f"{key}: expected NAME, module.NAME or module.NAME.key")
        done.append(key)
    return done


def parameters(package) -> dict:
    """Every overridable parameter: upper-case module constants (numbers, strings, dicts of them) of the package and of
    each module in MODULES, by `apply_set` key."""
    out = {k: v for k, v in vars(package).items() if k.isupper() and isinstance(v, (int, float, str, dict))}
    for name in MODULES:
        mod = getattr(package, name)
        for k, v in vars(mod).items():
            if k.isupper() and isinstance(v, (int, float, str, dict)) and not isinstance(v, bool):
                out[f"{name}.{k}"] = v
    return out
