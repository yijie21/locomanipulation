"""The scene RSI loop's skill library (TODO 27, plan docs/handoff/scene-rsi-plan-20261016.md D4, D6): task-level
functions `skill(ctx, ...) -> Outcome` over the motor layer and the privileged scene state, and `plan_for(task)`, the
plan for a goal shape. The loop's Engineer adds and changes skills here; the motor layer below only takes default-off
switches.

The package is loaded both as `locomani.tasks.scene.skills` and, by the loop, from a candidate's snapshot directory
under another name (scripts/experiments/scene_rsi_20261016/scene_round.py), so modules here import each other
relatively. `apply_set` is how a probe or a candidate overrides a parameter without editing code."""

from __future__ import annotations

from locomani.scene_discovery.predicates import parse

from . import clearance, cover, grasp_program, held, pick, place  # noqa: F401  (apply_set reaches the modules)

PLAN_HELD = "pick"  # held(X): "pick" (the ported closed-loop program, plan D4) or "pick_simple"


def plan_for(task):
    """The plan for a task's goal, or None when no skill covers it. Covered: held(X), alone or with upright(X) of the
    same X (`held.pick_held` over the PLAN_HELD pick); a conjunction of on(X, S) and upright(X) only (`place.place_on`
    for each on(X, S) in goal order, stopping at the first failure)."""
    preds = [parse(g) for g in task.goal]
    names, keys = {p.name for p in preds}, {p.args[0] for p in preds}
    if "held" in names and names <= {"held", "upright"} and len(keys) == 1:
        key = preds[0].args[0]
        skill = getattr(pick, PLAN_HELD)
        upright = "upright" in names
        return lambda ctx: held.pick_held(ctx, key, skill, upright)
    ons = [tuple(p.args) for p in preds if p.name == "on"]
    if ons and all(p.name in ("on", "upright") for p in preds):
        return lambda ctx: place_all(ctx, ons)
    return None


def place_all(ctx, ons):
    """`place.place_on` for each (object, surface) in order; the first failure ends the plan."""
    out = None
    for key, surface in ons:
        out = place.place_on(ctx, key, surface)
        if not out.ok:
            return out
    return out


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
    """Every overridable parameter: upper-case module constants (numbers, strings, dicts of them), by `apply_set` key."""
    out = {k: v for k, v in vars(package).items() if k.isupper() and isinstance(v, (int, float, str, dict))}
    for name in ("pick", "grasp_program", "place", "clearance", "held", "cover"):
        mod = getattr(package, name)
        for k, v in vars(mod).items():
            if k.isupper() and isinstance(v, (int, float, str, dict)) and not isinstance(v, bool):
                out[f"{name}.{k}"] = v
    return out
