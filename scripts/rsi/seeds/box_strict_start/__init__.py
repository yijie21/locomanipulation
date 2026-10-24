"""The scene RSI loop's skill library, grown from the atomic seed (TODO 31, user 2026-10-20): `held(X)` of an object
that takes two hands (bimanual.py, the hug pick). A sandbox built for a family whose `sandbox.seed` names the seed
directory starts its skills package from it (scripts/rsi/sandbox.py).

A skill is a task-level function `skill(ctx, ...) -> Outcome` over the motor layer and the privileged scene state;
`plan_for(task)` returns the plan for a goal, a callable `plan(ctx) -> Outcome`, or None when no skill covers it. What a
skill can use:
- `ctx.motor` (src/locomani/backends/scalebfm/motor.py): walk_to, turn, sidestep, posture, reach, grip, release, hold;
  per-tick hand commands (`hand`, `adopt_hand`, `tick`) and sensors (jaw centre, jaw opening, finger contacts);
- `locomani.tasks.scene.navigate.walk_path`: a walk around furniture on the occupancy map;
- `ctx.world` (src/locomani/tasks/scene/world.py): the scene file (`meta`: objects, surfaces), object poses, the free
  bodies, the occupancy map and navigator; `ctx.note` logs a step into the round record.

The package is loaded both as `locomani.tasks.scene.skills` and, by the loop, from a candidate's snapshot directory
under another name (scripts/rsi/scene_round.py), so modules here import each other relatively. `apply_set` is how a
probe or a candidate overrides a parameter without editing code; parameters are upper-case module constants."""

from __future__ import annotations

import types

from locomani.backends.scalebfm.motor import Outcome
from locomani.scene_discovery.predicates import parse

from . import bimanual


def _held_key(goal) -> str | None:
    """The object of a `held(X)` goal, or None. Other predicates may ride along when they are about X alone
    (`lifted(X, ...)`, `upright(X)`): the seed of box_bimanual's strict goal (TODO 38.19) is the 42ff17cb110f
    incumbent with only this dispatch widened by hand -- it accepted a single held(X) -- the pick itself unchanged."""
    preds = [parse(g) for g in goal]
    held = [p for p in preds if p.name == "held" and isinstance(p.args[0], str)]
    if len(held) != 1:
        return None
    key = held[0].args[0]
    if all(p.name in ("held", "lifted", "upright") and p.args[0] == key for p in preds):
        return key
    return None


def _pick(key: str):
    def plan(ctx):
        meta = ctx.world.meta["objects"][key]
        if (meta.get("semantic") or {}).get("graspable") == "bimanual" or meta.get("bimanual"):
            return bimanual.pick(ctx, key)
        return Outcome(False, "no one-hand pick in this package", ctx.motor.t, ctx.motor.t)

    return plan


def plan_for(task):
    """The plan for a task's goal, or None when no skill covers it. Covered: `held(X)`; an object whose features say
    it takes two hands (`semantic.graspable` bimanual) gets the hug pick (bimanual.py)."""
    key = _held_key(task.goal)
    return _pick(key) if key is not None else None


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


def _overridable(v) -> bool:
    return isinstance(v, (int, float, str, dict)) and not isinstance(v, bool)


def parameters(package) -> dict:
    """Every overridable parameter: upper-case constants (numbers, strings, dicts of them) of the package and of every
    submodule it has imported, by `apply_set` key."""
    out = {k: v for k, v in vars(package).items() if k.isupper() and _overridable(v)}
    for name, mod in vars(package).items():
        if isinstance(mod, types.ModuleType) and mod.__name__.startswith(package.__name__ + "."):
            out.update({f"{name}.{k}": v for k, v in vars(mod).items() if k.isupper() and _overridable(v)})
    return out
