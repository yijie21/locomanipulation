"""The scene RSI loop's skill library, atomic seed (TODO 31, user 2026-10-20): no skill yet. A sandbox built for a family
whose `sandbox.seed` names this directory starts its skills package from it (scripts/rsi/sandbox.py).

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

from . import doorway, pick, recipes, rim
from .recipes import tall_narrow


def plan_for(task):
    """The plan for a task's goal, or None when no skill covers it. Covered: held(X) of one free object (pick.pick:
    a side grasp, or a top-down pinch for a flat object; rim.pick: a pinch across one wall of an open container with
    thin box walls, decided on the loaded model when the plan runs). A side grasp on a high support whose forearm line
    crosses low fixed clutter is pitched nose-down (clutter.pitch). A side grasp of an uncovered non-container on a
    support in the hanging hand's band takes rim's stance and raised-hand walk (pick.banded). A level side grasp of a
    tall non-container whose section is wider than the open jaw on the walked approach, within reach, is raised to the
    lowest height under its top where the section fits (pick.raise_fit). After the pregrasp, a level side grasp hands an
    object whose features match a flow under recipes/ (recipes.choose; tall_narrow: a tall object whose hull section
    fits the open jaw with under 1 cm a side) to that flow. A high pinch whose lift leaves the object unlifted re-grasps
    it once from its new pose (pick.PINCH_REGRASP). No stance's pelvis stands within STANCE_CLEAR_M of a low fixed floor
    object the runner's map leaves out (pick.floor_statics). A spawn in another room than the object, with no stance in
    the object's room reachable on the runner's map, first walks through the doorway (doorway.walk)."""
    base = pick.plan(task)
    if base is None:
        return None
    key = pick.goal_object(task)

    def plan(ctx):
        door = doorway.walk(ctx, key)
        if door is not None and (not door.ok or ctx.motor.fell_at is not None):
            return door
        return rim.pick(ctx, key) if rim.walls(ctx.world, key) else base(ctx)

    return plan


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
