"""The scene RSI loop's skill library (TODO 27, plan docs/handoff/scene-rsi-plan-20261016.md D4, D6): task-level
functions `skill(ctx, ...) -> Outcome` over the motor layer and the privileged scene state. The loop's Engineer adds and
changes skills here; the motor layer below only takes default-off switches. `PLANS` maps a goal shape to the plan that
achieves it."""

from __future__ import annotations

from locomani.scene_discovery.predicates import parse
from locomani.tasks.scene.skills.pick import pick


def plan_for(task):
    """The plan for a task's goal, or None when no skill covers it. Covered: held(X)."""
    preds = [parse(g) for g in task.goal]
    if len(preds) == 1 and preds[0].name == "held":
        key = preds[0].args[0]
        return lambda ctx: pick(ctx, key)
    return None
