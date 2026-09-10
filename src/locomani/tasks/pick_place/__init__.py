"""Pick-and-place task: the bottle expert plus a place phase (docs/handoff/pick_place_plan.md).

Same scene sampler, expert entry and acceptance as `tasks.bottle` -- the
place phase (tasks/bottle/place_phase.py) is selected inside the bottle
round by `BottleTaskConfig.goal`, so this Task is the bottle Task with a
different config default (configs/tasks/pick_place.json sets target
soup_can, goal basket, prompt "pick up the red can and put it in the
basket") and four more failure tags: `carry` (object slipped while held),
`turn` (container never entered the placing band), `place` (missed, on the
rim, container knocked, hand still touching) and `retreat` (the retreat
swept the object). Schema is the bottle's 2-camera right-hand one.

Lazy grasp_gym imports for the same reason as tasks/bottle/__init__.py.
"""

from __future__ import annotations

from locomani.tasks.base import Task, bracket_label
from locomani.tasks.bottle import SCHEMA
from locomani.tasks.bottle.config import BottleTaskConfig

FAILURE_REASONS = ("stand", "view", "grasp", "hold", "gym", "carry", "turn", "place", "retreat")


def _sample_scene(*args, **kwargs):
    from locomani.tasks.bottle import grasp_gym

    return grasp_gym.plan_for_target(*args, **kwargs)


def _run_round(*args, **kwargs):
    from locomani.tasks.bottle import grasp_gym

    return grasp_gym.run_round(*args, **kwargs)


TASK = Task(
    name="pick_place",
    config_cls=BottleTaskConfig,
    sample_scene=_sample_scene,
    run_round=_run_round,
    acceptance=_run_round,
    schema=SCHEMA,
    classify_failure=bracket_label,
    failure_reasons=FAILURE_REASONS,
)
