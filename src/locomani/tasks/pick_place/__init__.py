"""Pick-and-place reuses bottle execution and schema with its goal-specific task profile."""

from __future__ import annotations

from locomani.tasks.base import Task, accept_result, bracket_label
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
    acceptance=accept_result,
    schema=SCHEMA,
    classify_failure=bracket_label,
    failure_reasons=FAILURE_REASONS,
)
