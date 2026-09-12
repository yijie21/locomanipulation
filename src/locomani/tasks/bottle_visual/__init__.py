"""Visual bottle entries and schema. Recorded-artifact checks live in audit.audit_case."""

from __future__ import annotations

from locomani.pipeline.execution_record import RoundResult
from locomani.schema import VISUAL_BOTTLE
from locomani.tasks.base import Task, accept_result
from locomani.tasks.bottle_visual.config import BottleVisualTaskConfig, GraspConfig  # noqa: F401

SCHEMA = VISUAL_BOTTLE

FAILURE_REASONS = ("timeout", "fall", "setup", "knock", "recovery", "visibility")
_RULES = (
    ("timeout", "timeout"),
    ("fall", "fall"),
    ("physics", "fall"),
    ("height changed", "setup"),
    ("setup", "setup"),
    ("knock", "knock"),
    ("recovery", "recovery"),
    ("visib", "visibility"),
)


def classify_failure(why: str) -> str:
    """round.py's `why` -> one of FAILURE_REASONS ("ok" for success, "other" for an unknown wording)."""
    if not why:
        return "ok"
    low = why.lower()
    for needle, label in _RULES:
        if needle in low:
            return label
    return "other"


def _sample_scene(*args, **kwargs):
    from locomani.tasks.bottle_visual import geometry, sampling

    return sampling.sample_reset(geometry, *args, **kwargs)


def _run_round(*args, **kwargs):
    from locomani.tasks.bottle_visual import round as round_module

    return RoundResult.from_mapping(round_module.run_case(*args, **kwargs))


TASK = Task(
    name="bottle_visual",
    config_cls=BottleVisualTaskConfig,
    sample_scene=_sample_scene,
    run_round=_run_round,
    acceptance=accept_result,
    schema=SCHEMA,
    classify_failure=classify_failure,
    failure_reasons=FAILURE_REASONS,
)
