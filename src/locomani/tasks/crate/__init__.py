"""Crate entries and dataset schema. Wall/carry drivers retain their task-specific inputs."""

from __future__ import annotations

from locomani.pipeline.execution_record import RoundResult
from locomani.schema import SONIC_CRATE
from locomani.tasks.base import Task, accept_result, bracket_label
from locomani.tasks.crate.config import CrateTaskConfig

SCHEMA = SONIC_CRATE

FAILURE_REASONS = ("stand", "grasp", "lift", "hold")


def _sample_scene(*args, **kwargs):
    from locomani.tasks.crate import bimanual_scene

    return bimanual_scene.main(*args, **kwargs)


def _run_round(*args, **kwargs):
    from locomani.tasks.crate import bimanual_carry

    return RoundResult.from_mapping(bimanual_carry.run_round(*args, **kwargs))


TASK = Task(
    name="crate",
    config_cls=CrateTaskConfig,
    sample_scene=_sample_scene,
    run_round=_run_round,
    acceptance=accept_result,
    schema=SCHEMA,
    classify_failure=bracket_label,
    failure_reasons=FAILURE_REASONS,
)
