"""bottle_visual task: the ScaleBFM visual bottle grasp (scene, expert, acceptance, schema, failures).

Second Task plugin (refactor-plan decision 6: ScaleBFM as the second backend). Like
tasks/bottle, this is a thin descriptor over functions that already exist: `sampling.sample_reset`
draws a round's geometry, `round.run_case` is the expert's round entry and returns the result dict
whose `ok`/`why` is the acceptance verdict, `audit.audit_case` re-checks a recorded case offline.
The three heavy modules import mujoco/torch, so they are imported lazily inside the wrappers
(gen_config_reference.py loads this package under plain python3).

`why` strings are plain sentences, not `[tag] ...` (round.py), so `classify_failure` matches on
their wording; FAILURE_REASONS lists the labels it can produce.
"""

from __future__ import annotations

from locomani.tasks.base import Task, TaskSchema
from locomani.tasks.bottle_visual.config import BottleVisualTaskConfig, GraspConfig  # noqa: F401

SCHEMA = TaskSchema(
    modality_config_path="configs/gr00t_bfm_visual_2cam.py",
    video_keys=("ego_view", "right_wrist"),
    state_keys=(
        "left_leg",
        "right_leg",
        "waist",
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
        "projected_gravity",
        "right_wrist_pos",
        "right_wrist_abs_quat",
        "robot_velocity",
        "previous_action",
    ),
    action_keys=("visual_command",),
)

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

    return round_module.run_case(*args, **kwargs)


def _acceptance(*args, **kwargs):
    from locomani.tasks.bottle_visual import audit

    return audit.audit_case(*args, **kwargs)


TASK = Task(
    name="bottle_visual",
    config_cls=BottleVisualTaskConfig,
    sample_scene=_sample_scene,
    run_round=_run_round,
    acceptance=_acceptance,
    schema=SCHEMA,
    classify_failure=classify_failure,
    failure_reasons=FAILURE_REASONS,
)
