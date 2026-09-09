"""Bottle task: scene, expert, acceptance, schema, failures.

Built entirely from grasp_gym.py's existing functions (decision 3): `plan_for_target`
samples a scene target's reach geometry (`sample_scene`), and `run_round` is the
scripted expert's round entry. Its `RoundResult` also serves as `acceptance` --
there is no separately-extractable ok/why function; the verify_* stages
(`execution.enter("verify_visibility")` / `"verify_terminal_hold"`) that decide
it live inside `_run_round`, which `run_round` wraps.

`failure_taxonomy.py` classifies a different thing -- CLOSED-LOOP EVAL traces.
Its `classify()` needs a full per-timestep `trace`; fed a bare `why` string with
an empty trace, its own first branch swallows almost everything into "A4 view
gate / pre-VLA" (`"[vla]" not in why and not tr"` is true for nearly any why).
So `classify_failure` here uses the `[tag] message` convention directly via
`tasks.base.bracket_label` instead of reusing failure_taxonomy -- see
docs/handoff/refactor-step4-layout.md "Executed (4b/5)" for this judgment call.
The five tags below are every prefix `grasp_gym.py`'s `return False, "[...] ..."`
sites (and `_run_round`'s one bare `"[gym] ..."` case) actually emit.

`sample_scene`/`run_round`/`acceptance` import grasp_gym lazily, inside the
wrapper, rather than at module level: grasp_gym.py imports `mujoco`, and
`scripts/gen_config_reference.py --check` (a required pre-commit hook) loads
`BottleTaskConfig` with plain `python3`, which does not have mujoco -- and any
import of anything under `locomani.tasks.bottle` runs this file first. Eager
`import grasp_gym` here broke that hook; this is the fix.
"""

from __future__ import annotations

from locomani.tasks.base import Task, TaskSchema, bracket_label
from locomani.tasks.bottle.config import BottleTaskConfig

SCHEMA = TaskSchema(
    modality_config_path="configs/gr00t_sonic_2cam.py",
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
    ),
    action_keys=("motion_token", "right_gripper"),
)

FAILURE_REASONS = ("stand", "view", "grasp", "hold", "gym")


def _sample_scene(*args, **kwargs):
    from locomani.tasks.bottle import grasp_gym

    return grasp_gym.plan_for_target(*args, **kwargs)


def _run_round(*args, **kwargs):
    from locomani.tasks.bottle import grasp_gym

    return grasp_gym.run_round(*args, **kwargs)


TASK = Task(
    name="bottle",
    config_cls=BottleTaskConfig,
    sample_scene=_sample_scene,
    run_round=_run_round,
    acceptance=_run_round,
    schema=SCHEMA,
    classify_failure=bracket_label,
    failure_reasons=FAILURE_REASONS,
)
