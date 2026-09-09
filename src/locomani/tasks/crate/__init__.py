"""Crate (bimanual) task: scene, expert, acceptance, schema, failures.

Built from bimanual_carry.py (decision 3). There is no crate equivalent of
grasp_gym's per-round `plan_for_target` reach-map sampling -- the crate scene
is a single fixed layout built ahead of time, not resampled per round -- so
`sample_scene` is `bimanual_scene.main`, the scene-building entry point
bimanual_carry.py's own `main()` points at when the loaded scene doesn't
match (`"build it with bimanual_scene.py"`). `run_round` also serves as
`acceptance`: it returns the `m` dict with `ok`/`why` directly, no separate
RoundResult class the way bottle has one.

`why` strings use the same `[tag] message` convention as bottle's (grepped
bimanual_carry.py's `m["why"] = ...` sites); `summarize_wall_pilot.py` prints
these prefixes per round but has no `classify` function of its own, so
`classify_failure` uses `tasks.base.bracket_label` here too -- see
docs/handoff/refactor-step4-layout.md "Executed (4b/5)" for this judgment call
(same one made for bottle, spelled out there).

`sample_scene`/`run_round`/`acceptance` import bimanual_scene/bimanual_carry
lazily, inside the wrapper, for the same reason bottle's do (see that
package's docstring): both pull in mujoco-dependent modules, and
`scripts/gen_config_reference.py --check` loads `CrateTaskConfig` with plain
`python3`, which does not have mujoco.
"""

from __future__ import annotations

from locomani.tasks.base import Task, TaskSchema, bracket_label
from locomani.tasks.crate.config import CrateTaskConfig

SCHEMA = TaskSchema(
    modality_config_path="configs/gr00t_sonic_3cam_bimanual.py",
    video_keys=("ego_view", "left_wrist", "right_wrist"),
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
        "left_wrist_pos",
        "left_wrist_abs_quat",
    ),
    action_keys=("motion_token", "left_gripper", "right_gripper"),
)

FAILURE_REASONS = ("stand", "grasp", "lift", "hold")


def _sample_scene(*args, **kwargs):
    from locomani.tasks.crate import bimanual_scene

    return bimanual_scene.main(*args, **kwargs)


def _run_round(*args, **kwargs):
    from locomani.tasks.crate import bimanual_carry

    return bimanual_carry.run_round(*args, **kwargs)


TASK = Task(
    name="crate",
    config_cls=CrateTaskConfig,
    sample_scene=_sample_scene,
    run_round=_run_round,
    acceptance=_run_round,
    schema=SCHEMA,
    classify_failure=bracket_label,
    failure_reasons=FAILURE_REASONS,
)
