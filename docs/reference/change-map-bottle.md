# Bottle, pick-and-place and ScaleBFM

Use the matching row; expand along actual callers only as the change requires.
Back to [domain navigation](change-map.md). Test names are relative to `tests/`.

| Change | Entry / owner | Configuration or example | Tests |
|---|---|---|---|
| Bottle arguments / profile precedence | [cli_args.py](../../src/locomani/tasks/bottle/cli_args.py) → [grasp_gym.py](../../src/locomani/tasks/bottle/grasp_gym.py) | [bottle/config.py](../../src/locomani/tasks/bottle/config.py) | `test_bottle_cli.py`, `test_reach_config.py` |
| Bottle strategy / grasp behavior | [policy.py](../../src/locomani/tasks/bottle/policy.py), then the selected implementation | [reach_config.py](../../src/locomani/tasks/bottle/reach_config.py), [teleop profile](../../configs/tasks/bottle_visible_teleop.json) | `test_reach_strategy.py`, `test_coupled_grasp_policy.py`, `test_teleop_grasp_policy.py` |
| Filtered walking / continuous supported bottle reach | [walking_reach.py](../../src/locomani/tasks/bottle/walking_reach.py), [whole_body_policy.py](../../src/locomani/tasks/bottle/whole_body_policy.py), [supported_motion.py](../../src/locomani/primitives/supported_motion.py) | [whole_body_config.py](../../src/locomani/tasks/bottle/whole_body_config.py), [candidate profile](../../configs/tasks/bottle_whole_body.json) | `test_bottle_walking_reach.py`, `test_bottle_whole_body.py`, `test_streamed_handoff.py` |
| Bottle sequencing / sampling | [round_phases.py](../../src/locomani/tasks/bottle/round_phases.py), [round_loop.py](../../src/locomani/tasks/bottle/round_loop.py) | [plan.py](../../src/locomani/tasks/bottle/plan.py), [runtime.py](../../src/locomani/tasks/bottle/runtime.py), `BottleTaskConfig.spawn` | `test_visible_expert.py`, `test_ready_vla_handover.py` |
| Pick and place | [place_phase.py](../../src/locomani/tasks/bottle/place_phase.py) | [pick_place.json](../../configs/tasks/pick_place.json), `BottleTaskConfig.place` | `test_place_phase.py` |
| ScaleBFM collection / evaluation | [collect.py](../../src/locomani/tasks/bottle_visual/pipeline/collect.py), [eval.py](../../src/locomani/tasks/bottle_visual/pipeline/eval.py) | [bottle_visual/config.py](../../src/locomani/tasks/bottle_visual/config.py), [scalebfm/config.py](../../src/locomani/backends/scalebfm/config.py) | `test_visual_sampling.py`, `test_visual_report.py` |

## Example: bottle grasp strategy or configuration

1. Define the field/validation in its owning config. Choose `reach.strategy` for an
   implementation choice; add a numeric field only for a distinct control.
2. Add the consumer and a test through it. For a new strategy update `policy.py`
   and its tests; keep selection out of round sequencing.
3. Profile fields represented on the gym CLI resolve in `cli_args.py`. Use explicit
   presence parsing and check both expert and VLA modes.
4. Run named tests and `make config-doc`; choose final validation through the [session workflow](../agents/session-workflow.md#verification). Inspect resolved gym options
   with `PYTHONPATH=src python -m locomani.tasks.bottle.cli_args --task-config <profile>`.
   This CPU-only command prints resolved values and sources without a simulator.
