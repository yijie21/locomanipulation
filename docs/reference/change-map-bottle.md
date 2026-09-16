# Bottle, pick-and-place and ScaleBFM

Natural walking preparation may raise the arm and acquire wrist ROI while a
stance solve is pending or stale, but it retains the farthest searched local
stop until a full-body result is ready. Live walking stance evaluation screens
the expected and forward/yaw uncertainty endpoints. Its result stays tied to
the selected normalized standing candidate despite unselected churn. Live settled
evaluation screens the full-body endpoint from the owned reference after
bounded command rates and a quiet measured joint/hand window; reference or
measured snapshot changes invalidate the request. Both live paths retain the
bounded Cartesian executor and its measured collision, foot, COM, and capture
guards on every tick. The CPU replay defaults to a full nominal Cartesian-path
audit and accepts `--verification endpoint_only` to reproduce the live planner
scope. POSE-period planner heartbeats send POSE ownership explicitly.

Loaded recovery clears the table in world frame before contact/tracking-gated
inward carry; swing supervision and rotation feedback start at liftoff, housing may touch. Its standing endpoint checks fixed-foot support, COM, hand IK and
attached-object collision. Handback measures loaded stability before one planner IDLE request and a post-switch window tolerating one switch-latency gap; no ack claimed.

Use the matching row; expand along actual callers only as the change requires.
Back to [domain navigation](change-map.md). Test names are relative to `tests/`.

| Change | Entry / owner | Configuration or example | Tests |
|---|---|---|---|
| Bottle arguments / profile precedence | [cli_args.py](../../src/locomani/tasks/bottle/cli_args.py) → [grasp_gym.py](../../src/locomani/tasks/bottle/grasp_gym.py) | [bottle/config.py](../../src/locomani/tasks/bottle/config.py) | `test_bottle_cli.py`, `test_reach_config.py` |
| Bottle strategy / grasp behavior | [policy.py](../../src/locomani/tasks/bottle/policy.py), then the selected implementation | [reach_config.py](../../src/locomani/tasks/bottle/reach_config.py), [teleop profile](../../configs/task/bottle_visible_teleop.yaml) | `test_reach_strategy.py`, `test_coupled_grasp_policy.py`, `test_teleop_grasp_policy.py` |
| Natural walking preparation / asynchronous stance reach / wrist ROI / continuous supported bottle reach | [walking_reach.py](../../src/locomani/tasks/bottle/walking_reach.py), [stance_reach.py](../../src/locomani/tasks/bottle/stance_reach.py), [assisted_target.py](../../src/locomani/tasks/bottle/assisted_target.py), [whole_body_policy.py](../../src/locomani/tasks/bottle/whole_body_policy.py), [supported_motion.py](../../src/locomani/primitives/supported_motion.py) | [whole_body_config.py](../../src/locomani/tasks/bottle/whole_body_config.py), [candidate profile](../../configs/task/bottle_whole_body.yaml), [CPU replay](../../scripts/experiments/bottle_stance_reach_probe_20260915.py) | `test_bottle_stance_reach.py`, `test_bottle_walking_reach.py`, `test_bottle_whole_body.py`, `test_streamed_handoff.py`, `test_pose_preview.py` |
| Central capture / measured hand braking / contact loading / supported tracking | [grasp_capture.py](../../src/locomani/tasks/bottle/grasp_capture.py), [whole_body_policy.py](../../src/locomani/tasks/bottle/whole_body_policy.py), [supported_motion.py](../../src/locomani/primitives/supported_motion.py), [whole_body_ik.py](../../src/locomani/primitives/whole_body_ik.py) | [whole_body_config.py](../../src/locomani/tasks/bottle/whole_body_config.py), [candidate profile](../../configs/task/bottle_whole_body.yaml) | `test_bottle_whole_body.py`, `test_bottle_supported_smoothing.py`, `test_crate_whole_body.py` |
| Loaded world path / standing endpoint / stable planner handback | [loaded_handback.py](../../src/locomani/tasks/bottle/loaded_handback.py), [whole_body_policy.py](../../src/locomani/tasks/bottle/whole_body_policy.py), [supported_motion.py](../../src/locomani/primitives/supported_motion.py), [streamed_arm.py](../../src/locomani/backends/sonic/streamed_arm.py) | [whole_body_config.py](../../src/locomani/tasks/bottle/whole_body_config.py), [loaded endpoint CPU replay](../../scripts/experiments/bottle_loaded_endpoint_probe_20260916.py) | `test_bottle_loaded_handback.py`, `test_bottle_whole_body.py`, `test_streamed_handoff.py` |
| Simple single-arm expert (hanging-arm walk, settled stance, one continuous reach, inward wrap entry, close on contact, lift) | [simple_policy.py](../../src/locomani/tasks/bottle/simple_policy.py), [simple_config.py](../../src/locomani/tasks/bottle/simple_config.py) | [bottle_simple.json](../../configs/task/bottle_simple.yaml), `reach.strategy=simple`, `reach.simple.*`; scene `build_sim_scene.py --object-com-drop` | `test_bottle_simple.py` |
| Bottle sequencing / braking / sampling | [round_phases.py](../../src/locomani/tasks/bottle/round_phases.py), [round_phases_approach.py](../../src/locomani/tasks/bottle/round_phases_approach.py), [round_loop.py](../../src/locomani/tasks/bottle/round_loop.py) | [plan.py](../../src/locomani/tasks/bottle/plan.py), [runtime.py](../../src/locomani/tasks/bottle/runtime.py), `BottleTaskConfig.spawn` | `test_bottle_approach_braking.py`, `test_visible_expert.py`, `test_ready_vla_handover.py` |
| Pick and place | [place_phase.py](../../src/locomani/tasks/bottle/place_phase.py) | [pick_place.json](../../configs/task/pick_place.yaml), `BottleTaskConfig.place` | `test_place_phase.py` |
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
