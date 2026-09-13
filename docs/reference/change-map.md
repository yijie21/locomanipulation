# Change map

Read the relevant row, its owning code and the named tests, using scoped searches
and relevant sections. Expand along callers, consumers and dependencies when needed.
Shared protocol changes, cross-layer failures and broad refactors can require wider
reading; explain the unresolved contract that warrants it. Routine local changes
start from this map. This map describes implemented entries, not experimental
validity. Test names below are relative to `tests/` and checked for existence.

| Change | Entry / owner | Configuration or example | Tests |
|---|---|---|---|
| Bottle arguments / profile precedence | [cli_args.py](../../src/locomani/tasks/bottle/cli_args.py) → [grasp_gym.py](../../src/locomani/tasks/bottle/grasp_gym.py) | [bottle/config.py](../../src/locomani/tasks/bottle/config.py) | `test_bottle_cli.py`, `test_reach_config.py` |
| Bottle strategy / grasp behavior | [policy.py](../../src/locomani/tasks/bottle/policy.py), then the selected implementation | [reach_config.py](../../src/locomani/tasks/bottle/reach_config.py), [teleop profile](../../configs/tasks/bottle_visible_teleop.json) | `test_reach_strategy.py`, `test_coupled_grasp_policy.py`, `test_teleop_grasp_policy.py` |
| Bottle sequencing / sampling | [round_phases.py](../../src/locomani/tasks/bottle/round_phases.py), [round_loop.py](../../src/locomani/tasks/bottle/round_loop.py) | [plan.py](../../src/locomani/tasks/bottle/plan.py), [runtime.py](../../src/locomani/tasks/bottle/runtime.py), `BottleTaskConfig.spawn` | `test_visible_expert.py`, `test_ready_vla_handover.py` |
| Pick and place | [place_phase.py](../../src/locomani/tasks/bottle/place_phase.py) | [pick_place.json](../../configs/tasks/pick_place.json), `BottleTaskConfig.place` | `test_place_phase.py` |
| Crate collection | [run_crate_pilot.py](../../src/locomani/pipeline/run_crate_pilot.py) → [round_runner.py](../../src/locomani/tasks/crate/round_runner.py) | [crate/config.py](../../src/locomani/tasks/crate/config.py) | `test_crate_pilot.py`, `test_crate_round_runner.py` |
| Crate carry / wall reach and waist | [bimanual_carry.py](../../src/locomani/tasks/crate/bimanual_carry.py), [bimanual_ik.py](../../src/locomani/tasks/crate/bimanual_ik.py), [carry_phases.py](../../src/locomani/tasks/crate/carry_phases.py) | [Reach contract](../crate_collection.md#waist-and-reach); [carry_config.py](../../src/locomani/tasks/crate/carry_config.py), [carry_motion.py](../../src/locomani/tasks/crate/carry_motion.py) | `test_crate_carry_waist.py`, `test_crate_bimanual_ik.py`, `test_crate_motion_flow.py`, `test_planner_reference_takeover.py` |
| Crate RGB approach / grasp | [expert.py](../../src/locomani/tasks/crate/expert.py), [perception.py](../../src/locomani/tasks/crate/perception.py) | [visual_config.py](../../src/locomani/tasks/crate/visual_config.py); [video pilot guide](../crate_visual_collection.md) | `test_crate_bimanual_ik.py`, `test_crate_visual_perception.py`, `test_crate_visual_policy.py`, `test_crate_visual_recording.py`, `test_crate_visual_approach.py`, `test_crate_joint_reference.py` |
| Crate planner squat comparison | [carry_crouch.py](../../src/locomani/tasks/crate/carry_crouch.py) → [bimanual_carry.py](../../src/locomani/tasks/crate/bimanual_carry.py) | [carry_config.py](../../src/locomani/tasks/crate/carry_config.py); [30 cm squat profile](../../configs/tasks/crate_table30.json) | `test_crate_crouch.py`, `test_configs.py` |
| Crate low reach with hip hinge | [carry_whole_body.py](../../src/locomani/tasks/crate/carry_whole_body.py), [whole_body_ik.py](../../src/locomani/tasks/crate/whole_body_ik.py) | [carry_config.py](../../src/locomani/tasks/crate/carry_config.py); [whole-body profile](../../configs/tasks/crate_low_reach.json) | `test_crate_whole_body.py`, `test_streamed_reference_packet.py`, `test_crate_bimanual_ik.py` |
| ScaleBFM collection / evaluation | [collect.py](../../src/locomani/tasks/bottle_visual/pipeline/collect.py), [eval.py](../../src/locomani/tasks/bottle_visual/pipeline/eval.py) | [bottle_visual/config.py](../../src/locomani/tasks/bottle_visual/config.py), [scalebfm/config.py](../../src/locomani/backends/scalebfm/config.py) | `test_visual_sampling.py`, `test_visual_report.py` |
| SONIC protocol / arm execution | [streamed_arm.py](../../src/locomani/backends/sonic/streamed_arm.py), [teleop_arm.py](../../src/locomani/backends/sonic/teleop_arm.py) | [sonic/config.py](../../src/locomani/backends/sonic/config.py) | `test_sonic_policy_default.py`, `test_teleop_arm.py` |
| Stack lifecycle / ports | [stack.py](../../src/locomani/pipeline/stack.py), [ports.py](../../src/locomani/sim/ports.py), [start_stack.sh](../../scripts/stack/start_stack.sh) | [stages.py](../../src/locomani/pipeline/stages.py); bottle/crate pilots share the seam | `test_stack_delivery.py`, `test_stack_start_failure.py`, `test_instance_ports.py` |
| Dataset fields / conversion | [schema.py](../../src/locomani/schema.py) → [convert_to_hf_format.py](../../src/locomani/pipeline/convert_to_hf_format.py) | [gr00t_sonic_2cam.py](../../configs/gr00t_sonic_2cam.py) | `test_schema_provenance.py`, `test_verification_tail.py` |
| Training / closed-loop evaluation | [run_learn_pilot.py](../../src/locomani/pipeline/run_learn_pilot.py), [eval_closed_loop.py](../../src/locomani/pipeline/eval_closed_loop.py) | [pilot_config.py](../../src/locomani/pipeline/pilot_config.py), [eval_config.py](../../src/locomani/pipeline/eval_config.py) | `test_runtime_consumers.py`, `test_eval_config_reaches_the_client.py`, `test_trace_round_join.py` |
| Task result / acceptance | [base.py](../../src/locomani/tasks/base.py), [execution_record.py](../../src/locomani/pipeline/execution_record.py) | `RoundResult.accepted(require_recording=...)` | `test_task_acceptance.py`, `test_execution_record.py` |
| Model / run provenance | [provenance.py](../../src/locomani/pipeline/provenance.py), [serve_policy.py](../../src/locomani/pipeline/serve_policy.py) | `config_snapshot.json`, `deploy_receipts.jsonl`, stage `commands.jsonl` | `test_schema_provenance.py`, `test_stack_delivery.py` |
| CPU geometry fixtures | [geometry_fixture.py](../../tests/geometry_fixture.py) | [fixture README](../../tests/fixtures/geometry/README.md) | `test_grasp_point_ik.py`, `test_ready_grasp_policy.py` |
| Shared config serialization | [config_base.py](../../src/locomani/config_base.py) | [bottle/config.py](../../src/locomani/tasks/bottle/config.py) | `test_configs.py`, `test_review_2026_09_10.py` |
| Scene / camera / rendering | [build_sim_scene.py](../../src/locomani/sim/build_sim_scene.py), [scene_lib.py](../../src/locomani/sim/scene_lib.py) | [rendering tutorial](../rendering.md) | `test_head_camera_spec.py`, `test_seat_on_floor.py` |
| Isaac source assets / cache | [scene_assets.py](../../src/locomani/sim/isaac/scene_assets.py), [build_isaac_assets.sh](../../scripts/build_isaac_assets.sh) | [rendering](../rendering.md#资产构建), [shared room source](../../assets/scene/README.md) | `test_isaac_scene_assets.py` |
| Navigation / architecture checks | [lint_repository.py](../../scripts/lint_repository.py) | [standards](standards.md) | `test_repository_rules.py` |

## Change workflow

1. Select the row matching the requested behavior; identify its owner, current
   interface, configuration and observable contract in the named tests. If no row
   fits, use the [module ownership rules](standards.md#module-ownership) to place it.
2. Implement through that interface and keep helpers with their owner. Interface,
   dependency or configuration changes follow [the standards](standards.md).
3. Check the changed behavior through its real consumer, including relevant failure
   paths and parameter delivery. Run `make test TESTS="tests/test_name.py"` with the
   actual relevant test paths. Regenerate config fields with `make config-doc` when
   configuration changes, then run `make check`. Apply the standards' live-validation
   requirements when physical commands, timing, sampling or acceptance change.
4. Update the affected map row when entries, configuration owners or tests change;
   add a row for a new change category. Update the owning documentation using the
   [document index](../README.md#documentation-ownership). Record checks actually run
   and remaining uncertainty; completion requires code, tests and navigation to agree.

## Example: bottle grasp strategy or configuration

1. Define the field/validation in its owning config. Choose `reach.strategy` for an
   implementation choice; add a numeric field only for a distinct control.
2. Add the consumer and a test through it. For a new strategy update `policy.py`
   and its tests; keep selection out of round sequencing.
3. Profile fields represented on the gym CLI resolve in `cli_args.py`. Use explicit
   presence parsing and check both expert and VLA modes.
4. Run named tests, `make config-doc`, then `make check`. Inspect resolved gym options
   with `PYTHONPATH=src python -m locomani.tasks.bottle.cli_args --task-config <profile>`.
   This CPU-only command prints resolved values and sources without a simulator.

## Current limitations

`locomani.cli` still dispatches to different drivers; consult the relevant row for
argument meanings. ScaleBFM has an independent in-process runner. SONIC consumers
still read derived compatibility properties of ReachConfig; five historical strategy
switches migrate only at the version-1 JSON boundary. GymRuntime and VlaConfig carry production runtime choices explicitly. Legacy
RTC diagnostics retain scoped instrumentation hooks. The gym-options report covers
its parser; child receipts and delivery tests cover subprocess settings.

Next migrations: [ROADMAP](../ROADMAP.md). Launch commands: [runbook](runbook.md).
Read [the document index](../README.md) for task detail and historical evidence.
