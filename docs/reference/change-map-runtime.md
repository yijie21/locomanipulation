# Shared runtime, data and development checks

Use the matching row; expand along actual callers only as the change requires.
Back to [domain navigation](change-map.md). Test names are relative to `tests/`.

| Change | Entry / owner | Configuration or example | Tests |
|---|---|---|---|
| SONIC protocol / arm execution | [streamed_arm.py](../../src/locomani/backends/sonic/streamed_arm.py), [teleop_arm.py](../../src/locomani/backends/sonic/teleop_arm.py), [pose_preview.py](../../src/locomani/backends/sonic/pose_preview.py) | [sonic/config.py](../../src/locomani/backends/sonic/config.py); [POSE deploy capability](../../patches/sonic_pose_ownership.md) | `test_sonic_policy_default.py`, `test_teleop_arm.py`, `test_streamed_handoff.py`, `test_pose_ownership.py`, `test_pose_preview.py`, `test_streamed_reference_packet.py` |
| Stack lifecycle / ports | [stack.py](../../src/locomani/pipeline/stack.py), [ports.py](../../src/locomani/sim/ports.py), [stack/manager.py](../../src/locomani/stack/manager.py) | [stages.py](../../src/locomani/pipeline/stages.py); bottle/crate pilots share the seam | `test_stack_delivery.py`, `test_stack_start_failure.py`, `test_instance_ports.py` |
| Dataset fields / conversion | [schema.py](../../src/locomani/schema.py) → [convert_to_hf_format.py](../../src/locomani/pipeline/convert_to_hf_format.py) | [gr00t_sonic_2cam.py](../../configs/gr00t_sonic_2cam.py) | `test_schema_provenance.py`, `test_verification_tail.py` |
| Training / closed-loop evaluation | [run_learn_pilot.py](../../src/locomani/pipeline/run_learn_pilot.py), [eval_closed_loop.py](../../src/locomani/pipeline/eval_closed_loop.py) | [pilot_config.py](../../src/locomani/pipeline/pilot_config.py), [eval_config.py](../../src/locomani/pipeline/eval_config.py) | `test_runtime_consumers.py`, `test_eval_config_reaches_the_client.py`, `test_trace_round_join.py` |
| Task result / acceptance | [base.py](../../src/locomani/tasks/base.py), [execution_record.py](../../src/locomani/pipeline/execution_record.py) | `RoundResult.accepted(require_recording=...)` | `test_task_acceptance.py`, `test_execution_record.py` |
| Model / run provenance | [provenance.py](../../src/locomani/pipeline/provenance.py), [serve_policy.py](../../src/locomani/pipeline/serve_policy.py) | `config_snapshot.json`, `deploy_receipts.jsonl`, stage `commands.jsonl` | `test_schema_provenance.py`, `test_stack_delivery.py` |
| Offline tracking-response identification | [response_identification.py](../../src/locomani/pipeline/response_identification.py), [upper_ref.py](../../src/locomani/backends/sonic/upper_ref.py), [privileged_pub.py](../../src/locomani/sim/privileged_pub.py) | Episode sidecars and traces; `--run-root`, `--split`, `--out`. Asynchronous reference receive times; held-out recursive prediction gates model export | `test_response_identification.py`, `test_response_instrumentation.py` |
| CPU geometry fixtures | [geometry_fixture.py](../../tests/geometry_fixture.py) | [fixture README](../../tests/fixtures/geometry/README.md) | `test_grasp_point_ik.py`, `test_ready_grasp_policy.py` |
| Stack lifecycle (sim, deploy, exporter as supervised children; lock, state file, process-group stop, deploy/exporter restart, health) | [stack/manager.py](../../src/locomani/stack/manager.py), [stack/cli.py](../../src/locomani/stack/cli.py), [pipeline/stack.py](../../src/locomani/pipeline/stack.py) (`start_stack`/`stop_instance` bridge for the pilots) | `python -m locomani.stack.cli up|down|status|restart-deploy|restart-exporter --instance N`, `<session_root>/stack.json` | `test_stack_start_failure.py`, `test_stack_delivery.py`, `test_sonic_policy_default.py` |
| Run composition (task × embodiment × scene × backend × collect × train × eval) and the one CLI | [conf.py](../../src/locomani/conf.py), [cli.py](../../src/locomani/cli.py), [embodiment_config.py](../../src/locomani/embodiment_config.py), [scene_config.py](../../src/locomani/scene_config.py) | [configs/config.yaml](../../configs/config.yaml), `configs/<group>/*.yaml`, `configs/experiment/*.yaml` (`+experiment=<name>`), [infra plan](../handoff/infra-plan-20260916.md) | `test_conf.py` |
| Shared config serialization | [config_base.py](../../src/locomani/config_base.py) | [bottle/config.py](../../src/locomani/tasks/bottle/config.py) | `test_configs.py`, `test_review_2026_09_10.py` |
| Verification routing / navigation / architecture | [check_changes.py](../../scripts/check_changes.py), [lint_repository.py](../../scripts/lint_repository.py) | [Verification](../agents/session-workflow.md#verification), [standards](standards.md) | `test_check_workflow.py`, `test_repository_rules.py` |

## Current limitations

`locomani.cli` composes a run with Hydra and still dispatches to the old drivers (`run_learn_pilot`, `run_crate_pilot`) by dumping the composed configs into the run directory; the collection backend and the crate expert kind are hard-coded in those drivers until step 3 of the infra plan. Crate convert/train/eval are not bridged. ScaleBFM has an independent in-process runner. SONIC consumers
still read derived compatibility properties of ReachConfig; five historical strategy
switches migrate only at the version-1 JSON boundary. GymRuntime and VlaConfig carry production runtime choices explicitly. Legacy
RTC diagnostics retain scoped instrumentation hooks. The gym-options report covers
its parser; child receipts and delivery tests cover subprocess settings.

Next migrations: [ROADMAP](../ROADMAP.md). Launch commands: [runbook](runbook.md).
