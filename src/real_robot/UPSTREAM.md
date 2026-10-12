# Provenance

The client (`inference.py`) follows NVIDIA's VLA inference runner from GR00T-WholeBodyControl function for function.

    upstream repo    https://github.com/NVlabs/GR00T-WholeBodyControl
    commit           a0732b6  (tag v1.1)
    source file      gear_sonic/scripts/run_vla_inference.py  (+ gear_sonic/utils/inference/vla_utils.py)
    read from        third_party/GR00T-WholeBodyControl  (the file itself has no local diffs at that commit)
    checked          2026-10-07: both files are identical on GitHub main

## What is vendored and what is imported

Only the client logic is rewritten here (`src/real_robot/`). Everything it stands on is imported at run time from the
`gear_sonic` package installed in SONIC's `.venv_sim` (`scripts/run_real_robot_inference.sh`), at the same commit:

| imported module | used for |
|---|---|
| `gear_sonic.camera.composed_camera.ComposedCameraClientSensor` | the camera server's merged image message |
| `gear_sonic.utils.data_collection.zmq_state_subscriber.ZMQStateSubscriber` | the deploy's `g1_debug` state (CONFLATE) |
| `gear_sonic.utils.data_collection.keyboard_subscriber.ZMQKeyboardSubscriber` | keyboard commands |
| `gear_sonic.utils.data_collection.transforms.compute_projected_gravity` | the `projected_gravity` state group |
| `gear_sonic.data.robot_model.instantiation.g1.instantiate_g1_robot_model` | the 43-joint model (the exporter's) |
| `gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message`, `build_command_message` | protocol v4 and start/stop |
| `gear_sonic.utils.inference.initial_poses.LATENT_INITIAL_MOTION_TOKEN` | the default initial token |

The GR00T side is this repository's vendored `src/gr00t` (Isaac-GR00T 376ba890, `src/gr00t/UPSTREAM.md`), served
by `real_robot.run_gr00t_server` (below); there is deliberately no second GR00T copy. From `locomani` itself the client
imports only pure-numpy modules: `schema` (field order), `backends.sonic.sonic_token` (codebook snap, gripper range)
and `backends.sonic.initial_pose.load_token_file`.

## Kept from upstream

Same layout and names: the tyro `InferenceConfig` (upstream's fields and defaults except where listed below),
`pack_latent_action_message`, `get_action_field`, `prepare_observation_from_sensors`,
`run_policy_inference_and_process`, `_inference_worker_loop`, `main(config)` with its closures
(`publish_initial_pose`, `blend_to_initial_pose`, `send_cpp_control_command`, `check_keyboard_input`) and
`_sleep_remaining`. In behaviour:

- Asynchronous inference thread; one action published per tick at `--action-publish-rate` (50 Hz); `Telemetry`
  timing, printed when a loop runs late or with `--verbose-timing`.
- Latency compensation, inference scheduling and prefix stripping are upstream's own functions, imported from
  `gear_sonic.utils.inference.vla_utils` (`calculate_latency_compensated_index`, `should_trigger_new_inference`,
  `concat_action`).
- The policy is called with Isaac-GR00T's `get_action(observation)`: nested `video` / `state` / `language`,
  (B, T, ...) with B = T = 1, and the reply is `(action, info)` with (B, T, D) actions.
- Chunks whose raw token leaves [-1.25, 1.25] are skipped; `task_progress` is dropped.
- Keyboard semantics `p k i t [ ] c s f`; the 1 s blend to the initial pose; while paused nothing is published
  and the loop polls at 5 Hz; protocol v4 `pose` messages with `token_state`, `frame_index`, `left_hand_joints`,
  `right_hand_joints`.

## Changed, and why (docs/handoff/real-robot-sim2real-plan-20261006.md, section 6)

- Three cameras under the dataset's keys (`right_wrist`, not `wrist_view`).
- The state is upstream's 46-d (43 joints + projected gravity, `locomani.schema.SONIC_STATE_KEYS`); the hand
  groups carry the Dex1 scalar; the Dex3 left-hand coupling hack is gone.
- Actions are token + one Dex1 gripper scalar per hand (upstream: token + two 7-d Dex3 hands); the scalar is
  broadcast to the 7-wide wire field. Tokens are snapped to the FSQ codebook.
  The chunk is converted once when it arrives (`chunk.ActionChunk`) instead of indexed in the raw dict every tick.
- The initial pose's hands are Dex1 open (5.5) or closed (0), broadcast; upstream sends Dex3 zeros or the Dex3 IK
  closed pose. The initial token can be a run's own file (`--init-token-file`); upstream always sends its own.
- `real_robot.policy_client.PolicyClient` instead of `gr00t.policy.server_client.PolicyClient`: SONIC's
  `.venv_sim` has no gr00t. Same wire format and methods, plus `identity`.
- An unreachable server or a served checkpoint whose field order differs from the schema stops the client before
  anything is published (upstream warns and carries on). `--no-identity-check` lifts the second. The gripper
  units are read from the checkpoint's statistics in `identity` (`--gripper-units auto`, `real_robot.statistics`);
  an explicit value that contradicts them stops the client too.
- Defaults: policy port 5590 (upstream 5550), the action socket binds `*` (upstream localhost), prompt
  "pick up the box". Added fields: `--schema`, `--gripper-units`, `--hand-wire`, `--snap-tokens`,
  `--identity-check`, `--init-token-file`.

## The policy server (`run_gr00t_server.py`)

Modelled on Isaac-GR00T's `gr00t/eval/run_gr00t_server.py` (https://github.com/NVIDIA/Isaac-GR00T, main; the same file
is vendored unchanged at `src/gr00t/eval/run_gr00t_server.py`). Kept: the tyro `ServerConfig` and its flag names,
`Gr00tPolicy` or `ReplayPolicy` (`--dataset-path`), the modality-config loading for replay, `PolicyServer` as a context
manager with its default endpoints. Changed: one extra endpoint (`identity`: file hashes, field order, the gripper
keys' training ranges), `--backbone-path` and
`--denoising-steps`, port 5590 and tag `unitree_g1_sonic` by default, no sim policy wrapper. The simulation's
`locomani.pipeline.serve_policy` (RTC, seeded denoising, run-directory lookup) is not used on the robot.
