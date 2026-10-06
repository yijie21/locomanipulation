# Provenance

The client is a restructuring of NVIDIA's VLA inference runner from GR00T-WholeBodyControl.

    upstream repo    https://github.com/NVlabs/GR00T-WholeBodyControl
    commit           a0732b6  (tag v1.1)
    source file      gear_sonic/scripts/run_vla_inference.py  (+ gear_sonic/utils/inference/vla_utils.py)
    read from        third_party/GR00T-WholeBodyControl  (the file itself has no local diffs at that commit)

## What is vendored and what is imported

Only the client logic is rewritten here (`real_robot/`). Everything it stands on is imported at run time from the
`gear_sonic` package installed in SONIC's `.venv_sim` (`run_inference.sh`), at the same commit:

| imported module | used for |
|---|---|
| `gear_sonic.camera.composed_camera.ComposedCameraClientSensor` | the camera server's merged image message |
| `gear_sonic.utils.data_collection.zmq_state_subscriber.ZMQStateSubscriber` | the deploy's `g1_debug` state (CONFLATE) |
| `gear_sonic.utils.data_collection.keyboard_subscriber.ZMQKeyboardSubscriber` | keyboard commands |
| `gear_sonic.utils.data_collection.transforms.compute_projected_gravity` | the `projected_gravity` state group |
| `gear_sonic.data.robot_model.instantiation.g1.instantiate_g1_robot_model` | the 43-joint model and wrist FK (the exporter's) |
| `gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message`, `build_command_message` | protocol v4 and start/stop |
| `gear_sonic.utils.inference.initial_poses.LATENT_INITIAL_MOTION_TOKEN` | the default initial token |

The GR00T side is this repository's vendored `src/gr00t` (Isaac-GR00T 376ba890, `src/gr00t/UPSTREAM.md`), served
by `locomani.pipeline.serve_policy`; there is deliberately no second GR00T copy. From `locomani` itself the client
imports only pure-numpy modules: `schema` (field order), `backends.sonic.sonic_token` (codebook snap, gripper range)
and `backends.sonic.initial_pose.load_token_file`.

## Kept from upstream, unchanged in behaviour

- Asynchronous inference thread; one action published per tick at `--action-publish-rate` (50 Hz).
- New chunk entered at `round(latency * publish_rate)`, clipped to the horizon (`chunk.latency_index`).
- Inference requested every `1 / --rate` seconds (2.5 Hz), never while one is running (`chunk.should_infer`).
- Chunks whose raw token leaves [-1.25, 1.25] are skipped.
- Keyboard semantics `p k i t [ ]`; the 1 s blend to the initial pose; protocol v4 `pose` messages with
  `token_state`, `frame_index`, `left_hand_joints`, `right_hand_joints`.

## Changed, and why (docs/handoff/real-robot-sim2real-plan-20261006.md, section 6)

- Three cameras under the dataset's keys (`right_wrist`, not `wrist_view`).
- The 60-d state of `locomani.schema.SONIC_CRATE`: wrist poses from FK (upstream sends 43 + 3 only); hand groups
  carry the Dex1 scalar; the Dex3 left-hand coupling hack is gone.
- Actions are token + one Dex1 gripper scalar per hand (upstream: token + two 7-d Dex3 hands); the scalar is
  broadcast to the 7-wide wire field. Tokens are snapped to the FSQ codebook.
- Talks to `serve_policy.py`'s flat `get_action_raw` and checks its `identity` against the schema.
- `argparse` instead of `tyro`; the policy server runs from this repository, not Isaac-GR00T's CLI.
