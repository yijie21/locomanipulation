# VLA action space and the exporter's planner gate

`third_party/` is a nested git repo the outer one does not track (`.gitignore:2`),
so these two edits live only in the working tree. `vla_action_space.patch` is
`git diff` from inside `third_party/GR00T-WholeBodyControl`; re-apply with

    cd third_party/GR00T-WholeBodyControl && git apply ../../patches/vla_action_space.patch

## 1. `run_data_exporter.py` — record planner fields in mode 3 too

The `teleop.planner_*` columns were gated on `stream_mode == 5`
(PLANNER_VR_3PT), written for the pico manager where 5 was the only planner
mode in use. The teleop-free pipeline runs the whole grasp in mode 3
(PLANNER_FROZEN_UPPER_BODY), so every grasp frame recorded `planner_mode = 0`
and `planner_height = -1`.

Measured before/after on a two-episode recording:

| column | gated on 5 | gated on (3, 5) |
|---|---|---|
| `planner_mode` values | `[0, 1]` | `[0, 1, 4]` — 4 is IDEL_SQUAT |
| `planner_height` values | `[-1.0]` | `[-1.0, 0.634]`, 656 frames crouching |
| `planner_movement` nonzero | 3.4 % | 20.1 % |

The crouch is commanded as `mode=IDEL_SQUAT` + `height=<pelvis target>`, so
the un-patched exporter dropped the entire squat from the recorded action.

## 2. `features_sonic_vla.py` — action is 66 dims, not 267

The action modality declared 267 dims. Measured on a recorded episode, **180
of them never changed in a single frame** (the SMPL slots alone are 135 dead
numbers without a Pico headset).

More to the point, the action is fixed by the wire protocol rather than by
taste. The deploy's pose protocol v4 is token-only streaming
(`zmq_endpoint_interface.hpp:82`, and `:678` — "Token-only mode - no motion
data, just tokens for the policy"; supplying motion alongside it is rejected):

    REQUIRED: token_state
    OPTIONAL: frame_index, left_hand_joints, right_hand_joints, body_quat_w

So a deployed VLA emits a token plus a gripper and nothing else has a channel
to travel on:

| component | dim | note |
|---|---|---|
| `motion_token` | 64 | the SONIC latent |
| `left_gripper` | 1 | slice 0 of the 7-wide Dex3-legacy column |
| `right_gripper` | 1 | dex1's two prismatic fingers always share one value |
| | **66** | 65 of which vary; the left gripper is constant today |

Everything the expert used to AUTHOR that token — planner mode / movement /
facing / speed / height — is upstream of the encoder and redundant with its
output: the crouch reaches the token as the k-planner's bent-leg reference
inside `motion_joint_positions_10frame_step5`, which is what encoder mode 0
reads. Those columns are still WRITTEN to the parquet, so using them as
auxiliary supervision later needs no re-recording; they are just not the
action.

`stream_mode` was dropped for a different reason: it is plumbing, not a
decision. Which ZMQ sub-mode a frame goes out in follows from the pipeline
stage, so the model would be regressing a discrete 3-or-5 label it has no
reason to choose.
