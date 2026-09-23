# VLA action space and the exporter's planner gate

`third_party/` is a nested git repo the outer one does not track (`.gitignore:2`),
so these two edits live only in the working tree. `vla_action_space.patch` is
`git diff` from inside `third_party/GR00T-WholeBodyControl`; re-apply with

    cd third_party/GR00T-WholeBodyControl \
      && git apply --include='gear_sonic/data/*' ../../patches/vla_action_space.patch \
      && git apply ../../patches/exporter_priv_seq.patch

The exporter hunks in this patch are superseded by `exporter_priv_seq.patch`
(a newer cumulative diff of `run_data_exporter.py` that contains them plus the
per-row priv_seq sidecar), hence the `--include`; applying this patch whole on
top of that one fails with `patch does not apply`. Check with
`git apply --check --reverse` on the same two commands.

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

## 2. `features_sonic_vla.py` — action is 78 dims, not 267

The action modality declared 267 dims. Measured on a recorded episode, **180
of them never changed in a single frame** (the SMPL slots alone are 135 dead
numbers without a Pico headset).

More to the point, the action is fixed by the wire protocol rather than by
taste. The deploy's pose protocol v4 is token-only streaming
(`zmq_endpoint_interface.hpp:82`, and `:678` — "Token-only mode - no motion
data, just tokens for the policy"; supplying motion alongside it is rejected):

    REQUIRED: token_state
    OPTIONAL: frame_index, left_hand_joints, right_hand_joints, body_quat_w

So a deployed VLA emits a token plus the hands and nothing else has a channel
to travel on:

| component | dim | note |
|---|---|---|
| `motion_token` | 64 | the SONIC latent |
| `left_hand_joints` | 7 | constant here — the gym is right-handed throughout |
| `right_hand_joints` | 7 | dex1 is a parallel jaw: components 0 and 1 move together (−0.020..0.024), 2..6 constant |
| | **78** | matches the action space the SONIC inference client expects |

**The group names and the width are load-bearing, not cosmetic.**
Isaac-GR00T's `unitree_g1_sonic` config
(`gr00t/configs/data/embodiment_configs.py`) selects action groups **by name** —
exactly `motion_token`, `left_hand_joints`, `right_hand_joints`. A name it
cannot find produces only

    Warning: Joint group 'left_hand_joints' not found in action modality.
    Available groups: [...]

and training then runs to completion on a 64-dim action with both hands
silently dropped. An earlier revision of this patch narrowed these to 1-dim
`left_gripper` / `right_gripper` — correct about dex1's kinematics, and exactly
the rename that triggers the silent drop.

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

## 3. Privileged pose columns follow the SCENE, not a hardcoded name

`observation.<key>_pose` is written by `run_data_exporter._add_object_poses`
and declared by `features_sonic_vla`. Those were two independent hardcoded
lists -- the writer defaulted `TF_POSE_OBJECTS` to `"can,plate"`, the schema
declared `observation.can_pose` / `observation.plate_pose` -- and disagreement
between them is silent in BOTH directions: a column the schema does not
declare is dropped, and one it declares but nobody writes stays all zeros.

A scene whose target is not called `can` therefore recorded 1008 rows of
`can_pose = 0.0` with nothing logged, and every downstream check read that as
"the object never moved". `check_dataset.py`'s whole grasp-causality pass is
built on that column.

Both sides now come from one `_pose_objects()` helper reading `TF_POSE_OBJECTS`,
the writer warns once per name the privileged stream does not carry, and
`start_stack.sh` defaults the variable from the booted scene's own
`scene_meta.json` so the warning should never be needed. Verified on a
bottle-only scene: `observation.water_bottle_pose`, 1133 rows, 693 distinct,
x 0.750 -> 0.259 as the robot walks in and z -0.120 -> +0.321 as it lifts.
