"""The policy's observation from the robot's state message and the camera server's images.

Flat keys, the wire format of this repository's policy server (`locomani.pipeline.serve_policy` `get_action_raw`):
`state.<group>` (1, D) float32 for every group of the schema, `video.<camera>` (1, H, W, 3) uint8 and the prompt
under `annotation.human.task_description`. The groups are the dataset's (`locomani.schema`): the seven joint groups
of the 43-joint URDF order, `projected_gravity`, and the wrist poses from forward kinematics -- the same pinocchio
model and frames the exporter used to write `observation.eef_state`, so what the policy sees on the robot is what
it saw in training. The hand groups carry the Dex1 scalar (`real_robot.gripper.hand_state`).

Differences from NVIDIA's `prepare_observation_from_sensors`: three cameras under the dataset's own keys (no
`wrist_view` rename), the wrist poses, the hand groups, and no Dex3 left-hand coupling hack.
"""

from __future__ import annotations

import numpy as np

from locomani.schema import FIELD_WIDTHS, TaskSchema
from real_robot.gripper import dex1_state, hand_state

BODY_GROUPS = ("left_leg", "right_leg", "waist", "left_arm", "right_arm")  # read from the URDF-order q
HAND_GROUPS = {"left_hand": "left_hand_q", "right_hand": "right_hand_q"}
WRIST_GROUPS = {"left": ("left_wrist_pos", "left_wrist_abs_quat"), "right": ("right_wrist_pos", "right_wrist_abs_quat")}


def projected_gravity(base_quat) -> np.ndarray:
    """World gravity [0, 0, -1] in the body frame: rotated by the inverse of the base quaternion (w, x, y, z).
    Same definition as gear_sonic's `compute_projected_gravity`; identity orientation gives [0, 0, -1]."""
    w, x, y, z = (float(v) for v in np.asarray(base_quat, dtype=np.float64))
    # third row of R(q)^T applied to -e_z: g_body = -R^T e_z = -(R[2, :])
    r20 = 2.0 * (x * z - w * y)
    r21 = 2.0 * (y * z + w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return np.array([-r20, -r21, -r22], dtype=np.float64)


def wrist_pose(robot_model, side: str) -> tuple[np.ndarray, np.ndarray]:
    """(position, quaternion w-first) of one wrist frame after `cache_forward_kinematics`, as the exporter writes
    `observation.eef_state` (`run_data_exporter`, scipy `as_quat(scalar_first=True)`)."""
    from scipy.spatial.transform import Rotation as R

    placement = robot_model.frame_placement(robot_model.supplemental_info.hand_frame_names[side])
    return np.asarray(placement.translation[:3], dtype=np.float64), R.from_matrix(placement.rotation).as_quat(
        scalar_first=True
    )


def build_observation(
    state_msg: dict,
    images: dict,
    prompt: str,
    robot_model,
    schema: TaskSchema,
    *,
    gripper_units: str = "dex1",
    gravity=projected_gravity,
) -> dict | None:
    """The flat observation, or None while a camera the schema needs has not published yet.

    `state_msg` is the deploy's `g1_debug` message (`body_q` 29, `left_hand_q` / `right_hand_q`, `base_quat` w-first);
    `images` the camera server's `{camera: HxWx3 RGB uint8}`. `gravity` is the projected-gravity function, so the
    caller may hand in gear_sonic's own."""
    missing = [k for k in schema.video_keys if k not in images]
    if missing:
        return None
    body = np.asarray(state_msg["body_q"], dtype=np.float64)
    hands = {g: np.asarray(state_msg[k], dtype=np.float64) for g, k in HAND_GROUPS.items()}
    whole = np.asarray(
        robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=body,
            left_hand_actuated_joint_values=hands["left_hand"],
            right_hand_actuated_joint_values=hands["right_hand"],
        ),
        dtype=np.float64,
    )
    f32 = lambda a: np.asarray(a, dtype=np.float32).reshape(1, -1)  # noqa: E731
    obs = {}
    for group in schema.state_keys:
        if group in BODY_GROUPS:
            obs[f"state.{group}"] = f32(whole[robot_model.get_joint_group_indices(group)])
        elif group in HAND_GROUPS:
            obs[f"state.{group}"] = f32(
                hand_state(dex1_state(hands[group]), FIELD_WIDTHS["state"][group], gripper_units)
            )
        elif group == "projected_gravity":
            obs["state.projected_gravity"] = f32(gravity(state_msg["base_quat"]))
    wanted_sides = [side for side, groups in WRIST_GROUPS.items() if any(g in schema.state_keys for g in groups)]
    if wanted_sides:
        robot_model.cache_forward_kinematics(whole)
        for side in wanted_sides:
            pos, quat = wrist_pose(robot_model, side)
            pos_key, quat_key = WRIST_GROUPS[side]
            obs[f"state.{pos_key}"] = f32(pos)
            obs[f"state.{quat_key}"] = f32(quat)
    for key in schema.video_keys:
        obs[f"video.{key}"] = np.asarray(images[key])[None].astype(np.uint8)
    obs["annotation.human.task_description"] = prompt
    schema.validate_observation(obs)
    return obs
