"""Dataset field ownership shared by conversion, task descriptors, training and deployment.

SONIC_MODALITY describes raw storage slices. TaskSchema selects their ordered names;
raw exporter order and model concatenation order are deliberately distinct.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSchema:
    modality_config_path: str
    video_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    action_keys: tuple[str, ...]

    def dimension(self, kind: str) -> int:
        keys = getattr(self, kind + "_keys")
        return sum(FIELD_WIDTHS[kind][key] for key in keys)

    def validate_observation(self, values: dict) -> None:
        for kind in ("video", "state"):
            for key in getattr(self, kind + "_keys"):
                name = f"{kind}.{key}"
                if name not in values:
                    raise ValueError(f"missing observation field {name}")
                if kind == "state":
                    if values[name].shape[-1] != FIELD_WIDTHS[kind][key]:
                        raise ValueError(f"wrong observation width for {name}: {values[name].shape}")


SONIC_MODALITY = {
    "state": {
        "left_leg": {"start": 0, "end": 6},
        "right_leg": {"start": 6, "end": 12},
        "waist": {"start": 12, "end": 15},
        "left_arm": {"start": 15, "end": 22},
        "left_hand": {"start": 22, "end": 29},
        "right_arm": {"start": 29, "end": 36},
        "right_hand": {"start": 36, "end": 43},
        "left_wrist_pos": {"start": 0, "end": 3, "original_key": "observation.eef_state"},
        "left_wrist_abs_quat": {
            "start": 3,
            "end": 7,
            "original_key": "observation.eef_state",
            "rotation_type": "quaternion",
        },
        "right_wrist_pos": {"start": 7, "end": 10, "original_key": "observation.eef_state"},
        "right_wrist_abs_quat": {
            "start": 10,
            "end": 14,
            "original_key": "observation.eef_state",
            "rotation_type": "quaternion",
        },
        "root_orientation": {
            "start": 0,
            "end": 4,
            "original_key": "observation.root_orientation",
            "rotation_type": "quaternion",
        },
        "projected_gravity": {"start": 0, "end": 3, "original_key": "observation.projected_gravity"},
        "cpp_rotation_offset": {
            "start": 0,
            "end": 4,
            "original_key": "observation.cpp_rotation_offset",
            "rotation_type": "quaternion",
        },
        "init_base_quat": {
            "start": 0,
            "end": 4,
            "original_key": "observation.init_base_quat",
            "rotation_type": "quaternion",
        },
    },
    "action": {
        "motion_token": {"start": 0, "end": 64, "original_key": "action.motion_token"},
        "left_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.left_hand_joints"},
        "right_hand_joints": {"start": 0, "end": 7, "original_key": "teleop.right_hand_joints"},
        "right_gripper": {"start": 0, "end": 1, "original_key": "teleop.right_hand_joints"},
        "left_gripper": {"start": 0, "end": 1, "original_key": "teleop.left_hand_joints"},
    },
    "video": {},
    "annotation": {"human.task_description": {"original_key": "task_index"}},
}

SONIC_BOTTLE = TaskSchema(
    modality_config_path="configs/gr00t_sonic_2cam.py",
    video_keys=("ego_view", "right_wrist"),
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
    ),
    action_keys=("motion_token", "right_gripper"),
)

SONIC_CRATE = TaskSchema(
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

VISUAL_BOTTLE = TaskSchema(
    modality_config_path="configs/gr00t_bfm_visual_2cam.py",
    video_keys=("ego_view", "right_wrist"),
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
        "robot_velocity",
        "previous_action",
    ),
    action_keys=("visual_command",),
)

FIELD_WIDTHS = {
    kind: {key: field["end"] - field["start"] for key, field in SONIC_MODALITY[kind].items()}
    for kind in ("state", "action")
}
FIELD_WIDTHS["state"].update(robot_velocity=9, previous_action=29)
FIELD_WIDTHS["action"]["visual_command"] = 29
VISUAL_STATE_DIMS = {key: FIELD_WIDTHS["state"][key] for key in VISUAL_BOTTLE.state_keys}
