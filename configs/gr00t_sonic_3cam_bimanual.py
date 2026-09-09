"""Modality config for the two-handed crate task: 3 cameras, both wrists, 66-D action.

Derived from gr00t_sonic_2cam.py; the three departures there stand, and
each of the three changes here undoes one of its *single-hand* measurements
because the crate task invalidates the measurement, not the reasoning.

1. video: + left_wrist
   2cam dropped left_wrist because over 23,103 bottle frames the left arm
   never moved and its camera was a static texture. In the crate task the
   left hand closes on its own rim bar; at the frame that decides the
   episode the bar fills the left-wrist view exactly as the bottle fills the
   right one. The per-hand alignment error is what each wrist camera is for.

2. state: 53 -> 60, adding the left wrist's pose
   left_wrist_pos + left_wrist_abs_quat are already declared in the dataset's
   modality.json (observation.eef_state 0:7); no re-conversion. Same argument
   as the right wrist in 2cam: FK of the robot's own encoders, available on
   hardware, not privileged.

3. action: 65 -> 66, motion_token + left_gripper + right_gripper
   The left jaw is now a real action. `left_gripper` slices component 0 of
   teleop.left_hand_joints the way right_gripper does -- one informative dim,
   not the 7-wide group whose other 6 columns are constants. It needs the
   converter to declare that key (tools/convert_to_hf_format.py MODALITY
   "action"); until then the dataset has only the 7-wide left_hand_joints.

Why the grippers stay two scalars and not one: single-hand data showed the
grip is bimodal (open/closed, 3% in between) and the model learns the discrete
state. Two hands make four states; the expert closes both in the same frame,
so (open,open) and (closed,closed) dominate and the mixed states appear only
in recovery demonstrations -- which is the point of recording them per hand.

Usage:
    --modality_config_path configs/gr00t_sonic_3cam_bimanual.py
"""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

_TAG = EmbodimentTag.UNITREE_G1_SONIC.value

_cfg = deepcopy(MODALITY_CONFIGS[_TAG])

_cfg["video"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=["ego_view", "left_wrist", "right_wrist"],
)

# Same layout as gr00t_sonic_2cam.py for 0..53, then the left wrist. The
# positional comments are the CONCATENATED state, not raw observation.state
# columns -- see 2cam for why they differ.
_cfg["state"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=[
        "left_leg",  # 0:6
        "right_leg",  # 6:12
        "waist",  # 12:15
        "left_arm",  # 15:22
        "right_arm",  # 22:29
        "left_hand",  # 29:36
        "right_hand",  # 36:43
        "projected_gravity",  # 43:46
        "right_wrist_pos",  # 46:49
        "right_wrist_abs_quat",  # 49:53
        "left_wrist_pos",  # 53:56
        "left_wrist_abs_quat",  # 56:60
    ],
)

_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
_cfg["action"] = ModalityConfig(
    delta_indices=list(range(40)),
    modality_keys=["motion_token", "left_gripper", "right_gripper"],
    action_configs=[_ABS, _ABS, _ABS],
)

MODALITY_CONFIGS[_TAG] = _cfg

_dims = {"state": 60, "action": 66}
print(
    f"[sonic_3cam_bimanual] {_TAG}: video={_cfg['video'].modality_keys} "
    f"state={_dims['state']} action={_dims['action']} "
    f"({_cfg['action'].modality_keys})"
)
