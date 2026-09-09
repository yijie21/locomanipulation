"""Modality config for the ScaleBFM action space WITH velocity columns: 2 cameras, 53-d state, 26-d action.

gr00t_bfm_2cam.py plus action.bfm_vel (7): pelvis target linear velocity (3) and yaw rate (1)
and right-wrist target linear velocity (3), all in the row's own pelvis frame, from
src/teleop_free/scalebfm/bfm_action.py. Without them a chunk of per-row pelvis-frame targets
cannot express walking speed (the plan leads the pelvis by a constant ~10 cm at any speed):
bfm_pilot_v3 trained on the 18-d action imitated the rows perfectly and never left the spawn.
The executor (eval_bfm_bottle.py --anchor live) re-anchors each row on the live pelvis and
builds ScaleBFM's six future frames from these velocities.

  bfm (18) | bfm_vel (7) | right_gripper (1)

Usage:
    --modality_config_path configs/gr00t_bfm_vel_2cam.py
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
    modality_keys=["ego_view", "right_wrist"],
)

_cfg["state"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=[
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
    ],
)

_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
_cfg["action"] = ModalityConfig(
    delta_indices=list(range(40)),
    modality_keys=["bfm", "bfm_vel", "right_gripper"],
    action_configs=[_ABS, _ABS, _ABS],
)

MODALITY_CONFIGS[_TAG] = _cfg

_dims = {"state": 53, "action": 26}
print(
    f"[bfm_vel_2cam] {_TAG}: video={_cfg['video'].modality_keys} "
    f"state={_dims['state']} action={_dims['action']} ({_cfg['action'].modality_keys})"
)
