"""Modality config for the ScaleBFM action space: 2 cameras, 53-d state, 19-d action.

Same observation as gr00t_sonic_2cam.py (ego_view + right_wrist, the 53-d state
with pelvis-frame right-wrist FK and projected gravity) so a policy trained here
sees exactly what the SONIC-era policies saw. What changes is the ACTION:

  bfm (18)        pelvis target (pos 3 + rot6d 6) and right wrist target
                  (pos 3 + rot6d 6), both in the CURRENT pelvis frame -- the
                  ScaleBFM mode-2 keypoint command that was actually sent
                  (feed-forward bias + proportional term included), recorded
                  by src/teleop_free/scalebfm/record_bottle.py.
  right_gripper (1)  the dex1 jaw target (GRIP_OPEN ... GRIP_CLOSED), a scalar
                  so the open/closed bimodality stays visible, as in 2cam.

Absolute, non-EEF, default format: plain per-dimension normalization. rot6d is
already a smooth parameterization; no delta/relative transform is wanted
because the target is re-expressed in the pelvis frame every frame.

Usage:
    --modality_config_path configs/gr00t_bfm_2cam.py
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

# Identical to gr00t_sonic_2cam.py (see there for the raw-vs-concatenated
# column caveat).
_cfg["state"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=[
        "left_leg", "right_leg", "waist", "left_arm", "right_arm",
        "left_hand", "right_hand", "projected_gravity",
        "right_wrist_pos", "right_wrist_abs_quat",
    ],
)

_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
_cfg["action"] = ModalityConfig(
    delta_indices=list(range(40)),
    modality_keys=["bfm", "right_gripper"],
    action_configs=[_ABS, _ABS],
)

MODALITY_CONFIGS[_TAG] = _cfg

_dims = {"state": 53, "action": 19}
print(f"[bfm_2cam] {_TAG}: video={_cfg['video'].modality_keys} "
      f"state={_dims['state']} action={_dims['action']} ({_cfg['action'].modality_keys})")
