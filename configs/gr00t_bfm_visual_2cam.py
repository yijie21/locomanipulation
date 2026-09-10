"""Exploratory visual-expert contract: two RGB times/views, state91, native action29.

Video history is -0.4 s and now at 25 Hz. The previous executed action is part
of proprioceptive history; object truth and phase labels are never modalities.
Freeze resolved training settings only after CPU qualification and replay.
"""
from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

_TAG = EmbodimentTag.UNITREE_G1_SONIC.value
_cfg = deepcopy(MODALITY_CONFIGS[_TAG])
_cfg['video'] = ModalityConfig(delta_indices=[-10, 0], modality_keys=['ego_view', 'right_wrist'])
_cfg['state'] = ModalityConfig(delta_indices=[0], modality_keys=[
    'left_leg', 'right_leg', 'waist', 'left_arm', 'right_arm', 'left_hand', 'right_hand',
    'projected_gravity', 'right_wrist_pos', 'right_wrist_abs_quat', 'robot_velocity', 'previous_action'])
_cfg['action'] = ModalityConfig(delta_indices=list(range(40)), modality_keys=['visual_command'],
    action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)])
MODALITY_CONFIGS[_TAG] = _cfg
