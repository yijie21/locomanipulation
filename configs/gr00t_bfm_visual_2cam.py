"""Exploratory visual-expert contract: two RGB times/views, state91, native action29.

Video history is -0.4 s and now at 25 Hz. The previous executed action is part
of proprioceptive history; object truth and phase labels are never modalities.
Freeze resolved training settings only after CPU qualification and replay.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from locomani.schema import VISUAL_BOTTLE  # noqa: E402

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

_TAG = EmbodimentTag.UNITREE_G1_SONIC.value
_cfg = deepcopy(MODALITY_CONFIGS[_TAG])
_cfg['video'] = ModalityConfig(delta_indices=[-10, 0], modality_keys=list(VISUAL_BOTTLE.video_keys))
_cfg['state'] = ModalityConfig(delta_indices=[0], modality_keys=list(VISUAL_BOTTLE.state_keys))
_cfg['action'] = ModalityConfig(delta_indices=list(range(40)), modality_keys=list(VISUAL_BOTTLE.action_keys),
    action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)])
MODALITY_CONFIGS[_TAG] = _cfg
