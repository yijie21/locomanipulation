"""The official unitree_g1_sonic modality config, with the two wrist cameras added.

The built-in config (Isaac-GR00T gr00t/configs/data/embodiment_configs.py)
declares video.modality_keys = ["ego_view"], so a session's wrist streams are
recorded by our exporter and then never read by training. ETHRC-humanoid's
public gr00t-g1-sonic-3cam checkpoint trains this same embodiment on all three,
which is what this file restores.

state and action are copied from the built-in config untouched, so the 46/78
layout and the slot-11 projector are exactly as NVIDIA declares them - only the
camera list changes.

Only usable on a dataset that actually has all three streams: the public
reference set (hojjunekim/g1_sonic_orange_to_plate_locomani) has ego_view only.

register_modality_config() asserts the tag is not already registered, and
unitree_g1_sonic is, so this rewrites the entry in place instead.

Usage:
    --modality_config_path configs/gr00t_sonic_3cam.py
"""

from copy import deepcopy

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig

_TAG = EmbodimentTag.UNITREE_G1_SONIC.value

_cfg = deepcopy(MODALITY_CONFIGS[_TAG])
_cfg["video"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=["ego_view", "left_wrist", "right_wrist"],
)
MODALITY_CONFIGS[_TAG] = _cfg

print(f"[sonic_3cam] {_TAG} video keys -> {_cfg['video'].modality_keys}")
