"""Modality config for the two-handed crate task: 3 cameras (ego + both wrists), 46-D state, 66-D action.

Derived from gr00t_sonic_2cam.py; its two departures stand, and each change
here undoes one of its *single-hand* measurements because the crate task
invalidates the measurement, not the reasoning.

1. video: + left_wrist
   2cam dropped left_wrist because over 23,103 bottle frames the left arm
   never moved and its camera was a static texture. In the crate task the
   left hand closes on its own rim bar; at the frame that decides the
   episode the bar fills the left-wrist view exactly as the bottle fills the
   right one. The per-hand alignment error is what each wrist camera is for.

2. state: the built-in 46, as in 2cam (2026-10-07, DECISIONS)
   Until 2026-10-07 this selected both wrist poses (60). They are FK of the
   joint angles already in state and were removed to match NVIDIA's SONIC +
   GR00T recipe; the dataset's modality.json still declares them
   (observation.eef_state), so restoring them needs no re-conversion.

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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
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
from locomani.schema import SONIC_CRATE  # noqa: E402

_TAG = EmbodimentTag.UNITREE_G1_SONIC.value

_cfg = deepcopy(MODALITY_CONFIGS[_TAG])

_cfg["video"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=list(SONIC_CRATE.video_keys),
)

# Same layout as gr00t_sonic_2cam.py: the built-in 0..45 byte for byte.
_cfg["state"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=list(SONIC_CRATE.state_keys),
)

_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
_cfg["action"] = ModalityConfig(
    delta_indices=list(range(40)),
    modality_keys=list(SONIC_CRATE.action_keys),
    action_configs=[_ABS, _ABS, _ABS],
)

MODALITY_CONFIGS[_TAG] = _cfg

_dims = {"state": 46, "action": 66}
print(
    f"[sonic_3cam_bimanual] {_TAG}: video={_cfg['video'].modality_keys} "
    f"state={_dims['state']} action={_dims['action']} "
    f"({_cfg['action'].modality_keys})"
)
