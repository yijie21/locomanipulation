"""The unitree_g1_sonic modality config this project actually trains on.

Three deliberate departures from Isaac-GR00T's built-in slot-11 config
(gr00t/configs/data/embodiment_configs.py:67). Each is a measurement, not a
preference -- the numbers below were taken over 23,103 frames of the bottle
dataset.

1. video: ego_view -> ego_view + right_wrist
   The built-in config reads ego_view only, so the wrist streams our exporter
   records were never seen by training. At the moment that decides the episode
   -- the jaws closing, consistently 54-58% of the way in -- the bottle is ~15
   px in the 256-crop ego view and part-occluded by the arm, while it fills the
   right-wrist frame between both visible fingers. That is the alignment error
   the close phase fails on, at two orders of magnitude more pixels.
   left_wrist is NOT added: the left arm never moves here, and its camera shows
   the table and its own idle gripper for the whole episode. Frames 85 apart
   are indistinguishable. It would be a third of the vision budget spent on a
   near-static texture.

2. state: 46 -> 53, adding the right wrist's pose
   right_wrist_pos + right_wrist_abs_quat are already declared in the dataset's
   modality.json (they slice observation.eef_state), so this needs no
   re-conversion. They carry no NEW information -- the exporter builds them by
   forward kinematics from the same joint angles already in state
   (run_data_exporter.py:656) -- but the error that decides this task is
   Cartesian, and handing the network an exact FK saves it from learning one.
   Being FK of the robot's own encoders, it is available on hardware; unlike
   observation.water_bottle_pose it is not privileged.
   NOT added, on purpose: root_orientation, init_base_quat, cpp_rotation_offset.
   All three are world-frame quaternions carrying absolute yaw, and episodes
   spawn at random yaw -- they would let the model index the spawn distribution
   instead of looking. projected_gravity, already in the built-in config, is
   the yaw-invariant way to get roll/pitch.

3. action: 78 -> 65, motion_token + one gripper scalar
   Measured over the whole dataset: teleop.left_hand_joints has std EXACTLY 0
   in all 7 components -- the left hand never moves. teleop.right_hand_joints
   is rank 1: components 0 and 1 are the same number (a mirrored parallel jaw,
   -0.020..0.024, negative meaning squeeze past contact) and 2..6 are
   identically 0. So the true action is 64 + 1, and the other 13 columns are
   constants.
   Keeping them is numerically safe -- state_action_processor.py:169 clamps a
   zero range to 1e-8 and clip_outliers pins the result -- but they would be
   predicted near-perfectly while carrying no information, diluting every MSE
   we report. The eval reads a gripper metric separately for the same reason:
   1 informative dim pooled with 64 disappears.

The dataset's modality.json still declares the 7-wide left_hand_joints and
right_hand_joints groups, so reverting to the 78-dim action is a config edit
with no re-conversion.

Usage:
    --modality_config_path configs/gr00t_sonic_2cam.py

register_modality_config() asserts the tag is not already registered, and
unitree_g1_sonic is, so this rewrites the entry in place.
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
from locomani.schema import SONIC_BOTTLE  # noqa: E402

_TAG = EmbodimentTag.UNITREE_G1_SONIC.value

_cfg = deepcopy(MODALITY_CONFIGS[_TAG])

_cfg["video"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=list(SONIC_BOTTLE.video_keys),
)

# Order is the layout: the state vector is these groups concatenated in this
# order, so the first eight entries keep the built-in 0..45 layout byte for
# byte and the wrist pose extends it to 53.
_cfg["state"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=list(SONIC_BOTTLE.state_keys),
)

_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
# action_configs is positional -- one per modality_key, same order.
_cfg["action"] = ModalityConfig(
    delta_indices=list(range(40)),
    modality_keys=list(SONIC_BOTTLE.action_keys),
    action_configs=[_ABS, _ABS],
)

MODALITY_CONFIGS[_TAG] = _cfg

_dims = {"state": 53, "action": 65}
print(
    f"[sonic_2cam] {_TAG}: video={_cfg['video'].modality_keys} "
    f"state={_dims['state']} action={_dims['action']} "
    f"({_cfg['action'].modality_keys})"
)
