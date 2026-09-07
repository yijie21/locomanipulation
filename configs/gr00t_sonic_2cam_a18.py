"""18-D action variant of configs/gr00t_sonic_2cam.py.

Same cameras and same 53-D state; only the ACTION changes, from the 64-D SONIC
motion token + gripper (65) to the 18-D command the expert actually issued:

    [right wrist pos(3) rotvec(3) grip(1) | left wrist pos(3) rotvec(3) grip(1)
     | base vx, vy, dyaw, dz]

Why this variant exists (docs/experiments.md, 2026-09-04). Measured on the same
12 episodes: the token's effective dimensionality is 11 of 64 (PCA, 95% var),
so 53 of its output directions carry combinations that never occur in data and
are unconstrained by the loss; and it is a staircase in time, with a median of
2 of 64 dims moving by exactly one FSQ step per frame. The 18-D action is 11 of
18 effective and continuous (median per-frame change 0.000 sd, p90 0.03). More
importantly the loss geometry matches the task: three of its dims ARE the palm
position in metres, whereas an MSE in token space has no metric relation to
end-effector error -- which is how a 0.048 training loss coexists with a 78%
knock-the-bottle-over rate.

Requires a dataset converted with `--a18` (convert_to_hf_format.py), i.e. one
carrying an `action.a18` column; the 65-D column is untouched, so one dataset
trains both variants.

    RUN=<name>_a18 DATA=... MODALITY=configs/gr00t_sonic_2cam_a18.py \
        bash scripts/train_bottle.sh
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
        # The comments are positions in the CONCATENATED state this config
        # builds, NOT columns of the raw observation.state. They differ:
        # meta/modality.json lays the raw vector out as left_hand 22:29 and
        # right_arm 29:36, i.e. the opposite order to the one listed here.
        # GR00T resolves each key by name, so the permutation is applied
        # identically at train and deploy and is harmless -- but reading these
        # numbers as raw columns is not. In the RAW vector the dex1 grippers
        # are at 26/27 (left) and 40/41 (right); 22/23 and 36/37 are
        # identically zero in every frame (verified on cover400: exactly 33 of
        # 43 columns are ever non-zero, and col 40 tracks the commanded right
        # grip at r=0.997). Anything that indexes observation.state directly
        # must use the raw layout, not this list.
        "left_leg",             # 0:6
        "right_leg",            # 6:12
        "waist",                # 12:15
        "left_arm",             # 15:22
        "right_arm",            # 22:29
        "left_hand",            # 29:36
        "right_hand",           # 36:43
        "projected_gravity",    # 43:46
        "right_wrist_pos",      # 46:49
        "right_wrist_abs_quat",  # 49:53
    ],
)

_ABS = ActionConfig(
    rep=ActionRepresentation.ABSOLUTE,
    type=ActionType.NON_EEF,
    format=ActionFormat.DEFAULT,
)
# One key, 18 dims. Left ABSOLUTE/NON_EEF like the token action: the wrist
# targets are already in the PELVIS frame, so GR00T's EEF machinery (which
# would re-frame them) must not touch them.
_cfg["action"] = ModalityConfig(
    delta_indices=list(range(40)),
    modality_keys=["a18"],
    action_configs=[_ABS],
)

MODALITY_CONFIGS[_TAG] = _cfg
print(f"[sonic_2cam_a18] {_TAG}: video={_cfg['video'].modality_keys} "
      f"state=53 action=18 ({_cfg['action'].modality_keys})")
