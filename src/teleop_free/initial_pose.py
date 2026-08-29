"""The token the deploy holds while the VLA's first chunk is computed.

NVIDIA ships one (`gear_sonic/utils/inference/initial_poses.py`,
LATENT_INITIAL_MOTION_TOKEN) and its own docstring says what the problem is:

    "CHECKPOINT-SPECIFIC: this value must be updated if the SONIC checkpoint
    changes ... this token will produce a different (and likely incorrect) pose
    in the new checkpoint's latent space."

Our SONIC checkpoint is theirs, so it decodes to a valid pose -- but a valid
pose is not the same as OUR pose. Theirs holds the arms forward and bent with
the torso pitched; every episode we trained on stands with the arms down. The
handover therefore started outside the training distribution and snapped into
it over the 1 s blend, which is the forward lean and the sudden straightening
visible in every third-person reel.

MEASURED, with sonic_decoder's validated joint_delta_deg (its joint_targets
absolute values are explicitly not validated, and an earlier pass of this
analysis that used them drew the wrong conclusion):

    mean joint command vs 12 sampled settled-stand tokens from our dataset
      this token (medoid of ours)   3.38 deg
      NVIDIA LATENT_INITIAL_...    23.02 deg      6.8x further
    the two against each other      23.75 deg mean, 133.8 deg max

WHY A MEDOID AND NOT A MEAN. Three synthesised candidates were tried first and
all three decoded FURTHER from our stand than NVIDIA's did:

  * per-dim median of the standing tokens -- the latent is not linear, so the
    median of tokens is not the token of the median pose (it decoded to a
    161 deg shoulder)
  * sonic_encoder's _stand_window() on DEFAULT_ANGLES
  * the same window on our dataset's measured mean stand

The encoder->decoder round trip is not the identity, so synthesis cannot be
trusted here. A medoid needs no round trip: it is an ACTUAL token our own stack
emitted while standing still, picked as the member with the smallest total L1
distance to the other 193.

Regenerate (194 settled-stand frames, planner mode IDLE, whole body below
0.05 rad/s, first 4 s of each episode):
    see docs/evaluation.md, "初始姿态"
"""

import os

import numpy as np

# Medoid of 194 settled-stand tokens in datasets/g1_locomani_ramp200_train.
# Mean L1 to the other 193: 1.11 (the worst member scores 11.46).
OURS_INITIAL_MOTION_TOKEN = np.array([
    -0.2500,  0.0625, -0.1250,  0.3125,  0.1250, -0.3750,  0.3750,
    -0.0625,  0.1875, -0.1250,  0.1875, -0.1250, -0.4375, -0.2500,
    -0.3125, -0.1250, -0.1875, -0.1875, -0.1875, -0.2500,  0.3750,
     0.3125,  0.0625, -0.1875,  0.0000,  0.1250,  0.1875, -0.4375,
     0.2500,  0.2500,  0.1875, -0.1875,  0.3750, -0.0625, -0.0625,
    -0.1250,  0.0625, -0.3750,  0.2500, -0.0625,  0.0000, -0.4375,
     0.0000,  0.0000, -0.2500, -0.0625,  0.3125,  0.0000,  0.0625,
     0.4375,  0.2500,  0.2500,  0.0000,  0.3750,  0.3125,  0.0000,
    -0.1250,  0.5625,  0.0000, -0.4375,  0.1250,  0.0000,  0.0000,
     0.0000,
], dtype=np.float32)


def initial_token() -> np.ndarray:
    """Ours by default; TF_INIT_TOKEN=official falls back to NVIDIA's."""
    if os.environ.get("TF_INIT_TOKEN", "ours") == "official":
        from gear_sonic.utils.inference.initial_poses import (
            LATENT_INITIAL_MOTION_TOKEN)

        return np.asarray(LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32)

    return OURS_INITIAL_MOTION_TOKEN
