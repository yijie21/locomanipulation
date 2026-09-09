"""The token the deploy holds while the VLA's first chunk is computed.

The rationale below describes the historical arms-down datasets. Ready-hand
collection uses an explicit TF_INIT_TOKEN_FILE, measured and held separately;
it does not replace the legacy default for older checkpoints.

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

import json
import os
import time
from pathlib import Path

import numpy as np

# Medoid of 194 settled-stand tokens in datasets/g1_locomani_ramp200_train.
# Mean L1 to the other 193: 1.11 (the worst member scores 11.46).
OURS_INITIAL_MOTION_TOKEN = np.array(
    [
        -0.2500,
        0.0625,
        -0.1250,
        0.3125,
        0.1250,
        -0.3750,
        0.3750,
        -0.0625,
        0.1875,
        -0.1250,
        0.1875,
        -0.1250,
        -0.4375,
        -0.2500,
        -0.3125,
        -0.1250,
        -0.1875,
        -0.1875,
        -0.1875,
        -0.2500,
        0.3750,
        0.3125,
        0.0625,
        -0.1875,
        0.0000,
        0.1250,
        0.1875,
        -0.4375,
        0.2500,
        0.2500,
        0.1875,
        -0.1875,
        0.3750,
        -0.0625,
        -0.0625,
        -0.1250,
        0.0625,
        -0.3750,
        0.2500,
        -0.0625,
        0.0000,
        -0.4375,
        0.0000,
        0.0000,
        -0.2500,
        -0.0625,
        0.3125,
        0.0000,
        0.0625,
        0.4375,
        0.2500,
        0.2500,
        0.0000,
        0.3750,
        0.3125,
        0.0000,
        -0.1250,
        0.5625,
        0.0000,
        -0.4375,
        0.1250,
        0.0000,
        0.0000,
        0.0000,
    ],
    dtype=np.float32,
)


def load_token_file(filename) -> np.ndarray:
    token = np.asarray(json.loads(Path(filename).read_text()), dtype=np.float32)
    if token.shape != (64,) or not np.isfinite(token).all() or np.max(abs(token)) > 1.25:
        raise ValueError("Initial token file must contain 64 finite values within [-1.25, 1.25]")
    return token


def initial_token() -> np.ndarray:
    """Legacy default, official token, or an explicitly validated pose file.

    A file is opt-in: a new collection posture must not silently change the
    initialization used by checkpoints trained on the old arms-down data.
    """
    filename = os.environ.get("TF_INIT_TOKEN_FILE")
    if filename:
        return load_token_file(filename)
    if os.environ.get("TF_INIT_TOKEN", "ours") == "official":
        from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN

        return np.asarray(LATENT_INITIAL_MOTION_TOKEN, dtype=np.float32)

    return OURS_INITIAL_MOTION_TOKEN


def prepare_token_pose(p, expected_arm_q, hold_seconds=3.0) -> bool:
    """Initialize a token-only ready posture before recording/visibility checks.

    Sends one fixed token with open hands. No target pose, arm IK or navigation
    enters the action. The existing ready-arm tolerance is checked afterwards.
    Deployment continues holding this same token during its usual prelude.
    """
    from tools.sonic_token import GRIPPER_MAX, expand_hands

    if not os.environ.get("TF_INIT_TOKEN_FILE"):
        raise ValueError("Token-pose preparation requires an explicit initial token file")
    expected = np.asarray(expected_arm_q, dtype=float)
    if expected.shape != (14,) or not np.isfinite(expected).all():
        raise ValueError("Expected ready pose must contain 14 finite arm joint values")
    if not np.isfinite(hold_seconds) or hold_seconds <= 0:
        raise ValueError("Initial pose hold must be positive and finite")
    token = initial_token()
    left, right = expand_hands(GRIPPER_MAX)
    # Match the deploy's normal prime-before-enter ordering.
    for _ in range(30):
        p.cmd.token_frame(token, right_hand=right, left_hand=left, frame_index=0)
        time.sleep(0.02)
    p.cmd.enter_streamed()
    end = time.monotonic() + hold_seconds
    while time.monotonic() < end:
        p.cmd.token_frame(token, right_hand=right, left_hand=left, frame_index=0)
        if p.obs()["fallen"]:
            return False
        time.sleep(0.04)
    state = p.obs()
    actual = np.asarray(state["joints"], dtype=float)[15:29]
    return not state["fallen"] and actual.shape == expected.shape and bool(np.max(abs(actual - expected)) < 0.30)
