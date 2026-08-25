"""The SONIC action wire format: the FSQ codebook, and 65 <-> protocol v4.

One module because these two conversions must agree in three places -- the
open-loop metrics, the closed-loop client, and anything that inspects a
checkpoint's output. A snap that differs between eval and deployment makes the
eval a measurement of something we do not run.

THE CODEBOOK

`action.motion_token` is not a continuous latent. It is the output of a Finite
Scalar Quantizer inside the SONIC encoder, and the quantizer is part of the
trained graph, sitting BEFORE the decoder's input. Read straight off
policy/release/model_encoder.onnx, whose nodes are literally named
/quantizer, /quantizer_1, /quantizer_2 (3 groups x 32 dims = our 64):

    z   = tanh(x) * 15.5155      # Constant_2, = (L-1)/2 for L=32
    z   = z - 0.5                # Constant_3
    zq  = Round(z)               # -> integers
    out = zq / 16                # Constant_4

(The Sub/Add pair around the Round is the straight-through estimator; exported
for inference it is just a hard round.)

So every legal token component is n/16 for an integer n in [-16, 15]: 32 levels
per dim, step 0.0625, range [-1.0, 0.9375]. The decoder has never seen anything
else -- structurally cannot have. A predicted value between grid points is a
codeword that does not exist, the same way a non-codebook vector is meaningless
to a VQ-VAE decoder.

That is why snapping is not a cosmetic post-process and not a way to flatter a
metric: it projects the model's output back onto the set of values the
interface can actually carry. It uses only the fixed 32-level grid -- known a
priori, identical for every episode, nothing from the labels and nothing from
the test set. Snapping to the nearest GROUND-TRUTH token would be cheating;
restricting the grid to the 20 levels this dataset happens to use would be
cheating with the test set. Neither is done here.

The grid is coarse relative to the signal: measured over the bottle dataset,
per-dim std is 0.08-0.15 against a step of 0.0625, so a dim uses only 2-5 grid
points. "Nearly right" is therefore not a small error here, which is also why
the eval reports a snapped exact-match rate next to the MSE -- in a space this
discrete, a hit rate reads and an MSE does not.

THE WIRE FORMAT

The deploy's pose protocol v4 carries left_hand_joints and right_hand_joints as
7 wide each, but dex1 is a parallel jaw and this task is right-handed: over
23,103 frames the left hand never moves at all (std exactly 0 in all 7) and the
right hand is rank 1 (components 0 and 1 are the same number, 2..6 identically
0). The model therefore predicts one gripper scalar, and expand_hands() puts it
back on the wire.
"""

from __future__ import annotations

import numpy as np

# FSQ levels per dimension, from model_encoder.onnx (Constant_4 = 16 with
# L = 32 levels, so the integer index runs -L/2 .. L/2-1).
FSQ_LEVELS = 32
FSQ_STEP = 1.0 / 16.0
FSQ_MIN_INDEX = -FSQ_LEVELS // 2      # -16  -> -1.0
FSQ_MAX_INDEX = FSQ_LEVELS // 2 - 1   #  15  ->  0.9375
TOKEN_DIM = 64

# The left hand's frozen pose, read off the dataset (float32 0.0245 in the two
# jaw components). Sent verbatim so the deployed robot holds the same open left
# hand every training episode had.
LEFT_HAND_CONST = np.array(
    [0.02449999935925007, 0.02449999935925007, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
# Observed command range of the right jaw. The negative end is not a bug: the
# scripted grasp commands past contact to load the jaw against the bottle.
GRIPPER_MIN, GRIPPER_MAX = -0.0200000014, 0.0244999994


def snap(token, *, clip_range: bool = True):
    """Project a predicted token onto the FSQ codebook.

    Args:
        token: array-like, last axis of length TOKEN_DIM (any leading shape).
        clip_range: also clamp to the codebook's own [-1.0, 0.9375]. Leave on;
            off is for measuring how far outside the model wanders.

    Returns:
        float32 array of the same shape, every element an exact multiple of
        1/16 inside the legal range.
    """
    t = np.asarray(token, dtype=np.float64)
    if t.shape[-1] != TOKEN_DIM:
        raise ValueError(f"token last axis must be {TOKEN_DIM}, got {t.shape[-1]}")
    idx = np.rint(t / FSQ_STEP)
    if clip_range:
        idx = np.clip(idx, FSQ_MIN_INDEX, FSQ_MAX_INDEX)
    return (idx * FSQ_STEP).astype(np.float32)


def token_index(token):
    """The integer codebook index per dimension, in [-16, 15].

    What a hit-rate or a confusion matrix should be computed on: comparing
    integers avoids asking whether two floats are the same grid point.
    """
    return np.clip(
        np.rint(np.asarray(token, dtype=np.float64) / FSQ_STEP),
        FSQ_MIN_INDEX, FSQ_MAX_INDEX,
    ).astype(np.int16)


def expand_hands(gripper):
    """One predicted gripper scalar -> the two 7-wide protocol v4 fields.

    Mirrors the scalar onto jaw components 0 and 1 (which the dataset shows are
    always the same number) and leaves 2..6 at zero, exactly as recorded.
    """
    g = float(np.clip(gripper, GRIPPER_MIN, GRIPPER_MAX))
    right = np.zeros(7, dtype=np.float32)
    right[0] = right[1] = g
    return LEFT_HAND_CONST.copy(), right


def split_action(action):
    """A 65-wide model action -> (token[64], gripper scalar), snapped.

    The single place the 65-dim layout of configs/gr00t_sonic_2cam.py is
    decoded, so a change to the action order breaks here rather than silently
    somewhere downstream.
    """
    a = np.asarray(action, dtype=np.float64)
    if a.shape[-1] != TOKEN_DIM + 1:
        raise ValueError(f"action last axis must be {TOKEN_DIM + 1}, got {a.shape[-1]}")
    return snap(a[..., :TOKEN_DIM]), a[..., TOKEN_DIM]
