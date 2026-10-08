"""One action chunk from the policy, in the form the publish loop replays it.

NVIDIA's client (`run_vla_inference.py`) keeps the policy's action dict and indexes it every publish tick; latency
compensation and inference scheduling are its own `vla_utils` functions, which `real_robot.inference` imports.
What differs is what a chunk holds, this repository's action format: the 64-wide SONIC token, snapped to the FSQ
codebook (`locomani.backends.sonic.sonic_token`), and one gripper scalar per hand, converted to Dex1 units once
per chunk instead of once per tick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from locomani.backends.sonic.sonic_token import TOKEN_DIM, snap
from real_robot.gripper import to_dex1

TOKEN_BOUND = 1.25  # NVIDIA's client drops a chunk whose raw token leaves this bound (the codebook ends at 1.0)


def action_field(out: dict, key: str) -> np.ndarray:
    """A field of the policy server's reply, with or without the `action.` prefix."""
    for name in (f"action.{key}", key):
        if name in out:
            return np.asarray(out[name])
    raise KeyError(f"the policy reply has no {key!r}; it carries {sorted(out)}")


@dataclass(frozen=True)
class ActionChunk:
    token: np.ndarray  # (T, 64) float32, on the codebook
    left: np.ndarray  # (T,) Dex1 units
    right: np.ndarray  # (T,) Dex1 units

    @classmethod
    def from_policy(cls, out: dict, *, gripper_units: str, snap_tokens: bool = True) -> ActionChunk:
        """The policy's action, prefixed or not (`motion_token` (T, 64) or (1, T, 64), `left_gripper` /
        `right_gripper` (T,), (T, 1) or (1, T, 1)) as one chunk. Raises ValueError on a token outside the bound, which
        the caller skips the way NVIDIA's client does."""
        token = action_field(out, "motion_token").astype(np.float64)
        if token.ndim == 3:
            token = token[0]
        if token.ndim != 2 or token.shape[1] != TOKEN_DIM:
            raise ValueError(f"motion_token must be (T, {TOKEN_DIM}), got {token.shape}")
        if np.abs(token).max() > TOKEN_BOUND:
            raise ValueError(f"motion_token leaves [-{TOKEN_BOUND}, {TOKEN_BOUND}] (max |t| {np.abs(token).max():.3f})")
        hands = []
        for side in ("left", "right"):
            g = action_field(out, f"{side}_gripper").astype(np.float64).reshape(-1)
            if g.shape[0] != token.shape[0]:
                raise ValueError(f"{side}_gripper has {g.shape[0]} steps, the token {token.shape[0]}")
            hands.append(to_dex1(g, gripper_units).astype(np.float32))
        return cls((snap(token) if snap_tokens else token).astype(np.float32), hands[0], hands[1])

    def __len__(self) -> int:
        return int(self.token.shape[0])

    def frame(self, index: int) -> tuple[np.ndarray, float, float]:
        """(token, left Dex1, right Dex1) at `index`, held at the last action past the horizon."""
        i = int(np.clip(index, 0, len(self) - 1))
        return self.token[i], float(self.left[i]), float(self.right[i])
