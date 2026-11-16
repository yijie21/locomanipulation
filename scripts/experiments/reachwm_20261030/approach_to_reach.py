"""The reaches inside the walk-then-reach recordings as reach-model data (2026-11-03, after stance_wrist.py: reaches that
start right after a turn-in-place were missing from the reach model's data; on the 108 turned moves its position error
went 0.87 -> 1.27 cm while it predicted 0.30).

data_approach / data_approach0 segments start before the walk; each one's reach starts at k_reach (after the post-walk
hold), commanded in the body frame where the robot really stands (gen_reach.approach_segment). Rewritten in gen_reach's
reach format: k0 = k_reach, start = the true pelvis pose there (the W row), target = target_b, the wrist command
Rz(dyaw) Ry(pitch) in that heading frame (no extra wrist turn in these datasets), the plan the reach really got
(ref_*_post), walked. The step arrays (S, W, J, C) are kept whole, so data.derive reads the 3 frames before k_reach.

    python3 scripts/experiments/reachwm_20261030/approach_to_reach.py
Writes outputs/reachwm_20261030/data_approach_reach/all.npz and data_approach0_reach/all.npz.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("REACHWM_ROOT") or Path(__file__).resolve().parents[3] / "outputs" / "reachwm_20261030")


def rot_h6(dyaw: np.ndarray, pitch: np.ndarray) -> np.ndarray:
    """First two columns of Rz(dyaw) Ry(pitch), (N, 6)."""
    cz, sz, cp, sp = np.cos(dyaw), np.sin(dyaw), np.cos(pitch), np.sin(pitch)
    c0 = np.stack([cz * cp, sz * cp, -sp], -1)
    c1 = np.stack([-sz, cz, np.zeros_like(cz)], -1)
    return np.concatenate([c0, c1], -1).astype(np.float32)


def convert(name: str) -> None:
    z = dict(np.load(OUT / f"data_{name}" / "all.npz"))
    assert not np.abs(z["wrist_dev"]).any(), "an extra wrist turn: the command rotation needs it"
    n = z["n_steps"].astype(int)
    so = np.r_[0, np.cumsum(n)[:-1]]
    kr = z["k_reach"].astype(int)
    W = z["W"]
    rows = so + kr
    out = {k: z[k] for k in ("S", "W", "J", "C", "n_steps", "side", "dyaw", "pitch", "pelvis_z", "fell", "ok", "jaw_err_cm",
                             "att_err_deg", "ik_err_cm", "reach_s", "reset", "homed", "n_sidesteps", "sidestep", "wrist_dev")}  # fmt: skip
    out.update(k0=kr, walked=np.ones(len(n), bool), chain=np.zeros(len(n), int), target=z["target_b"].astype(np.float32),
               rot6=rot_h6(z["dyaw"].astype(np.float64), z["pitch"].astype(np.float64)),
               start=np.c_[W[rows, :3], W[rows, 3]].astype(np.float32),
               ref_q=z["ref_q_post"], ref_pel=z["ref_pel_post"], ref_s=z["ref_s_post"])  # fmt: skip
    d = OUT / f"data_{name}_reach"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "all.npz", **out)
    turn = np.degrees(np.abs(z["walk"][:, 2]))
    print(
        f"{name}: {len(n)} reaches -> {d / 'all.npz'}; walk turn median {np.median(turn):.0f} deg, no turn {int((turn == 0).sum())}"
    )


if __name__ == "__main__":
    for nm in sys.argv[1:] or ("approach", "approach0"):
        convert(nm)
    print("check: REACHWM_GOAL_REF=1 data.py check outputs/reachwm_20261030/data_approach_reach/all.npz", math.pi > 3)
