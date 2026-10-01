"""TODO 3.14.1: fit the phase brake's band from diag_brake trials that braked on the marker (no guard).

    python3 scripts/experiments/fit_phase_band_20261001.py outputs/diagnostics/brake_055_phasecal_20261001/results.json

band = how far ahead of the pelvis-at-brake the robot's frontmost low-body point gets after IDLE
     = frontmost_max_past_marker + marker - brake_fwd                 (m, along the walk heading)
model: band = c0 + a cos(2 pi phase) + b sin(2 pi phase), phase = brake_phase (time since the last landing / step period).
Prints c0, a, b, the leave-one-out residual sd and worst case, and the margin that would have kept every trial
behind the line; `--phase-brake c0,a,b --margin M` is what diag_brake and `visual.guard_phase_band` take.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def design(phase):
    return np.c_[np.ones_like(phase), np.cos(2 * np.pi * phase), np.sin(2 * np.pi * phase)]


def main() -> None:
    d = json.load(open(sys.argv[1]))
    rows = [r for r in d["rows"] if r.get("fell") is None and r.get("brake_phase") is not None]
    phase = np.array([r["brake_phase"] for r in rows])
    band = np.array([r["frontmost_max_past_marker"] + d["marker"] - r["brake_fwd"] for r in rows])
    carry = np.array([r["carry"] for r in rows])
    X = design(phase)
    coef = np.linalg.lstsq(X, band, rcond=None)[0]
    loo = []
    for i in range(len(rows)):
        keep = np.arange(len(rows)) != i
        c = np.linalg.lstsq(X[keep], band[keep], rcond=None)[0]
        loo.append(band[i] - X[i] @ c)
    loo = np.array(loo)
    print(
        json.dumps(
            dict(
                trials=len(rows),
                falls=sum(1 for r in d["rows"] if r.get("fell")),
                band_mean_cm=round(100 * band.mean(), 1),
                band_sd_cm=round(100 * band.std(), 1),
                carry_mean_cm=round(100 * carry.mean(), 1),
                coef_m=[round(float(c), 4) for c in coef],
                loo_sd_cm=round(100 * loo.std(), 1),
                loo_worst_cm=round(100 * loo.max(), 1),
                margin_for_no_crossing_cm=round(100 * max(0.0, loo.max()), 1),
                phase_brake_arg=",".join(f"{c:.4f}" for c in coef),
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
