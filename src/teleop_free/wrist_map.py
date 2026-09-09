"""What to command so the palm ends up where it is wanted.

The deploy does not track wrist targets one-to-one, and the constants that used
to stand in for that fact were wrong by a quarter of a metre. Measured over 64
commands spanning the region the grasp uses (teleop_free/diag_map.py):

    the constants in use before   mean 24.6 cm   max 47.4 cm
    affine fit                    mean  6.5 cm
    quadratic fit                 mean  3.5 cm   (4.3 cm cross-validated)

The gains were not merely mis-tuned. z had the wrong SIGN — the model predicted
-0.07 m where the palm actually sits at +0.068 — and y was assumed to track
one-to-one when its real gain is about 0.6 with a strong dependence on x. The
reach's bias integrator was quietly making up the whole 25 cm through a second
of dead time, which is why it either crawled or rang.

Two models are stored, both quadratic in their three inputs and both fitted
from the same paired samples: `Wf` predicts where the palm lands given a
command, `Wg` predicts the command for a wanted palm position. What matters is
the composition — command g(want), then see where f puts it — which is 2.8 cm
in-sample and 3.8 cm held out, against 30.2 cm for what was there before.

The remaining 3.8 cm is left to the closed loop. That is the right division of
labour: diag_repeat.py measured the mapping's repeatability at 2-7 mm, so the
residual is unmodelled structure a richer fit could take further, but a seed
this close means the integrator trims rather than searches.

Re-measure with diag_map.py after any change to the checkpoint, the controller
config, or the observation space — these numbers are properties of a specific
deploy, not of the robot.
"""

from __future__ import annotations

import os

import numpy as np
import paths

MAP_PATH = f"{paths.REPO}/outputs/scene/wrist_map.npz"

_WG: np.ndarray | None = None
_WF: np.ndarray | None = None


def _quad(v: np.ndarray) -> np.ndarray:
    """The feature vector both fits use: linear, square, cross, constant."""
    v = np.atleast_2d(np.asarray(v, dtype=float))
    x, y, z = v[:, 0], v[:, 1], v[:, 2]

    return np.stack([x, y, z, x * x, y * y, z * z, x * y, x * z, y * z, np.ones(len(v))], axis=1)


def _load():
    global _WG, _WF
    if _WG is None:
        if not os.path.exists(MAP_PATH):
            raise FileNotFoundError(f"{MAP_PATH} missing — run teleop_free/diag_map.py")
        d = np.load(MAP_PATH)
        if "Wg" not in d:
            raise KeyError(f"{MAP_PATH} predates the quadratic fit — re-run teleop_free/diag_map.py")
        _WG, _WF = d["Wg"], d["Wf"]

    return _WG, _WF


def available() -> bool:
    try:
        _load()
    except (FileNotFoundError, KeyError):
        return False

    return True


def _jacobian(c: np.ndarray, wf: np.ndarray) -> np.ndarray:
    """d(palm)/d(command) for the quadratic forward model."""
    x, y, z = c
    d = np.array(
        [
            [1.0, 0.0, 0.0, 2 * x, 0.0, 0.0, y, z, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 2 * y, 0.0, x, 0.0, z, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 2 * z, 0.0, x, y, 0.0],
        ]
    )

    return (d @ wf).T  # rows: palm axes, cols: command axes


def command_for(palm_wanted, box_lo=None, box_hi=None, iters: int = 12) -> np.ndarray:
    """The wrist command whose settled palm position is `palm_wanted`.

    Both are in the pelvis frame, which is the frame the wire format uses.

    Gauss-Newton on the FORWARD model rather than a fitted inverse. Fitting
    palm -> command directly is the obvious thing and it is worse: that fit
    carries its own 7 cm residual, in command space, where the forward model
    then amplifies it — asking for a palm at y = -0.30 produced a command of
    -0.64, hard against the box floor, and a palm 20 cm from the request. The
    request was well inside the reachable set; the inverse fit simply is not
    accurate enough to be used open-loop.

    Inverting f numerically optimises the quantity that actually matters — the
    composition, f(command) against the request — and stays consistent with the
    forward model by construction. The fitted inverse is still useful as the
    starting guess, which is why it is stored.

    Damped, and it keeps the best iterate rather than the last. Undamped
    Gauss-Newton on a quadratic diverges on a minority of requests — over the
    64 measured palms it gave a p95 of 1.1 cm and a worst case of 68, the mean
    wrecked by a handful of runaways. Backtracking on the step and returning
    the best point seen makes the result never worse than the starting guess,
    which is the fitted inverse, which is already better than what this
    replaces.
    """
    wg, wf = _load()
    want = np.asarray(palm_wanted, dtype=float).reshape(3)
    c = (_quad(want) @ wg)[0]
    if box_lo is not None:
        c = np.clip(c, box_lo, box_hi)
    best_c = c.copy()
    best_e = float(np.linalg.norm((_quad(c) @ wf)[0] - want))

    for _ in range(iters):
        err = (_quad(c) @ wf)[0] - want
        if float(np.linalg.norm(err)) < 1e-4:
            break
        step = np.linalg.lstsq(_jacobian(c, wf), err, rcond=None)[0]
        moved = False
        for scale in (1.0, 0.5, 0.25, 0.1):
            trial = c - step * scale
            if box_lo is not None:
                trial = np.clip(trial, box_lo, box_hi)
            e = float(np.linalg.norm((_quad(trial) @ wf)[0] - want))
            if e < best_e:
                best_c, best_e, c, moved = trial.copy(), e, trial, True
                break
        if not moved:
            break

    return best_c


def palm_for(command) -> np.ndarray:
    """Where the palm settles for a given command — the forward direction.

    Useful for checking a command before sending it, and for the offline
    reachability work, which otherwise has to assume the command IS the pose.
    """
    _, wf = _load()

    return (_quad(command) @ wf)[0]


__all__ = ["command_for", "palm_for", "available", "MAP_PATH"]
