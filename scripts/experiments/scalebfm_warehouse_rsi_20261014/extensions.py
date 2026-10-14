"""TODO 25.9.1: behaviour the loop's Engineer adds to the ScaleBFM warehouse round, switched on per candidate by
bfm_round.py's spec key `extensions: {name: params}` and off otherwise -- so an extension cannot change the incumbent
(the gate's sentinel re-runs the incumbent and would catch it).

Register a hook with `@hook("name", "reference")` or `@hook("name", "futures")`:
* reference(q, root, params, *, t, phase, sim, st8) -> (q, root): the expert's 29-joint reference and pelvis position,
  each tick of the ready and reach phases, before FK to ScaleBFM's keypoints;
* futures(fut, params, *, t, phase, sim, st8) -> fut: ScaleBFM's keypoint targets {body: (positions (F,3), wxyz
  quaternions (F,4))} each reach tick, after the pad servo.
`st8` is the round's state dict: hooks may keep their own state under a key named after themselves. Hooks must be
deterministic (no wall clock, no unseeded randomness).
"""

from __future__ import annotations

import numpy as np

HOOKS: dict[str, dict] = {}


def hook(name: str, point: str):
    if point not in ("reference", "futures"):
        raise ValueError(point)

    def register(fn):
        HOOKS.setdefault(name, {})[point] = fn
        return fn

    return register


@hook("pelvis_floor", "futures")
def pelvis_floor(fut, params, *, t, phase, sim, st8):
    """Phase 1 (2026-10-14): ScaleBFM's pelvis keypoint never below params["z"] (m, world) while the hand keypoints
    stay where the expert put them -- SONIC's delivery (its stream carries no pelvis height; it stood at ~0.75 m
    through the reach) without moving the hands, which the FK-coupled `reference: root_z_min` would."""
    if "pelvis" not in fut:
        return fut
    p, q = fut["pelvis"]
    p = p.copy()
    p[:, 2] = np.maximum(p[:, 2], float(params["z"]))
    return {**fut, "pelvis": (p, q)}
