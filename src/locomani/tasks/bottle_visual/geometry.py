"""Reset geometry of the ScaleBFM bottle scene: where the bottle, the stand and the spawn go.

The SONIC gym's edge geometry (2026-09-08, docs/ledger/scalebfm.md): the bottle hugs its
nearest table edge, the robot approaches that edge face-on from 1.1-1.3 m with +-20 deg of yaw
error and +-0.20 m of lateral offset, and stands `stand_d` from the bottle. `sample_scene` is
the `legacy_edges_v1` profile of sampling.py; the current `yplus_center_v1` profile
(sampling.sample_reset) reuses the constants below with the same rng draw order.

The v3 expert that lived here (run_round, 187 lines) and its corner-approach scene were deleted
2026-09-10 -- its datasets and models are gone, the visual expert (round.py) replaced it.
"""

from __future__ import annotations

import numpy as np

STAND_D = (0.45, 0.55)  # pelvis-to-bottle distance band (measured 2026-09-08; the visual expert pins 0.375)
STAND_BACKSTEP = 0.06  # the body steps back this much when the arm rises during the walk-in; aim closer by it
STAND_LEFT = 0.15  # pelvis offset to the robot's left, so the bottle is in front of the right shoulder
SPAWN_LAT = 0.20  # lateral spawn offset band (m)
SPAWN_YAW_ERR = np.radians(20)
BOTTLE_JITTER = 0.08  # bottle jitter along the edge (m)
EDGE_INBOARD = (
    0.04,
    0.08,
)  # bottle centre this far in from its nearest edge (grasp_gym CAN_INBOARD); strata override it
SPAWN_EDGE = (1.1, 1.3)  # pelvis-to-edge distance at spawn (grasp_gym --spawn-edge 1.1 1.3)


def sample_scene(sim, rng, edge="auto", stand_d=STAND_D, inboard=EDGE_INBOARD):
    """One round's bottle, stand and spawn. Consumes the rng in a fixed order, so a replay can burn a
    round's draws by calling this and discarding the result. `edge` x-|x+|y-|y+ forces the edge
    (diagnostics); "auto" picks the edge the nominal bottle spot hugs.
    """
    bottle = sim.crate0.copy()
    d = float(rng.uniform(*stand_d))
    surf = sim.smeta["surfaces"]["A"]
    cx, cy = surf["xy"]
    hx, hy = surf["flat_half_extent"]
    nom = sim.crate0[:2]
    dists = {
        (0, -1): nom[0] - (cx - hx),
        (0, 1): (cx + hx) - nom[0],
        (1, -1): nom[1] - (cy - hy),
        (1, 1): (cy + hy) - nom[1],
    }
    ax, sgn = min(dists, key=dists.get)  # the edge the nominal spot hugs
    if edge != "auto":
        ax, sgn = "xy".index(edge[0]), (1 if edge[1] == "+" else -1)
    inboard_m = float(rng.uniform(*inboard))
    along = float(rng.uniform(-BOTTLE_JITTER, BOTTLE_JITTER))
    edge_coord = (cx, cy)[ax] + sgn * (hx, hy)[ax]
    bottle[ax] = edge_coord - sgn * inboard_m
    bottle[1 - ax] = nom[1 - ax] + along
    appr0 = np.zeros(2)
    appr0[ax] = -sgn  # inward normal of that edge
    left = np.array([-appr0[1], appr0[0]])
    stand_xy = bottle[:2] - appr0 * (d - STAND_BACKSTEP) + left * STAND_LEFT
    spawn_edge = float(rng.uniform(*SPAWN_EDGE))
    spawn_lat = float(rng.uniform(-SPAWN_LAT, SPAWN_LAT))
    edge_pt = bottle[:2] - appr0 * inboard_m  # the edge point in front of the bottle
    spawn_xy = edge_pt - appr0 * spawn_edge + left * spawn_lat
    meta = dict(
        scene="edge",
        edge=f"{'xy'[ax]}{'+' if sgn > 0 else '-'}",
        inboard=inboard_m,
        stand_d=d,
        stand_edge=d - STAND_BACKSTEP - inboard_m,
        spawn_edge=spawn_edge,
        spawn_lat=spawn_lat,
    )
    yaw0 = float(np.arctan2(*(bottle[:2] - spawn_xy)[::-1])) + float(rng.uniform(-SPAWN_YAW_ERR, SPAWN_YAW_ERR))
    meta.update(bottle_xy=[round(float(v), 3) for v in bottle[:2]], spawn_yaw_deg=float(np.degrees(yaw0)))
    return bottle, stand_xy, spawn_xy, yaw0, meta
