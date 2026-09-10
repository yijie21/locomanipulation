"""ScaleBFM single-hand bottle grasp expert (image-to-action design, 2026-09-08).

    CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl $VENV_SIM_PY src/teleop_free/scalebfm/run_bottle.py \
        --rounds 10 --seed 1 --video --out outputs/videos/scalebfm_bottle

Every command is a function of the bottle pose and the robot's own visible state:
  * stand point   = bottle - d * approach + 0.15 * left   (d ~ U(0.45, 0.55): the band
                    where the feed-forward+proportional hand lands within 1.5 cm and the
                    head camera still sees the bottle, bottle_probe.py)
  * walk          = pelvis reference along the straight line, trapezoid 0.5 m/s, hands at
                    the sides; the REACH STARTS while still walking (last 0.5 m), one
                    min-jerk curve to the pre-grasp point (grasp - 0.10 along approach)
  * entry         = straight line 2 s to the grasp point (bottle centre - 2 cm), level palm
                    + 13 deg nose-up feed-forward, one global position bias + proportional
                    overdrive on the measured pad error (no integrator, no hidden state)
  * close cue     = pad centre within 1.5 cm of the grasp point across/height and 2 cm
                    along (what the wrist camera shows: bottle between the pads)
  * lift + carry  = +0.10 m in 1.5 s, then 0.15 m toward the body while descending 0.05 m
                    (2 s), hold 1 s. Success: bottle >= 8 cm above its start at the end,
                    upright (< 30 deg), never knocked before the close.
No hover hold, no squat command, no pelvis servo, no re-approach. The torso lean is
emergent. Static-frame fraction is reported (the summariser reads the manifest).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.runtime import (
    DT,
    GRIP_CLOSED,
    GRIP_OPEN,
    OFF,
    PELVIS_Z,
    SETTLE_S,
    WALK_ACC,
    WALK_V,
    Policy,
    Refs,
    Sim,
    mat,
    rot_y,
    rot_z,
    smoothstep,
    wrap,
    wrist_from_pad,
    yaw_of,
)

R.SCENE = os.environ.get("SCENE_DIR", os.path.join(R.ROOT, "outputs/scene/sim"))
R.OBJ = "water_bottle"

STAND_D = (0.45, 0.55)  # pelvis distance to the bottle (measured band; BFM_STAND_D="lo,hi" overrides for sweeps)
if os.environ.get("BFM_STAND_D"):
    STAND_D = tuple(float(v) for v in os.environ["BFM_STAND_D"].split(","))
STAND_BACKSTEP = (
    0.06  # the body steps back this much when the arm rises during the walk-in (debug rounds); aim closer by it
)
STAND_LEFT = 0.15  # pelvis offset to the robot's left, so the bottle is in front of the right shoulder
SPAWN_BACK = (0.9, 1.3)  # behind the stand
SPAWN_LAT = 0.20
SPAWN_YAW_ERR = np.radians(20)
BOTTLE_JITTER = 0.08  # bottle placement jitter (x, y), stays >= 10 cm inboard of the edge
REACH_START_DIST = float(
    os.environ.get("BFM_REACH_START", 0.5)
)  # start the reach when the pelvis reference has this far to go (0 = after arrival)
REACH_MIN_S = float(os.environ.get("BFM_REACH_MIN_S", 2.5))
GRIP_BELOW_CENTRE = 0.02
PRE_GRASP_BACK = 0.10
ENTRY_S = 2.0
ENTRY_TIMEOUT = 4.0
CLOSE_TOL_ACROSS, CLOSE_TOL_ALONG = 0.015, 0.02
CLOSE_S, CLOSE_HOLD_S = 0.6, 0.3
LIFT_H, LIFT_S = 0.10, 1.5
CARRY_BACK, CARRY_DOWN, CARRY_S, CARRY_HOLD_S = (
    0.15,
    0.0,
    2.0,
    1.0,
)  # no lowering: the bottle rides with the hand, lowering just gave the 8 cm rule away
LIFT_OK, TILT_OK_DEG = 0.08, 30.0
BIAS = np.array(
    [0.100, -0.030, 0.028]
)  # bottle_probe open-loop mean (command = ref + BIAS); y widened by the +1.6 cm residual of pilot 1
# BIAS was measured on the +x approach and is a BODY-frame offset (ahead, right, up): it is rotated by
# the stand yaw before use (BIAS_w below). Applied as a world constant it sent the hand 10 cm sideways
# on the +-y approaches of the edge geometry (2026-09-08: 'bottle pushed' 3/3, stand err 13 cm).
PITCH_FF_DEG = 13.0
KP, CAP = 1.0, 0.05
GRIP_CLIP = float(os.environ.get("GRIP_CLIP", 0.6))  # x 250 N clamp on the gripper PD (0.6 = 150 N)
EGO_EVERY = 25


SCENE_MODE = os.environ.get("BFM_SCENE", "edge")  # edge = the SONIC gym's geometry (default); corner = v3/v4 legacy
EDGE_INBOARD = (0.04, 0.08)  # bottle centre this far in from its nearest edge (grasp_gym CAN_INBOARD)
SPAWN_EDGE = (1.1, 1.3)  # pelvis-to-edge distance at spawn (grasp_gym --spawn-edge 1.1 1.3)


def sample_scene(sim: Sim, rng):
    """One round's bottle, stand and spawn. Consumes the rng in a fixed order, so a replay can burn a
    round's draws by calling this and discarding the result.

    edge (default): the SONIC gym's geometry -- the bottle hugs its nearest table edge (4-8 cm inboard,
    jittered along the edge), the robot approaches that edge face-on from 1.1-1.3 m with +-20 deg of
    yaw error and +-0.20 m lateral offset, and stands STAND_D from the bottle. corner (BFM_SCENE=corner):
    the v3/v4 pilots' geometry -- the bottle at its nominal spot 34 cm in from the -x edge and 12 cm from
    the +y edge, always approached along +x, i.e. from the table corner with a 0.45-0.49 m reach; kept
    only to reproduce those datasets.
    """
    bottle = sim.crate0.copy()
    d = float(rng.uniform(*STAND_D))
    if SCENE_MODE == "corner":
        bottle[:2] += rng.uniform(-BOTTLE_JITTER, BOTTLE_JITTER, 2)
        appr0 = np.array([1.0, 0.0])
        left = np.array([-appr0[1], appr0[0]])
        stand_xy = bottle[:2] - appr0 * (d - STAND_BACKSTEP) + left * STAND_LEFT
        spawn_back = float(rng.uniform(*SPAWN_BACK))
        spawn_lat = float(rng.uniform(-SPAWN_LAT, SPAWN_LAT))
        spawn_xy = stand_xy - appr0 * spawn_back + left * spawn_lat
        meta = dict(scene="corner", stand_d=d, spawn_back=spawn_back, spawn_lat=spawn_lat)
    else:
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
        force = os.environ.get("BFM_EDGE", "auto")  # x-|x+|y-|y+ overrides (diagnostics)
        if force != "auto":
            ax, sgn = "xy".index(force[0]), (1 if force[1] == "+" else -1)
        inboard = float(rng.uniform(*EDGE_INBOARD))
        along = float(rng.uniform(-BOTTLE_JITTER, BOTTLE_JITTER))
        edge_coord = (cx, cy)[ax] + sgn * (hx, hy)[ax]
        bottle[ax] = edge_coord - sgn * inboard
        bottle[1 - ax] = nom[1 - ax] + along
        appr0 = np.zeros(2)
        appr0[ax] = -sgn  # inward normal of that edge
        left = np.array([-appr0[1], appr0[0]])
        stand_xy = bottle[:2] - appr0 * (d - STAND_BACKSTEP) + left * STAND_LEFT
        spawn_edge = float(rng.uniform(*SPAWN_EDGE))
        spawn_lat = float(rng.uniform(-SPAWN_LAT, SPAWN_LAT))
        edge_pt = bottle[:2] - appr0 * inboard  # the edge point in front of the bottle
        spawn_xy = edge_pt - appr0 * spawn_edge + left * spawn_lat
        meta = dict(
            scene="edge",
            edge=f"{'xy'[ax]}{'+' if sgn > 0 else '-'}",
            inboard=inboard,
            stand_d=d,
            stand_edge=d - STAND_BACKSTEP - inboard,
            spawn_edge=spawn_edge,
            spawn_lat=spawn_lat,
        )
    yaw0 = float(np.arctan2(*(bottle[:2] - spawn_xy)[::-1])) + float(rng.uniform(-SPAWN_YAW_ERR, SPAWN_YAW_ERR))
    meta.update(bottle_xy=[round(float(v), 3) for v in bottle[:2]], spawn_yaw_deg=float(np.degrees(yaw0)))
    return bottle, stand_xy, spawn_xy, yaw0, meta
