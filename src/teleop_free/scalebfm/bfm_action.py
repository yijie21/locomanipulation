"""Velocity columns for the ScaleBFM action space.

action.bfm (18) stores, per 25 Hz row, the pelvis and right-wrist keypoint targets in THAT
row's own pelvis frame. That drops the walking speed: while the expert walks at 0.4 m/s the
plan leads the pelvis by a near-constant ~10 cm, so a 40-row chunk reads "+0.10 ahead" in
every row and a policy that imitates it perfectly can only creep (bfm_pilot_v3, 2026-09-08:
0/12 with the robot never leaving the spawn). ScaleBFM takes its speed from the spacing of
its six future frames, which the expert filled from plan(t + j*DT).

These 7 columns put the speed back, computed offline from columns every recording already
has (the recorded root pose per row and action.bfm):

  0:3  pelvis target linear velocity, in the row's pelvis frame   [m/s]
  3    pelvis target yaw rate                                       [rad/s]
  4:7  right-wrist target linear velocity, in the row's pelvis frame [m/s]

An executor re-anchors each chunk row on the LIVE pelvis pose and extrapolates the future
frames with these velocities (eval_bfm_bottle.py --anchor live).
"""
import numpy as np

import mujoco


def mat(q):
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def rot6d_to_mat(d6):
    a1, a2 = np.asarray(d6[:3], float), np.asarray(d6[3:6], float)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-9)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def _yaw(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def world_targets(act, root_pos, root_q):
    """Per-row pelvis/wrist targets in WORLD (n,3), (n,3,3), (n,3) from the row-frame action."""
    n = len(act)
    pel_p = np.zeros((n, 3)); pel_R = np.zeros((n, 3, 3)); wr_p = np.zeros((n, 3))
    for r in range(n):
        R = mat(root_q[r]); p = np.asarray(root_pos[r], float)
        pel_p[r] = p + R @ act[r, 0:3]; pel_R[r] = R @ rot6d_to_mat(act[r, 3:9]); wr_p[r] = p + R @ act[r, 9:12]
    return pel_p, pel_R, wr_p


def velocity_columns(act, root_pos, root_q, fps=25.0):
    """(n,7) float32: see the module docstring. Central differences of the world-frame plan,
    re-expressed in each row's pelvis frame; one-sided at the ends."""
    act = np.asarray(act, float); n = len(act)
    pel_p, pel_R, wr_p = world_targets(act, root_pos, root_q)
    yaw = np.unwrap(np.array([_yaw(R) for R in pel_R]))
    out = np.zeros((n, 7), np.float32)
    if n < 2:
        return out
    dt = 1.0 / fps
    v_pel = np.gradient(pel_p, dt, axis=0); v_wr = np.gradient(wr_p, dt, axis=0); wz = np.gradient(yaw, dt)
    for r in range(n):
        Rt = mat(root_q[r]).T
        out[r, 0:3] = Rt @ v_pel[r]; out[r, 3] = wz[r]; out[r, 4:7] = Rt @ v_wr[r]
    return out
