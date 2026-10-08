"""Settle every free body of a robot-free scene and report which ones do not stay put (TODO 19.6.5).

An object placed at its ADT pose on a 2 cm slab, with Claude's mass and friction and a hull collider, should
not move. One that does (slides off a tilted top, pops out of an initial penetration, tips over, falls through)
cannot be a task object as built; the report says which and by how much. Runs on the sim venv's MuJoCo.
"""

from __future__ import annotations

import mujoco
import numpy as np

MOVE_MAX = 0.02  # m
TILT_MAX = 10.0  # deg
FLOOR_Z = -0.05  # below this the body fell through the floor


def _tilt_deg(q0, q1) -> float:
    """Angle between the body z axes of two (w, x, y, z) orientations."""
    z0 = np.zeros(9)
    z1 = np.zeros(9)
    mujoco.mju_quat2Mat(z0, np.asarray(q0, dtype=float))
    mujoco.mju_quat2Mat(z1, np.asarray(q1, dtype=float))
    c = float(np.clip(z0.reshape(3, 3)[:, 2] @ z1.reshape(3, 3)[:, 2], -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def settle(model: mujoco.MjModel, seconds: float = 2.0) -> dict:
    """Step `model` from its initial state; per free body: {moved_m, tilt_deg, final_z, penetration_m, ok}."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    start = {j: data.qpos[model.jnt_qposadr[j] : model.jnt_qposadr[j] + 7].copy() for j in free}
    deepest = {}
    for c in data.contact[: data.ncon]:
        for g in (c.geom1, c.geom2):
            b = model.geom_bodyid[g]
            deepest[b] = min(deepest.get(b, 0.0), float(c.dist))
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    out = {}
    for j in free:
        a = model.jnt_qposadr[j]
        q0, q1 = start[j], data.qpos[a : a + 7]
        body = model.jnt_bodyid[j]
        moved = float(np.linalg.norm(q1[:3] - q0[:3]))
        tilt = _tilt_deg(q0[3:], q1[3:])
        out[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)] = {
            "moved_m": round(moved, 4),
            "tilt_deg": round(tilt, 1),
            "final_z": round(float(q1[2]), 4),
            "penetration_m": round(-deepest.get(body, 0.0), 4),
            "ok": bool(moved < MOVE_MAX and tilt < TILT_MAX and q1[2] > FLOOR_Z),
        }
    return out
