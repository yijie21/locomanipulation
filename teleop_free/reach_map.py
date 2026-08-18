"""The graspable-region map: where can the hands GRASP, in the pelvis frame.

Navigation needs a stand-position criterion, and the scattered scalar
thresholds it uses today (GRIP_X band, REACH_XY_TOL, aim()'s shoulder
distances) are hand-tuned shadows of one object: the set R of pelvis-frame
points where some arm can put the palm WITH a grasp-quality orientation.
This module computes R once, offline, and exposes a lookup.

Method: forward Monte-Carlo. Sample arm joint configurations uniformly,
FK them on the scene model (ArmIK.fk — the same model the sim runs), keep
samples whose palm orientation could actually take an upright object:

    - approach (the palm-frame grasp-offset direction, world) dives at most
      MAX_DESCEND below horizontal — steeper is a top-down strike;
    - the jaw axis (palm +y) leans at most JAW_TILT_MAX out of the
      horizontal plane — past that it closes along the can, not across it.

Torso: the pelvis frame makes legs irrelevant here — squatting moves R
DOWN IN THE WORLD, not in the pelvis frame — so world-frame coverage is the
pelvis-frame map extruded over the delivered pelvis-height range
[PELVIS_MIN, PELVIS_MAX]. The waist joints DO change R in the pelvis frame,
so waist pitch is enumerated over the range SONIC actually delivers.

Everything here is SONIC-legal by construction: the arms are exact (the
sim-side pin runs them in position mode), and the torso parameters are the
measured delivered ranges, not commanded ones.
"""

from __future__ import annotations

import os

import numpy as np

from arm_ik import ArmIK
from g1_joints import ARM_MJ
from primitives_const import GRASP_CENTRE_OFFSET_CONST

MAP_PATH = "/workspace/code/locomanipulation/outputs/scene/reach_map.npz"

OFFSET_DIR = (np.asarray(GRASP_CENTRE_OFFSET_CONST)
              / np.linalg.norm(GRASP_CENTRE_OFFSET_CONST))

MAX_DESCEND = np.deg2rad(25.0)
JAW_TILT_MAX = np.deg2rad(25.0)

# Delivered torso capability (measured on this stack, not commanded):
PELVIS_MAX = 0.793            # standing pelvis height
PELVIS_MIN = 0.660            # deepest DELIVERED squat (~13 cm, stream_reach)
WAIST_PITCH = (0.0, 0.05)     # DELIVERED lean while standing: commanding
                              # 0.25 measures 0.03-0.06, and the policy also
                              # wanders waist yaw ~0.2 rad in response to
                              # unusual arm poses — the map keeps the waist
                              # nearly neutral and the navigator erodes the
                              # lookup by 5 cm instead of trusting lean

# Voxel grid, pelvis frame.
VOX = 0.025
X0, X1 = 0.00, 0.75
Y0, Y1 = -0.75, 0.75
Z0, Z1 = -0.55, 0.45

N_SAMPLES = 120_000           # per arm per waist config


def _grid_shape():
    nx = int(round((X1 - X0) / VOX))
    ny = int(round((Y1 - Y0) / VOX))
    nz = int(round((Z1 - Z0) / VOX))
    return nx, ny, nz


def build(seed: int = 0) -> dict:
    import mujoco
    ik = ArmIK()
    rng = np.random.default_rng(seed)
    # Self-collision filter. mj_kinematics alone accepted configurations
    # with the forearm through the chest — the probe measured 30+ cm of
    # "delivery error" there because the sim, unlike bare FK, has a torso.
    # Robot-robot contact pairs (non-adjacent bodies) reject a sample.
    m = ik.model
    robot_geoms = set()
    for g in range(m.ngeom):
        nm = m.body(m.geom_bodyid[g]).name
        if nm.startswith(("left_", "right_", "pelvis", "torso", "waist",
                          "logo", "head", "imu")):
            robot_geoms.add(g)

    def self_collides() -> bool:
        mujoco.mj_collision(m, ik.data)
        d = ik.data
        for ci in range(d.ncon):
            g1, g2 = d.contact.geom1[ci], d.contact.geom2[ci]
            if g1 not in robot_geoms or g2 not in robot_geoms:
                continue
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if b1 == b2 or m.body_parentid[b1] == b2 or m.body_parentid[b2] == b1:
                continue
            return True
        return False
    nx, ny, nz = _grid_shape()
    occ = {"left": np.zeros((nx, ny, nz), dtype=bool),
           "right": np.zeros((nx, ny, nz), dtype=bool)}
    # Exemplar evidence per voxel: the joint solution that made it feasible.
    # The verification probe replays exactly this — the promise, not a
    # re-derivation of it.
    ex_q = {s2: np.zeros((nx, ny, nz, 7), dtype=np.float32)
            for s2 in ("left", "right")}
    ex_wp = {s2: np.full((nx, ny, nz), np.nan, dtype=np.float32)
             for s2 in ("left", "right")}
    ex_pos = {s2: np.zeros((nx, ny, nz, 3), dtype=np.float32)
              for s2 in ("left", "right")}
    ex_appr = {s2: np.zeros((nx, ny, nz, 3), dtype=np.float32)
               for s2 in ("left", "right")}

    for side in ("left", "right"):
        arm = ARM_MJ[side]
        lo, hi = ik.lo[arm].copy(), ik.hi[arm].copy()
        for wp in WAIST_PITCH:
            qs = rng.uniform(lo, hi, size=(N_SAMPLES, 7))
            q29 = np.zeros(29)
            q29[14] = wp                       # waist_pitch
            kept = 0
            for row in qs:
                q29[arm] = row
                pos, rot = ik.fk(q29, side)
                appr = rot @ OFFSET_DIR
                dh = float(np.hypot(appr[0], appr[1]))
                if -appr[2] > np.tan(MAX_DESCEND) * max(dh, 1e-9):
                    continue                   # diving too steeply
                if appr[2] > 0.5:
                    continue                   # grasping upward: not a table
                jaw_tilt = abs(float(np.arcsin(np.clip(rot[2, 1], -1, 1))))
                if jaw_tilt > JAW_TILT_MAX:
                    continue
                if self_collides():
                    continue
                i = int((pos[0] - X0) / VOX)
                j = int((pos[1] - Y0) / VOX)
                k = int((pos[2] - Z0) / VOX)
                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
                    if not occ[side][i, j, k]:
                        ex_q[side][i, j, k] = row
                        ex_wp[side][i, j, k] = wp
                        ex_pos[side][i, j, k] = pos
                        ex_appr[side][i, j, k] = appr
                    occ[side][i, j, k] = True
                    kept += 1
            print(f"  {side} waist {wp:.2f}: kept {kept}/{N_SAMPLES}")

    both = occ["left"] | occ["right"]
    np.savez_compressed(
        MAP_PATH, left=occ["left"], right=occ["right"], both=both,
        vox=VOX, origin=np.array([X0, Y0, Z0]),
        pelvis_range=np.array([PELVIS_MIN, PELVIS_MAX]),
        ex_q_left=ex_q["left"], ex_q_right=ex_q["right"],
        ex_wp_left=ex_wp["left"], ex_wp_right=ex_wp["right"],
        ex_pos_left=ex_pos["left"], ex_pos_right=ex_pos["right"],
        ex_appr_left=ex_appr["left"], ex_appr_right=ex_appr["right"])
    print(f"saved {MAP_PATH}: both {both.sum()} voxels "
          f"({both.sum() * VOX**3 * 1e3:.1f} litres)")
    return {"left": occ["left"], "right": occ["right"], "both": both}


class ReachMap:
    """Runtime lookup: is a world point graspable from a candidate stand?"""

    def __init__(self, path: str = MAP_PATH):
        z = np.load(path)
        self.left, self.right, self.both = z["left"], z["right"], z["both"]
        self.vox = float(z["vox"])
        self.origin = z["origin"]
        self.pelvis_min, self.pelvis_max = z["pelvis_range"]

    def _at(self, grid, p) -> bool:
        idx = ((np.asarray(p) - self.origin) / self.vox).astype(int)
        if ((idx < 0).any() or (idx >= np.array(grid.shape)).any()):
            return False
        return bool(grid[tuple(idx)])

    def graspable(self, p_pelvis, arm: str = "both") -> bool:
        """p in the pelvis frame at STANDING height. The squat extends reach
        DOWNWARD: a point below the map is re-tested as if the pelvis had
        come down by up to (PELVIS_MAX - PELVIS_MIN)."""
        grid = {"left": self.left, "right": self.right,
                "both": self.both}[arm]
        p = np.asarray(p_pelvis, dtype=float)
        drop = self.pelvis_max - self.pelvis_min
        for dz in np.arange(0.0, drop + 1e-9, self.vox):
            if self._at(grid, p + [0.0, 0.0, dz]):
                return True
        return False

    def margin(self, p_pelvis, arm: str = "both") -> float:
        """Distance (m) to the nearest non-graspable voxel, xy-plane, at the
        best squat level — the stand-quality score the navigator maximises."""
        if not self.graspable(p_pelvis, arm):
            return 0.0
        m = 0.0
        while m < 0.15:
            m += self.vox
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                q = np.asarray(p_pelvis) + [m * np.cos(a), m * np.sin(a), 0]
                if not self.graspable(q, arm):
                    return m - self.vox
        return m


if __name__ == "__main__":
    build()
