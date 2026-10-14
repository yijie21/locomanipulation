"""Pelvis pose from leg kinematics + the IMU (TODO 25.5, plan §4.5): what ScaleBFM's world-anchored references would
stand on without the simulator's ground truth, and what the real-robot deploy program carries over.

Inputs every control tick are what the robot has: the 29 joint encoders (MuJoCo order) and the IMU orientation of the
pelvis (wxyz). FK with the pelvis at the origin and the IMU's orientation puts the eight sole corners (the bottom
corners of each foot's collision box) in a world-aligned frame. Corners within `tol_m` of the lowest are in contact;
a corner that comes into contact is latched where the current estimate puts it, and while it stays in contact
(until it rises 2 x tol_m above the lowest) the pelvis estimate is its latched position minus its current offset,
averaged over the contact corners. Latching corners rather than the ankle keeps heel strike and toe-off (the foot
rolls about a corner that does not move) from dragging the estimate. Drift comes from slip and encoder noise; yaw is
the IMU's; a flat floor at the reset's height is assumed.

    odo = LegOdometry(scene_xml); odo.reset(pelvis_xyz, joints29, imu_quat)   # the start pose is the one thing told
    est = odo.update(joints29, imu_quat_wxyz)                                  # -> pelvis position estimate (3,)
"""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


class LegOdometry:
    def __init__(self, scene_xml: str, tol_m: float = 0.004):
        self.m = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.d = mujoco.MjData(self.m)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in MUJOCO_JOINT_NAMES])
        self.corners = []  # (body id, point in the body frame): the bottom corners of each foot's collision boxes
        for f in FEET:
            b = self.m.body(f).id
            for g in range(self.m.ngeom):
                if (
                    self.m.geom_bodyid[g] != b
                    or not self.m.geom_contype[g]
                    or self.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX
                ):
                    continue
                Rg = np.empty(9)
                mujoco.mju_quat2Mat(Rg, self.m.geom_quat[g])
                Rg = Rg.reshape(3, 3)
                sx, sy, sz = self.m.geom_size[g]
                for ex, ey in itertools.product((-1, 1), (-1, 1)):
                    self.corners.append((b, self.m.geom_pos[g] + Rg @ np.array([ex * sx, ey * sy, -sz])))
        assert len(self.corners) >= 8, "no box soles on the feet"
        self.tol = tol_m
        self.pelvis = np.zeros(3)
        self.latched: dict[int, np.ndarray] = {}

    def _corners_rel(self, joints, quat) -> np.ndarray:
        """Sole corners relative to the pelvis, world-aligned (the IMU's orientation)."""
        self.d.qpos[:3] = 0.0
        self.d.qpos[3:7] = quat
        self.d.qpos[self.qadr] = joints
        mujoco.mj_kinematics(self.m, self.d)
        return np.array([self.d.xpos[b] + self.d.xmat[b].reshape(3, 3) @ p for b, p in self.corners])

    def reset(self, pelvis_xyz, joints, quat) -> None:
        self.pelvis = np.asarray(pelvis_xyz, dtype=float).copy()
        self.latched = {}
        self.update(joints, quat)

    def update(self, joints, quat) -> np.ndarray:
        rel = self._corners_rel(np.asarray(joints, dtype=float), np.asarray(quat, dtype=float))
        low = rel[:, 2].min()
        for k in range(len(rel)):
            z = rel[k, 2] - low
            if k in self.latched and z > 2 * self.tol:
                del self.latched[k]
            elif k not in self.latched and z <= self.tol:
                self.latched[k] = self.pelvis + rel[k]
        if self.latched:
            self.pelvis = np.mean([self.latched[k] - rel[k] for k in self.latched], axis=0)
        return self.pelvis.copy()


def attach(sim, scene_xml: str, meta_joint_names) -> LegOdometry:
    """Make `sim.state()` report the odometry's pelvis position instead of the simulator's (TODO 25.5, the estimated-pose
    arm): ScaleBFM's world-anchored targets are then expressed relative to where the robot thinks it is. Everything
    else in the state (IMU orientation and rate, joints) is the robot's own sensing, perturbed or not. Call
    `odo.reset(...)` with the true start pose before the first step; one state() call per control tick advances it."""
    odo = LegOdometry(scene_xml)
    order = [list(meta_joint_names).index(n) for n in MUJOCO_JOINT_NAMES]  # policy (Isaac) order -> MuJoCo order
    true_state = sim.state

    def state():
        st = true_state()
        st["root_pos"] = odo.update(np.asarray(st["dof_pos"])[order], st["root_quat"])
        st["root_pos_true"] = sim.d.qpos[:3].copy()
        return st

    sim.state = state
    odo.order = order
    return odo
