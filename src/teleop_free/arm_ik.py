"""Damped-least-squares IK for one G1 arm, in the pelvis frame.

The streamed-motion path (diag_stream.py) needs arm *joint angles* whose FK
puts the palm at a wanted pose — the opposite direction from everything the
planner-mode code did. Legs, waist, and the other arm are held at whatever the
robot currently measures; only the 7 joints of one arm move.

The model is the same scene XML the simulator runs, so FK here is exactly the
FK the privileged publisher reports palms from (PALM_BODY = wrist_yaw_link).
The base free joint is parked at the origin with identity orientation, which
makes body frames pelvis-frame poses directly — no world involved.
"""

from __future__ import annotations

import numpy as np

from g1_joints import ARM_MJ, MUJOCO_JOINT_NAMES
import paths

SCENE_XML = f"{paths.REPO}/outputs/scene/sim/scene_locomani.xml"
PALM_BODY = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}


class ArmIK:
    def __init__(self, xml: str = SCENE_XML):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        def jadr(name):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(f"joint {name} not in {xml}")
            return jid

        self.jids = [jadr(n) for n in MUJOCO_JOINT_NAMES]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([self.model.jnt_dofadr[j] for j in self.jids])
        base = jadr("floating_base_joint")
        self.base_qadr = int(self.model.jnt_qposadr[base])

        self.palm_bid = {}
        for side, bname in PALM_BODY.items():
            b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if b < 0:
                raise KeyError(f"body {bname} not in {xml}")
            self.palm_bid[side] = b

        self.lo = np.array([self.model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([self.model.jnt_range[j][1] for j in self.jids])
        # Waist bounds are deliberately WIDER than the model's: the policy
        # delivers only ~2/3 of a commanded lean (measured: waist_pitch asked
        # 30 deg, tracked 21 deg short of nothing near that), so the reference
        # may exaggerate — the robot saturates at what it can actually do,
        # like a teleop human overdriving their motion. The rest of the body
        # keeps physical limits.
        for j in (13, 14):
            self.lo[j] *= 1.6
            self.hi[j] *= 1.6

    def fk(self, q_mj29: np.ndarray, side: str):
        """Palm (pos, rot 3x3) in the pelvis frame for a full joint vector."""
        mj = self._mj
        self.data.qpos[:] = 0
        self.data.qpos[self.base_qadr + 3] = 1.0  # identity base quat
        self.data.qpos[self.qadr] = q_mj29
        mj.mj_kinematics(self.model, self.data)
        b = self.palm_bid[side]

        return (self.data.xpos[b].copy(),
                self.data.xmat[b].reshape(3, 3).copy())

    def solve(self, side: str, target_pos, target_rot=None,
              q_mj29=None, iters: int = 120, damping: float = 5e-3,
              rot_weight: float = 0.5, use_waist: bool = False,
              waist_cost: float = 10.0):
        """Joints putting the palm at `target_pos` (pelvis frame).

        Returns (q_mj29_solved, pos_err_m, rot_err_rad) — the FULL joint
        vector, since with `use_waist` the waist moves too. With the shoulder
        at pelvis-frame z +0.29 the arm alone tops out at x ~0.20 in the grasp
        band; table grasps need the torso lean, so `use_waist` adds the three
        waist joints with `waist_cost`-times the damping — the solver leans
        only as much as the target requires.
        """
        mj = self._mj
        q = np.array(q_mj29, dtype=float).copy()
        arm = list(ARM_MJ[side]) + ([12, 13, 14] if use_waist else [])
        target_pos = np.asarray(target_pos, dtype=float)

        b = self.palm_bid[side]
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        cols = self.dadr[arm]
        d_diag = damping * np.ones(len(arm))
        if use_waist:
            d_diag[-3:] *= waist_cost

        for _ in range(iters):
            self.data.qpos[:] = 0
            self.data.qpos[self.base_qadr + 3] = 1.0
            self.data.qpos[self.qadr] = q
            mj.mj_kinematics(self.model, self.data)
            mj.mj_comPos(self.model, self.data)

            pos = self.data.xpos[b]
            rot = self.data.xmat[b].reshape(3, 3)
            e_pos = target_pos - pos
            if target_rot is not None:
                # axis-angle of R_t @ R^T, the world-frame rotation still needed
                dR = np.asarray(target_rot) @ rot.T
                aa = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0],
                               dR[1, 0] - dR[0, 1]])
                s = np.linalg.norm(aa) / 2.0
                c = (np.trace(dR) - 1.0) / 2.0
                ang = np.arctan2(s, c)
                e_rot = (aa / (2.0 * max(s / max(ang, 1e-9), 1e-9))
                         if s > 1e-9 else np.zeros(3))
            else:
                e_rot = np.zeros(3)

            if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_rot) < 1e-3:
                break

            mj.mj_jacBody(self.model, self.data, jacp, jacr, b)
            J = np.vstack([jacp[:, cols], rot_weight * jacr[:, cols]])
            e = np.concatenate([e_pos, rot_weight * e_rot])
            dq = np.linalg.solve(J.T @ J + np.diag(d_diag), J.T @ e)
            step = np.linalg.norm(dq)
            if step > 0.5:
                dq *= 0.5 / step
            q[arm] = np.clip(q[arm] + dq, self.lo[arm], self.hi[arm])

        pos, rot = self.fk(q, side)
        pos_err = float(np.linalg.norm(pos - target_pos))
        if target_rot is not None:
            dR = np.asarray(target_rot) @ rot.T
            rot_err = float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        else:
            rot_err = 0.0

        return q, pos_err, rot_err
