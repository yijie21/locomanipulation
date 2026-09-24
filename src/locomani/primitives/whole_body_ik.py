"""Low reach geometry with planted feet, COM support and a shallow pelvis posture.

The result is a geometric reference, not a guarantee that SONIC will track it.
All robot joint limits come from the loaded model; no object pose is altered.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from locomani.primitives.arm_ik import ArmIK


@dataclass(frozen=True)
class WholeBodyPose:
    joints: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    hand_errors: np.ndarray
    foot_errors: np.ndarray
    com_shift: np.ndarray
    com_shift_max: float = 0.015

    @property
    def reachable(self):
        return (
            max(self.hand_errors) < 0.02
            and max(self.foot_errors) < 0.005
            and np.linalg.norm(self.com_shift) < self.com_shift_max
        )


class WholeBodyIK:
    """A fixed support problem, reusable for admission and continuous paired motion."""

    def __init__(
        self,
        ik: ArmIK,
        state,
        config,
        *,
        tool_offset,
        joint_limits=None,
        hand_weights=(1.0, 1.0),
        com_weight=5.0,
        com_shift_max=0.015,
        com_free_radius=0.0,
        knee_max=0.85,
        pelvis_forward_max=0.20,
        pelvis_height_target=None,
        elbow_min_deg=5.0,
    ):
        self.ik = ik
        self.hand_weights = np.asarray(hand_weights, dtype=float)
        if not np.isfinite(com_weight) or com_weight <= 0:
            raise ValueError("COM weight must be positive and finite")
        self.com_weight = float(com_weight)
        # Support envelope. com_shift_max is the admission threshold; com_free_radius
        # is a dead band in the residual inside which the horizontal COM is not pulled
        # back to its initial position at all (0 keeps the historical linear pull).
        # Knee flexion and forward pelvis travel are capped explicitly.
        for name, value in (
            ("com_shift_max", com_shift_max),
            ("knee_max", knee_max),
            ("pelvis_forward_max", pelvis_forward_max),
        ):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not np.isfinite(com_free_radius) or com_free_radius < 0:
            raise ValueError("com_free_radius must be finite and nonnegative")
        self.com_shift_max, self.com_free_radius = float(com_shift_max), float(com_free_radius)
        self.knee_max, self.pelvis_forward_max = float(knee_max), float(pelvis_forward_max)
        # The height residual pulls the pelvis toward this value; None keeps the initial height,
        # a lower value asks the solver to crouch (the streamed leg reference is what delivers it).
        if pelvis_height_target is not None and (not np.isfinite(pelvis_height_target) or pelvis_height_target <= 0):
            raise ValueError("pelvis_height_target must be positive and finite")
        self.pelvis_height_target = pelvis_height_target
        if (
            self.hand_weights.shape != (2,)
            or not np.isfinite(self.hand_weights).all()
            or np.any(self.hand_weights <= 0)
        ):
            raise ValueError("hand weights must be two positive finite values")
        self.q0 = np.asarray(state["joints"], dtype=float)
        self.base0 = np.asarray(state["pelvis"]["pos"], dtype=float)
        rot0 = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
        pitch0 = Rotation.from_matrix(rot0).as_euler("xyz")[1]
        self.heading = rot0 @ Rotation.from_euler("y", -pitch0).as_matrix()
        m, d, mj = ik.model, ik.data, ik._mj
        self.feet = [m.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
        self.hands = [ik.palm_bid[side] for side in ("left", "right")]
        self.pelvis = m.body("pelvis").id
        self.torso = m.body("torso_link").id
        self.offset = np.asarray(tool_offset)
        d.qpos[:] = m.qpos0
        self.initial = np.r_[self.base0, pitch0, self.q0]
        self.put(self.initial)
        self.feet_pos = d.xpos[self.feet].copy()
        self.feet_rot = d.xmat[self.feet].reshape(2, 3, 3).copy()
        self.com0 = d.subtree_com[self.pelvis, :2].copy()
        ranges = m.jnt_range[ik.jids].copy()
        ranges[14, 1] = min(ranges[14, 1], config.waist_pitch_max)
        ranges[[3, 9], 0] = 0.05
        ranges[[3, 9], 1] = np.minimum(ranges[[3, 9], 1], self.knee_max)
        ranges[[18, 25], 0] = np.maximum(ranges[[18, 25], 0], np.radians(elbow_min_deg))
        if joint_limits is not None:
            ranges[:, 0] = np.maximum(ranges[:, 0], joint_limits[0])
            ranges[:, 1] = np.minimum(ranges[:, 1], joint_limits[1])
        if config.waist_dofs == "pitch":
            ranges[12:14] = [-1e-9, 1e-9]
        self.lo = np.r_[self.base0[:2] - [0.3, 0.15], config.pelvis_height_min, min(0, pitch0), ranges[:, 0]]
        self.hi = np.r_[
            self.base0[:2] + [self.pelvis_forward_max, 0.15],
            max(self.base0[2] + 0.02, config.pelvis_height_min + 0.01),
            config.pelvis_pitch_max,
            ranges[:, 1],
        ]
        self.seed = self.initial.copy()
        self.seed[3] = 0.3
        self.seed[4 + np.array([0, 6])] -= 0.3
        self.seed[18] = min(0.45, config.waist_pitch_max)
        self.seed = np.clip(self.seed, self.lo, self.hi)
        self.last_diagnostics = {}

    def put(self, x):
        ik = self.ik
        base = self.heading @ Rotation.from_euler("y", x[3]).as_matrix()
        ik.data.qpos[ik.base_qadr : ik.base_qadr + 3] = x[:3]
        ik.data.qpos[ik.base_qadr + 3 : ik.base_qadr + 7] = Rotation.from_matrix(base).as_quat()[[3, 0, 1, 2]]
        ik.data.qpos[ik.qadr] = x[4:]
        ik._mj.mj_kinematics(ik.model, ik.data)
        ik._mj.mj_comPos(ik.model, ik.data)
        return base

    def tools(self, x):
        self.put(x)
        d = self.ik.data
        return np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ self.offset for b in self.hands])

    def solve(
        self,
        positions,
        rotations,
        *,
        seed=None,
        bounds=None,
        posture=None,
        posture_weights=None,
        continuity_reference=None,
        continuity_dt=None,
        continuity_weight=0.0,
        torso_rotation=None,
        iterations=120,
        hand_rotation_weight=0.6,
    ):
        """`hand_rotation_weight` scales the hand orientation residual (rad) against position (10 per metre):
        at 0.6, 1 cm of hand position costs about as much as 10 degrees of hand orientation."""
        d = self.ik.data
        positions, rotations = np.asarray(positions), np.asarray(rotations)
        prior = self.q0 if posture is None else posture
        if posture_weights is None:
            posture_scale = 0.008 if posture is None else 0.04
        else:
            posture_scale = np.asarray(posture_weights, dtype=float)
            if posture_scale.shape != (29,) or not np.isfinite(posture_scale).all() or np.any(posture_scale < 0):
                raise ValueError("posture_weights must contain 29 finite nonnegative values")
        if not np.isfinite(continuity_weight) or continuity_weight < 0:
            raise ValueError("continuity_weight must be finite and nonnegative")
        if continuity_weight > 0:
            continuity_reference = np.asarray(continuity_reference, dtype=float)
            if continuity_reference.shape != (33,) or not np.isfinite(continuity_reference).all():
                raise ValueError("continuity_reference must contain 33 finite values")
            if continuity_dt is None or not np.isfinite(continuity_dt) or continuity_dt <= 0:
                raise ValueError("continuity_dt must be positive and finite")

        def residual(x):
            points = self.tools(x)
            foot_rot = np.concatenate(
                [
                    Rotation.from_matrix(d.xmat[b].reshape(3, 3) @ r.T).as_rotvec()
                    for b, r in zip(self.feet, self.feet_rot)
                ]
            )
            hand_rot = np.concatenate(
                [Rotation.from_matrix(d.xmat[b].reshape(3, 3) @ r.T).as_rotvec() for b, r in zip(self.hands, rotations)]
            )
            continuity = (
                np.empty(0)
                if continuity_weight == 0
                else continuity_weight * (x - continuity_reference) / continuity_dt
            )
            com_shift = d.subtree_com[self.pelvis, :2] - self.com0
            if self.com_free_radius > 0:
                # Inside the free disc only a tenth of the pull remains, so the solver
                # spends COM margin when the hands need it and otherwise stays central.
                com_norm = np.linalg.norm(com_shift)
                com_shift = com_shift * (
                    (0.1 * min(com_norm, self.com_free_radius) + max(0.0, com_norm - self.com_free_radius))
                    / max(com_norm, 1e-9)
                )
            return np.r_[
                20 * (d.xpos[self.feet] - self.feet_pos).ravel(),
                3 * foot_rot,
                10 * ((points - positions) * self.hand_weights[:, None]).ravel(),
                hand_rotation_weight * hand_rot * np.repeat(self.hand_weights, 3),
                self.com_weight * com_shift,
                posture_scale * (x[4:] - prior),
                0.3 * (x[2] - (self.base0[2] if self.pelvis_height_target is None else self.pelvis_height_target)),
                np.zeros(3)
                if torso_rotation is None
                else 1.5 * Rotation.from_matrix(d.xmat[self.torso].reshape(3, 3) @ torso_rotation.T).as_rotvec(),
                continuity,
            ]

        lo, hi = (self.lo, self.hi) if bounds is None else bounds
        x0 = np.clip(self.seed if seed is None else seed, lo, hi)
        result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=iterations, ftol=1e-6, xtol=1e-6, gtol=1e-6)
        self.seed = result.x.copy()
        continuity_velocity_error = (
            0.0 if continuity_weight == 0 else float(np.max(np.abs((result.x - continuity_reference) / continuity_dt)))
        )
        self.last_diagnostics = dict(
            continuity_weight=float(continuity_weight),
            continuity_velocity_error=continuity_velocity_error,
            posture_residual_norm=float(np.linalg.norm(np.asarray(posture_scale) * (result.x[4:] - prior))),
        )
        rotation = self.put(result.x)
        return WholeBodyPose(
            result.x[4:].copy(),
            result.x[:3].copy(),
            rotation,
            np.linalg.norm(self.tools(result.x) - positions, axis=1),
            np.linalg.norm(d.xpos[self.feet] - self.feet_pos, axis=1),
            d.subtree_com[self.pelvis, :2].copy() - self.com0,
            self.com_shift_max,
        )


def plan_low_reach(
    ik: ArmIK,
    state,
    positions,
    rotations,
    config,
    *,
    tool_offset,
    joint_limits=None,
):
    """Solve from observed feet, hands and pelvis; minimize lowering and posture change."""
    return WholeBodyIK(
        ik,
        state,
        config,
        tool_offset=tool_offset,
        joint_limits=joint_limits,
    ).solve(positions, rotations)
