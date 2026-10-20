"""AMO motor layer (TODO 28): the scene skills' eight primitives on AMO instead of ScaleBFM.

`AmoMotor` keeps `backends.scalebfm.motor.Motor`'s whole interface and reference logic -- walk_to / turn / sidestep build
the same pelvis path, posture / reach / adopt_hand the same whole-body IK poses, grip / release / hold the same jaw ramps,
`hand` the same per-tick jaw-centre override -- and replaces only how a reference reaches the joints:

  legs + waist  AMO's command (runtime.Command). While walking: pelvis-path speed x ff_gain plus kp x the pelvis error,
                in the measured heading frame, the path heading as the target yaw, stepping on (`turning`) until the
                stop is inside tolerance. Standing: speed 0, the gait frozen. Base height = the reference pelvis height
                plus an integral on the measured height; torso yaw / pitch / roll = the reference torso_link attitude in
                the reference heading frame plus an integral on the measured pitch.
  arms          not AMO's: our PD on joint targets. While walking the reference's arm joints (the arms ride with the
                body). Otherwise each tick an arm IK (primitives.arm_ik) puts the jaw centre at its world target from
                the *measured* pelvis and waist, so AMO's own choice of hips and waist does not move the hand; the
                target is the reference's jaw centre (FK) or `hand`'s override, plus the same jaw-centre / wrist-attitude
                integral servo ScaleBFM's motor runs (servo_ki, servo_ki_rot).

Measured facts this rests on (scripts/experiments/amo_compare_20261020, mars G1 on the flat walk_mars floor, 2026-10-20):
AMO delivers about half the commanded forward speed (0.5 -> 0.26 m/s, 0.3 -> 0.14); a turn in place drifts the pelvis
~0.4 m; a 0.45 m height command lands at 0.58 m; a 1.0 rad torso pitch command at 58 deg. From ScaleBFM's bent-knee
reset pose it walks off (60 deg in 10 s), so `reset` starts from AMO's default pose (all joints 0, arms at ARM_READY)
and settles 1 s before the round's clock starts.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from locomani.backends.amo import runtime as A
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import (
    ANKLE,
    JAW_OPEN,
    MODE_WALK,
    SIDES,
    WRIST,
    Motor,
    MotorConfig,
    Pose,
    expmap,
    lowest_foot_point,
)
from locomani.primitives.g1_joints import ARM_MJ
from locomani.primitives.whole_body_ik import rotvec

# shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw: hands beside the thighs, clear of the torso and legs
ARM_READY = np.array([0.1, 0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.1, -0.2, 0.0, 0.5, 0.0, 0.0, 0.0])


@dataclass(frozen=True)
class AmoConfig:
    ff_gain: float = 2.0  # x the path speed: AMO delivers about half its speed command
    kp_xy: float = 3.0  # 1/s on the pelvis error while walking
    v_min: float = 0.18  # m/s: a pelvis error past stop_tolerance gets at least this speed command (AMO barely moves
    # below ~0.15: 0.15 -> 0.05 m/s, 2026-10-20)
    ki_height: float = 1.0  # 1/s base-height integral
    cap_height: float = 0.30  # m
    ki_pitch: float = 1.0  # 1/s torso-pitch integral: AMO delivers a 1.0 rad lean on flat floor but 35 of 60 deg
    # reaching into the display shelf (pick_thick_novel s000, 2026-10-20)
    cap_pitch: float = 0.30  # rad
    torso_yaw_max: float = 0.40  # rad: the reference torso yaw / roll sent, capped (a 45 deg IK waist yaw under a
    torso_roll_max: float = 0.30  # raised arm made AMO walk off 1.5 m; with neither sent low reaches missed 11-36 cm)
    pitch_relative: bool = True  # pitch command relative to the nominal standing torso (pitch0)
    pitch_range: tuple = (-0.10, 1.30)  # rad: the reference torso pitch used; whole-body IK leans a high reach back
    # 45 deg (ScaleBFM never tracked the torso), and AMO walks backwards 1.5 m under that (motor_check reach_high)
    ik_iters: int = 30
    ik_rot_weight: float = 0.5
    ik_posture_weight: float = 0.05  # pull toward the reference's arm joints: unseeded, the per-tick arm IK slid into a
    # twisted branch (shoulder yaw -1.9) and AMO walked off under it (scene rounds 2026-10-20; ArmIK's own warning)
    hand_lead_s: float = 0.04  # s: `hand`'s velocity leads the arm target this far (the arm PD lags)
    settle_s: float = 1.0  # s standing after reset, before t = 0


# AMO's own motor defaults: slower turns and a longer stop than ScaleBFM's (a turn in place drifted the pelvis 9-12 cm at
# 0.8 rad/s, motor_check 2026-10-20), and whole-body IK that leans the torso only by waist pitch: with waist yaw / roll
# free the reference twisted the torso 23 deg and rolled it 17 deg for a counter-top reach, which ScaleBFM never
# tracked but AMO does, and it stepped back 30 cm doing it (orange_pickup s300003, 2026-10-20)
# Asked 60-78 deg into the display shelf AMO leans 36 deg (waist at its 0.52 rad limit, the rest as a squat;
# pick_thick_novel, 2026-10-20); capping the planned pelvis pitch at 0.15 rad did not get the book either.
AMO_MOTOR = MotorConfig(turn_rate=0.6, stop_timeout=5.0, waist_dofs="pitch")


class AmoMotor(Motor):
    """`Motor` on AMO. `sim` is an AmoSim, `policy` an AmoPolicy; `model_xml` the robot-only model for IK."""

    def __init__(self, sim: A.AmoSim, policy: A.AmoPolicy, config: MotorConfig | None = None, *, model_xml: str,
                 track: tuple[str, ...] = (), jacobian: str = "analytic", amo: AmoConfig | None = None):  # fmt: skip
        super().__init__(sim, policy, config or AMO_MOTOR, model_xml=model_xml, track=track, jacobian=jacobian)
        self.amo = amo or AmoConfig()
        self.torso_id = sim.m.body("torso_link").id
        self._reset_servos()

    def _reset_servos(self) -> None:
        self.h_int = 0.0
        self.p_int = 0.0
        self.arm_q = np.r_[ARM_READY]
        self.cmd_jaw: dict = {}
        self.stepping = False
        self.last_cmd = A.Command()
        self.pitch0 = 0.0
        self.nominal = None

    # -- reset ------------------------------------------------------------------------------------------------------
    def reset(self, xy, yaw: float, *, warm_up: int = 0) -> None:
        sim = self.sim
        sim.reset(np.asarray(xy, float), float(yaw), 0.0)
        sim.d.qpos[sim.qadr] = 0.0
        sim.d.qpos[sim.qadr[A.ARMS]] = ARM_READY
        mujoco.mj_forward(sim.m, sim.d)
        sim.d.qpos[2] += 0.005 - lowest_foot_point(sim.m, sim.d, self.foot_bodies)
        mujoco.mj_forward(sim.m, sim.d)
        self.pol.reset()
        self._reset_servos()
        self.k, self.trace, self.fell_at = 0, [], None
        self.jaw[:] = JAW_OPEN
        self.servo, self.servo_target, self.hand_override = {}, {}, {}
        self.servo_on = {s: False for s in SIDES}
        y = float(yaw)
        for _ in range(int(self.amo.settle_s / A.DT)):  # AMO stands up from its default pose; not on the round's clock
            sim.grip_target[:] = JAW_OPEN
            sim.step(self.pol(sim.state(), A.Command(yaw=y), ARM_READY))
        self.nominal = self.joints()[:15].copy()
        self._stand()
        self.stand_pose = self.ref(0.0)
        # AMO stands with a steady forward torso lean (~9 deg on the mars G1); the pitch command is relative to it, so
        # the standing reference (the measured pose) does not ask for more lean
        self.pitch0 = float(self._torso_attitude(self.stand_pose)[1])

    # -- the tick ---------------------------------------------------------------------------------------------------
    def _set_mode(self, mode: int) -> None:
        self.mode = mode

    def _stand(self, pose: Pose | None = None) -> None:
        """Motor._stand, but a pose built from the measurement keeps the legs and waist of the nominal standing pose.
        Standing, AMO bends its waist forward to the 0.52 rad limit and pitches the pelvis back; Motor's measured
        reference (pelvis upright, measured waist) then reads as a 30 deg lean, AMO leans into it, and every later
        walk asked for more (a whole route walked at 30 deg, scene round 2026-10-20)."""
        if pose is None and getattr(self, "nominal", None) is not None:
            pos, quat = self.pelvis()
            q = self.joints()
            q[:15] = self.nominal
            pose = Pose(pos, R.quat_wxyz(R.rot_z(R.yaw_of(quat))), q, np.zeros(29))
        super()._stand(pose)

    def _torso_attitude(self, pose: Pose) -> np.ndarray:
        """(roll, pitch, yaw) of the reference torso_link in the reference's heading frame."""
        R_t = R.mat(self.kp(pose.pos, pose.quat, pose.joints)["torso_link"][1])
        return A.euler(R.quat_wxyz(R.rot_z(-R.yaw_of(pose.quat)) @ R_t))

    def _command(self, ref: Pose) -> A.Command:
        a, cfg = self.amo, self.cfg
        d = self.sim.d
        yaw_ref = R.yaw_of(ref.quat)
        yaw = self.yaw()
        e_xy = ref.pos[:2] - d.qpos[:2]
        c, s = np.cos(yaw), np.sin(yaw)
        to_body = np.array([[c, s], [-s, c]])
        vx = vy = 0.0
        if self.mode == MODE_WALK:
            v_ref = (self.ref(self.t + A.DT).pos[:2] - ref.pos[:2]) / A.DT
            v = a.ff_gain * v_ref + a.kp_xy * e_xy
            off = float(np.linalg.norm(e_xy)) > cfg.stop_tolerance
            n = float(np.linalg.norm(v))
            if off and 1e-9 < n < a.v_min:
                v = v * (a.v_min / n)
            vx, vy = to_body @ v
            moving = float(np.linalg.norm(v_ref)) > 1e-4
            turning = abs(R.wrap(yaw - yaw_ref)) > np.radians(1.5)
            self.stepping = moving or off or turning
        else:
            self.stepping = False  # standing primitives keep the feet where they are, as ScaleBFM's pinned ankles
        # base height and torso pitch: reference + integral on the measurement (AMO under-delivers deep squats)
        roll, pitch, tyaw = self._torso_attitude(ref)
        off = self.pitch0 if a.pitch_relative else 0.0
        pitch = float(np.clip(pitch - off, *a.pitch_range))  # the command; it stands for the absolute lean pitch + off
        tyaw = float(np.clip(tyaw, -a.torso_yaw_max, a.torso_yaw_max))
        roll = float(np.clip(roll, -a.torso_roll_max, a.torso_roll_max))
        if self.mode != MODE_WALK:
            self.h_int = float(
                np.clip(self.h_int + a.ki_height * A.DT * (ref.pos[2] - d.qpos[2]), -a.cap_height, a.cap_height)
            )
            meas = A.euler(R.quat_wxyz(R.rot_z(-yaw) @ R.mat(d.xquat[self.torso_id])))[1]
            self.p_int = float(
                np.clip(self.p_int + a.ki_pitch * A.DT * (pitch + off - meas), -a.cap_pitch, a.cap_pitch)
            )
        cmd = A.Command(vx=float(vx), vy=float(vy), yaw=yaw_ref, height=float(ref.pos[2] + self.h_int),
                        torso_yaw=float(tyaw), torso_pitch=float(pitch + self.p_int), torso_roll=float(roll),
                        turning=self.stepping)  # fmt: skip
        if not self.stepping:
            cmd.vx = cmd.vy = 0.0
        return cmd

    def _jaw_target(self, side: str, ref: Pose):
        """World jaw-centre target (pos, rot) of `side` this tick: `hand`'s override, else the reference's FK plus the
        servo's lead."""
        if side in self.hand_override:
            jaw, rot, vel = self.hand_override[side]
            return jaw + self.amo.hand_lead_s * vel, rot
        kp = self.kp(ref.pos, ref.quat, ref.joints + ref.velocity * self.amo.hand_lead_s)
        p_w, q_w = kp[WRIST[side]]
        R_ref = R.mat(q_w)
        p_ref = p_w + R_ref @ R.OFF
        if not self.servo_on[side]:
            return p_ref, R_ref
        cfg = self.cfg
        jaw_ref = self.servo_target.get(side, p_ref)
        p_meas, R_meas = self.jaw_centre(side)
        turn = self.servo.get(("turn", side), np.zeros(3)) + cfg.servo_ki_rot * R.DT * rotvec(R_ref @ R_meas.T)
        n = float(np.linalg.norm(turn))
        turn = turn * (cfg.servo_cap_rot / n) if n > cfg.servo_cap_rot else turn
        self.servo[("turn", side)] = turn
        lead = np.clip(self.servo.get(("pos", side), np.zeros(3)) + cfg.servo_ki * R.DT * (jaw_ref - p_meas),
                       -cfg.servo_cap, cfg.servo_cap)  # fmt: skip
        self.servo[("pos", side)] = lead
        return jaw_ref + lead, expmap(turn) @ R_ref

    def _arms(self, ref: Pose) -> np.ndarray:
        q_meas = self.joints()
        if self.mode == MODE_WALK and not self.hand_override:
            self.arm_q = ref.joints[A.ARMS].copy()
            for s in SIDES:  # what the reference's FK puts there, for commanded_jaw and the trace
                p_w, q_w = self.kp(ref.pos, ref.quat, ref.joints)[WRIST[s]]
                self.cmd_jaw[s] = (p_w + R.mat(q_w) @ R.OFF, R.mat(q_w))
            return self.arm_q
        pel_p, pel_q = self.pelvis()
        Rp = R.mat(pel_q)
        q = q_meas.copy()
        q[A.ARMS] = ref.joints[A.ARMS]  # seeded at the reference's (planned) arm, held near it by the posture prior
        for s in SIDES:
            p, Rw = self._jaw_target(s, ref)
            self.cmd_jaw[s] = (p, Rw)
            sol, _, _ = self.ik.solve(s, Rp.T @ (p - pel_p), Rp.T @ Rw, q_mj29=q, iters=self.amo.ik_iters,
                                      rot_weight=self.amo.ik_rot_weight, point_offset=R.OFF, posture=ref.joints,
                                      posture_weight=self.amo.ik_posture_weight)  # fmt: skip
            q[ARM_MJ[s]] = sol[ARM_MJ[s]]
        self.arm_q = q[A.ARMS].copy()
        return self.arm_q

    def tick(self) -> bool:
        sim = self.sim
        ref = self.ref(self.t)
        cmd = self._command(ref)
        arms = self._arms(ref)
        sim.grip_target[:] = np.repeat(self.jaw, 2)
        if sim.fallen():
            self.fell_at = self.t if self.fell_at is None else self.fell_at
            return False
        self.trace.append(
            dict(
                t=self.t,
                primitive=self.primitive,
                mode=self.mode,
                robot=sim.d.qpos[:7].copy(),
                joints=self.joints(),
                ref_pelvis=ref.pos.copy(),
                ref_wrists=np.array([self.cmd_jaw[s][0] - self.cmd_jaw[s][1] @ R.OFF for s in SIDES]),
                wrists=np.array([sim.body(WRIST[s])[0] for s in SIDES]),
                jaw=self.jaw.copy(),
                fingers=sim.d.qpos[sim.gq].copy(),
                objects={n: np.r_[sim.d.xpos[b], sim.d.xquat[b]] for n, b in self.track.items()},
                cmd=np.array(
                    [
                        cmd.vx,
                        cmd.vy,
                        cmd.yaw,
                        cmd.height,
                        cmd.torso_yaw,
                        cmd.torso_pitch,
                        cmd.torso_roll,
                        float(cmd.turning),
                    ]
                ),  # fmt: skip
            )
        )
        sim.step(self.pol(sim.state(), cmd, arms))
        self.last_cmd = cmd
        self.k += 1
        return True

    def commanded_jaw(self, side: str) -> np.ndarray:
        return self.cmd_jaw[side][0].copy()

    def arrays(self) -> dict:
        out = super().arrays()
        out["amo_cmd"] = np.array([r["cmd"] for r in self.trace], dtype=np.float32)
        return out


__all__ = ["AMO_MOTOR", "ANKLE", "AmoConfig", "AmoMotor", "ARM_READY"]
