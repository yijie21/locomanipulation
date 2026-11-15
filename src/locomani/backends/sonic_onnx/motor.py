"""`Motor` on SONIC (teleop mode, in process): the scene skills' primitives on SONIC instead of ScaleBFM.

`SonicMotor` keeps `backends.scalebfm.motor.Motor`'s interface and reference logic -- walk_to / turn / sidestep build the
same pelvis path, posture / reach the same whole-body IK poses, grip / release / hold the same jaw ramps -- and replaces
only how a reference reaches the controller, every tick:

  upper body  the VR 3-point targets of the reference: its left and right wrist and its head point (torso + 0.35 m), in
              the reference pelvis frame, from FK of the reference pose. This is what SONIC's teleop encoder reads; it
              decides the joints itself, as ScaleBFM decides them from the reference keypoints.
  legs        SONIC's own velocity planner. While walking (MODE_WALK): SLOW_WALK along the reference path's velocity
              plus kp_xy x the pelvis error, the pelvis from `root_est` (leg odometry when it is on, else the
              simulator's), facing the reference heading; a finished path within stop_tolerance -> IDLE. Standing
              primitives: IDEL_SQUAT at the reference pelvis height when it is below squat_below (the only planner mode
              that honours a height, backends.sonic.agent_server.LocomotionMode), else IDLE; the facing is the
              reference heading either way, so a turn in place is the planner's.

No servo, hand program or arm IK of ours runs between the reference and SONIC (the self-model data record the
controller, not a correction on top of it). `last_fut` holds the reference wrist poses so `commanded_jaw` -- what the
self-model data record as the command -- is the reference jaw centre, as under ScaleBFM.

Reset: SONIC's default pose (shoulder roll +-0.2), feet 5 mm above the floor, the policy's history and the planner's
Initialize, then `settle_s` of SONIC holding its first (IDLE) plan with the VR targets of that plan's own pose, off the
round's clock (the deploy starts the same way: the planner motion plays before VR takes the arms, zmq_manager.hpp:
686-700).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import (
    JAW_OPEN,
    MODE_WALK,
    SIDES,
    WRIST,
    Motor,
    MotorConfig,
    Pose,
    lowest_foot_point,
)
from locomani.backends.sonic_onnx import runtime as S


@dataclass(frozen=True)
class SonicMotorConfig:
    kp_xy: float = 1.5  # 1/s on the pelvis error while walking
    v_min: float = 0.10  # m/s: SLOW_WALK's slowest useful speed (planner probe 2026-11-16: 0.10 -> 0.074 m/s)
    v_max: float = 0.60  # m/s: SLOW_WALK's top (0.1 + 0.5 x stick in the official VR sender)
    squat_below: float = 0.775  # m: a reference pelvis lower than this is a squat at that height (IDLE stands ~0.78)
    squat_min: float = 0.40  # m: the squat mode's documented lower end (planner_onnx.md)
    settle_s: float = 2.0  # s of SONIC holding its first plan after reset, before t = 0


class SonicMotor(Motor):
    """`Motor` on SONIC. `sim` is a SonicSim, `policy` a SonicPolicy; `model_xml` the robot-only model for IK."""

    def __init__(self, sim: S.SonicSim, policy: S.SonicPolicy, config: MotorConfig | None = None, *,
                 model_xml: str, track: tuple[str, ...] = (), jacobian: str = "analytic",
                 sonic: SonicMotorConfig | None = None):  # fmt: skip
        super().__init__(sim, policy, config, model_xml=model_xml, track=track, jacobian=jacobian)
        self.sonic = sonic or SonicMotorConfig()
        self.cmd: S.SonicCommand | None = None

    # -- reset ------------------------------------------------------------------------------------------------------
    def reset(self, xy, yaw: float, *, warm_up: int = 0) -> None:
        sim = self.sim
        sim.reset(np.asarray(xy, float), float(yaw), 0.0)  # SonicSim's default pose
        sim.d.qpos[2] += 0.005 - lowest_foot_point(sim.m, sim.d, self.foot_bodies)
        mujoco.mj_forward(sim.m, sim.d)
        self.k, self.trace, self.fell_at = 0, [], None
        self.jaw[:] = JAW_OPEN
        self.servo, self.servo_target, self.hand_override = {}, {}, {}
        self.servo_on = {s: False for s in SIDES}
        self.pol.reset(sim.state())
        for _ in range(int(round(self.sonic.settle_s / R.DT))):
            sim.set_jaw(self.jaw)
            sim.step(self.pol(sim.state(), self._plan_command()))
        self._stand()
        self.stand_pose = self.ref(0.0)

    def _plan_command(self) -> S.SonicCommand:
        """IDLE, and the VR targets of the playing plan's own pose (what the g1 mode would track)."""
        pos, quat, q = self.pol.plan_pose()
        links = self.kp(pos, quat, q)
        vr_pos, vr_quat = S.vr_targets(links, pos, quat)
        return S.SonicCommand(S.Movement(S.IDLE, -1.0, np.zeros(3), self._facing(R.yaw_of(self.sim.d.qpos[3:7])), -1.0),
                              vr_pos, vr_quat)  # fmt: skip

    # -- the tick ---------------------------------------------------------------------------------------------------
    def _walk_overshoot(self, speed: float) -> float:
        return 0.0  # the walk closes on the pelvis estimate here; ScaleBFM's stop shortfall is not SONIC's

    def _set_mode(self, mode: int) -> None:
        self.mode = mode

    @staticmethod
    def _facing(yaw: float) -> np.ndarray:
        return np.array([math.cos(yaw), math.sin(yaw), 0.0])

    def _pelvis_estimate(self) -> np.ndarray:
        est = self.pol.root_est
        return self.sim.d.qpos[:3].copy() if est is None else np.asarray(est, float)

    def _movement(self, ref: Pose) -> S.Movement:
        c, cfg = self.sonic, self.cfg
        facing = self._facing(R.yaw_of(ref.quat))
        if self.mode == MODE_WALK:
            v_ref = (self.ref(self.t + R.DT).pos[:2] - ref.pos[:2]) / R.DT
            err = ref.pos[:2] - self._pelvis_estimate()[:2]
            v = v_ref + c.kp_xy * err
            moving = float(np.linalg.norm(v_ref)) > 1e-4
            if moving or float(np.linalg.norm(err)) > cfg.stop_tolerance:
                n = float(np.linalg.norm(v))
                speed = float(np.clip(n, c.v_min, c.v_max))
                return S.Movement(S.SLOW_WALK, speed, np.r_[v / max(n, 1e-9), 0.0], facing, -1.0)
            return S.Movement(S.IDLE, -1.0, np.zeros(3), facing, -1.0)
        z = float(ref.pos[2])
        if z < c.squat_below:
            return S.Movement(S.IDEL_SQUAT, -1.0, np.zeros(3), facing, max(z, c.squat_min))
        return S.Movement(S.IDLE, -1.0, np.zeros(3), facing, -1.0)

    def _command(self, ref: Pose) -> S.SonicCommand:
        links = self.kp(ref.pos, ref.quat, ref.joints)
        vr_pos, vr_quat = S.vr_targets(links, ref.pos, ref.quat)
        self.last_fut = {WRIST[s]: (links[WRIST[s]][0][None], links[WRIST[s]][1][None]) for s in SIDES}
        return S.SonicCommand(self._movement(ref), vr_pos, vr_quat)

    def tick(self) -> bool:
        sim = self.sim
        ref = self.ref(self.t)
        cmd = self._command(ref)
        sim.set_jaw(self.jaw)
        if sim.fallen():
            self.fell_at = self.t if self.fell_at is None else self.fell_at
            return False
        mv = cmd.movement
        self.trace.append(
            dict(
                t=self.t,
                primitive=self.primitive,
                mode=self.mode,
                robot=sim.d.qpos[:7].copy(),
                joints=self.joints(),
                ref_pelvis=ref.pos.copy(),
                ref_wrists=np.array([self.last_fut[WRIST[s]][0][0] for s in SIDES]),
                wrists=np.array([sim.body(WRIST[s])[0] for s in SIDES]),
                jaw=self.jaw.copy(),
                fingers=sim.d.qpos[sim.gq].copy(),
                objects={n: np.r_[sim.d.xpos[b], sim.d.xquat[b]] for n, b in self.track.items()},
                sonic=np.r_[mv.mode, mv.speed, mv.movement, mv.facing, mv.height, self.pol.frame],
            )
        )
        sim.step(self.pol(sim.state(), cmd))
        self.cmd = cmd
        self.k += 1
        return True

    def arrays(self) -> dict:
        out = super().arrays()
        out["sonic_cmd"] = np.array([r["sonic"] for r in self.trace], dtype=np.float32)
        return out


__all__ = ["SonicMotor", "SonicMotorConfig"]
