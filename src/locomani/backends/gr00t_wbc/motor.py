"""GR00T WBC motor layer (TODO 34): the scene skills' eight primitives on NVIDIA's gear WBC instead of ScaleBFM.

The gear WBC takes the same kind of command AMO does -- planar speed, a heading, base height, torso roll / pitch / yaw
-- and leaves the arms to the caller, so `Gr00tMotor` is the AMO adapter (`backends.amo.motor.AmoMotor`: pelvis path ->
speed with feed-forward and a pelvis-error gain, height and torso pitch with integrals on the measurement, a per-tick
arm IK on the measured pelvis and waist with ScaleBFM's jaw-centre servo) with three changes:
  reset     from the gear WBC's default pose (knees bent 0.3 rad, hips -0.1, ankles -0.2; arms at ARM_READY), standing
            1 s on the Balance net before the round's clock starts;
  gains     its own feed-forward and minimum speed (`Gr00tConfig`, set from the admission runs);
  posture   optionally rhoyn's walking posture: while walking, the base-height and torso-pitch commands fixed at
            `walk_height` / `walk_pitch` (the board's gr00t_wbc_h066_p012: 0.66 m, 0.12 rad), standing primitives
            unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

from locomani.backends.amo.motor import AMO_MOTOR, ARM_READY, AmoConfig, AmoMotor
from locomani.backends.gr00t_wbc import runtime as G
from locomani.backends.scalebfm.motor import JAW_OPEN, MODE_WALK, SIDES, MotorConfig, lowest_foot_point


@dataclass(frozen=True)
class Gr00tConfig(AmoConfig):
    ff_gain: float = 1.0  # x the path speed (AMO needed 2.0; the gear WBC's tracking is measured in admission)
    v_min: float = 0.10  # m/s floor on a pelvis error past the stop tolerance
    pitch_relative: bool = False  # the gear WBC stands upright at a 0 pitch command
    walk_height: float | None = None  # m: base-height command while walking (rhoyn's best row: 0.66)
    walk_pitch: float | None = None  # rad: torso-pitch command while walking (rhoyn's best row: 0.12)


GR00T_MOTOR = AMO_MOTOR
VARIANTS = {"gr00t_wbc": Gr00tConfig(), "gr00t_wbc_h066_p012": Gr00tConfig(walk_height=0.66, walk_pitch=0.12)}


class Gr00tMotor(AmoMotor):
    """`Motor` on the gear WBC. `sim` is a Gr00tSim, `policy` a Gr00tPolicy; `model_xml` the robot-only model for IK."""

    def __init__(self, sim: G.Gr00tSim, policy: G.Gr00tPolicy, config: MotorConfig | None = None, *, model_xml: str,
                 track: tuple[str, ...] = (), jacobian: str = "analytic", gr00t: Gr00tConfig | None = None):  # fmt: skip
        super().__init__(sim, policy, config or GR00T_MOTOR, model_xml=model_xml, track=track, jacobian=jacobian,
                         amo=gr00t or Gr00tConfig())  # fmt: skip

    def reset(self, xy, yaw: float, *, warm_up: int = 0) -> None:
        sim = self.sim
        sim.reset(np.asarray(xy, float), float(yaw), 0.0)
        sim.d.qpos[sim.qadr] = 0.0
        sim.d.qpos[self.kp.qadr[:15]] = G.DEFAULT
        sim.d.qpos[self.kp.qadr[G.ARMS]] = ARM_READY
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
        stand = G.Command(yaw=y, height=G.HEIGHT0)
        for _ in range(int(self.amo.settle_s / G.DT)):  # not on the round's clock
            sim.grip_target[:] = JAW_OPEN
            self._step(self.pol(self._mj_state(), stand, ARM_READY))
        self.nominal = self.joints()[:15].copy()
        self._stand()
        self.stand_pose = self.ref(0.0)
        self.pitch0 = 0.0

    def _command(self, ref):
        cmd = super()._command(ref)
        a = self.amo
        if self.mode == MODE_WALK and a.walk_height is not None:
            cmd = replace(cmd, height=float(a.walk_height), torso_pitch=float(a.walk_pitch or 0.0))
        return cmd

    def arrays(self) -> dict:
        out = super().arrays()
        out["gr00t_cmd"] = out.pop("amo_cmd")
        return out


__all__ = ["GR00T_MOTOR", "Gr00tConfig", "Gr00tMotor", "VARIANTS"]
