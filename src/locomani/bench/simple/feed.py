"""The hand estimate's feed for the camera track (round 6, 2026-11-18; moved from escalation_20261116/feed_lml.py, TODO 49.2): perception/odometry.py's JawFeed with what
this bed's robot can read and the estimator was missing.

1. `patch_torques(sim)`: the mirrored model's qfrc_actuator is never written (the environment holds the physics), so leg_odo's load
   weights were all zero and the feet were averaged 50/50. A robot knows its own motors' torques: here the PD command it sent,
   clip(kp (q_des - q) - kd qdot, limits), written where leg_odo reads it.
2. `SlipFeed(JawFeed)` (default off): weights the two anchors by load and by how far each foot is from slipping, from the ankle force
   the leg torques give (horizontal against vertical), and re-seats a foot's anchor from the kinematics while it is slipping.
"""

import numpy as np

from locomani.perception import odometry as LO


def patch_torques(sim) -> None:
    """After every step, qfrc_actuator[vadr] = the PD torque the robot commanded (its own motor reading)."""
    _step = sim.step

    def step(tgt):
        d = sim.d
        q, qd = d.qpos[sim.qadr].copy(), d.qvel[sim.vadr].copy()
        tau = np.clip(sim.kp * (np.asarray(tgt, float) - q) - sim.kd * qd, -sim.tlim, sim.tlim)
        _step(tgt)
        sim.d.qfrc_actuator[sim.vadr] = tau

    sim.step = step


def ankle_forces(odo: "LO.LegOdometry", q, tau, grav):
    """(2, 3) force at each ankle in the level frame, from its leg's torques through the transpose of the ankle Jacobian
    (leg_odo.feet's computation, vertical part sign flipped = the load)."""
    import mujoco

    odo.feet(q, tau, grav)  # leaves the private model at q
    L = LO.level(grav)
    out = np.zeros((2, 3))
    for k, s in enumerate(LO.SIDES):
        mujoco.mj_jacBody(odo.m, odo.d, odo.jp, odo.jr, odo.ank[s])
        J = np.r_[odo.jp, odo.jr][:, odo.vadr[LO.LEGS[s]]]
        w = np.linalg.lstsq(J.T, np.asarray(tau, float)[LO.LEGS[s]], rcond=None)[0]
        out[k] = L @ w[:3]
    return out


class SeatedOdometry(LO.LegOdometry):
    """leg_odo.LegOdometry in load mode, with each foot's anchor re-seated while the foot is off the floor.

    The diagnostic (round 6): while the arm goes out ScaleBFM steps the right foot 10 cm sideways in 4 ticks, both feet unloaded;
    leg_odo took the step as the pelvis moving (50/50 fallback when both loads are zero, and the torques that give the loads were
    never filled in on this bed). Here a foot counts only after `n_on` ticks with its load over `on_n` newtons; until then its anchor
    is where the kinematics put it (its contribution to the pelvis is zero), so a stepped foot is planted where it lands; with
    neither foot down the pelvis estimate holds."""

    def __init__(self, *a, on_n: float = 40.0, n_on: int = 5, hold_frac: float = 0.0, **k):
        super().__init__(*a, **k)
        self.hold_frac = hold_frac
        self.on_n, self.n_on, self.cnt, self.w0, self.nw = on_n, n_on, np.zeros(2, int), 0.0, 0

    def start(self, q, tau, gyro, grav) -> None:
        super().start(q, tau, gyro, grav)
        self.cnt, self.w0, self.nw = np.full(2, self.n_on), 0.0, 0

    def step(self, q, tau, gyro, grav, dt: float) -> None:
        rate = self._rate(gyro, grav)
        self.yaw += 0.5 * (rate + self.last_rate) * dt
        self.last_rate = rate
        f, load = self.feet(q, tau, grav)
        fw = (self._rot() @ f.T).T
        self.cnt = np.where(load >= self.on_n, self.cnt + 1, 0)
        w = load * (self.cnt >= self.n_on)
        tot = float(load.sum())
        if self.nw < 10:  # the weight the feet carry at the tie: the robot standing
            self.w0, self.nw = (self.w0 * self.nw + tot) / (self.nw + 1), self.nw + 1
        elif (
            tot < self.hold_frac * self.w0
        ):  # most of the weight is off both feet (a step in progress): no foot is a fixed point
            w = np.zeros(2)
        if w.sum() > 0:
            self.p = self.p - ((w / w.sum())[:, None] * (fw - self.prev)).sum(0)
        self.prev = fw


class SlipFeed(LO.JawFeed):
    """JawFeed whose odometry is the SeatedOdometry (default off: bed_lml_cam's setting `seat_feed`)."""

    def anchor(self, p, yaw: float, wrist: dict, off) -> None:
        if not isinstance(self.odo, SeatedOdometry):
            self.odo = SeatedOdometry(self.env.sim.m, self.env.sim.qadr, self.env.sim.vadr, mode="load", pow_=1.0)
        super().anchor(p, yaw, wrist, off)
