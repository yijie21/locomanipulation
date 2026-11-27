"""`Motor` on HEFT: the scene skills' primitives on LocoManiLab's whole-body motion tracker instead of ScaleBFM.

`HeftMotor` keeps `backends.scalebfm.motor.Motor`'s interface and reference logic -- posture / reach / reach_joints build
the same whole-body IK poses, grip / release / hold the same jaw ramps, `_stand` holds the measured pose -- and replaces
how a reference reaches the controller and how the robot walks:

  reference  every tick the motor's reference pose (pelvis position, pelvis wxyz, 29 joints) becomes one qpos frame, the
             frame LocoManiLab pushes to HEFT each tick (lml_host.HostDriver.push); the policy reads the frames at its
             13 steps (0..6 ahead, 1..16 behind). The frames behind are the ones fed on earlier ticks (a log, as
             RealtimeMotionBuffer keeps them), the ones ahead come from the reference itself. HeftConfig.ref_delay_s
             shifts the whole window back: 0 (default) reads the motor's reference from now on, as ScaleBFM's keypoints
             do on this bed; 0.16 reads it as LocoManiLab's buffer does (frame 0 = the frame fed 8 ticks ago, nothing
             newer than the frame fed 2 ticks ago).
  root xy    HEFT reads the reference pelvis only relative to itself (displacements between its frames in frame 0's
             heading, attitude against the IMU, height), so the frames are kept continuous in xy: when a primitive's
             reference starts somewhere else than the last frame fed (a walk's end -> the measured stance), the jump is
             absorbed into an xy offset, as LocoManiLab's drivers shift each new plan onto the current reference root
             (d3_collect.plan_reach: delta = reference root - measured root). Height and attitude are fed as they are.
  walk       HEFT has no velocity command: walk_to / turn / sidestep play a footstep gait reference, carried from the
             measured pelvis onto the current reference: gait_heft.HEFT by default (hg_walk's plan with step shapes HEFT
             keeps up with: high, early-landing forward steps, low turn steps, a move under 0.30 m stepped aside
             directly), or gait.py (vendored from hardgrasp_20261125's hg_walk: turn toward the goal, walk straight in
             20 cm / 0.55 s steps, turn to the goal heading) with `gait=None`; both end in the start stance held 1.5 s.
             Open loop: no stop servo (stop_ki), `speed` and `face` are the gait's own.

No servo, hand program or arm IK of ours runs between the reference and HEFT (`servo` and `hand` are ignored, as under
SONIC): the reference joints are what HEFT tracks. `last_fut` holds the reference wrist poses (FK of the reference) so
`commanded_jaw` is the reference jaw centre, as under ScaleBFM and SONIC.

Reset: HeftSim's reset pose (LocoManiLab's stance by default, runtime.HeftConfig.reset_pose), feet 5 mm above the floor,
the policy's history cleared, then `settle_s` of HEFT holding that measured stance off the round's clock (its boot
indicator runs out after 25 steps, 0.5 s), then the measured stance.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from locomani.backends.heft import gait as G
from locomani.backends.heft import gait_heft as GH
from locomani.backends.heft import runtime as H
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import (
    JAW_OPEN,
    MODE_WALK,
    SIDES,
    WRIST,
    Motor,
    MotorConfig,
    Outcome,
    Pose,
    lowest_foot_point,
)


@dataclass(frozen=True)
class HeftMotorConfig:
    settle_s: float = 1.0  # s of HEFT holding its reset stance before t = 0 (covers the 0.5 s boot indicator)
    gait: GH.HeftGait | None = GH.HEFT  # walk_to's gait (gait_heft); None: gait.py, hg_walk's reference as it ran


def frame_of(pose: Pose) -> np.ndarray:
    """A reference pose as a qpos frame: pelvis xyz, wxyz, 29 joints (MuJoCo order)."""
    return np.r_[np.asarray(pose.pos, float), np.asarray(pose.quat, float), np.asarray(pose.joints, float)]


class HeftMotor(Motor):
    """`Motor` on HEFT. `sim` is a HeftSim, `policy` a HeftPolicy; `model_xml` the robot-only model for IK and the gait."""

    def __init__(self, sim: H.HeftSim, policy: H.HeftPolicy, config: MotorConfig | None = None, *, model_xml: str,
                 track: tuple[str, ...] = (), jacobian: str = "analytic", heft: HeftMotorConfig | None = None):  # fmt: skip
        super().__init__(sim, policy, config, model_xml=model_xml, track=track, jacobian=jacobian)
        self.heft = heft or HeftMotorConfig()
        self.steps = np.asarray(policy.future_steps, int)
        self.delay_f = int(round(policy.cfg.ref_delay_s / R.DT))
        self.legs: G.LegModel | None = None  # the gait's leg IK, on the IK model (built on the first walk)
        self.log: list[np.ndarray] = []  # the frames fed, one a tick (settle included), oldest first
        self.offset = np.zeros(2)  # xy added to the motor's reference to keep the fed frames continuous
        self._ref_seen = None
        self.frames: np.ndarray | None = None  # the last tick's 13 frames (the policy's input reference)

    # -- reset ------------------------------------------------------------------------------------------------------
    def reset(self, xy, yaw: float, *, warm_up: int = 0) -> None:
        sim = self.sim
        sim.reset(np.asarray(xy, float), float(yaw), 0.0)  # HeftSim's reset pose (HeftConfig.reset_pose)
        sim.d.qpos[2] += 0.005 - lowest_foot_point(sim.m, sim.d, self.foot_bodies)
        mujoco.mj_forward(sim.m, sim.d)
        self.k, self.trace, self.fell_at = 0, [], None
        self.jaw[:] = JAW_OPEN
        self.servo, self.servo_target, self.hand_override = {}, {}, {}
        self.servo_on = {s: False for s in SIDES}
        self.log, self.offset, self._ref_seen = [], np.zeros(2), None
        self.pol.reset(sim.state())
        self._stand()  # leg odometry (OdoRoot wraps _stand) anchors here, before the settle
        for _ in range(int(round(self.heft.settle_s / R.DT))):
            sim.set_jaw(self.jaw)
            sim.step(self.pol(sim.state(), self._frames()))
        self._stand()
        self.stand_pose = self.ref(0.0)

    # -- the reference ------------------------------------------------------------------------------------------------
    def _set_mode(self, mode: int) -> None:
        self.mode = mode

    def _walk_overshoot(self, speed: float) -> float:
        return 0.0  # the gait walks its footsteps; ScaleBFM's stop shortfall is not HEFT's

    def _fed(self, pose: Pose) -> np.ndarray:
        f = frame_of(pose)
        f[:2] += self.offset
        return f

    def _frames(self) -> np.ndarray:
        """This tick's frame onto the log, and the policy's 13 frames (future_steps order)."""
        if self.ref is not self._ref_seen:  # a new primitive's reference: absorb its xy jump from the old one, now
            if self._ref_seen is not None:
                jump = np.asarray(self._ref_seen(self.t).pos[:2], float) - np.asarray(self.ref(self.t).pos[:2], float)
                self.offset = self.offset + jump
            self._ref_seen = self.ref
        self.log.append(self._fed(self.ref(self.t)))
        del self.log[: -(int(-self.steps.min()) + self.delay_f + 1)]
        out = np.empty((len(self.steps), 36))
        for i, k in enumerate(self.steps):
            j = int(k) - self.delay_f  # ticks from now
            out[i] = self._fed(self.ref(self.t + j * R.DT)) if j > 0 else self.log[max(j - 1, -len(self.log))]
        self.frames = out
        return out

    def tick(self) -> bool:
        sim = self.sim
        ref = self.ref(self.t)
        frames = self._frames()
        links = self.kp(ref.pos, ref.quat, ref.joints)
        self.last_fut = {WRIST[s]: (links[WRIST[s]][0][None], links[WRIST[s]][1][None]) for s in SIDES}
        sim.set_jaw(self.jaw)
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
                ref_wrists=np.array([self.last_fut[WRIST[s]][0][0] for s in SIDES]),
                wrists=np.array([sim.body(WRIST[s])[0] for s in SIDES]),
                jaw=self.jaw.copy(),
                fingers=sim.d.qpos[sim.gq].copy(),
                objects={n: np.r_[sim.d.xpos[b], sim.d.xquat[b]] for n, b in self.track.items()},
                heft_ref=frames[0].copy(),  # the frame the policy tracks now (frame 0, after the delay and offset)
            )
        )
        sim.step(self.pol(sim.state(), frames))
        self.k += 1
        return True

    # -- locomotion -------------------------------------------------------------------------------------------------
    def walk_to(self, xy, yaw: float | None = None, *, speed: float | None = None, face: str = "travel") -> Outcome:
        """Walk the pelvis to `xy` (world) and end facing `yaw` (None: the heading of the move, or the current one for a
        move under the gait's MOVE_MIN) on the footstep gait. `speed` and `face` are not used (see the module doc)."""
        cfg = self.cfg
        goal = np.asarray(xy, float)
        p0, q0 = self.pelvis()
        yaw0 = R.yaw_of(q0)
        delta = goal - p0[:2]
        dist = float(np.linalg.norm(delta))
        move_min = G.MOVE_MIN if self.heft.gait is None else self.heft.gait.move_min
        if yaw is None:
            yaw = float(np.arctan2(delta[1], delta[0])) if dist >= move_min else yaw0
        if dist < 0.005 and abs(R.wrap(float(yaw) - yaw0)) < np.radians(1.0):
            return self.hold(cfg.settle_s)
        if self.legs is None:
            self.legs = G.LegModel(self.ik)
        start = self.ref(self.t)
        info: dict = {}
        if self.heft.gait is None:
            frames = G.walk_frames(self.legs, frame_of(start), (p0[:2], yaw0), goal, float(yaw), info=info)
        else:
            frames = GH.walk_frames(self.legs, frame_of(start), (p0[:2], yaw0), goal, float(yaw), self.heft.gait, info)
        t_start, n = self.t, len(frames)

        def at(t):
            i = int(round((t - t_start) / R.DT)) - 1  # frame i plays at t_start + (i + 1) DT
            if i < 0:
                return start
            f = frames[min(i, n - 1)]
            return Pose(f[:3].astype(float), f[3:7].astype(float), f[7:].astype(float), np.zeros(29))

        self.pinned = None
        self._set_mode(MODE_WALK)
        self.ref = at
        total = n * R.DT + self.delay_f * R.DT

        def update(t):
            if t - t_start < total:
                return None
            err = float(np.linalg.norm(goal - self.sim.d.qpos[:2]))
            return Outcome(
                True,
                "" if err <= cfg.stop_tolerance else "stopped outside tolerance",
                info=dict(stop_err_cm=round(100 * err, 2), yaw_err_deg=round(float(np.degrees(R.wrap(self.yaw() - yaw))), 2),
                          walked_m=round(dist, 3), steps=info["n_steps"], gait_s=round(info["total_s"], 2),
                          gait_ik_cm=round(info["ik_err_cm"], 2)),
            )  # fmt: skip

        out = self._run("walk_to", update, total + 1.0)
        if self.fell_at is None:
            self._stand()
        return out

    def arrays(self) -> dict:
        out = super().arrays()
        out["heft_ref"] = np.array([r["heft_ref"] for r in self.trace], dtype=np.float32)
        return out


__all__ = ["HeftMotor", "HeftMotorConfig", "frame_of"]
