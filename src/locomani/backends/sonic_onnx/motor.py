"""`Motor` on SONIC (teleop mode, in process): the scene skills' primitives on SONIC instead of ScaleBFM.

`SonicMotor` keeps `backends.scalebfm.motor.Motor`'s interface and reference logic -- walk_to / turn / sidestep build the
same pelvis path, posture / reach the same whole-body IK poses, grip / release / hold the same jaw ramps -- and replaces
only how a reference reaches the controller, every tick:

  upper body  the VR 3-point targets of the reference: its left and right wrist and its head point (torso + 0.35 m),
              from FK of the reference pose, expressed in the pelvis frame the robot believes it has (`vr_frame`
              "robot": the pelvis estimate and the IMU attitude), so SONIC is asked for the reference's world wrist
              poses, as ScaleBFM is (its input is the world keypoints minus its root estimate). SONIC's teleop encoder
              reads these and decides the joints itself.
  legs        SONIC's own velocity planner. While walking (MODE_WALK, `walk` "goal"): SLOW_WALK toward the walk's end
              pose, braking as sqrt(2 decel d) from v_cruise, IDLE within stop_radius, on `root_est` (leg odometry when
              it is on, else the simulator's), facing the reference heading. Standing primitives: IDEL_SQUAT at the
              reference pelvis height (rounded to height_step) once it drops below squat_enter, back to IDLE above
              squat_exit (IDEL_SQUAT is the only planner mode that honours a height,
              backends.sonic.agent_server.LocomotionMode), else IDLE; the facing is the reference heading either way, so
              a turn in place is the planner's.

The walk and VR-frame defaults were chosen on a800 2026-11-16 (sonic_selfmodel_20261115_a800, "预测修订"): 24
approach-like walks (APPROACH_BOX, seed 99000000, apart from the data seeds) per setting, decel 0.3 / v_cruise 0.45
stop error median 9.4 cm (kp_goal 1.0 / 0.35: 17.8 cm, 1 fall); 5 smoke reaches, "robot" frame 2.0-7.3 cm against
"reference" 1.0-45 cm (the squat reach lost the pelvis' own 7 cm in the reference frame).

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
    # walking. "goal": head for the walk's end pose at a speed that falls with the distance left, IDLE within
    # stop_radius; "track": follow the moving reference point (path velocity + kp_xy x error). Both on the pelvis
    # estimate. The commands are rounded (speed_step, dir_step_deg) so the 10 Hz replan rule does not restart the
    # gait on every float change.
    walk: str = "goal"
    v_cruise: float = 0.45  # m/s, "goal": the top speed (SLOW_WALK delivers ~0.8 of it, planner probe 2026-11-16)
    kp_goal: float = 1.0  # 1/s, "goal" with decel 0: speed = kp_goal x distance left, within [v_min, v_cruise]
    decel: float = 0.3  # m/s^2, "goal": > 0 brakes as sqrt(2 decel d) instead (cruise until close)
    stop_radius: float = 0.03  # m, "goal": IDLE inside this
    kp_xy: float = 1.5  # 1/s, "track"
    v_min: float = 0.10  # m/s: SLOW_WALK's slowest useful speed (planner probe 2026-11-16: 0.10 -> 0.074 m/s)
    v_max: float = 0.60  # m/s: SLOW_WALK's top (0.1 + 0.5 x stick in the official VR sender)
    speed_step: float = 0.05  # m/s
    dir_step_deg: float = 5.0
    # the frame the VR 3-point targets are expressed in. "robot": the pelvis the robot believes it has (the pelvis
    # estimate, the IMU attitude), so SONIC is asked for the reference's WORLD wrist poses -- what ScaleBFM's input is
    # (world keypoints minus its root estimate); "reference": the reference pose's own pelvis (the teleop sender's
    # convention, which leaves the pelvis' own error in the hand)
    vr_frame: str = "robot"
    # standing vs squatting, with hysteresis. IDLE stands its pelvis at ~0.785 whatever the reference says. The motor's
    # standing reference is the MEASURED pelvis (Motor._stand), so a squat threshold just under the standing height
    # (0.775 until 2026-11-16) turned every stand into a squat at that height, the robot sagged a little, the next
    # stand re-anchored lower, and the unrounded height replanned every 0.1 s: 75 % of stand commands ended in squat
    # mode at 0.746 m and 5-7 % of segments settled (sonic_selfmodel_20261115_a800, independent review)
    squat_enter: float = 0.70  # m: reference pelvis below this -> IDEL_SQUAT
    squat_exit: float = 0.73  # m: ... above this -> IDLE again
    height_step: float = 0.02  # m: the squat height sent, rounded (a gamepad's resolution, not a float stream)
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
        self.squatting = False

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
        self.squatting = False
        self.pol.reset(sim.state())
        # anchor the pelvis estimate here before the settle (leg odometry restarts on _stand, as Motor.reset's does at
        # once): otherwise the settle's ticks integrate onto the PREVIOUS episode's anchor, and two runs of the same
        # reset differ by what the worker ran before it (a800 2026-11-16: 24.74 / 20.00 cm alone, 24.71 / 20.77 after
        # another reset)
        self._stand()
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

    def _walk(self, v: np.ndarray, top: float, facing: np.ndarray) -> S.Movement:
        """SLOW_WALK along `v` (m/s, world xy), its speed in [v_min, top] and rounded, its direction rounded."""
        c = self.sonic
        n = float(np.linalg.norm(v))
        speed = round(float(np.clip(n, c.v_min, top)) / c.speed_step) * c.speed_step
        step = math.radians(c.dir_step_deg)
        a = round(math.atan2(v[1], v[0]) / step) * step
        return S.Movement(S.SLOW_WALK, speed, np.array([math.cos(a), math.sin(a), 0.0]), facing, -1.0)

    def _movement(self, ref: Pose) -> S.Movement:
        c, cfg = self.sonic, self.cfg
        facing = self._facing(R.yaw_of(ref.quat))  # constant except while the path turns: not rounded
        idle = S.Movement(S.IDLE, -1.0, np.zeros(3), facing, -1.0)
        if self.mode == MODE_WALK:
            self.squatting = False
            est = self._pelvis_estimate()[:2]
            if c.walk == "goal":
                left = self.ref(self.t + 1e3).pos[:2] - est  # walk_to's reference clamps to its end pose
                dist = float(np.linalg.norm(left))
                if dist <= c.stop_radius:
                    return idle
                speed = math.sqrt(2.0 * c.decel * dist) if c.decel > 0 else c.kp_goal * dist
                return self._walk(left / dist * speed, c.v_cruise, facing)
            v_ref = (self.ref(self.t + R.DT).pos[:2] - ref.pos[:2]) / R.DT
            err = ref.pos[:2] - est
            if float(np.linalg.norm(v_ref)) > 1e-4 or float(np.linalg.norm(err)) > cfg.stop_tolerance:
                return self._walk(v_ref + c.kp_xy * err, c.v_max, facing)
            return idle
        z = float(ref.pos[2])
        self.squatting = z < (c.squat_exit if self.squatting else c.squat_enter)
        if self.squatting:
            h = max(round(z / c.height_step) * c.height_step, c.squat_min)
            return S.Movement(S.IDEL_SQUAT, -1.0, np.zeros(3), facing, h)
        return idle

    def _command(self, ref: Pose) -> S.SonicCommand:
        links = self.kp(ref.pos, ref.quat, ref.joints)
        if self.sonic.vr_frame == "robot":
            vr_pos, vr_quat = S.vr_targets(links, self._pelvis_estimate(), self.sim.d.qpos[3:7])
        else:
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
