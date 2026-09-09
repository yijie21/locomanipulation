"""The grasp, on SONIC's streamed-motion channel.

Planner-mode wrist targets pass through the kinematic planner and arrive with
a 25 cm warp and a second of lag; three weeks of tuning got the closed loop to
3.7 cm, which is not a grasp. Pico teleop never touches that channel — it
streams full 29-DoF joint references on the `pose` topic and the policy
imitates them. diag_stream.py measured this channel at ~1 cm palm error with
one or two IK refinement steps, standing rock-solid the whole time.

This module replaces the planner-era crouch/track/servo_grasp/grasp/lift chain
for the grasp stage only: walking and carrying stay in planner mode, and the
deploy is switched back the moment the object is lifted.

The servo logic is servo_grasp's (align across the jaw at a safe depth, then
advance gated on alignment) — what changes is the actuation underneath it:
IK on the scene model streamed as joint references, instead of wrist targets
nudged through the planner.
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
from arm_ik import ArmIK
from execution_record import ExecutionRecord, GraspResult
from g1_joints import ARM_MJ, mujoco_to_isaac
from primitives import (
    DT,
    GRASP_CENTRE_OFFSET,
    HAND_CLOSED_LEFT,
    HAND_CLOSED_RIGHT,
    HAND_OPEN,
    LIFT_HEIGHT,
    LocomotionMode,
    Primitives,
    StreamMode,
    wrap,
)
from reach_config import DEFAULT_REACH, ReachConfig

LIFT_MIN = 0.06  # the object must actually leave the table


class ApproachStatus(str, Enum):
    READY = "ready"
    TIMEOUT = "timeout"
    FALLEN = "fallen"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ApproachResult:
    status: ApproachStatus
    depth: float
    elapsed: float
    azimuth: float


# Compatibility snapshot for legacy grasp_streamed callers only. The normal
# collector owns its ExecutionRecord; depth gates use yaw-only coordinates.
LAST_TRANSITION = {}
# Shared depth limit, independent of the sampled navigation waypoint. The
# bottle pilot physically grasped at yaw-frame depths 0.294..0.381 m. This
# 0.40 m gate is a screening bound, not a collision-free IK certificate;
# post-crouch replanning and actual grasp verification remain mandatory.
TRANSITION_POST_MAX = DEFAULT_REACH.post_max


def approach_depth_band(post_max, backshift=0.08, margin=0.01):
    """Invert an estimated crouch backshift, not a reachability certificate.

    8 cm is the existing measured prior, not a guaranteed displacement.
    Actual post-crouch geometry remains the mandatory acceptance gate.
    """
    return ReachConfig(post_max=post_max, backshift=backshift, depth_margin=margin).approach_band()


CHUNK = 5  # pose frames per message, like the pico manager
JOINT_SPEED = 0.8  # rad/s for large repositioning moves
FINE_SPEED = 0.4  # rad/s inside the servo
# TRIMMED FOR IMITATION DATA (2026-08-25). These settles were tuned for
# scripted-expert reliability, and for the script they are free -- but they are
# recorded, and a VLA trained on them learns them: measured over the bottle
# datasets, arrival->reach sat at median 4.6 s of quasi-static standing (the
# human-teleop reference the official recipe trains from has 0.2 s), and every
# pause longer than one 40-frame chunk is a state where "keep standing" is the
# majority label -- the exact standing-still attractor both closed-loop rounds
# died in. The 50 Hz servo re-aims on the measured object continuously, so a
# settled base is a comfort, not a requirement.
SETTLE = 0.8  # s after a repositioning move (was 2.0)
REFINE_TOL = 0.015  # m, palm error to stop refining the pregrasp at
REFINE_MAX = 3
from primitives_const import HAND as _HAND  # noqa: E402

# hold the object this far out along the entry axis. dex1's fingertips
# LEAD the grasp centre by 3.3 cm along the approach (tips at wrist+18.5,
# grasp centre at wrist+15.15) — the whole slide-in geometry needs that
# lead budgeted or the tips reach the can while the servo still thinks
# it is 4-6 cm out.
SAFE_GAP = 0.17 if _HAND == "dex1" else 0.13
# (0.10 left only 5 cm of margin over a bad pregrasp, and 5 cm pregrasp
# errors do happen — the fingertips grazed the can twice)
DEPTH_TOL = 0.03

# Fraction of servo ticks with the object inside the ego camera's central 90%
# (legacy callers only). New task code reads GraspResult.visibility so a
# failed or camera-less attempt cannot inherit a preceding episode's value.
LAST_EGO_VIS = None
# Looser than servo_grasp's 6 mm: that loop nudged a 50 Hz drifting command,
# this one hops between IK solutions with a ~1 cm delivery floor, so an 8 mm
# gate just oscillates. The advance gate keeps re-centring on the way in.
ALIGN_TOL = 0.013
GATE_TOL = 0.018
GRASP_TOL = 0.020
STEP_MAX = 0.03  # m, largest palm correction per servo step
NUDGE_MAX = 0.05  # m, object displacement that aborts the attempt
# Squat whenever the grip sits below this pelvis-frame height. Generous on
# purpose: the palm's sweet spot is at or above pelvis height minus a few cm,
# and every centimetre of squat is a centimetre of waist lean the policy does
# not have to deliver (it delivers only ~2/3 of a commanded lean).
# Off-hand guard (2026-09-03): while the right hand grasps, the torso yaws
# toward the object and the FROZEN left arm sweeps across the table -- a
# recurring "bottle toppled" failure. Every pushed frame checks the left
# palm's distance to the object and, inside LEFT_SAFE_M, solves left-arm IK
# to a retreat target radially away from the object (rate-limited, no
# waist). GYM_LEFT_GUARD=0 disables it (A/B).
LEFT_SAFE_M = 0.24  # start retreating inside this palm-object distance
LEFT_RETREAT_M = 0.10  # retreat target sits SAFE + this from the object
LEFT_GUARD_RATE = 1.2  # rad/s cap on the guarded left-arm joints
CROUCH_BACK = 0.09  # m the planner crouch moves the pelvis BACK (measured 4-13 cm, 2026-09-03)
SQUAT_DROP_MAX = 0.20  # deepest crouch the planner squat is asked for
SQUAT_HEIGHT_BIAS = 0.02  # delivered pelvis_z - commanded height (measured)
SQUAT_SETTLE = (
    1.6  # s; covers the k-planner squat's own t90 of 1.1-1.6 s (was 2.5 -- the extra 0.9 s was recorded dead time)
)
SQUAT_TRIGGER = 0.03
SQUAT_GAIN = 6.8  # rad of squat per metre of wanted drop (measured)
SQUAT_MAX = 0.65


def make_arm(p: Primitives, side: str):
    """Build the arm driver for the channel this run is testing.

    The default is now `planner`: it keeps the deploy in planner mode for
    the whole grasp, so there is no mode switch, no latch-clear window and
    no stand twitch. Measured 12/12 with the sim pin OFF against 2/12 for
    the streamed channel with its pin ON, same seed (see the Phase 0
    commit). GYM_ARM_CHANNEL=streamed restores the old path.
    """
    if os.environ.get("GYM_ARM_CHANNEL", "planner") == "planner":
        return PlannerArm(p, side, send_vel=os.environ.get("GYM_ARM_VEL", "1") == "1")

    return StreamedArm(p, side)


def yaw_only(quat_wxyz):
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


class StreamedArm:
    """Owns the pose stream: joint references, hand targets, pacing."""

    def __init__(self, p: Primitives, side: str):
        self.p = p
        self.side = side
        self.ik = ArmIK()
        # The waist may HELP the reach but only within a human lean. With the
        # stock limits the IK bridged any height/side gap by folding the
        # torso flat onto the table (head down, palm never reaching its
        # lifted reference); with the waist excluded entirely, side-offset
        # pregrasps went unreachable (res 0.13). Indices 12/13/14 = waist
        # yaw/roll/pitch in MuJoCo order.
        for j, (lo, hi) in {12: (-0.35, 0.35), 13: (-0.15, 0.15), 14: (-0.05, 0.20)}.items():
            self.ik.lo[j], self.ik.hi[j] = lo, hi
        self.buf: deque = deque(maxlen=CHUNK)
        self.i = 0
        # Start from the commander's CURRENT hand targets, not open: entering
        # streamed mode mid-carry with zeroed hands would uncurl the fingers
        # and drop whatever they hold.
        self.hands = {
            "left": np.array(p.cmd.left_hand, dtype=np.float32),
            "right": np.array(p.cmd.right_hand, dtype=np.float32),
        }
        self.q = None  # current reference, MuJoCo order
        self.root_quat = None
        self.next_t = None
        self.guard = None  # object key to keep the off hand away from
        self._left_now = None  # guarded left-arm joints, carried across move_to
        self.guard_stats = {"frames": 0, "min_d": 9.0}

    # -- plumbing ----------------------------------------------------------

    def start(self) -> None:
        s = self.p.obs()
        # Release the planner's upper-body pin before streaming: the deploy
        # OVERWRITES the upper-body joints of any reference motion while
        # `has_upper_body_control_` is latched (g1_deploy_onnx_ref.cpp:780),
        # and the latch updates per planner message — so one frame WITHOUT
        # the field clears it. Skipping this froze the arm mid-grasp with
        # IK res 0.000 and a motionless palm.
        #
        # This window is what makes the arms twitch at the stand (the deploy
        # is still driving the body, hears "no upper-body target", and hands
        # the arms to the gait prior until the first streamed frame snaps
        # them back), and two ways round it have now been measured and
        # reverted, both 0/6 on seed 11/7 against 5/6 here:
        #   - hold the sim-side pin through the handoff: the arm comes out
        #     of a stiff position loop and the servo cannot absorb it;
        #   - skip the clear entirely and let PLANNER_TIMEOUT do it (the
        #     official Pico manager's order, pico_manager_thread_server.py
        #     :2098): the deploy only evaluates that timeout while the
        #     planner is ENABLED (zmq_manager.hpp:623), so once
        #     enter_streamed() disables it the latch never expires and the
        #     upper body stays pinned to the last frozen target — exactly
        #     the "arm not tracking, res 0.000" failure above.
        # The latch can therefore only be cleared from inside planner mode.
        # What has NOT been tried: hold the arms through the window with
        # VR-3pt wrist targets at the measured palm pose (switch
        # stream_mode to PLANNER_VR_3PT for these 0.4 s) — same control
        # regime as the streamed tracking that follows, so no stiff-pin
        # discontinuity, and the planner stays enabled so the clear still
        # happens.
        self.p.cmd.upper_body = None
        end = time.time() + 0.4
        while time.time() < end:  # one frame raced the deploy's reader;
            self.p.cmd.frame(facing=self.p._facing(s))
            time.sleep(DT)
        self.q = np.array(s["joints"])
        self.root_quat = yaw_only(s["pelvis"]["quat"])
        self.next_t = time.time()
        # Prime the buffer BEFORE the switch — the deploy stops control if
        # streamed mode goes active over an empty motion buffer.
        self.hold(0.3)
        self.p.cmd.enter_streamed()
        # 0.4 s, not 1.0: the replay diag validated the switch with 0.6 s of
        # held frames; what the deploy needs is a non-empty buffer and a few
        # frames of continuity, not a second of recorded standing.
        self.hold(0.4)

    def stop(self) -> None:
        """Hand control back to the planner without a discontinuity."""
        from primitives import upper_body_from_mj

        s = self.p.obs()
        palm_pf = self.p.world_to_root_full(s["palms"][self.side]["pos"], s)
        self.p.cmd.set_wrists(**{f"{self.side}_pos": np.asarray(palm_pf, dtype=np.float32)})
        # In FROZEN_UPPER_BODY mode the planner pins the arms to the 17-value
        # target, so hand back the pose the streamed reach actually ended in
        # — otherwise the switch snaps the arms to the stale target and yanks
        # whatever the hand is holding.
        self.p.cmd.set_upper_body(upper_body_from_mj(s["joints"]))
        self.p.cmd.set_hands(**{self.side: self.hands[self.side]})
        self.p.cmd.exit_streamed()
        # The switch re-anchors the deploy's planner heading at the robot's
        # current yaw (diag_heading.py); record it so frame() can undo the
        # rotation, or the next navigation walks confidently sideways.
        #
        # Read the yaw HERE, not from `s` at the top of this method. The
        # deploy anchors at the instant of the switch, and `s` is several
        # hundred milliseconds older -- which is invisible while the robot
        # is standing still and catastrophic while it is not. reanchor()
        # runs right after start_control, with the robot still rising out of
        # the reset crouch and turning as it rises: measured 50.6 deg of
        # error between the recorded offset and the robot's actual yaw. The
        # settle then quietly turned the robot 10 deg off the heading it was
        # being commanded to hold, and every later facing command was skewed
        # by the same 50 deg, which is what made a 70 deg spawn turn
        # unreachable.
        self.p.cmd.heading_offset = float(self.p.obs()["pelvis"]["yaw"])

    def _push(self, q_mj29) -> None:
        q_mj29 = self._guard_left(np.asarray(q_mj29, dtype=float).copy())
        self.buf.append(mujoco_to_isaac(q_mj29).astype(np.float32))
        self.p.cmd.pose_chunk(
            np.stack(self.buf),
            self.root_quat,
            self.i - len(self.buf) + 1,
            left_hand=self.hands["left"],
            right_hand=self.hands["right"],
        )
        self.i += 1
        self.next_t += DT
        lag = self.next_t - time.time()
        if lag > 0:
            time.sleep(lag)
        else:
            self.next_t = time.time()

    def hold(self, seconds: float) -> None:
        for _ in range(int(round(seconds / DT))):
            self._push(self.q)

    def _guard_left(self, q: np.ndarray, observed=None) -> np.ndarray:
        """Keep the off (left) palm outside LEFT_SAFE_M of the object.

        Reads the live state (palm + object, world), retreats the palm
        radially in the table plane to SAFE + RETREAT, solves left-arm IK
        in the pelvis frame and rate-limits the joints. The retreated pose
        persists (no return to the frozen snapshot) and is carried into
        every later reference via _left_now."""
        if self.guard is None or self.side != "right" or os.environ.get("GYM_LEFT_GUARD", "1") != "1":
            return q
        L = ARM_MJ["left"]
        if self._left_now is not None:
            q[L] = self._left_now
        if observed is not None and time.time() - observed["wall"] > 0.2:
            raise ValueError("stale observation for offhand guard")
        s = self.p.obs() if observed is None else observed
        obj = np.asarray(self.p.object_pos(self.guard, s), dtype=float)
        palm = np.asarray(s["palms"]["left"]["pos"], dtype=float)
        d = float(np.linalg.norm(palm - obj))
        self.guard_stats["min_d"] = min(self.guard_stats["min_d"], d)
        if d >= LEFT_SAFE_M:
            return q
        away = palm[:2] - obj[:2]
        n = np.linalg.norm(away)
        away = away / n if n > 1e-3 else np.array([0.0, 1.0])
        target_w = obj.copy()
        target_w[:2] += away * (LEFT_SAFE_M + LEFT_RETREAT_M)
        target_w[2] = max(palm[2], obj[2] + 0.05)
        target_pf = self.p.world_to_root_full(target_w, s)
        q_sol, res, _ = self.ik.solve("left", target_pf, None, q_mj29=q, use_waist=False)
        step = LEFT_GUARD_RATE * DT
        q[L] = q[L] + np.clip(q_sol[L] - q[L], -step, step)
        self._left_now = q[L].copy()
        self.guard_stats["frames"] += 1
        if self.guard_stats["frames"] == 1:
            self.p.log(f"[guard] left palm {d:.2f} m from {self.guard}: retreating (ik res {res:.3f})")
        return q

    def move_to(self, q_target, speed: float = JOINT_SPEED) -> None:
        q0 = self.q.copy()
        q_target = np.asarray(q_target, dtype=float).copy()
        if self._left_now is not None:
            q0[ARM_MJ["left"]] = self._left_now
            q_target[ARM_MJ["left"]] = self._left_now
        dur = max(0.4, float(np.abs(q_target - q0).max()) / speed)
        n = max(1, int(round(dur / DT)))
        for k in range(1, n + 1):
            u = k / n
            sm = 3 * u * u - 2 * u**3
            self._push(q0 + sm * (q_target - q0))
        self.q = q_target
        if self._left_now is not None:
            self.q[ARM_MJ["left"]] = self._left_now

    # -- behaviours --------------------------------------------------------

    def squat(self, drop_m: float, track: bool = False) -> float:
        """Bend the legs in the reference; returns the measured pelvis drop.

        track=True re-solves the servo arm against the LIVE pelvis pose a
        few times per second so the palm holds its world pose while the
        body sinks — the legs fold under the hand. Without it the frozen
        arm rides down with the body into the can's height band (the old
        pre-lift bounce existed only to survive that sink)."""
        a = float(np.clip(drop_m * SQUAT_GAIN, 0.0, SQUAT_MAX))
        if a < 0.05:
            return 0.0
        s = self.p.obs()
        z0 = s["pelvis"]["pos"][2]
        q_squat = self.q.copy()
        for hip, knee, ank in ((0, 3, 4), (6, 9, 10)):
            q_squat[hip] -= a
            q_squat[knee] += 2 * a
            q_squat[ank] -= a
        if not track:
            self.move_to(q_squat, speed=0.6)
            self.hold(SETTLE)
            s = self.p.obs()
            return float(z0 - s["pelvis"]["pos"][2])
        palm_w = np.array(s["palms"][self.side]["pos"])
        rot_w = Primitives._rot(s["palms"][self.side]["quat"])
        q0 = self.q.copy()
        arm_idx = ARM_MJ[self.side]
        arm_hold = q0[arm_idx].copy()
        dur = max(0.4, float(np.abs(q_squat - q0).max()) / 0.6)
        n = max(1, int(round(dur / DT)))
        for k in range(1, n + 1):
            u = k / n
            sm = 3 * u * u - 2 * u**3
            q_ref = q0 + sm * (q_squat - q0)
            if k % 5 == 1:
                s = self.p.obs()
                aim_pf = self.p.world_to_root_full(palm_w, s)
                rot_pf = Primitives._rot(s["pelvis"]["quat"]).T @ rot_w
                q_sol, res, _ = self.ik.solve(self.side, aim_pf, rot_pf, q_mj29=q_ref, use_waist=False)
                if res < 0.08:
                    arm_hold = q_sol[arm_idx]
            q_ref[arm_idx] = arm_hold
            self.q = q_ref
            self._push(q_ref)
        self.q = q_squat.copy()
        self.q[arm_idx] = arm_hold
        self.hold(SETTLE)
        s = self.p.obs()

        return float(z0 - s["pelvis"]["pos"][2])

    def goto(
        self,
        palm_world,
        rot_world=None,
        refine: int = REFINE_MAX,
        tol: float = REFINE_TOL,
        speed: float = JOINT_SPEED,
        obj_key: str | None = None,
        nudge_max: float = NUDGE_MAX,
    ):
        """Palm to a world pose; re-aim on the measured residual. -> err or None."""
        p = self.p
        obj0 = p.object_pos(obj_key) if obj_key else None
        want_w = np.asarray(palm_world, dtype=float)
        for it in range(refine + 1):
            s = p.obs()
            if s.get("furniture_touch"):
                p.log(f"    TABLE CONTACT during reach: {', '.join(s['furniture_touch'])}")
            if s["fallen"]:
                p.aborted = "fell during the streamed reach"
                return None
            if obj0 is not None:
                moved = float(np.linalg.norm(p.object_pos(obj_key, s) - obj0))
                if moved > nudge_max:
                    p.aborted = f"{obj_key} moved {moved * 100:.0f} cm en route"
                    return None
            palm_w = np.array(s["palms"][self.side]["pos"])
            err = float(np.linalg.norm(palm_w - want_w))
            if it > 0:
                p.log(f"    stream goto it{it}: err {err * 100:.1f} cm")
            if it > 0 and err < tol:
                return err
            # aim at want + measured residual (current residual only — past
            # corrections are transients, accumulating them diverges), the
            # correction capped: mirroring a 20 cm miss overshoots by as much.
            corr = np.clip(want_w - palm_w, -0.05, 0.05) if it > 0 else 0.0
            aim_w = want_w + corr
            aim_pf = p.world_to_root_full(aim_w, s)
            rot_pf = None
            if rot_world is not None:
                rot_pf = Primitives._rot(s["pelvis"]["quat"]).T @ np.asarray(rot_world)
            q_sol, res, _ = self.ik.solve(self.side, aim_pf, rot_pf, q_mj29=self.q, use_waist=True)
            if res > 0.04:
                p.log(f"    stream goto: aim unreachable by {res * 100:.1f} cm")
                if it > 0:
                    return err
            self.move_to(q_sol, speed=speed if it == 0 else FINE_SPEED)
            self.hold(SETTLE if it == 0 else 1.2)
        s = p.obs()

        return float(np.linalg.norm(np.array(s["palms"][self.side]["pos"]) - want_w))

    def close_hand(self, obj_key: str, squeeze: float = 0.28, ramp: float = 1.6) -> bool:
        closed = (HAND_CLOSED_RIGHT if self.side == "right" else HAND_CLOSED_LEFT).astype(np.float32)
        self._closed = closed
        p = self.p
        contact_alpha = None
        n = int(round(ramp / DT))
        for k in range(n + 1):
            alpha = k / n
            if contact_alpha is not None and alpha >= contact_alpha + squeeze:
                break
            self.hands[self.side] = HAND_OPEN + alpha * (closed - HAND_OPEN)
            self._push(self.q)
            if k % 5 == 0 and contact_alpha is None:
                s = p.obs()
                if self.side in s["objects"][obj_key]["touching"]:
                    contact_alpha = alpha
        self.hold(0.8)
        self.last_alpha = alpha
        s = p.obs()

        return self.side in s["objects"][obj_key]["touching"]


class PlannerArm(StreamedArm):
    """The same arm interface, driven through the PLANNER channel.

    StreamedArm authors the whole 29-DoF reference and therefore has to
    switch the deploy out of planner mode, which forces the 0.4 s
    latch-clear window that makes the arms twitch (see StreamedArm.start).
    This class never switches: it stays in PLANNER_FROZEN_UPPER_BODY and
    writes only the 17 upper-body joints (waist 3 + arms 14), which the
    deploy substitutes into the k-planner's reference every tick
    (g1_deploy_onnx_ref.cpp:781-787). `has_upper_body_control_` therefore
    never clears and no twitch is possible.

    The cost is the legs: they belong to the k-planner animation, so
    `squat()` is a no-op here and pelvis height is only reachable through
    the planner message's `height` field.

    `send_vel=True` also fills `upper_body_velocity` by finite difference.
    The encoder observes motion_joint_velocities_* but every channel we
    have used so far sends zeros, which makes a moving reference look
    stationary to it.
    """

    def __init__(self, p: Primitives, side: str, send_vel: bool = False):
        super().__init__(p, side)
        self.send_vel = send_vel
        self.height = -1.0  # planner-message height; -1 = default
        self.mode = LocomotionMode.IDLE
        self._q_prev = None
        self._face = (1.0, 0.0, 0.0)

    def start(self, *, upper_reference=None) -> None:
        """Take over, optionally preserving an already owned frozen reference.

        The explicit 17-value reference is the caller's last commanded upper
        body, not the deploy's unmodified planner motion or measured joints.
        Default entry keeps the historical measured seed for existing callers.
        """
        from primitives import upper_body_from_mj

        if upper_reference is not None:
            upper_reference = np.asarray(upper_reference, dtype=float)
            if upper_reference.shape != (17,) or not np.isfinite(upper_reference).all():
                raise ValueError("upper_reference must contain 17 finite joint targets")
        s = self.p.obs()
        self.q = np.array(s["joints"])
        if upper_reference is not None:
            self.q[12:15] = upper_reference[:3]
            self.q[15:22] = upper_reference[3::2]
            self.q[22:29] = upper_reference[4::2]
        self._q_prev = self.q.copy()
        self.root_quat = yaw_only(s["pelvis"]["quat"])
        self._face = self.p._facing(s)
        self.p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
        self.p.cmd.set_upper_body(upper_body_from_mj(self.q))
        # Zeros, not None, even when send_vel is off. The deploy substitutes
        # the LAST upper_body_velocity it received for as long as position
        # data keeps arriving (input_interface.hpp:381-390 has no staleness
        # cutoff, g1_deploy_onnx_ref.cpp:2985 drops the validity flag), so
        # omitting the key leaves whatever the approach ramp last sent
        # standing in the reference.
        self.p.cmd.set_upper_body_vel(np.zeros(17, dtype=np.float32))
        self.next_t = time.time()
        self.hold(0.6)

    def stop(self) -> None:
        """No mode switch to make; just leave the target where the arm is."""
        from primitives import upper_body_from_mj

        if self.guard is not None:
            g = self.guard_stats
            self.p.log(f"[guard] left palm min {g['min_d']:.2f} m from {self.guard}, guard active {g['frames']} frames")
        d = getattr(self, "servo_dbg", None)
        if d:
            parts = []
            for ph, v in d.items():
                if ph in ("last", "pel0") or not v["n"]:
                    continue
                parts.append(
                    f"{ph}: n={v['n']} err {v['err'] / v['n'] * 100:.1f}/"
                    f"{v['errmax'] * 100:.1f}cm dz {(max(v['z']) - min(v['z'])) * 100:.1f}cm "
                    f"pitch {np.degrees(max(v['pitch']) - min(v['pitch'])):.1f}deg "
                    f"drift {v.get('drift', 0) * 100:.0f}cm dyaw {np.degrees(v.get('dyaw', 0)):.0f}deg"
                )
            last = d.get("last")
            self.p.log(
                "[servo-dbg] "
                + " | ".join(parts)
                + (f" | last {last[0]} err {last[1] * 100:.1f}cm obj_moved {last[2] * 100:.1f}cm" if last else "")
            )
        if self.mode != LocomotionMode.IDLE:
            self.stand()
        s = self.p.obs()
        self.p.cmd.set_upper_body(upper_body_from_mj(s["joints"]))
        # Zeros, not None -- see start(). Handing back with None left the
        # servo's last non-zero velocity in the reference for the whole
        # carry/stand tail of the episode, with the arm standing still.
        self.p.cmd.set_upper_body_vel(np.zeros(17, dtype=np.float32))

    def _push(self, q_mj29, observed=None) -> None:
        from primitives import upper_body_from_mj

        q = self._guard_left(np.asarray(q_mj29, dtype=float).copy(), observed=observed)
        u = upper_body_from_mj(q)
        self.p.cmd.set_upper_body(u)
        if self.send_vel:
            self.p.cmd.set_upper_body_vel((u - upper_body_from_mj(self._q_prev)) / DT)
        self._q_prev = q.copy()
        self.p.cmd.set_hands(left=self.hands["left"], right=self.hands["right"])
        self.p.cmd.frame(mode=self.mode, facing=self._face, height=self.height)
        self.i += 1
        self.next_t += DT
        lag = self.next_t - time.time()
        if lag > 0:
            time.sleep(lag)
        else:
            self.next_t = time.time()

    def squat(self, drop_m: float, track: bool = True) -> float:
        """Crouch through the k-planner's own squat gait. -> measured drop.

        The legs are not ours in this channel, so the depth comes from
        LocomotionMode::IDEL_SQUAT plus the planner message's `height`,
        which is an ABSOLUTE pelvis-height command and is honoured ONLY in
        a squat mode (in IDLE / SLOW_WALK it has 0.2 cm of authority).
        Calibrated 2026-08-21 (diag_squatmode.py --sweep): delivered
        pelvis_z ~= height + 0.02 over 0.75..0.50, i.e. 2.6..20 cm of drop,
        saturating near 27 cm below 0.40.

        IDEL_SQUAT is a STATIC mode, so nothing may walk until stand() puts
        the mode back. `track` re-solves the arm against the live pelvis
        while the body sinks, which matters more here than in the streamed
        channel: the held 17-joint target drifts up to 0.24 rad during the
        descent.
        """
        s = self.p.obs()
        z0 = float(s["pelvis"]["pos"][2])
        target_z = z0 - float(np.clip(drop_m, 0.0, SQUAT_DROP_MAX))
        if z0 - target_z < 0.02:
            return 0.0
        self.height = float(np.clip(target_z - SQUAT_HEIGHT_BIAS, 0.40, 0.78))
        self.mode = LocomotionMode.IDEL_SQUAT
        palm_w = np.array(s["palms"][self.side]["pos"])
        rot_w = Primitives._rot(s["palms"][self.side]["quat"])
        arm_idx = ARM_MJ[self.side]
        n = int(round(SQUAT_SETTLE / DT))
        for k in range(n):
            if track and k % 5 == 1:
                st = self.p.obs()
                aim_pf = self.p.world_to_root_full(palm_w, st)
                rot_pf = Primitives._rot(st["pelvis"]["quat"]).T @ rot_w
                q_sol, res, _ = self.ik.solve(self.side, aim_pf, rot_pf, q_mj29=self.q, use_waist=False)
                if res < 0.08:
                    self.q[arm_idx] = q_sol[arm_idx]
            self._push(self.q)
        st = self.p.obs()

        return float(z0 - st["pelvis"]["pos"][2])

    def stand(self) -> None:
        """Back to the standing gait, so walking is possible again."""
        self.mode = LocomotionMode.IDLE
        self.height = -1.0
        self.hold(1.5)


def reanchor(p: Primitives, side: str = "right") -> None:
    """Force a planner heading re-anchor at a yaw this process knows.

    The deploy may carry a stale anchor from a previous process's streamed
    session (the anchor lives in the deploy, the compensation in the
    commander). A two-second streamed toggle re-anchors it at the robot's
    current yaw and records that yaw in cmd.heading_offset.

    This one is ALWAYS a StreamedArm, never make_arm(): the re-anchor IS the
    mode switch, so a PlannerArm here is a no-op and the deploy keeps the
    stale anchor -- the robot then walks off closing a standing heading
    error (measured 1.01 m of drift through the 4 s settle, which the gym
    reports as "teleport did not land"). The twitch this toggle costs
    happens before the recording starts, so it is free.
    """
    arm = StreamedArm(p, side)
    arm.start()
    arm.hold(0.3)
    arm.stop()


GRIP_X_MAX = 0.30  # the reach map's outer band: [0.22, 0.30] is the
# only overlap of the graspable region with stands
# whose hip does not press the table edge
GRIP_X_MIN = DEFAULT_REACH.post_min  # closer, and the hip leans on the table edge
# and shoves it (FURN[R.hip_roll] forensics)

RAISE_CLEAR = 0.10  # palm above the tabletop while walking the last bit


def _raise_hand(p: Primitives, side: str, surface_top: float | None, grip_world=None) -> None:
    """Lift the palm above the tabletop BEFORE walking in past the edge.

    The squat-then-raise order used to put the hand below the rim and drag it
    up into the table. Raising first has to happen out at the approach ring —
    inside the ring the swing itself can clip the edge — and it has to happen
    on the PLANNER channel, because the walk that follows is planner-mode and
    the wrist target persists through it (the carry leg has walked 2.4 m on
    exactly this mechanism for every successful episode so far). Position-only
    and ~5 cm sloppy, but "above the table" needs no better.
    """
    if surface_top is None:
        return
    from agent_server import LocomotionMode
    from primitives import READY_WRISTS

    s = p.obs()
    # The arms already travel in the READY pose (forearms level, hands above
    # tabletop height) for the whole episode, so this is a trim, not a move:
    # keep the ready x/y and only push the height up if THIS surface or can
    # rim demands more than the ready pose gives.
    top = surface_top + RAISE_CLEAR
    if grip_world is not None:
        top = max(top, float(grip_world[2]) + 0.12)
    ready = np.asarray(READY_WRISTS[side], dtype=np.float32).copy()
    z_pf = float(top - s["pelvis"]["pos"][2])
    if z_pf <= ready[2] + 0.005:
        return  # ready height already clears everything
    ready[2] = z_pf
    p.cmd.set_wrists(**{f"{side}_pos": ready})
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.log(f"  stream: ready hand trimmed up to clear the surface (z_pf {z_pf:+.2f})")


def _nudge_closer(p: Primitives, grip_world, side: str, x_max: float = None, x_min: float = None) -> None:
    """Small guarded forward steps until the grip is within the lean budget.

    The approach ring leaves the pelvis anywhere from 0.28 to 0.45 m from
    the can; beyond grip x ~ 0.36 the policy's delivered lean saturates and
    the align phase plateaus with the same undelivered 5 cm step forever.
    The potential field cannot help (its inflation ring is what stopped the
    walk), so shuffle straight at the can in open loop, one short step at a
    time, re-measuring after each.
    """
    from agent_server import LocomotionMode

    for _pass in range(3):
        s = p.obs()
        x0 = p.world_to_root_full(grip_world, s)[0]
        if (x_min or GRIP_X_MIN) <= x0 <= (x_max or GRIP_X_MAX):
            return
        p.log(f"  stream: grip {x0:.2f} m out, {'walking in' if x0 > (x_max or GRIP_X_MAX) else 'backing off'}")
        _nudge_pass(p, grip_world, side, x_max=x_max, x_min=x_min)
    s = p.obs()
    p.log(f"  stream: grip still {p.world_to_root_full(grip_world, s)[0]:.2f} m out after 3 nudge passes")


def _nudge_pass(p: Primitives, grip_world, side: str, x_max: float = None, x_min: float = None) -> None:
    """One guarded walk toward the band, then a settle and a re-measure.

    Split out of _nudge_closer so overshoot is recoverable: the gait carries
    ~0.2 m of momentum past the stop command, and a single pass that backed
    off from 0.22 m sailed through the whole [0.24, 0.34] band to 0.50 m —
    outside the palm envelope, which the servo then times out against. The
    caller simply runs another pass; walking in from an overshoot converges
    because each pass ends slower than it started.
    """
    from agent_server import LocomotionMode

    # One continuous guarded walk, not step bursts — 0.9 s of SLOW_WALK never
    # even starts the gait, so six bursts moved the robot nowhere. Too far
    # saturates the lean; too close drives the knees into the table edge when
    # the squat comes (a 0.21 m stand launched the can 78 cm without a hand
    # near it), so both directions are corrected.
    end = time.time() + 8.0
    while time.time() < end:
        s = p.obs()
        grip_pf = p.world_to_root_full(grip_world, s)
        if (x_min or GRIP_X_MIN) <= grip_pf[0] <= (x_max or GRIP_X_MAX) or s["fallen"]:
            break
        # Walk along the commanded FACING, not a world movement vector: the
        # facing channel's frame is measured and compensated
        # (diag_heading.py) while movement's is not, and an open-loop world
        # vector walked the robot 4 cm AWAY from the can. Sending
        # movement == facing keeps them parallel in whatever frame the
        # deploy uses, so the robot walks where its nose points, and the
        # nose is steered by ground truth.
        yaw = p.pelvis_yaw(s)
        d = np.asarray(grip_world[:2]) - p.pelvis_xy(s)
        bearing_can = float(np.arctan2(d[1], d[0]))
        if p.hold_yaw is not None and abs(wrap(float(p.hold_yaw) - bearing_can)) < 0.8:
            # Keep the AIM heading. Steering at the can's bearing re-centred
            # it azimuthally by the end of every walk-in (video: can at
            # pelvis y -0.05, hand pinned against the thigh trying to cross
            # to it), throwing away exactly the 10-20 deg side offset the
            # aim chose. Walking straight along the held heading closes the
            # depth just as well and keeps the can on the arm's side — but
            # only while the can is still roughly ahead on that heading; a
            # stale anchor once marched the robot straight PAST the can
            # (final target at pelvis x -0.09, behind the hip).
            bearing = float(p.hold_yaw)
        else:
            bearing = bearing_can
        face = yaw + float(np.clip(wrap(bearing - yaw), -0.4, 0.4))
        fv = (np.cos(face), np.sin(face), 0.0)
        backing = grip_pf[0] <= (x_max or GRIP_X_MAX)
        mv = (-fv[0], -fv[1], 0.0) if backing else fv
        # Backing off slower than walking in: the stop takes ~0.2 m of gait
        # momentum either way, but overshooting outward puts the grip beyond
        # the palm envelope while overshooting inward just costs a re-pass.
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.22 if backing else 0.35, movement=mv, facing=fv)
        time.sleep(DT)
    stop = time.time() + 1.5
    while time.time() < stop:
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(s))
        time.sleep(DT)
    s = p.obs()
    p.log(f"  stream: grip now {p.world_to_root_full(grip_world, s)[0]:.2f} m out")


def _creep_to_band(
    p,
    grip_world,
    side: str,
    x_min: float = 0.18,
    x_max: float = 0.27,
    max_s: float = 8.0,
    target_rel=None,
    envelope=None,
    update_upper=None,
    target_lateral=None,
) -> ApproachResult:
    """Reach an observable depth/cone gate; timeout and falling are failures.

    target_rel is accepted for old callers but never controls termination.
    Random approach waypoints may vary; identical live geometry has the same
    transition rule. Both this depth and the post-crouch depth are yaw-only.
    """
    t0 = time.time()
    ready_since = None
    status = ApproachStatus.TIMEOUT
    aim_max = float(envelope[2]) if envelope is not None else np.deg2rad(36.5)
    while time.time() - t0 < max_s:
        s = p.obs()
        if s["fallen"]:
            status = ApproachStatus.FALLEN
            break
        if update_upper is not None and not update_upper(s):
            status = ApproachStatus.BLOCKED
            break
        grip = np.asarray(p.world_to_root(grip_world, s))
        gx = float(grip[0])
        az = float(np.arctan2(grip[1], grip[0]))
        in_band = x_min <= gx <= x_max
        in_cone = abs(az) <= aim_max
        if target_lateral is not None:
            in_cone = in_cone and abs(grip[1] - target_lateral) <= 0.025
        yaw = p.pelvis_yaw(s)
        # Stop translating while validating the band. The old branch kept
        # walking forward even while accumulating its ready counter.
        if in_band and in_cone:
            if ready_since is None:
                ready_since = time.time()
            if time.time() - ready_since >= 0.2:
                status = ApproachStatus.READY
                break
            face = yaw
            mv = (0.0, 0.0, 0.0)
            speed = 0.20
        else:
            ready_since = None
            face = yaw + float(np.clip(az, -0.5, 0.5)) if not in_cone else yaw
            fv = (np.cos(face), np.sin(face), 0.0)
            if target_lateral is not None:
                # Align the body to put the object ahead of the active hand.
                # The released gait tracks forward/turn commands more reliably
                # than tiny lateral shuffles with the arms held in front.
                desired_az = float(np.arctan2(target_lateral, max(gx, 0.10)))
                face = yaw + float(np.clip(az - desired_az, -0.35, 0.35))
                direction = 0.0 if in_band else (-1.0 if gx < x_min else 1.0)
                mv = (direction * np.cos(face), direction * np.sin(face), 0.0)
                speed = 0.215
            elif not in_cone:
                mv, speed = (0.0, 0.0, 0.0), 0.215
            else:
                backing = gx < x_min
                err = x_min - gx if backing else gx - x_max
                mv = (-fv[0], -fv[1], 0.0) if backing else fv
                speed = 0.20 if err < 0.05 else 0.215
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=speed, movement=mv, facing=(np.cos(face), np.sin(face), 0.0))
        time.sleep(DT)
    s = p.obs()
    if s["fallen"]:
        status = ApproachStatus.FALLEN
    grip = np.asarray(p.world_to_root(grip_world, s))
    az = float(np.arctan2(grip[1], grip[0]))
    if status == ApproachStatus.READY and not (
        x_min <= grip[0] <= x_max
        and abs(az) <= aim_max
        and (target_lateral is None or abs(grip[1] - target_lateral) <= 0.025)
    ):
        status = ApproachStatus.TIMEOUT
    result = ApproachResult(status, float(grip[0]), time.time() - t0, az)
    # Never leave a walking command active on failure.
    if status != ApproachStatus.READY:
        p.cmd.frame(
            mode=LocomotionMode.IDLE,
            speed=0.0,
            movement=(0.0, 0.0, 0.0),
            facing=(np.cos(p.pelvis_yaw(s)), np.sin(p.pelvis_yaw(s)), 0.0),
        )
    p.log(
        f"  stream: approach {status.value}, depth {result.depth:.3f} m "
        f"in [{x_min:.3f}, {x_max:.3f}], az {np.degrees(az):+.1f} deg, "
        f"{result.elapsed:.2f} s"
    )
    return result


def toppled(s, obj_key: str) -> bool:
    """Has the (upright-cylinder) object fallen over? Its body z-axis more
    than ~60 deg off vertical. Failure ATTRIBUTION: a toppled can later
    surfaces as 'grip height unreachable' (the grasp line follows the body
    origin down to lying height), which blames the arm for a knock that
    happened during the approach."""
    rot = Primitives._rot(s["objects"][obj_key]["quat"])
    return float(rot[2, 2]) < 0.5


def final_approach(
    p: Primitives, obj_key: str, rm, edge_dist_fn, z_pf: float, side_pref: str = "right", timeout: float = 25.0
):
    """Stage-2 navigation: slow, guarded, facing the object the whole way.

    Advances straight at the object (facing == movement == bearing, the
    heading-compensated pattern), re-measuring at 50 Hz, and stops when the
    object reaches grasping depth or the hip reaches the table edge. Safety
    is REACTIVE, not omniscient: any disturbance of the object (>2 cm) backs
    the robot off and re-anchors — no assumption about how tall anything is.
    A final minimal side-turn then places the object at the smallest lateral
    offset the reach map accepts (dead-centre is kinematically empty beyond
    the hip-contact line, so "as centred as physically possible").

    -> (ok, side, why)
    """
    from agent_server import LocomotionMode

    X_STOP = 0.26  # object depth at which grasping is possible
    EDGE_STOP = 0.16  # hip-to-edge, incl. ~3 cm of reach-lean sway
    s = p.obs()
    can0 = p.object_pos(obj_key, s)
    backoffs = 0
    end = time.time() + timeout
    while time.time() < end:
        s = p.obs()
        if s["fallen"]:
            return False, side_pref, "fell during the final approach"
        if toppled(s, obj_key):
            return False, side_pref, "object toppled during the final approach"
        can = p.object_pos(obj_key, s)
        moved = float(np.linalg.norm(can - can0))
        if moved > 0.10:
            return False, side_pref, (f"object knocked {moved * 100:.0f} cm away during the final approach")
        yaw = p.pelvis_yaw(s)
        d = can[:2] - p.pelvis_xy(s)
        c, sn = np.cos(yaw), np.sin(yaw)
        x_pf = c * d[0] + sn * d[1]
        if moved > 0.02 and backoffs < 3:
            # Reactive guard: something (us, most likely) disturbed it.
            # Back straight off, let it settle, re-anchor, resume.
            backoffs += 1
            p.log(f"  approach2: object stirred {moved * 100:.1f} cm, backing off ({backoffs}/3)")
            stop_t = time.time() + 0.8
            fv = (c, sn, 0.0)
            while time.time() < stop_t:
                p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.22, movement=(-fv[0], -fv[1], 0.0), facing=fv)
                time.sleep(DT)
            p.cmd.hold(0.8, mode=LocomotionMode.IDLE, facing=fv)
            can0 = p.object_pos(obj_key)
            continue
        if x_pf <= X_STOP or edge_dist_fn(p.pelvis_xy(s)) <= EDGE_STOP:
            break
        bearing = float(np.arctan2(d[1], d[0]))
        face = yaw + float(np.clip(wrap(bearing - yaw), -0.6, 0.6))
        fv = (np.cos(face), np.sin(face), 0.0)
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.25, movement=fv, facing=fv)
        time.sleep(DT)
    # Settle into the depth WINDOW [0.22, 0.30] with short bidirectional
    # pulses: the stop carries ~0.2 m of momentum inward, and a fixed-length
    # back-step overshoots outward just as badly (measured 0.42-0.65 m
    # stands after "stepping back"). Pulse length scales with the error;
    # every pulse is followed by a full settle before re-measuring.
    for _settle in range(6):
        stop_t = time.time() + 1.8
        while time.time() < stop_t:
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(p.obs()))
            time.sleep(DT)
        s = p.obs()
        can = p.object_pos(obj_key, s)
        yaw = p.pelvis_yaw(s)
        d2 = can[:2] - p.pelvis_xy(s)
        x_now = float(np.cos(yaw) * d2[0] + np.sin(yaw) * d2[1])
        edge_ok = edge_dist_fn(p.pelvis_xy(s)) >= EDGE_STOP - 0.02
        if 0.22 <= x_now <= 0.30 and edge_ok:
            break
        err = x_now - 0.26
        sign = 1.0 if (err > 0 and edge_ok) else -1.0
        dur = float(np.clip(abs(err) * 3.0, 0.25, 0.7))
        p.log(f"  approach2: depth {x_now:.2f}, pulsing {'in' if sign > 0 else 'out'} {dur:.2f} s")
        c2, s2 = np.cos(yaw), np.sin(yaw)
        end_t = time.time() + dur
        while time.time() < end_t:
            p.cmd.frame(
                mode=LocomotionMode.SLOW_WALK, speed=0.22, movement=(sign * c2, sign * s2, 0.0), facing=(c2, s2, 0.0)
            )
            time.sleep(DT)

    # Minimal side-turn into the reach map: try facing offsets outward from
    # zero; the first (smallest) that lands the object in a graspable voxel
    # with a voxel of margin wins. Prefers the requested arm, falls back to
    # the other.
    s = p.obs()
    can = p.object_pos(obj_key, s)
    d = can[:2] - p.pelvis_xy(s)
    dist = float(np.linalg.norm(d))
    bearing = float(np.arctan2(d[1], d[0]))
    sides = [side_pref, "left" if side_pref == "right" else "right"]
    for delta_deg in range(0, 42, 3):
        for sgn in (0,) if delta_deg == 0 else (1, -1):
            delta = np.deg2rad(delta_deg) * (sgn or 1)
            rel = np.array([dist * np.cos(delta), -dist * np.sin(delta), z_pf])
            if abs(rel[1]) < 0.12:
                # Kinematically the map allows a centred object; the SERVO
                # does not — the policy under-delivers in the chest zone
                # (every centred attempt stalls or shoves the can). "As
                # centred as possible" bottoms out at |y| = 0.12.
                continue
            for side in sides:
                if rm.graspable(rel, side):
                    psi = bearing + delta
                    # Closed-loop turn (2026-09-03): turn_to's controller
                    # holds a 6-8 deg residual and the planner under-turns
                    # from a walk; one open-loop turn left the bottle at a
                    # -60 deg bearing ("stand is wrong", 1 in 8 rounds).
                    # Re-measure where the object actually ended up and turn
                    # again; give up to the stand-stage retry if it is still
                    # beside the body.
                    for _try in range(3):
                        p.turn_to(psi)
                        s = p.obs()
                        rel_m = p.world_to_root(p.object_pos(obj_key, s), s)
                        b_m = float(np.arctan2(rel_m[1], rel_m[0]))
                        if abs(b_m - (-delta)) < np.deg2rad(8.0):
                            break
                        psi = p.pelvis_yaw(s) + (b_m + delta)  # what is left to turn
                        p.log(
                            f"  approach2: turn landed at bearing "
                            f"{np.degrees(b_m):+.0f} deg (wanted "
                            f"{np.degrees(-delta):+.0f}), turning again"
                        )
                    p.hold_yaw = p.pelvis_yaw()
                    if abs(b_m) > np.deg2rad(40.0) or rel_m[0] < 0.12:
                        return (
                            False,
                            side,
                            (
                                f"object still at bearing {np.degrees(b_m):+.0f} deg "
                                f"(rel {np.round(rel_m[:2], 2)}) after the side-turn"
                            ),
                        )
                    p.log(
                        f"  approach2: stopped at depth {dist:.2f} m, "
                        f"turned {np.degrees(delta):+.0f} deg -> {side} "
                        f"arm, object at rel {np.round(rel_m, 2)} "
                        f"(planned {np.round(rel, 2)})"
                    )
                    return True, side, ""
    return (
        False,
        side_pref,
        (
            f"object at {dist:.2f} m is outside the reach map at every facing "
            f"the body can take without touching the table"
        ),
    )


def place_streamed(p: Primitives, obj_key: str, drop_world, side: str = "right") -> tuple[bool, str]:
    """Carry the palm over the drop point and open the hand. -> (ok, why)."""
    # Walk in past the inflation ring first: the approach parks ~0.68 m from
    # the plate, beyond even the lean envelope. Same guarded walk as the
    # grasp, aimed at XY distance instead of pelvis-frame depth.
    from agent_server import LocomotionMode

    end = time.time() + 6.0
    s = p.obs()
    while time.time() < end:
        s = p.obs()
        if float(np.linalg.norm(np.asarray(drop_world[:2]) - p.pelvis_xy(s))) <= 0.42 or s["fallen"]:
            break
        yaw = p.pelvis_yaw(s)
        d = np.asarray(drop_world[:2]) - p.pelvis_xy(s)
        bearing = float(np.arctan2(d[1], d[0]))
        face = yaw + float(np.clip(wrap(bearing - yaw), -0.4, 0.4))
        fv = (np.cos(face), np.sin(face), 0.0)
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.35, movement=fv, facing=fv)
        time.sleep(DT)
    stop = time.time() + 1.5
    while time.time() < stop:
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(s))
        time.sleep(DT)

    arm = make_arm(p, side)
    arm.start()
    try:
        # Tuck the FREE arm against the chest first: it hangs at walking
        # height, and the contact log shows its wrist brushing the counter
        # four times per place (L.wrist_yaw) as the robot works at the edge.
        free = "left" if side == "right" else "right"
        q_t = arm.q.copy()
        fa = ARM_MJ[free]
        sign = 1.0 if free == "left" else -1.0
        q_t[fa] = [0.35, 0.25 * sign, 0.0, 1.2, 0.0, 0.0, 0.0]
        arm.move_to(q_t, speed=0.6)
        arm.hold(0.5)

        s = p.obs()
        # Release the can UPRIGHT. The grasp tilts it and the carry pose adds
        # more; released lying sideways it lands on its side and rolls off
        # the counter (video, twice). The grip is rigid, so rotate the whole
        # palm by whatever rotation brings the can's axis back to vertical.
        rot_palm = Primitives._rot(s["palms"][side]["quat"])
        axis_now = Primitives._rot(s["objects"][obj_key]["quat"])[:, 2]
        v = np.cross(axis_now, [0.0, 0.0, 1.0])
        sv, cv = np.linalg.norm(v), float(axis_now[2])
        if sv > 1e-6:
            k = v / sv
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            ang = np.arctan2(sv, cv)
            r_fix = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
        else:
            r_fix = np.eye(3)
        rot_hold = r_fix @ rot_palm
        drop_pf = p.world_to_root_full(drop_world, s)
        if drop_pf[2] < SQUAT_TRIGGER:
            arm.squat(SQUAT_TRIGGER - drop_pf[2])
        # The CAN has to end up over the plate, and the can is NOT at the
        # nominal grasp offset any more — it shifts in the grip during the
        # carry (measured 8-10 cm of drift, which released it onto the
        # plate's rim). Use its MEASURED palm-frame offset, and after the
        # descent servo on the can's own position, not the palm's.
        obj_off_pf = rot_palm.T @ (p.object_pos(obj_key, s) - np.array(s["palms"][side]["pos"]))
        drop_w = np.asarray(drop_world, dtype=float)
        palm_target = drop_w - rot_hold @ obj_off_pf
        if arm.goto(palm_target + [0.0, 0.0, 0.12], rot_world=rot_hold, refine=1, tol=0.05) is None:
            return False, p.aborted or "streamed staging over the plate failed"
        err = arm.goto(palm_target, rot_world=rot_hold, refine=1, tol=0.03, speed=FINE_SPEED)
        if err is None:
            return False, p.aborted or "streamed reach over the plate failed"
        for _ in range(2):
            s = p.obs()
            can_err = drop_w - p.object_pos(obj_key, s)
            if float(np.linalg.norm(can_err[:2])) < 0.03:
                break
            palm_target = np.array(s["palms"][side]["pos"]) + np.clip(can_err, -0.05, 0.05)
            if arm.goto(palm_target, rot_world=rot_hold, refine=0, speed=FINE_SPEED) is None:
                return False, p.aborted or "streamed can-centering failed"
        # FINAL live correction right before the fingers move: the last
        # servo measurement goes stale during the settling holds, and the
        # video shows the can released over the plate RIM after a clean
        # 2 cm reading. Recompute the upright twist from the can's live
        # axis at the same time.
        s = p.obs()
        axis2 = Primitives._rot(s["objects"][obj_key]["quat"])[:, 2]
        v2 = np.cross(axis2, [0.0, 0.0, 1.0])
        s2, c2 = np.linalg.norm(v2), float(axis2[2])
        if s2 > 1e-6:
            k2 = v2 / s2
            K2 = np.array([[0, -k2[2], k2[1]], [k2[2], 0, -k2[0]], [-k2[1], k2[0], 0]])
            a2 = np.arctan2(s2, c2)
            rot_hold = (np.eye(3) + np.sin(a2) * K2 + (1 - np.cos(a2)) * K2 @ K2) @ Primitives._rot(
                s["palms"][side]["quat"]
            )
        can_err = drop_w - p.object_pos(obj_key, s)
        if float(np.linalg.norm(can_err[:2])) > 0.015:
            arm.goto(
                np.array(s["palms"][side]["pos"]) + np.clip(can_err, -0.05, 0.05),
                rot_world=rot_hold,
                refine=0,
                speed=FINE_SPEED,
            )
            arm.hold(0.8)
        s = p.obs()
        can_xy_err = float(np.linalg.norm(drop_w[:2] - p.object_pos(obj_key, s)[:2]))
        p.log(f"  stream: can over the plate, xy err {can_xy_err * 100:.1f} cm")
        if can_xy_err > 0.07:
            return False, (f"can never centred over the plate ({can_xy_err * 100:.1f} cm)")
        # Open SLOWLY, and HOLD STATION while doing it: with the reference
        # frozen the palm sags back toward its natural delivery point —
        # 10-15 cm at this stretch — and the can gets released over the
        # counter's near edge instead of the plate (video: four episodes,
        # all landing a systematic ~37 cm short). Warm IK each tick, same
        # as the grasp servo's close phase.
        start = arm.hands[side].copy()
        s = p.obs()
        hold_w = np.array(s["palms"][side]["pos"])
        n = int(round(2.5 / DT))
        for k in range(n + 1):
            arm.hands[side] = start + (HAND_OPEN - start) * (k / n)
            if k % 3 == 0:
                s = p.state.latest(5) or s
                palm_now = np.array(s["palms"][side]["pos"])
                cmd = hold_w + np.clip(1.0 * (hold_w - palm_now), -0.08, 0.08)
                q_sol, res, _ = arm.ik.solve(
                    side, p.world_to_root_full(cmd, s), None, q_mj29=arm.q, use_waist=True, iters=8
                )
                if res < 0.10:
                    arm.q = q_sol
            arm._push(arm.q)
        arm.hold(1.0)
        # Retreat along the entry axis REVERSED, not straight up: the open
        # upright palm is a cradle, and lifting it vertically carried the can
        # away still sitting in the hand (dz +0.70 at the landing check).
        # Sliding the palm back out from under it leaves the can behind.
        s = p.obs()
        back = -(rot_hold[:, 1])
        arm.goto(np.array(s["palms"][side]["pos"]) + back * 0.12 + [0.0, 0.0, 0.03], refine=0, speed=0.2)
        arm.hold(0.5)
        s = p.obs()
        if side in s["objects"][obj_key]["touching"]:
            arm.goto(np.array(s["palms"][side]["pos"]) + back * 0.10, refine=0, speed=FINE_SPEED)
        # Tuck BOTH arms fully while still in streamed mode. Handing a
        # half-extended arm back to the planner and letting tuck_arms finish
        # the job sent the wrist through the can (forensics: RETREAT-TUCK
        # moved the can 9.4 cm, touch R.wrist_yaw) — the planner channel
        # cannot be trusted anywhere near the placed object. Go up-and-back
        # first so the retraction path clears the counter.
        s = p.obs()
        arm.goto(np.array(s["palms"][side]["pos"]) + back * 0.10 + [0.0, 0.0, 0.18], refine=0, speed=0.5)
        q_t2 = arm.q.copy()
        for sd, sgn in (("left", 1.0), ("right", -1.0)):
            q_t2[ARM_MJ[sd]] = [0.35, 0.25 * sgn, 0.0, 1.2, 0.0, 0.0, 0.0]
        arm.move_to(q_t2, speed=0.7)
        arm.hold(0.5)

        return True, ""
    finally:
        arm.stop()


# Continuous-servo tuning. One 50 Hz loop replaces the walk-stop-look-walk
# discrete servo: every tick reads the freshest privileged state, advances a
# cartesian reference a fraction of a millimetre, and re-solves warm-started
# IK — the way a teleoperator's hand moves, not the way a surveyor works.
V_TRAVEL = 0.15  # m/s toward the staging point (was 0.30: the WBC arm lags
# the reference by ~0.2 s x speed -- 6 cm at 0.30, measured 2026-09-03)
V_DESCEND = 0.10  # m/s dropping to the pregrasp (was 0.18, same lag argument)
V_APPROACH = 0.10  # m/s inside the jaw-alignment approach
# Pre-narrow the jaw while sliding in. Calibrated kinematically: fingertip
# gap ~ 6.1 cm at alpha 0, closing ~0.6 cm per 0.1 alpha, so 0.10 leaves
# ~5.6 cm over the 4.8 cm can — 8 mm of total margin instead of 13, but the
# first contact now lands on the closing envelope (which wraps) instead of a
# flat finger (which shoves): the graze-and-roll that killed 8 of 15
# episodes happens at the OPEN aperture's finger tips.
PRE_NARROW = 0.10
SAG_COMP = 0.018  # free-air sag is ~1.6-2 cm; 0.035 was measured
# against a WEDGED hand (the table held it down) and
# overcompensated: signed telemetry showed the palm
# riding 15-19 mm HIGH, closing the jaw at the rim
V_LIFT = 0.22
LEAD_MAX = 0.08  # m the reference may run ahead of the real palm
KP_TRACK = 0.7  # extra command past the reference, fights the sag
CLOSE_RAMP_S = 2.0  # 1.0 s fingers punched the can off the table


def _clipn(v, m):
    n = float(np.linalg.norm(v))

    return v if n <= m else v * (m / n)


# How much entry travel may remain when a CONTACT is allowed to trigger the
# close. Geometry, measured off the dex1 collision meshes: the pads span
# wrist+0.1206..0.1848 along the entry axis and the grasp centre sits at
# wrist+0.1515, so with e_ent still to travel the can centre sits at
# 0.1515+e_ent from the wrist. It is inside the pad span only for
# e_ent <= 0.033, while the fingertips (0.1848) first touch a 24 mm-radius
# can at e_ent ~ 0.0575. The old 0.06 gate therefore fired at the exact
# moment the TIPS grazed the can, with the can centre still 27 mm beyond
# them — closing there dealt the can a one-sided shove. Mined over 158
# labelled rounds: reaching the close phase through the servo's tolerance
# gate succeeded 70/70, while this contact shortcut succeeded 26/64, and
# every "toppled in the close phase" in the corpus came through it.
CLOSE_ENT_MAX = 0.02

# How far the can centre may sit from the grasp centre ALONG THE ENTRY AXIS
# and still be between the pads. Straight off the dex1 collision meshes: the
# pads span wrist+0.1206..0.1848 and the grasp centre is wrist+0.1515, so the
# can is seated for -0.031..+0.033; 0.030 keeps 3 mm at the near end and
# 3 mm at the far end.
#
# The tolerance gate used to ask for 0.012 (0.018 once riding or patient),
# which was tuned when the sim-side pin executed the IK solution exactly and
# the arm really could reach 0.012. Without the pin the arm stalls 2-3.7 cm
# short -- measured palm->ref pinned at LEAD_MAX with res 0.000, i.e. the
# joints solve and the policy does not deliver, which is the p90 of the
# 216-sample pin-off error. The can was then SEATED and the gate still
# refused, so the servo hovered until the 6 s wedge timer lifted it off, twice
# (one 118 s episode against 66-71 s for the successes). This is not the old
# 0.06 contact shortcut that succeeded 26/64: that one fired with the can
# centre 2.7 cm BEYOND the fingertips, where closing is a one-sided shove.
CLOSE_SEATED = 0.030

FREEZE_TRAVEL = 0.10  # residual travel below which the plan is frozen
STAGE_UP = 0.15  # staging point height above the pregrasp (was 0.10: only ~5 cm
# over the bottle top, eaten by the 6-7 cm tracking lag -> 'up' topples)
# Deliverable staging box, pelvis frame. The scene-model IK reaches further
# than the policy delivers (res 0.000 while the palm stalls): a pregrasp
# 0.5 m to the side solves fine and then never arrives — and so does one at
# the chest (x 0.10, z +0.08), where retracting the palm is equally
# undelivered. The box is the region the planner-era reach demonstrably
# worked in; candidates whose staging point leaves it are rejected before
# the arm ever chases them.
PRE_X_MIN, PRE_X_MAX = 0.17, 0.38
PRE_Z_MIN, PRE_Z_MAX = -0.24, 0.04
PRE_XY_MAX = 0.46  # horizontal reach from the pelvis


def _servo_continuous(
    p: Primitives,
    arm: StreamedArm,
    obj_key: str,
    plan,
    side: str,
    replan=None,
    ghost=None,
    pin=None,
    surface_top=None,
    execution=None,
    transport_guard=None,
    start_at_pregrasp=False,
    rotation_bias=None,
    reference_guard=None,
    pregrasp_gap=None,
    close_from_current=False,
    grip_target=None,
    require_bilateral=False,
    capture_feedback=None,
    grasp_point_control=False,
    preserve_capture_reference=False,
) -> tuple[bool, str]:
    """Travel, align, slide in, close, lift — one uninterrupted 50 Hz loop.

    `replan(wrist_now, palm_rot) -> Grasp` re-derives the candidate pose from
    the REAL wrist, so the approach is always the current shortest line; it is
    called at ~5 Hz during the travel phases and frozen once the residual
    travel drops under FREEZE_TRAVEL (or the slide-in begins) — past that
    point a moving target only feeds the push-away chatter. `ghost` mirrors
    every accepted replan into the video's translucent hand.
    """
    gap = SAFE_GAP if pregrasp_gap is None else float(pregrasp_gap)
    if preserve_capture_reference and (not close_from_current or grasp_point_control or capture_feedback is not None):
        raise ValueError("Reference handover requires guarded measured wrist closure")
    if grasp_point_control and (not close_from_current or capture_feedback is not None):
        raise ValueError("Grasp-point servo requires measured closure without legacy wrist feedback")
    if not np.isfinite(gap) or not 0.07 <= gap <= 0.25:
        raise ValueError("pregrasp gap must be between .07 and .25 m")
    rot_t = np.asarray(plan.rot)
    # Entry axis: dex3 slides along palm +y (the finger cage sweeps the can
    # in sideways); dex1 is a parallel jaw and MUST come in along palm +x,
    # between the fingers — sliding along +y leads with one finger and rams
    # the can (TOUCH[dex1_finger_2] on every approach).
    ENTRY_COL = 0 if _HAND == "dex1" else 1
    entry = rot_t[:, ENTRY_COL]
    pre_w = np.asarray(plan.wrist) - entry * gap
    above_w = pre_w + np.array([0.0, 0.0, STAGE_UP])
    frozen = replan is None
    last_replan = 0.0
    offset = np.asarray(GRASP_CENTRE_OFFSET, dtype=float)
    closed = (HAND_CLOSED_RIGHT if side == "right" else HAND_CLOSED_LEFT).astype(np.float32)

    s = p.obs()
    obj0 = p.object_pos(obj_key, s)
    z0 = float(obj0[2])
    ref = np.array(s["palms"][side]["pos"], dtype=float)
    if start_at_pregrasp:
        # Diagnostic joint transport may hand over here only after actual
        # pregrasp delivery. Never skip transport based on a planned pose.
        rot_now = Primitives._rot(s["palms"][side]["quat"])
        angle = float(np.arccos(np.clip((np.trace(rot_t @ rot_now.T) - 1) / 2, -1.0, 1.0)))
        if np.linalg.norm(ref - pre_w) >= 0.045 or angle >= 0.25:
            return False, "pregrasp handover geometry not delivered"
    phase, ok_ticks, alpha, contact_alpha = ("in" if start_at_pregrasp else "up"), 0, 0.0, None
    capture_bias = np.zeros(3)
    if close_from_current:
        from measured_grasp import closure_geometry

        if start_at_pregrasp or _HAND != "dex1" or reference_guard is None:
            raise ValueError("measured closure requires guarded Dex1 control")
        if grip_target is None:
            raise ValueError("measured closure requires the shared live grasp target")
        if not closure_geometry(s, side, grip_target(s))["ready"]:
            return False, "measured closure handover geometry not delivered"
        phase = "close"
        if grasp_point_control:
            ref += Primitives._rot(s["palms"][side]["quat"]) @ offset
    full_f = None  # low-passed palm-frame error (the servo's eyes)
    obj_f = None  # low-passed can position
    close_tgt = ref.copy() if close_from_current else None
    if close_from_current:
        obj_f = obj0.copy()
    obj_in0 = None  # can position when the slide-in began
    backoff_until = 0.0
    touch_ticks = 0
    trim_w = np.zeros(3)  # slow world-frame trim: absorbs the gap between
    # the PLANNED palm rotation the absolute target
    # assumes and the rotation the policy delivers
    lift_target, settle_until = None, None
    deadline = time.time() + 60.0  # wedge class fixed; a 90 s hover is a
    # real failure, not a slow success
    t_prev = time.time()
    last_log = 0.0
    res = 0.0
    in_since = None
    ent_mark = None
    stall_palm, stall_t = None, time.time()
    vis_n = tot_n = 0  # ego-camera visibility, for the VLA data gate
    in_lift = 0.0  # slide-in extra height, set in the in phase
    dbg_z = ""  # z-chain telemetry (line/cmd/palm/floor)
    touch_seen: set = set()  # object-contact links since the last status line

    recorded_phase = None
    while time.time() < deadline:
        transport_phase = phase if phase in ("up", "down") else None
        if execution is not None and phase != recorded_phase:
            execution.enter("grasp." + phase)
            recorded_phase = phase
        s = p.state.latest(5) or s
        now = time.time()
        dt = float(np.clip(now - t_prev, 0.005, 0.06))
        t_prev = now
        if s["fallen"]:
            return False, "fell during the continuous servo"
        if toppled(s, obj_key):
            return False, f"{obj_key} toppled in the {phase} phase"
        palm = np.array(s["palms"][side]["pos"])
        rot_p = Primitives._rot(s["palms"][side]["quat"])
        controlled = palm + rot_p @ offset if grasp_point_control else palm
        obj = p.object_pos(obj_key, s)
        # servo telemetry (2026-09-03 topple study): palm-vs-reference
        # tracking error and torso motion per phase, printed by arm.stop()
        _d = arm.__dict__.setdefault("servo_dbg", {})
        _ph = _d.setdefault(phase, {"n": 0, "err": 0.0, "errmax": 0.0, "z": [], "pitch": []})
        _e = float(np.linalg.norm(controlled - ref))
        _ph["n"] += 1
        _ph["err"] += _e
        _ph["errmax"] = max(_ph["errmax"], _e)
        _pq = s["pelvis"]["quat"]
        _ph["z"].append(float(s["pelvis"]["pos"][2]))
        _ph["pitch"].append(float(np.arcsin(np.clip(2.0 * (_pq[0] * _pq[2] - _pq[3] * _pq[1]), -1, 1))))
        _d["last"] = (phase, _e, float(np.linalg.norm(obj - obj0)))
        if "pel0" not in _d:
            _d["pel0"] = (np.array(s["pelvis"]["pos"][:2]), float(s["pelvis"]["yaw"]))
        _ph.setdefault("drift", 0.0)
        _ph["drift"] = max(_ph["drift"], float(np.linalg.norm(np.array(s["pelvis"]["pos"][:2]) - _d["pel0"][0])))
        _ph.setdefault("dyaw", 0.0)
        _ph["dyaw"] = max(_ph["dyaw"], abs(float(wrap(s["pelvis"]["yaw"] - _d["pel0"][1]))))

        tl0 = s["objects"][obj_key].get("touch_links")
        if tl0:
            touch_seen.update(tl0)
        if "cam" in s["objects"][obj_key]:
            global LAST_EGO_VIS
            cam = s["objects"][obj_key]["cam"]
            tot_n += 1
            if cam is not None and abs(cam[0]) < 0.9 and abs(cam[1]) < 0.9:
                vis_n += 1
            if execution is None:
                LAST_EGO_VIS = vis_n / tot_n  # legacy direct servo callers
            else:
                execution.metrics["ego_visibility"] = vis_n / tot_n

        if not frozen and phase in ("up", "down") and now - last_replan > 0.2:
            last_replan = now
            npl = replan(palm, rot_p)
            if npl.ok and npl.travel > FREEZE_TRAVEL:
                cand_rot = np.asarray(npl.rot)
                cand_pre = np.asarray(npl.wrist) - cand_rot[:, ENTRY_COL] * gap
                pre_pf = p.world_to_root_full(cand_pre, s)
                if (
                    float(np.hypot(pre_pf[0], pre_pf[1])) <= PRE_XY_MAX
                    and PRE_X_MIN <= pre_pf[0] <= PRE_X_MAX
                    and PRE_Z_MIN <= pre_pf[2] <= PRE_Z_MAX
                ):
                    rot_t = cand_rot
                    entry = rot_t[:, ENTRY_COL]
                    pre_w = cand_pre
                    above_w = pre_w + np.array([0.0, 0.0, STAGE_UP])
                    if ghost is not None:
                        ghost(npl)
            elif npl.ok:
                frozen = True  # committed: inside the freeze radius
                p.log(f"  stream: plan frozen {npl.travel * 100:.0f} cm out")
        if phase == "in":
            frozen = True
        if phase in ("up", "down", "in"):
            moved = float(np.linalg.norm(obj - obj0))
            # The in-phase tolerance is double: the servo target tracks the
            # LIVE can, so a thumb-graze that scoots it 5 cm is recoverable —
            # the hand simply re-centres on where it now is. Five episodes
            # died at exactly "5 cm in the in phase" for no lasting reason.
            # Up/down tolerate a table-shove: the close stands lean the hip
            # on the edge and slide the can a few cm with no robot contact;
            # the live replan re-anchors on it, so only a big displacement
            # (roll-away) is fatal before the slide-in begins.
            limit = NUDGE_MAX * 3 if phase == "in" else NUDGE_MAX * 2.4
            if moved > limit:
                hit = s["objects"][obj_key].get("touch_links") or ["nothing"]
                ft = s.get("furniture_touch") or []
                return False, (
                    f"{obj_key} moved {moved * 100:.0f} cm in the "
                    f"{phase} phase (touched {', '.join(hit)}"
                    f"{'; furniture: ' + ', '.join(ft) if ft else ''})"
                )

        if phase == "up":
            tgt, v = above_w, V_TRAVEL
            if float(np.linalg.norm(ref - above_w)) < 0.03:
                phase = "down"
        elif phase == "down":
            tgt, v = pre_w, V_DESCEND
            if float(np.linalg.norm(ref - pre_w)) < 0.015 and float(np.linalg.norm(palm - pre_w)) < 0.045:
                phase = "in"
                p.log("  stream: at pregrasp, sliding in")
        elif phase == "in" and now < backoff_until:
            # Contact flinch: the thumb has started pushing the can, so
            # retreat a few centimetres along the entry axis, let it settle,
            # and re-approach the place it ACTUALLY is now. Without this the
            # servo chases the can it is pushing until the guard kills the
            # episode — seven of ten in batch13.
            tgt, v = palm - entry * 0.05 + np.array([0, 0, 0.03]), 0.15
            if now >= backoff_until - 0.02:
                obj_in0 = None
        elif phase == "in":
            if obj_in0 is None:
                obj_in0 = obj.copy()
                obj_f = obj.copy()
                if in_since is None:
                    # Once per servo call, NOT per flinch re-entry: resetting
                    # it every backoff meant the patience close-gate needed
                    # 12 uninterrupted seconds and never fired on episodes
                    # that flinched every few seconds — they hovered at the
                    # can until the 75 s deadline instead.
                    in_since = now
                alpha = PRE_NARROW
                arm.hands[side] = HAND_OPEN + alpha * (closed - HAND_OPEN)
            if float(np.linalg.norm(obj - obj_in0)) > 0.015:
                if _HAND == "dex1" and full_f is not None and full_f[1] < CLOSE_ENT_MAX:
                    # Parallel jaw, can already between the pads (entry
                    # nearly spent): contact means CLOSE NOW and funnel it
                    # to the palm. The dex3-era flinch re-dives and pokes
                    # the can over, once per cycle, until it topples.
                    p.log(
                        f"  stream: contact inside the jaw — closing "
                        f"(ent {full_f[1] * 100:.1f}cm "
                        f"lat {full_f[0] * 1000:.0f}mm)"
                    )
                    phase = "close"
                    continue
                backoff_until = now + 1.0
                p.log("  stream: contact flinch, backing off to re-centre")
                continue
            if ent_mark is None or abs(full_f[1] - ent_mark[0]) > 0.005:
                ent_mark = (full_f[1] if full_f is not None else 1.0, now)
            elif now - ent_mark[1] > 6.0:
                # Wedged: no entry progress for 6 s with the fingertip on
                # the table — lift-and-retreat to break the friction lock.
                backoff_until = now + 1.2
                ent_mark = None
                p.log("  stream: wedged (no entry progress 6 s), lifting off")
                continue
            # ABSOLUTE target, built from the (low-passed) can and the
            # PLANNED palm rotation — never from the current palm. A target
            # anchored on the live palm chases its own sway (telemetry: lat
            # wobbling 2-6 cm, self-excited), and a target chasing the raw
            # can position bulldozes a nudged can. rot_t is constant, obj_f
            # moves at EMA speed: the target is calm.
            obj_f = obj if obj_f is None else 0.85 * obj_f + 0.15 * obj
            q_obj = rot_p.T @ (obj - palm)
            full = q_obj - offset
            # The trim integrates the MEASURED palm-frame error (which is
            # rotation-truth) into the absolute target at 0.5/s — absolute
            # geometry keeps the loop calm, the trim keeps it honest.
            can_still = float(np.linalg.norm(obj - obj_in0)) < 0.008
            if can_still:
                # Trim only while the can is stationary: integrating the
                # error of a can the hand is pushing turns the trim into a
                # positive-feedback bulldozer.
                err_w = rot_p @ full
                near = float(np.linalg.norm((obj_f - rot_t @ offset + trim_w) - palm)) < 0.06
                if near:
                    d_err = err_w
                else:
                    # Far out, the entry component of the error is just the
                    # un-traveled distance — project it out. The VERTICAL
                    # error is genuine delivery bias (the policy under-
                    # delivers the waist lean by ~2/3, pin covers only the
                    # arm joints — measured pal 4-5 cm above cmd) except for
                    # the deliberate slide-in lift, which is added back.
                    d_err = err_w - float(np.dot(err_w, entry)) * entry
                    d_err = d_err.copy()
                    d_err[2] = err_w[2] + in_lift
                trim_w = trim_w + 0.5 * dt * d_err
                trim_w[:2] = np.clip(trim_w[:2], -0.04, 0.04)
                # z gets more headroom: the waist-lean shortfall alone is
                # 4-5 cm.
                trim_w[2] = float(np.clip(trim_w[2], -0.08, 0.08))
            base = obj_f - rot_t @ offset
            if surface_top is not None:
                # Grip-height feasibility. The z-floor (hand lowest point
                # over the table) moves with the PLAN's rotation, and some
                # plans put it above the grasp line — the palm then hovers
                # at the floor while the thumb sweeps the can at rim height
                # (TOUCH[R:thumb_1] + can_moved telemetry). Raise the grip
                # on the can by up to 2 cm to clear the floor; a grip that
                # needs more is unreachable — fail instead of bulldozing.
                import adaptive_grasp as _ag

                low_c2 = float((_ag.hand_points() @ rot_t.T)[:, 2].min())
                zf2 = surface_top + (0.030 if _HAND == "dex1" else 0.012) - low_c2 - float((rot_t @ _ag.OFFSET)[2])
                need = zf2 + 0.003 - float(base[2])
                if need > 0.02:
                    return False, (
                        "grip height unreachable over the table: the hand's "
                        f"low point forces the palm {need * 100:.1f} cm "
                        f"above the grasp line (obj z {float(obj_f[2]):.3f}, "
                        f"low_c {low_c2:.3f})"
                    )
                if need > 0:
                    base = base + np.array([0.0, 0.0, need])
            final_w = base + trim_w
            # The gate measures the DELIVERED palm against the TRUE grip
            # line (trim excluded): the trim deliberately biases the COMMAND
            # to cancel the policy's delivery offset, so measuring against
            # the biased line would hold the gate shut by that offset
            # forever.
            to_true = base - palm
            e_ent = float(np.dot(to_true, entry))
            perp = to_true - e_ent * entry
            lat = float(np.linalg.norm(perp))
            lat_h = float(np.hypot(perp[0], perp[1]))
            raw_f = np.array([lat, e_ent, lat_h, perp[2]])
            full_f = raw_f if full_f is None else 0.8 * full_f + 0.2 * raw_f
            # HARD align precondition, like the discrete servo's separate
            # align phase: zero forward motion until laterally within 12 mm.
            # The soft 18 mm gate let the hand creep in 17%-speed while still
            # 15 mm off line, and that is exactly the thumb-graze band —
            # four episodes per batch died re-flinching their way to the
            # 10 cm guard.
            if float(p.world_to_root_full(final_w, s)[0]) < 0.05:
                _pf = p.world_to_root_full(final_w, s)
                _opf = p.world_to_root_full(obj, s)
                _pel0 = arm.servo_dbg.get("pel0")
                _drift = (
                    (
                        float(np.linalg.norm(np.array(s["pelvis"]["pos"][:2]) - _pel0[0])),
                        float(np.degrees(wrap(s["pelvis"]["yaw"] - _pel0[1]))),
                    )
                    if _pel0
                    else (0, 0)
                )
                return False, (
                    "grip target ended up at the body/behind it "
                    "— the stand is wrong, not the arm "
                    f"(final_pf {np.round(_pf, 2).tolist()} obj_pf {np.round(_opf, 2).tolist()} "
                    f"trim {np.round(trim_w, 2).tolist()} rot_off {np.round(rot_t @ offset, 2).tolist()} "
                    f"pelvis drift {_drift[0]:.2f} m yaw {_drift[1]:+.0f} deg since servo start)"
                )
            # 18 mm, was 12: the delivery floor wobbles at 11-16 mm, and a
            # 12 mm hard gate kept zeroing the advance — hands hovered a
            # full 14 cm out until the deadline. The pre-narrowed jaw wraps
            # from 18 mm of lateral error.
            # The gate keys on the HORIZONTAL lateral error only: the
            # vertical offset during the slide-in is the deliberate lift
            # (managed by SAG_COMP / the z-floor / the ramp below), and the
            # thumb-graze the gate exists to prevent is a horizontal miss.
            gate = 0.0 if lat_h > 0.018 else float(np.clip(1.0 - lat_h / 0.018, 0.0, 1.0))
            if e_ent < 0.08 and lat > 0.022:
                # Final 8 cm only fully ON the line, vertical included: the
                # batch that gated horizontally all the way in finished the
                # entry 26-31 mm off and bulldozed the can 15 cm through
                # flinch loops (10/15). The trim integrates while this
                # holds, so the hold is convergent, not a deadlock.
                gate = 0.0
            # Hold back the un-gated part of the entry distance: laterally
            # free, forward only as far as alignment has earned.
            tgt = final_w - (1.0 - gate) * max(e_ent, 0.0) * entry
            # Slide in HIGH, drop late: full 3 cm of extra height while the
            # entry residual is >20 cm, blending to zero by 8 cm — the
            # fingertips clear the tabletop for the whole travel instead of
            # dragging across it (the wedge that froze lat at 21-30 mm).
            # dex1: no slide-in lift. Its fingertips hang only 1.45 cm
            # under the grasp centre (vs dex3's table-dragging wedge), and
            # any extra height puts the leading finger's UNDERSIDE at the
            # can's rim — it rides the rim and knocks the can over
            # (TOUCH[dex1_finger_2] at ent 5-12 cm with pal 3-7 cm high).
            in_lift = 0.0 if _HAND == "dex1" else 0.03 * float(np.clip((e_ent - 0.08) / 0.12, 0.0, 1.0))
            v = V_APPROACH if e_ent > 0.05 else V_APPROACH * 0.5
            # Close on the palm-frame TRUTH (`full` = measured object minus
            # the wanted offset), not on distance to the planned line: with
            # the pinned arm the hover sits 15-21 mm off the LINE while the
            # can is already inside the jaw envelope, and every extra hover
            # second is another fingertip graze inching the can away (the
            # 15 cm-cumulative aborts). If the can is within the envelope,
            # closing funnels it to the palm — that is what fingers curl FOR.
            # Palm-frame gate axes follow the entry column: depth along
            # the entry axis, lateral = the other two.
            la, lb = (1, 2) if ENTRY_COL == 0 else (0, 2)
            g_lat = float(np.hypot(full[la], full[lb]))
            good = g_lat < 0.015 and abs(full[ENTRY_COL]) < 0.012
            if e_ent < (0.06 if _HAND == "dex1" else 0.02) and side in s["objects"][obj_key]["touching"]:
                # The hand is already riding the can with the entry spent:
                # the can is inside the jaw envelope but the wrist presses
                # its top, the moving can freezes the trim (anti-bulldozer),
                # and every extra hover second pushes it further (4x
                # "toppled in the in phase"/batch, lat hovering 8-18 mm).
                # Closing now funnels it to the palm — take the patient
                # tolerance immediately instead of after 8 s.
                good = g_lat < 0.020 and abs(full[ENTRY_COL]) < CLOSE_SEATED
            patient = now - in_since > 8.0 and g_lat < 0.020 and abs(full[ENTRY_COL]) < CLOSE_SEATED
            if good or patient:
                ok_ticks += 1
                if ok_ticks > 20:  # ~0.4 s in tolerance
                    phase = "close"
                    p.log(f"  stream: pre-close object@palm {np.round(q_obj, 3)} want {np.round(offset, 3)}")
            else:
                ok_ticks = max(0, ok_ticks - 2)
        elif phase == "close":
            # Keep station actively while the fingers close — a frozen
            # reference lets the palm sag and the closing sweep tips the can.
            if close_tgt is None:
                close_tgt = obj_f - rot_t @ offset + trim_w
            tgt, v = close_tgt, 0.03
            advance = True
            if capture_feedback is not None:
                tgt, advance, lost = capture_feedback.update(s, dt)
                if execution is not None:
                    stats = execution.metrics.setdefault("feedback_close", dict(active_ticks=0, paused_ticks=0))
                    stats["active_ticks"] += 1
                    stats["paused_ticks"] += int(not advance)
                    counts = stats.setdefault("pause_reasons", {})
                    for reason in capture_feedback.blocked_reasons:
                        counts[reason] = counts.get(reason, 0) + 1
                if lost:
                    return False, "capture geometry or relative motion lost during closure"
            if advance:
                alpha = min(1.0, alpha + dt / CLOSE_RAMP_S)
            arm.hands[side] = HAND_OPEN + alpha * (closed - HAND_OPEN)
            # Latch contact only when SUSTAINED: a single-tick graze latched
            # contact_alpha early, capped the squeeze low, and the under-
            # gripped can dropped out during the lift.
            from grasp_capture import bilateral_dex1_contact

            paired = require_bilateral and _HAND == "dex1"
            contact = (
                bilateral_dex1_contact(s["objects"][obj_key], side)
                if paired
                else side in s["objects"][obj_key]["touching"]
            )
            touch_ticks = touch_ticks + 1 if contact else 0
            if contact_alpha is None and touch_ticks >= 3:
                contact_alpha = alpha
            # Past contact, squeeze another half — the travel grip that
            # survives 2.4 m of gait bounce (the tighten-for-the-road step,
            # folded into the same ramp).
            lift_ready = (
                alpha >= 1.0 and touch_ticks >= 3
                if paired
                else contact_alpha is not None and alpha >= min(1.0, contact_alpha + 0.6)
            )
            if lift_ready:
                phase = "lift"
                lift_target = controlled + np.array([0.0, 0.0, LIFT_HEIGHT])
            elif paired and alpha >= 1.0 and touch_ticks < 3:
                return False, "no sustained bilateral finger contact after closing"
            elif alpha >= 1.0 and contact_alpha is None:
                return False, "no contact after closing"
        else:  # lift
            tgt, v = lift_target, V_LIFT
            if float(np.linalg.norm(ref - lift_target)) < 0.02:
                settle_until = settle_until or (now + 0.8)
                if now >= settle_until:
                    lifted = float(p.object_pos(obj_key, s)[2] - z0)
                    if lifted < LIFT_MIN:
                        return False, (f"{obj_key} rose only {lifted:.3f} m, need > {LIFT_MIN}")
                    p.log(f"  stream: lifted {lifted * 100:.1f} cm")
                    return True, ""

        ref = ref + _clipn(np.asarray(tgt, dtype=float) - ref, v * dt)
        lead = ref - controlled
        if float(np.linalg.norm(lead)) > LEAD_MAX:
            # The policy is behind; wait for it instead of running away.
            ref = controlled + _clipn(lead, LEAD_MAX)
            lead = ref - controlled
        cmd = ref + np.clip(KP_TRACK * lead, -0.05, 0.05) + capture_bias
        if phase in ("down", "in", "close") and not (close_from_current and phase == "close"):
            # The policy arm SAGS ~2 cm below its reference; geometry that
            # clears the table on paper drags the middle fingertip across it
            # in the flesh (FURN[R:middle_1] wedged, lat/ent frozen for a
            # minute). Bias the commanded height so the DELIVERED height is
            # the planned one.
            cmd = cmd + np.array([0.0, 0.0, SAG_COMP + (in_lift if phase == "in" else 0.0)])
        if surface_top is not None:
            # Hard geometric floor: with the pinned arm executing IK heights
            # EXACTLY, any command whose HAND (not wrist — the fingertips
            # hang below) would dip under tabletop+1.2 cm turns into a
            # scrape that bulldozes the can. Computed from the measured
            # hand point cloud at the CURRENT planned rotation.
            import adaptive_grasp as _ag

            low_c = float((_ag.hand_points() @ rot_t.T)[:, 2].min())
            z_floor = surface_top + (0.030 if _HAND == "dex1" else 0.012) - low_c - float((rot_t @ _ag.OFFSET)[2])
            if grasp_point_control:
                z_floor += float((rot_t @ _ag.OFFSET)[2])
            if rotation_bias is not None:
                from scipy.spatial.transform import Rotation

                biased_rot = Rotation.from_rotvec(rotation_bias).as_matrix() @ rot_t
                biased_low = float((_ag.hand_points() @ biased_rot.T)[:, 2].min())
                z_floor = max(z_floor, surface_top + 0.030 - biased_low - float((biased_rot @ _ag.OFFSET)[2]))
            if cmd[2] < z_floor:
                cmd = cmd.copy()
                cmd[2] = z_floor
            if phase == "in" and obj_f is not None:
                line_z = float((obj_f - rot_t @ offset)[2])
                dbg_z = f"  z(pal{palm[2] - line_z:+.3f} cmd{float(cmd[2]) - line_z:+.3f} fl{z_floor - line_z:+.3f})"
        aim_pf = p.world_to_root_full(cmd, s)
        rot_use = rot_t
        if _HAND == "dex1" and phase in ("up", "down"):
            # Dex1's fingers reach 18.5 cm past the wrist: flying to the
            # pregrasp with them already pointing down the entry sweeps a
            # tip arc through the can's column (video: topple with the
            # grasp centre still 14 cm out). Keep the tips pitched UP
            # while far from the pregrasp and blend flat over the last
            # 12 cm of the flight — the rotation then happens ~17 cm out,
            # where the arc is clear of the object.
            # Remaining travel, not command tracking error. A well-tracked
            # short reference step can be millimetres from the hand while
            # the pregrasp is still far away; flattening there sweeps the
            # fingers through the object during transport.
            d_pre = float(np.linalg.norm(np.asarray(pre_w) - palm))
            k_up = float(np.clip((d_pre - 0.02) / 0.12, 0.0, 1.0))
            th = np.deg2rad(55.0) * k_up
            if th > 1e-3:
                c, sn = np.cos(th), np.sin(th)
                for sgn in (1.0, -1.0):
                    ry = np.array([[c, 0.0, sgn * sn], [0.0, 1.0, 0.0], [-sgn * sn, 0.0, c]])
                    cand = rot_t @ ry
                    if (cand @ np.array([1.0, 0.0, 0.0]))[2] > 0:
                        rot_use = cand
                        break
        if rotation_bias is not None:
            from scipy.spatial.transform import Rotation

            rot_use = Rotation.from_rotvec(rotation_bias).as_matrix() @ rot_use
        rot_pf = Primitives._rot(s["pelvis"]["quat"]).T @ rot_use
        q_sol, res, _ = arm.ik.solve(
            side,
            aim_pf,
            rot_pf,
            q_mj29=arm.q,
            use_waist=True,
            iters=10,
            **(dict(point_offset=offset, rot_weight=0.1) if grasp_point_control else {}),
        )
        if rotation_bias is not None or reference_guard is not None:
            q_sol = arm.q + np.clip(q_sol - arm.q, -np.deg2rad(2.0), np.deg2rad(2.0))
        if transport_guard is not None and transport_phase is not None:
            candidate = q_sol if res < 0.10 else arm.q
            checked = np.asarray(candidate).copy()
            checked[:12] = np.asarray(s["joints"])[:12]  # legs are controlled by WBC
            guard_start = time.perf_counter()
            swept = transport_guard.check(s, checked, side)
            if execution is not None:
                stats = execution.metrics.setdefault("transport_guard", dict(checks=0, max_ms=0.0, rejected=None))
                stats["checks"] += 1
                stats["max_ms"] = max(stats["max_ms"], (time.perf_counter() - guard_start) * 1000)
                if not swept.clear:
                    stats["rejected"] = swept.to_dict()
            if not swept.clear:
                return False, (
                    f"transport clearance in {transport_phase}: {swept.link} vs "
                    f"{swept.obstacle}, {swept.distance:.4f} m"
                )
        if reference_guard is not None:
            rejection = reference_guard(s, q_sol if res < 0.10 else arm.q, phase)
            if rejection:
                return False, rejection
        if res < 0.10:
            arm.q = q_sol
        arm._push(arm.q)
        if pin is not None:
            # Position-mode arm: the sim-side per-motor PD tracks the IK
            # solution exactly (+-0.03 rad measured), instead of the policy's
            # ~1 cm-and-worse delivery. The streamed reference above still
            # runs so the policy's balance model sees the arm where it is;
            # its arm torques are overridden at the actuator.
            # Waist stays with the policy: pinning it stiff fought the
            # balance controller (torso jitter, 9/15 toppled) — the arm pin
            # plus the vertical trim absorb the lean shortfall instead.
            pin(**{side: [float(v) for v in arm.q[ARM_MJ[side]]]})

        # A palm that has not moved 2 cm in 8 s during the travel phases is
        # not slow, it is stuck — either the IK is rejecting every solution
        # (res stays high) or the policy is not delivering the reference.
        # Fail fast and SAY WHICH: three episodes burned their whole 60 s
        # deadline pinned at the lead cap with no diagnosis in the log.
        if stall_palm is None or float(np.linalg.norm(palm - stall_palm)) > 0.02:
            stall_palm, stall_t = palm.copy(), now
        elif phase in ("up", "down") and now - stall_t > 8.0:
            return False, (
                f"arm not tracking in the {phase} phase "
                f"(palm stalled 8 s, IK res {res:.3f}, "
                f"target {float(np.linalg.norm(tgt - palm)):.2f} m "
                f"away)"
            )

        if now - last_log > 2.0:
            extra = ""
            ft = s.get("furniture_touch")
            if ft:
                extra += f"  FURN[{','.join(ft)}]"
            if touch_seen:
                extra += f"  TOUCH[{','.join(sorted(touch_seen))}]"
                touch_seen.clear()
            dcan = float(np.linalg.norm(obj - obj0))
            if dcan > 0.01:
                extra += f"  can_moved {dcan * 100:.1f}cm"
            if phase == "in" and full_f is not None:
                extra += (
                    f"  lat {full_f[0] * 1000:.0f}mm"
                    f"(h{full_f[2] * 1000:.0f}"
                    f" v{full_f[3] * 1000:+.0f})"
                    f" ent {full_f[1] * 100:.1f}cm" + (dbg_z if dbg_z else "")
                )
                # jaw-axis vertical components: planned rot vs MEASURED palm
                extra += f"  jawz(t{rot_t[2, 1]:+.2f} m{rot_p[2, 1]:+.2f})"
                hq = s["palms"][side].get("hand_q")
                if hq:
                    extra += "  hq[" + ",".join(f"{v:+.3f}" for v in hq) + "]"
            tpf = p.world_to_root_full(np.asarray(tgt, dtype=float), s)
            point_name = "grasp_center" if grasp_point_control else "palm"
            p.log(
                f"    stream cont[{phase}] {point_name}->ref "
                f"{float(np.linalg.norm(lead)) * 100:.1f} cm  res {res:.3f}"
                f"  tgt_pf [{tpf[0]:+.2f} {tpf[1]:+.2f} {tpf[2]:+.2f}]{extra}"
            )
            last_log = now
    return False, f"continuous servo timed out in the {phase} phase"


def grasp_streamed(
    p: Primitives,
    obj_key: str,
    grip_world,
    plan,
    side: str = "right",
    surface_top: float | None = None,
    replan=None,
    ghost=None,
    nudge: bool = True,
    pin=None,
    squat_hint: float | None = None,
    stand_rel=None,
    envelope=None,
    plan_r: float | None = None,
) -> tuple[bool, str]:
    """Legacy tuple interface; new task code consumes execute_grasp's result."""
    global LAST_EGO_VIS
    LAST_TRANSITION.clear()
    LAST_EGO_VIS = None
    result = execute_grasp(
        p,
        obj_key,
        grip_world,
        plan,
        side=side,
        surface_top=surface_top,
        replan=replan,
        ghost=ghost,
        nudge=nudge,
        pin=pin,
        squat_hint=squat_hint,
        stand_rel=stand_rel,
        envelope=envelope,
        plan_r=plan_r,
    )
    LAST_TRANSITION.update(result.execution["transition"])
    LAST_EGO_VIS = result.visibility
    return result.ok, result.why


def execute_grasp(p, obj_key, grip_world, plan, *, execution=None, **options):
    """Approach, posture delivery, measured replan, servo; return owned results.

    Optional execution joins a containing task's journal. No caller reads globals.
    Failure stage is the first control failure, not a post-hoc physical taxonomy.
    """
    execution = execution if execution is not None else ExecutionRecord()
    execution.transition.clear()
    execution.metrics.pop("ego_visibility", None)
    try:
        ok, why = _execute_grasp(p, obj_key, grip_world, plan, execution=execution, **options)
    except Exception as exc:
        execution.fail(f"{type(exc).__name__}: {exc}")
        raise
    if not ok:
        execution.fail(why)
    return GraspResult(ok, why, execution.snapshot())


def _execute_grasp(
    p: Primitives,
    obj_key: str,
    grip_world,
    plan,
    side: str = "right",
    surface_top: float | None = None,
    replan=None,
    ghost=None,
    nudge: bool = True,
    pin=None,
    squat_hint: float | None = None,
    stand_rel=None,
    envelope=None,
    plan_r: float | None = None,
    *,
    execution,
    config=DEFAULT_REACH,
    transport_guard_factory=None,
    coordinated=None,
) -> tuple[bool, str]:
    """The whole grasp: raise, walk in, servo, close, lift. -> (ok, why)."""
    # Raise BEFORE walking in: from here on the hand is above the tabletop
    # and the approach can only ever come down onto the can, never up through
    # the table from under its rim.
    # No arm-drop before the reach. The hands travel and STAY above the
    # tabletop (READY height ~0.85 m): dropping them at the table put the
    # hanging hand BELOW the tabletop right at the edge, and the servo's
    # up-phase then hooked the fingers on the table (FURN[R:middle_*],
    # palm pinned at the lead cap for 8 s). The servo's first target is
    # above the pregrasp at ~the same height the hands already are.
    # plan_r is retained for old callers, but cannot set a transition gate:
    # it is sampled per episode and is not observable to the student.
    x_band = config.post_max - 0.02
    execution.metrics["reach_config"] = config.to_dict()
    # NO settle hold here (2026-09-03, user rule: the sequence may be slow,
    # but it must never visibly pause). A settle-until-still loop (<= 2 s,
    # nearly always hitting the cap) was measured at +1.5 s of standing
    # between the gait's end and the crouch (flow200 0.9 s, still/shift
    # batches 2.4 s), and it bought nothing: the post-crouch grip x spread
    # was the same with it (0.20-0.38) as without (0.19-0.37). The brake's
    # momentum slide (5-18 cm) is absorbed by braking ~4 cm earlier instead
    # (grasp_gym MARGIN_STOP), and the crouch itself stops the base; the
    # post-crouch band check below is the only stand gate. Under one-shot
    # the pre-crouch nudge is skipped too (it never fired in 24 rounds and
    # a walk-stop-walk shuffle here is exactly the hesitation to keep out
    # of the data); outside one-shot it still centres the stand.
    # Closed-loop, pause-free last centimetres (2026-09-03): shuffle until the
    # grip's pelvis-frame x is in [0.18, 0.27] and go straight into the
    # crouch (which moves the pelvis 3-15 cm back, typically 8). Replaces the
    # open-loop brake offset, which was load-dependent (see _creep_to_band).
    # Not gated on `nudge`: grasp_gym passes nudge=False (that flag covered the
    # old walk-in shuffles), and the first creep batch silently ran without it
    # (12/29, every discard a stand outside the band).
    execution.enter("approach")
    post_max = x_band + 0.02
    pre_min, pre_max = config.approach_band()
    aim_max = float(envelope[2]) if envelope is not None else np.deg2rad(config.azimuth_deg)
    effective_envelope = (None, 0.0, aim_max)
    if config.ready_grasp_recovery:
        if coordinated is None or not hasattr(coordinated, "execute"):
            return False, "recovery configuration requires a recovery controller"
        return coordinated.execute(grip_world=grip_world, squat_hint=squat_hint, ghost=ghost)
    approach = _creep_to_band(
        p,
        grip_world,
        side,
        x_min=pre_min,
        x_max=pre_max,
        target_rel=stand_rel,
        envelope=effective_envelope,
        max_s=config.approach_timeout,
        update_upper=coordinated.walk_tick if coordinated is not None else None,
        target_lateral=getattr(coordinated, "approach_lateral", None),
    )
    execution.transition.update(
        approach_status=approach.status.value,
        approach_seconds=approach.elapsed,
        pre_depth=approach.depth,
        pre_azimuth=approach.azimuth,
        pre_band=[pre_min, pre_max],
        post_band=[config.post_min, post_max],
        expected_backshift=config.backshift,
        depth_frame="pelvis_yaw",
    )
    if approach.status != ApproachStatus.READY:
        return False, f"[approach] {approach.status.value}: depth {approach.depth:.3f} m"
    if coordinated is not None:
        return coordinated.finish(grip_world=grip_world, squat_hint=squat_hint, ghost=ghost)
    execution.enter("arm_control")
    arm = make_arm(p, side)
    arm.guard = obj_key  # off-hand guard (see _guard_left)
    guard = transport_guard_factory(arm.ik) if config.transport_guard and transport_guard_factory else None
    execution.metrics["transport_guard_enabled"] = guard is not None
    arm.start()
    # NOTE: the sim-side pin must already be RELEASED before this call.
    # Holding it through start() and releasing here removes the visible
    # arm twitch (the gait prior pulls the unheld arms in during the
    # 0.7 s latch-clear window, then the first streamed frame snaps them
    # back) — but it also cost every grasp: 0/6 on seed 11 versus 5/6
    # with the early release, with or without a 1 s settle after the
    # release. Whatever the arm carries out of a stiff pin, the servo
    # cannot absorb it. Fix the twitch some other way.
    try:
        # A/B knob (GYM_IDLE_SWEEP=1): sweep the idle arm to the free flank
        # with a shoulder-yaw turn before the crouch. Removing this sweep is
        # the only grasp-path change at the batch boundary where the success
        # rate fell (19/21 -> 9/24), but the mechanism I proposed for it (a
        # centre-of-mass shift) measures just 6.1 mm on a 35.9 kg robot, so
        # the link is unproven — this switch exists to settle it by
        # experiment rather than by argument.
        if os.environ.get("GYM_IDLE_SWEEP") == "1":
            other = "left" if side == "right" else "right"
            q_sw = arm.q.copy()
            q_sw[ARM_MJ[other]] = [
                0.0,
                0.35 if other == "left" else -0.35,
                1.2 if other == "left" else -1.2,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            arm.move_to(q_sw, speed=1.0)
        # The IDLE arm is left ALONE — no sweep, no pre-lift. It rides the
        # crouch at the READY pose, and that clears the tabletop with room
        # to spare: measured on the true hand meshes (not the bounding
        # spheres, which sit 6.2 cm low and drove an earlier sideways
        # sweep), its lowest point is pelvis +0.065, i.e. 0.698 m world
        # after a 16 cm squat and 0.663 m at the 19.5 cm floor — above
        # both tabletops (0.62 / 0.66). The old FURN[wrist_yaw] contacts
        # came from the deep-squat era standing 7 cm closer.
        # The SERVO arm gets no pre-lift either: squat(track=True) below
        # holds its palm at the CURRENT world pose via live IK while the
        # legs fold under it — the grasp trajectory then starts from
        # exactly where the hand already is.
        # PLANNED crouch before reaching: bring the body down until the
        # wrist's natural height is near the grip, so the (limit-clamped)
        # waist only fine-tunes instead of folding to bridge the gap. The
        # wrist rides ~7.5 cm above the grip point, so the drop is the
        # palm's current height minus (grip + that margin).
        execution.enter("posture")
        s0 = p.obs()
        p.log(f"  stream: grip x {p.world_to_root(grip_world, s0)[0]:.2f} m before the crouch")
        palm_z = float(s0["palms"][side]["pos"][2])
        drop = float(np.clip(palm_z - (float(grip_world[2]) + 0.06), 0.0, config.max_drop))
        if squat_hint is not None:
            # Navigation planned this crouch so the sweet core's THICK band
            # lands on the object — never crouch less than the plan.
            drop = float(np.clip(max(drop, squat_hint), 0.0, config.max_drop))
        if drop > 0.03:
            _sb = p.obs()
            _pb = np.array(_sb["pelvis"]["pos"][:2])
            _yb = float(_sb["pelvis"]["yaw"])
            _gb = p.world_to_root(grip_world, _sb)
            got = arm.squat(drop, track=True)
            _sa = p.obs()
            _pa = np.array(_sa["pelvis"]["pos"][:2])
            _ga = p.world_to_root(grip_world, _sa)
            execution.transition.update(
                squat_requested=drop,
                squat_delivered=got,
                pre_squat_depth=float(_gb[0]),
                post_depth=float(_ga[0]),
                actual_depth_shift=float(_ga[0] - _gb[0]),
                post_full_grip=list(map(float, p.world_to_root_full(grip_world, _sa))),
            )
            if _sa["fallen"] or got < min(0.06, 0.6 * drop):
                return False, "[transition] crouch not delivered"
            _fwd = float(np.dot(_pa - _pb, [np.cos(_yb), np.sin(_yb)]))
            p.log(
                f"  stream: crouch moved the pelvis {_fwd * 100:+.0f} cm forward "
                f"({float(np.linalg.norm(_pa - _pb)) * 100:.0f} cm total, yaw "
                f"{np.degrees(wrap(float(_sa['pelvis']['yaw']) - _yb)):+.0f} deg); "
                f"grip x {_gb[0]:.2f} -> {_ga[0]:.2f} m"
            )
            # Post-crouch band check (2026-09-03): between the approach and
            # the crouch the IDLE hold drifts the pelvis ~9 cm forward and
            # the planner crouch then moves it 4-13 cm BACK; the grip's
            # pelvis-frame x after the crouch spreads 0.19-0.42 m against a
            # 0.22-0.36 band, and the low tail is every "stand is wrong"
            # failure. Verify after the crouch; if outside the band, stand,
            # nudge against a virtual grip shifted by the mean crouch
            # back-shift, and crouch again (at most twice).
            for _fix in range(2):
                _sa = p.obs()
                _gx = float(p.world_to_root(grip_world, _sa)[0])
                if config.post_min <= _gx <= post_max:
                    break
                if os.environ.get("GYM_ONE_SHOT", "0") == "1":
                    # flow200 rule: no repair segments in the data -- a stand
                    # that needs correcting is a discarded round, not a
                    # stand-up/shuffle/crouch-again sequence for the VLA to learn
                    return False, (
                        f"[stand] band correction needed: grip x {_gx:.2f} m "
                        f"after the crouch, band [{config.post_min:.2f}, {x_band + 0.02:.2f}]"
                    )
                p.log(
                    f"  stream: grip x {_gx:.2f} m after the crouch is outside "
                    f"[{config.post_min:.2f}, {x_band + 0.02:.2f}] -- stand, nudge, crouch again"
                )
                execution.enter("posture_correction", attempt=_fix + 1)
                arm.stand()
                _s = p.obs()
                _yaw = p.pelvis_yaw(_s)
                _fwd = np.array([np.cos(_yaw), np.sin(_yaw), 0.0])
                # this robot's own crouch back-shift (not the mean: the one
                # failing round crouched back only 3 cm, and the mean prior
                # then judged 0.18 m standing as "fine"), minus a margin so
                # the nudge backs off until the NEXT crouch lands in-band
                _back = float(_ga[0] - _gb[0])
                _virtual = np.asarray(grip_world, dtype=float) + max(_back - 0.04, -0.05) * _fwd
                p.log(
                    f"  stream: nudging against a grip shifted "
                    f"{max(_back - 0.04, -0.05) * 100:+.0f} cm (crouch back-shift {_back * 100:+.0f} cm)"
                )
                _nudge_pass(p, _virtual, side, x_min=config.post_min, x_max=x_band)
                arm.q = np.array(p.obs()["joints"])
                arm._q_prev = arm.q.copy()
                _gb = p.world_to_root(grip_world, p.obs())
                got = arm.squat(drop, track=True)
                _ga = p.world_to_root(grip_world, p.obs())
            p.log(f"  stream: planned crouch {drop * 100:.0f} cm wanted, {got * 100:.0f} cm delivered")
        # Replan against the measured pose after crouching, before any reach.
        execution.enter("reach_validation")
        measured = p.obs()
        if measured["fallen"] or (drop > 0.03 and got < min(0.06, 0.6 * drop)):
            return False, "[transition] crouch not delivered after correction"
        if toppled(measured, obj_key):
            return False, "[transition] object toppled before reach"
        live_grip = np.asarray(grip_world, dtype=float).copy()
        live_grip[:2] = measured["objects"][obj_key]["pos"][:2]
        actual_grip = p.world_to_root(live_grip, measured)
        execution.transition["reach_check_depth"] = float(actual_grip[0])
        execution.transition["reach_check_azimuth"] = float(np.arctan2(actual_grip[1], actual_grip[0]))
        if not (
            config.post_min <= actual_grip[0] <= post_max and abs(np.arctan2(actual_grip[1], actual_grip[0])) <= aim_max
        ):
            return False, "[transition] post-crouch geometry outside depth/cone"
        if replan is not None:
            plan = replan(measured["palms"][side]["pos"], Primitives._rot(measured["palms"][side]["quat"]))
            if not plan.ok:
                return False, f"[transition] post-crouch replan: {plan.why}"
        execution.transition["reach_started"] = True
        # One continuous 50 Hz servo does everything: down the live
        # replanned line to the pregrasp, jaw-gated slide onto the can,
        # close, lift — no stop-and-settle anywhere.
        ok, why = _servo_continuous(
            p,
            arm,
            obj_key,
            plan,
            side,
            replan=replan,
            ghost=ghost,
            pin=pin,
            surface_top=surface_top,
            execution=execution,
            transport_guard=guard,
        )
        if not ok and os.environ.get("GYM_ONE_SHOT", "0") != "1" and ("timed out" in why or "not tracking" in why):
            # Diagnose before medicating: "cannot reach" is usually the grip
            # sitting outside the depth band (a nudge overshoot), and walking
            # back fixes that; the squat only helps when the grip is LOW, and
            # squatting at the wrong depth just adds a knee into the table.
            execution.enter("reach_retry")
            s = p.obs()
            grip_pf = p.world_to_root(grip_world, s)
            if not (config.post_min <= grip_pf[0] <= x_band):
                arm.stop()
                _nudge_closer(p, grip_world, side, x_min=config.post_min, x_max=x_band)
                arm = make_arm(p, side)
                arm.start()
                s = p.obs()
                grip_pf = p.world_to_root(grip_world, s)
            # No squat here. It was tried as the can't-reach fallback and made
            # things strictly worse: a 13 cm drop leaves the palm targets
            # ABOVE the pelvis, which is the policy's worst delivery zone,
            # and the episode still stalled 0.27 m short. The raised hand
            # plus the tilted approach covers this table; a genuinely lower
            # surface needs a planned crouch, not a panic one.
            p.log("  stream: retrying the servo after re-nudging")
            ok, why = _servo_continuous(
                p,
                arm,
                obj_key,
                plan,
                side,
                replan=replan,
                ghost=ghost,
                pin=pin,
                surface_top=surface_top,
                execution=execution,
                transport_guard=guard,
            )
        return ok, why
    finally:
        if pin is not None:
            pin()  # release the arm motors back to the policy
        arm.stop()
