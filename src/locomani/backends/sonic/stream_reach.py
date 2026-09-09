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

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from locomani.backends.sonic.primitives import (
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
from locomani.pipeline.execution_record import ExecutionRecord, GraspResult
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.g1_joints import ARM_MJ, mujoco_to_isaac
from locomani.tasks.bottle.config import DEFAULT_BOTTLE_TASK
from locomani.tasks.bottle.reach_config import DEFAULT_REACH, ReachConfig

LIFT_MIN = 0.06  # the object must actually leave the table

# BottleTaskConfig.{arm_channel,arm_vel,left_guard} defaults (decision 8, were
# GYM_ARM_CHANNEL/GYM_ARM_VEL/GYM_LEFT_GUARD); module-level so a caller can
# override by assignment, same pattern as grasp_gym.RESTART_DEPLOY.
ARM_CHANNEL = DEFAULT_BOTTLE_TASK.arm_channel
ARM_VEL = DEFAULT_BOTTLE_TASK.arm_vel
LEFT_GUARD = DEFAULT_BOTTLE_TASK.left_guard


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


# Shared depth limit, independent of the sampled navigation waypoint. The
# bottle pilot physically grasped at yaw-frame depths 0.294..0.381 m. This
# 0.40 m gate is a screening bound, not a collision-free IK certificate;
# post-crouch replanning and actual grasp verification remain mandatory.
TRANSITION_POST_MAX = DEFAULT_REACH.post_max


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
from locomani.backends.sonic.primitives_const import HAND as _HAND  # noqa: E402

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
    commit). BottleTaskConfig.arm_channel="streamed" restores the old path.
    """
    if ARM_CHANNEL == "planner":
        return PlannerArm(p, side, send_vel=ARM_VEL)

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
        from locomani.backends.sonic.primitives import upper_body_from_mj

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
        if self.guard is None or self.side != "right" or not LEFT_GUARD:
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
        from locomani.backends.sonic.primitives import upper_body_from_mj

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
        from locomani.backends.sonic.primitives import upper_body_from_mj

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
        from locomani.backends.sonic.primitives import upper_body_from_mj

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
    from locomani.backends.sonic.agent_server import LocomotionMode
    from locomani.backends.sonic.primitives import READY_WRISTS

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
        from locomani.tasks.bottle.measured_grasp import closure_geometry

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
                from locomani.tasks.bottle import adaptive_grasp as _ag

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
            from locomani.tasks.bottle.grasp_capture import bilateral_dex1_contact

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
            from locomani.tasks.bottle import adaptive_grasp as _ag

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
    """Dispatch the grasp to the recovery controller. -> (ok, why).

    Refactor step 2, family 2 (2026-09-09): the old non-recovery approach/
    crouch/servo tail (_creep_to_band, GYM_ONE_SHOT, _nudge_pass/_nudge_closer,
    reach retry, GYM_IDLE_SWEEP) is retired -- every surviving collection
    profile sets ready_grasp_recovery=True and grasp_gym always builds a
    recovery controller when ready_grasp_baseline is True. See
    docs/handoff/refactor-step2-candidates.md.
    """
    execution.metrics["reach_config"] = config.to_dict()
    if not config.ready_grasp_recovery:
        return False, "ready_grasp_recovery=False is retired; every collection profile now uses a recovery controller"
    if coordinated is None or not hasattr(coordinated, "execute"):
        return False, "recovery configuration requires a recovery controller"
    return coordinated.execute(grip_world=grip_world, squat_hint=squat_hint, ghost=ghost)
