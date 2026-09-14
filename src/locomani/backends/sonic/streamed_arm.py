"""Planner and POSE arm channels, with explicit reference ownership at takeover.

Whole-body callers can preserve the consumed planner pose during mode changes.
Legacy single-arm callers keep the established default handoff and guard behavior.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from locomani.backends.sonic.primitives import (
    DT,
    HAND_CLOSED_LEFT,
    HAND_CLOSED_RIGHT,
    HAND_OPEN,
    LocomotionMode,
    Primitives,
    StreamMode,
)
from locomani.primitives.arm_ik import ArmIK
from locomani.primitives.g1_joints import ARM_MJ, mujoco_to_isaac

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

# m, object displacement that aborts the attempt. Shared with servo.py's
# _servo_continuous (the moved-too-much guard and the "in" phase's tolerance
# multiples), which imports it from here.
NUDGE_MAX = 0.05

# BottleTaskConfig.left_guard default (decision 8, was GYM_LEFT_GUARD);
# module-level so a caller can override by assignment (ARM_CHANNEL/ARM_VEL,
# the same pattern, stay in stream_reach.py next to make_arm()).
# Off-hand guard (2026-09-03): while the right hand grasps, the torso yaws
# toward the object and the FROZEN left arm sweeps across the table -- a
# recurring "bottle toppled" failure. Every pushed frame checks the left
# palm's distance to the object and, inside LEFT_SAFE_M, solves left-arm IK
# to a retreat target radially away from the object (rate-limited, no
# waist). GYM_LEFT_GUARD=0 disables it (A/B).
LEFT_SAFE_M = 0.24  # start retreating inside this palm-object distance
LEFT_RETREAT_M = 0.10  # retreat target sits SAFE + this from the object
LEFT_GUARD_RATE = 1.2  # rad/s cap on the guarded left-arm joints

# Squat whenever the grip sits below this pelvis-frame height. Generous on
# purpose: the palm's sweet spot is at or above pelvis height minus a few cm,
# and every centimetre of squat is a centimetre of waist lean the policy does
# not have to deliver (it delivers only ~2/3 of a commanded lean).
SQUAT_DROP_MAX = 0.20  # deepest crouch the planner squat is asked for
SQUAT_HEIGHT_BIAS = 0.02  # delivered pelvis_z - commanded height (measured)
SQUAT_SETTLE = (
    1.6  # s; covers the k-planner squat's own t90 of 1.1-1.6 s (was 2.5 -- the extra 0.9 s was recorded dead time)
)
SQUAT_GAIN = 6.8  # rad of squat per metre of wanted drop (measured)
SQUAT_MAX = 0.65


def yaw_only(quat_wxyz):
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)], dtype=np.float32)


class StreamedArm:
    """Owns the pose stream: joint references, hand targets, pacing."""

    def __init__(self, p: Primitives, side: str, *, ik=None, left_guard: bool = True, preview=None):
        self.p = p
        self.side = side
        self.ik = ArmIK() if ik is None else ik
        self.left_guard = left_guard
        self.preview = preview
        # The waist may HELP the reach but only within a human lean. With the
        # stock limits the IK bridged any height/side gap by folding the
        # torso flat onto the table (head down, palm never reaching its
        # lifted reference); with the waist excluded entirely, side-offset
        # pregrasps went unreachable (res 0.13). Indices 12/13/14 = waist
        # yaw/roll/pitch in MuJoCo order.
        for j, (lo, hi) in {12: (-0.35, 0.35), 13: (-0.15, 0.15), 14: (-0.05, 0.20)}.items():
            self.ik.lo[j], self.ik.hi[j] = lo, hi
        self.buf: deque = deque(maxlen=CHUNK)
        self.root_buf: deque = deque(maxlen=CHUNK)
        self.velocity_buf: deque = deque(maxlen=CHUNK)
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

    def start(self, *, hold_planner_pose=False) -> None:
        s = self.p.obs()
        # A primed POSE stream atomically takes the planner override in the deploy.
        if hold_planner_pose:
            # Fill the POSE buffer while the old planner reference is still held.
            # No intermediate VR encoder mode or released planner arm frame is needed.
            from locomani.backends.sonic import upper_ref

            incoming = upper_ref.shared().pose()
            if incoming is None:
                raise RuntimeError("continuous POSE takeover needs a fresh planner joint/root reference")
            self.q, self.root_quat = (np.asarray(value).copy() for value in incoming)
            # body_q_target excludes the frozen override. Compose the actual
            # reference being consumed, not the measured tracking error.
            held = self.p.cmd.upper_body
            if held is not None:
                self.q[12:15], self.q[15:22], self.q[22:29] = held[:3], held[3::2], held[4::2]
            self.next_t = time.time()
            self.hold(0.1)
            self.p.cmd.set_upper_body_vel(None)
            self.p.cmd.enter_streamed()
            self.p.cmd.upper_body = None
            self.hold(0.1)
            return
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

    def stop(self, *, preserve_reference=False) -> None:
        """Hand control back to the planner; optionally retain the owned reference."""
        from locomani.backends.sonic.primitives import upper_body_from_mj

        s = self.p.obs()
        palm_pf = self.p.world_to_root_full(s["palms"][self.side]["pos"], s)
        self.p.cmd.set_wrists(**{f"{self.side}_pos": np.asarray(palm_pf, dtype=np.float32)})
        # In FROZEN_UPPER_BODY mode the planner pins the arms to the 17-value
        # target, so hand back the pose the streamed reach actually ended in
        # — otherwise the switch snaps the arms to the stale target and yanks
        # whatever the hand is holding.
        self.p.cmd.set_upper_body(upper_body_from_mj(self.q if preserve_reference else s["joints"]))
        self.p.cmd.set_hands(**{self.side: self.hands[self.side]})
        if preserve_reference:
            # The deploy reads its latest planner packet when switching modes.
            # Prime that packet before the switch; setting the Python cache
            # alone leaves a gap with stale or absent arm/hand ownership.
            self.p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
            self.p.cmd.vr_targets = False
            self.p.cmd.set_upper_body_vel(np.zeros(17))
            self.p.cmd.frame(
                mode=LocomotionMode.IDLE,
                height=-1.0,
                speed=0.0,
                movement=(0, 0, 0),
                facing=(np.cos(s["pelvis"]["yaw"]), np.sin(s["pelvis"]["yaw"]), 0),
            )
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

    def _push(self, q_mj29, *, reference_velocity=None, reference_time=None) -> None:
        """Buffer each pose with its own root rotation and optional 29-joint velocity."""
        q_mj29 = self._guard_left(np.asarray(q_mj29, dtype=float).copy())
        velocity = np.zeros(29) if reference_velocity is None else np.asarray(reference_velocity, dtype=float)
        if velocity.shape != (29,) or not np.isfinite(velocity).all():
            raise ValueError("reference_velocity must contain 29 finite values")
        if reference_velocity is not None and self.guard is not None and self.left_guard:
            raise ValueError("explicit velocity requires an unguarded reference")
        self.buf.append(mujoco_to_isaac(q_mj29).astype(np.float32))
        self.root_buf.append(np.asarray(self.root_quat, dtype=np.float32).copy())
        self.velocity_buf.append(mujoco_to_isaac(velocity).astype(np.float32))
        positions, roots, velocities = np.stack(self.buf), np.stack(self.root_buf), np.stack(self.velocity_buf)
        first_frame = self.i - len(self.buf) + 1
        if self.preview is not None:
            # Explicit physical time for control; priming samples the same clock.
            stamp = float(self.p.obs()["t"]) if reference_time is None else float(reference_time)
            poses, speeds, roots, first_frame = self.preview.sample(stamp, q_mj29, velocity, self.root_quat)
            positions = np.array([mujoco_to_isaac(q) for q in poses], dtype=np.float32)
            velocities = np.array([mujoco_to_isaac(v) for v in speeds], dtype=np.float32)
        self.p.cmd.pose_chunk(
            positions,
            roots,
            first_frame,
            left_hand=self.hands["left"],
            right_hand=self.hands["right"],
            joint_vel_isaac=velocities,
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
        if self.guard is None or self.side != "right" or not self.left_guard:
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
    never clears, avoiding that latch-clear transition. The caller still needs
    a continuous reference to avoid position and velocity jumps on this channel.

    The cost is the legs: they belong to the k-planner animation, so
    `squat()` is a no-op here and pelvis height is only reachable through
    the planner message's `height` field.

    `send_vel=True` also fills `upper_body_velocity` by finite difference.
    The encoder observes motion_joint_velocities_*. Sending zeros for a
    changing reference makes that reference look
    stationary to it.
    """

    def __init__(self, p: Primitives, side: str, send_vel: bool = False, *, ik=None, left_guard: bool = True):
        super().__init__(p, side, ik=ik, left_guard=left_guard)
        self.send_vel = send_vel
        self.height = -1.0  # planner-message height; -1 = default
        self.mode = LocomotionMode.IDLE
        self._q_prev = None
        self._face = (1.0, 0.0, 0.0)

    def start(self, *, upper_reference=None, settle_seconds=0.6) -> None:
        """Take over, optionally preserving an already owned frozen reference.

        The explicit 17-value reference is the caller's last commanded upper
        body, not the deploy's unmodified planner motion or measured joints.
        Default entry keeps the historical measured seed and 0.6 s hold.
        A caller that already owns and has settled this channel can explicitly
        provide its reference and settle_seconds=0 for immediate takeover.
        """
        from locomani.backends.sonic.primitives import upper_body_from_mj

        if not np.isfinite(settle_seconds) or settle_seconds < 0:
            raise ValueError("settle_seconds must be finite and non-negative")
        if settle_seconds == 0 and upper_reference is None:
            raise ValueError("immediate takeover requires an explicit owned upper_reference")
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
        self.hold(settle_seconds)

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

    def _push(self, q_mj29, observed=None, *, movement=(0.0, 0.0, 0.0), speed=-1.0, reference_velocity=None) -> None:
        """Publish and pace one reference; explicit velocity uses 29-joint MuJoCo order.

        An unguarded trajectory owner can supply its elapsed-time derivative.
        Other callers retain the existing send_vel finite-difference behavior.
        """
        from locomani.backends.sonic.primitives import upper_body_from_mj

        if reference_velocity is not None:
            reference_velocity = np.asarray(reference_velocity, dtype=float)
            if reference_velocity.shape != (29,) or not np.isfinite(reference_velocity).all():
                raise ValueError("reference_velocity must contain 29 finite joint velocities")
            if self.guard is not None and self.left_guard:
                raise ValueError("explicit velocity requires an unguarded reference")
        q = self._guard_left(np.asarray(q_mj29, dtype=float).copy(), observed=observed)
        u = upper_body_from_mj(q)
        self.p.cmd.set_upper_body(u)
        if reference_velocity is not None:
            self.p.cmd.set_upper_body_vel(upper_body_from_mj(reference_velocity))
        elif self.send_vel:
            self.p.cmd.set_upper_body_vel((u - upper_body_from_mj(self._q_prev)) / DT)
        self._q_prev = q.copy()
        self.p.cmd.set_hands(left=self.hands["left"], right=self.hands["right"])
        self.p.cmd.frame(mode=self.mode, facing=self._face, height=self.height, movement=movement, speed=speed)
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
