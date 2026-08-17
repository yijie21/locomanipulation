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

import numpy as np

from arm_ik import ArmIK
from g1_joints import ARM_MJ, mujoco_to_isaac
from primitives import (
    DT,
    GRASP_CENTRE_OFFSET,
    HAND_CLOSED_LEFT,
    HAND_CLOSED_RIGHT,
    LIFT_HEIGHT,
    Primitives,
    wrap,
)

LIFT_MIN = 0.06           # the object must actually leave the table

CHUNK = 5                 # pose frames per message, like the pico manager
JOINT_SPEED = 0.8         # rad/s for large repositioning moves
FINE_SPEED = 0.4          # rad/s inside the servo
SETTLE = 2.0              # s after a repositioning move
REFINE_TOL = 0.015        # m, palm error to stop refining the pregrasp at
REFINE_MAX = 3
SAFE_GAP = 0.13           # hold the object this far out along the entry axis
# (0.10 left only 5 cm of margin over a bad pregrasp, and 5 cm pregrasp
# errors do happen — the fingertips grazed the can twice)
DEPTH_TOL = 0.03
# Looser than servo_grasp's 6 mm: that loop nudged a 50 Hz drifting command,
# this one hops between IK solutions with a ~1 cm delivery floor, so an 8 mm
# gate just oscillates. The advance gate keeps re-centring on the way in.
ALIGN_TOL = 0.013
GATE_TOL = 0.018
GRASP_TOL = 0.020
STEP_MAX = 0.03           # m, largest palm correction per servo step
NUDGE_MAX = 0.05          # m, object displacement that aborts the attempt
# Squat whenever the grip sits below this pelvis-frame height. Generous on
# purpose: the palm's sweet spot is at or above pelvis height minus a few cm,
# and every centimetre of squat is a centimetre of waist lean the policy does
# not have to deliver (it delivers only ~2/3 of a commanded lean).
SQUAT_TRIGGER = 0.03
SQUAT_GAIN = 6.8          # rad of squat per metre of wanted drop (measured)
SQUAT_MAX = 0.65


def yaw_only(quat_wxyz):
    w, x, y, z = quat_wxyz
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
                    dtype=np.float32)


class StreamedArm:
    """Owns the pose stream: joint references, hand targets, pacing."""

    def __init__(self, p: Primitives, side: str):
        self.p = p
        self.side = side
        self.ik = ArmIK()
        self.buf: deque = deque(maxlen=CHUNK)
        self.i = 0
        # Start from the commander's CURRENT hand targets, not open: entering
        # streamed mode mid-carry with zeroed hands would uncurl the fingers
        # and drop whatever they hold.
        self.hands = {"left": np.array(p.cmd.left_hand, dtype=np.float32),
                      "right": np.array(p.cmd.right_hand, dtype=np.float32)}
        self.q = None            # current reference, MuJoCo order
        self.root_quat = None
        self.next_t = None

    # -- plumbing ----------------------------------------------------------

    def start(self) -> None:
        s = self.p.obs()
        self.q = np.array(s["joints"])
        self.root_quat = yaw_only(s["pelvis"]["quat"])
        self.next_t = time.time()
        # Prime the buffer BEFORE the switch — the deploy stops control if
        # streamed mode goes active over an empty motion buffer.
        self.hold(0.3)
        self.p.cmd.enter_streamed()
        self.hold(1.0)

    def stop(self) -> None:
        """Hand control back to the planner without a discontinuity."""
        s = self.p.obs()
        palm_pf = self.p.world_to_root_full(s["palms"][self.side]["pos"], s)
        self.p.cmd.set_wrists(**{f"{self.side}_pos":
                                 np.asarray(palm_pf, dtype=np.float32)})
        self.p.cmd.set_hands(**{self.side: self.hands[self.side]})
        self.p.cmd.exit_streamed()
        # The switch re-anchors the deploy's planner heading at the robot's
        # current yaw (diag_heading.py); record it so frame() can undo the
        # rotation, or the next navigation walks confidently sideways.
        self.p.cmd.heading_offset = float(s["pelvis"]["yaw"])

    def _push(self, q_mj29) -> None:
        self.buf.append(mujoco_to_isaac(q_mj29).astype(np.float32))
        self.p.cmd.pose_chunk(np.stack(self.buf), self.root_quat,
                              self.i - len(self.buf) + 1,
                              left_hand=self.hands["left"],
                              right_hand=self.hands["right"])
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

    def move_to(self, q_target, speed: float = JOINT_SPEED) -> None:
        q0 = self.q.copy()
        q_target = np.asarray(q_target, dtype=float)
        dur = max(0.4, float(np.abs(q_target - q0).max()) / speed)
        n = max(1, int(round(dur / DT)))
        for k in range(1, n + 1):
            u = k / n
            sm = 3 * u * u - 2 * u ** 3
            self._push(q0 + sm * (q_target - q0))
        self.q = q_target

    # -- behaviours --------------------------------------------------------

    def squat(self, drop_m: float) -> float:
        """Bend the legs in the reference; returns the measured pelvis drop."""
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
        self.move_to(q_squat, speed=0.6)
        self.hold(2.0)
        s = self.p.obs()

        return float(z0 - s["pelvis"]["pos"][2])

    def goto(self, palm_world, rot_world=None, refine: int = REFINE_MAX,
             tol: float = REFINE_TOL, speed: float = JOINT_SPEED,
             obj_key: str | None = None, nudge_max: float = NUDGE_MAX):
        """Palm to a world pose; re-aim on the measured residual. -> err or None."""
        p = self.p
        obj0 = p.object_pos(obj_key) if obj_key else None
        want_w = np.asarray(palm_world, dtype=float)
        for it in range(refine + 1):
            s = p.obs()
            if s.get("furniture_touch"):
                p.log(f"    TABLE CONTACT during reach: "
                      f"{', '.join(s['furniture_touch'])}")
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
                rot_pf = (Primitives._rot(s["pelvis"]["quat"]).T
                          @ np.asarray(rot_world))
            q_sol, res, _ = self.ik.solve(
                self.side, aim_pf, rot_pf, q_mj29=self.q, use_waist=True)
            if res > 0.04:
                p.log(f"    stream goto: aim unreachable by {res * 100:.1f} cm")
                if it > 0:
                    return err
            self.move_to(q_sol, speed=speed if it == 0 else FINE_SPEED)
            self.hold(SETTLE if it == 0 else 1.2)
        s = p.obs()

        return float(np.linalg.norm(
            np.array(s["palms"][self.side]["pos"]) - want_w))

    def close_hand(self, obj_key: str, squeeze: float = 0.28,
                   ramp: float = 1.6) -> bool:
        closed = (HAND_CLOSED_RIGHT if self.side == "right"
                  else HAND_CLOSED_LEFT).astype(np.float32)
        self._closed = closed
        p = self.p
        contact_alpha = None
        n = int(round(ramp / DT))
        for k in range(n + 1):
            alpha = k / n
            if contact_alpha is not None and alpha >= contact_alpha + squeeze:
                break
            self.hands[self.side] = closed * alpha
            self._push(self.q)
            if k % 5 == 0 and contact_alpha is None:
                s = p.obs()
                if self.side in s["objects"][obj_key]["touching"]:
                    contact_alpha = alpha
        self.hold(0.8)
        self.last_alpha = alpha
        s = p.obs()

        return self.side in s["objects"][obj_key]["touching"]


def reanchor(p: Primitives, side: str = "right") -> None:
    """Force a planner heading re-anchor at a yaw this process knows.

    The deploy may carry a stale anchor from a previous process's streamed
    session (the anchor lives in the deploy, the compensation in the
    commander). A two-second streamed toggle re-anchors it at the robot's
    current yaw and records that yaw in cmd.heading_offset.
    """
    arm = StreamedArm(p, side)
    arm.start()
    arm.hold(0.3)
    arm.stop()


GRIP_X_MAX = 0.34         # pelvis-frame grip depth the lean reliably covers
GRIP_X_MIN = 0.24         # closer than this, the squatting knees hit the table


def _nudge_closer(p: Primitives, grip_world, side: str) -> None:
    """Small guarded forward steps until the grip is within the lean budget.

    The approach ring leaves the pelvis anywhere from 0.28 to 0.45 m from
    the can; beyond grip x ~ 0.36 the policy's delivered lean saturates and
    the align phase plateaus with the same undelivered 5 cm step forever.
    The potential field cannot help (its inflation ring is what stopped the
    walk), so shuffle straight at the can in open loop, one short step at a
    time, re-measuring after each.
    """
    from agent_server import LocomotionMode
    s = p.obs()
    x0 = p.world_to_root_full(grip_world, s)[0]
    if GRIP_X_MIN <= x0 <= GRIP_X_MAX:
        return
    p.log(f"  stream: grip {x0:.2f} m out, "
          f"{'walking in' if x0 > GRIP_X_MAX else 'backing off'}")
    # One continuous guarded walk, not step bursts — 0.9 s of SLOW_WALK never
    # even starts the gait, so six bursts moved the robot nowhere. Too far
    # saturates the lean; too close drives the knees into the table edge when
    # the squat comes (a 0.21 m stand launched the can 78 cm without a hand
    # near it), so both directions are corrected.
    end = time.time() + 8.0
    while time.time() < end:
        s = p.obs()
        grip_pf = p.world_to_root_full(grip_world, s)
        if GRIP_X_MIN <= grip_pf[0] <= GRIP_X_MAX or s["fallen"]:
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
        bearing = float(np.arctan2(d[1], d[0]))
        face = yaw + float(np.clip(wrap(bearing - yaw), -0.4, 0.4))
        fv = (np.cos(face), np.sin(face), 0.0)
        mv = fv if grip_pf[0] > GRIP_X_MAX else (-fv[0], -fv[1], 0.0)
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.35,
                    movement=mv, facing=fv)
        time.sleep(DT)
    stop = time.time() + 1.5
    while time.time() < stop:
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(s))
        time.sleep(DT)
    s = p.obs()
    p.log(f"  stream: grip now {p.world_to_root_full(grip_world, s)[0]:.2f} m out")


def place_streamed(p: Primitives, obj_key: str, drop_world,
                   side: str = "right") -> tuple[bool, str]:
    """Carry the palm over the drop point and open the hand. -> (ok, why)."""
    # Walk in past the inflation ring first: the approach parks ~0.68 m from
    # the plate, beyond even the lean envelope. Same guarded walk as the
    # grasp, aimed at XY distance instead of pelvis-frame depth.
    from agent_server import LocomotionMode
    end = time.time() + 6.0
    s = p.obs()
    while time.time() < end:
        s = p.obs()
        if (float(np.linalg.norm(np.asarray(drop_world[:2]) - p.pelvis_xy(s)))
                <= 0.42 or s["fallen"]):
            break
        yaw = p.pelvis_yaw(s)
        d = np.asarray(drop_world[:2]) - p.pelvis_xy(s)
        bearing = float(np.arctan2(d[1], d[0]))
        face = yaw + float(np.clip(wrap(bearing - yaw), -0.4, 0.4))
        fv = (np.cos(face), np.sin(face), 0.0)
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.35,
                    movement=fv, facing=fv)
        time.sleep(DT)
    stop = time.time() + 1.5
    while time.time() < stop:
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(s))
        time.sleep(DT)

    arm = StreamedArm(p, side)
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
            K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]],
                          [-k[1], k[0], 0]])
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
        obj_off_pf = rot_palm.T @ (p.object_pos(obj_key, s)
                                   - np.array(s["palms"][side]["pos"]))
        drop_w = np.asarray(drop_world, dtype=float)
        palm_target = drop_w - rot_hold @ obj_off_pf
        if arm.goto(palm_target + [0.0, 0.0, 0.12], rot_world=rot_hold,
                    refine=1, tol=0.05) is None:
            return False, p.aborted or "streamed staging over the plate failed"
        err = arm.goto(palm_target, rot_world=rot_hold, refine=1, tol=0.03,
                       speed=FINE_SPEED)
        if err is None:
            return False, p.aborted or "streamed reach over the plate failed"
        for _ in range(2):
            s = p.obs()
            can_err = drop_w - p.object_pos(obj_key, s)
            if float(np.linalg.norm(can_err[:2])) < 0.03:
                break
            palm_target = (np.array(s["palms"][side]["pos"])
                           + np.clip(can_err, -0.05, 0.05))
            if arm.goto(palm_target, rot_world=rot_hold, refine=0,
                        speed=FINE_SPEED) is None:
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
            K2 = np.array([[0, -k2[2], k2[1]], [k2[2], 0, -k2[0]],
                           [-k2[1], k2[0], 0]])
            a2 = np.arctan2(s2, c2)
            rot_hold = (np.eye(3) + np.sin(a2) * K2
                        + (1 - np.cos(a2)) * K2 @ K2) @ Primitives._rot(
                s["palms"][side]["quat"])
        can_err = drop_w - p.object_pos(obj_key, s)
        if float(np.linalg.norm(can_err[:2])) > 0.015:
            arm.goto(np.array(s["palms"][side]["pos"])
                     + np.clip(can_err, -0.05, 0.05),
                     rot_world=rot_hold, refine=0, speed=FINE_SPEED)
            arm.hold(0.8)
        s = p.obs()
        can_xy_err = float(np.linalg.norm(
            drop_w[:2] - p.object_pos(obj_key, s)[:2]))
        p.log(f"  stream: can over the plate, xy err {can_xy_err * 100:.1f} cm")
        if can_xy_err > 0.07:
            return False, (f"can never centred over the plate "
                           f"({can_xy_err * 100:.1f} cm)")
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
            arm.hands[side] = start * (1.0 - k / n)
            if k % 3 == 0:
                s = p.state.latest(5) or s
                palm_now = np.array(s["palms"][side]["pos"])
                cmd = hold_w + np.clip(1.0 * (hold_w - palm_now), -0.08, 0.08)
                q_sol, res, _ = arm.ik.solve(
                    side, p.world_to_root_full(cmd, s), None,
                    q_mj29=arm.q, use_waist=True, iters=8)
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
        arm.goto(np.array(s["palms"][side]["pos"]) + back * 0.12
                 + [0.0, 0.0, 0.03], refine=0, speed=0.2)
        arm.hold(0.5)
        s = p.obs()
        if side in s["objects"][obj_key]["touching"]:
            arm.goto(np.array(s["palms"][side]["pos"]) + back * 0.10,
                     refine=0, speed=FINE_SPEED)
        # Tuck BOTH arms fully while still in streamed mode. Handing a
        # half-extended arm back to the planner and letting tuck_arms finish
        # the job sent the wrist through the can (forensics: RETREAT-TUCK
        # moved the can 9.4 cm, touch R.wrist_yaw) — the planner channel
        # cannot be trusted anywhere near the placed object. Go up-and-back
        # first so the retraction path clears the counter.
        s = p.obs()
        arm.goto(np.array(s["palms"][side]["pos"]) + back * 0.10
                 + [0.0, 0.0, 0.18], refine=0, speed=0.5)
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
V_TRAVEL = 0.30           # m/s toward the staging point
V_DESCEND = 0.18          # m/s dropping to the pregrasp
V_APPROACH = 0.10         # m/s inside the jaw-alignment approach
V_LIFT = 0.22
LEAD_MAX = 0.08           # m the reference may run ahead of the real palm
KP_TRACK = 0.7            # extra command past the reference, fights the sag
CLOSE_RAMP_S = 2.0       # 1.0 s fingers punched the can off the table


def _clipn(v, m):
    n = float(np.linalg.norm(v))

    return v if n <= m else v * (m / n)


def _servo_continuous(p: Primitives, arm: StreamedArm, obj_key: str, plan,
                      side: str) -> tuple[bool, str]:
    """Travel, align, slide in, close, lift — one uninterrupted 50 Hz loop."""
    rot_t = np.asarray(plan.rot)
    entry = rot_t[:, 1]
    pre_w = np.asarray(plan.wrist) - entry * SAFE_GAP
    above_w = pre_w + np.array([0.0, 0.0, 0.15])
    offset = np.asarray(GRASP_CENTRE_OFFSET, dtype=float)
    closed = (HAND_CLOSED_RIGHT if side == "right"
              else HAND_CLOSED_LEFT).astype(np.float32)

    s = p.obs()
    obj0 = p.object_pos(obj_key, s)
    z0 = float(obj0[2])
    ref = np.array(s["palms"][side]["pos"], dtype=float)
    phase, ok_ticks, alpha, contact_alpha = "up", 0, 0.0, None
    full_f = None            # low-passed palm-frame error (the servo's eyes)
    obj_f = None             # low-passed can position
    close_tgt = None
    obj_in0 = None           # can position when the slide-in began
    backoff_until = 0.0
    touch_ticks = 0
    trim_w = np.zeros(3)     # slow world-frame trim: absorbs the gap between
                             # the PLANNED palm rotation the absolute target
                             # assumes and the rotation the policy delivers
    lift_target, settle_until = None, None
    deadline = time.time() + 60.0
    t_prev = time.time()
    last_log = 0.0

    while time.time() < deadline:
        s = p.state.latest(5) or s
        now = time.time()
        dt = float(np.clip(now - t_prev, 0.005, 0.06))
        t_prev = now
        if s["fallen"]:
            return False, "fell during the continuous servo"
        palm = np.array(s["palms"][side]["pos"])
        rot_p = Primitives._rot(s["palms"][side]["quat"])
        obj = p.object_pos(obj_key, s)
        if phase in ("up", "down", "in"):
            moved = float(np.linalg.norm(obj - obj0))
            # The in-phase tolerance is double: the servo target tracks the
            # LIVE can, so a thumb-graze that scoots it 5 cm is recoverable —
            # the hand simply re-centres on where it now is. Five episodes
            # died at exactly "5 cm in the in phase" for no lasting reason.
            limit = NUDGE_MAX * 3 if phase == "in" else NUDGE_MAX
            if moved > limit:
                hit = s["objects"][obj_key].get("touch_links") or ["nothing"]
                return False, (f"{obj_key} moved {moved * 100:.0f} cm in the "
                               f"{phase} phase (touched {', '.join(hit)})")

        if phase == "up":
            tgt, v = above_w, V_TRAVEL
            if float(np.linalg.norm(ref - above_w)) < 0.03:
                phase = "down"
        elif phase == "down":
            tgt, v = pre_w, V_DESCEND
            if (float(np.linalg.norm(ref - pre_w)) < 0.015
                    and float(np.linalg.norm(palm - pre_w)) < 0.045):
                phase = "in"
                p.log("  stream: at pregrasp, sliding in")
        elif phase == "in" and now < backoff_until:
            # Contact flinch: the thumb has started pushing the can, so
            # retreat a few centimetres along the entry axis, let it settle,
            # and re-approach the place it ACTUALLY is now. Without this the
            # servo chases the can it is pushing until the guard kills the
            # episode — seven of ten in batch13.
            tgt, v = palm - entry * 0.05, 0.15
            if now >= backoff_until - 0.02:
                obj_in0 = None
        elif phase == "in":
            if obj_in0 is None:
                obj_in0 = obj.copy()
                obj_f = obj.copy()
            if float(np.linalg.norm(obj - obj_in0)) > 0.015:
                backoff_until = now + 1.0
                p.log("  stream: contact flinch, backing off to re-centre")
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
            if can_still and float(np.linalg.norm(
                    (obj_f - rot_t @ offset + trim_w) - palm)) < 0.06:
                # Trim only while the can is stationary: integrating the
                # error of a can the hand is pushing turns the trim into a
                # positive-feedback bulldozer.
                trim_w = np.clip(trim_w + 0.5 * dt * (rot_p @ full),
                                 -0.04, 0.04)
            final_w = obj_f - rot_t @ offset + trim_w
            to_final = final_w - palm
            e_ent = float(np.dot(to_final, entry))
            lat = float(np.linalg.norm(to_final - e_ent * entry))
            # HARD align precondition, like the discrete servo's separate
            # align phase: zero forward motion until laterally within 12 mm.
            # The soft 18 mm gate let the hand creep in 17%-speed while still
            # 15 mm off line, and that is exactly the thumb-graze band —
            # four episodes per batch died re-flinching their way to the
            # 10 cm guard.
            gate = 0.0 if lat > 0.012 else float(
                np.clip(1.0 - lat / 0.012, 0.0, 1.0))
            # Hold back the un-gated part of the entry distance: laterally
            # free, forward only as far as alignment has earned.
            tgt = final_w - (1.0 - gate) * max(e_ent, 0.0) * entry
            v = V_APPROACH if e_ent > 0.05 else V_APPROACH * 0.5
            if lat < 0.012 and abs(full[1]) < 0.010:
                ok_ticks += 1
                if ok_ticks > 20:                     # ~0.4 s in tolerance
                    phase = "close"
                    p.log(f"  stream: pre-close object@palm "
                          f"{np.round(q_obj, 3)} want {np.round(offset, 3)}")
            else:
                ok_ticks = max(0, ok_ticks - 2)
        elif phase == "close":
            # Keep station actively while the fingers close — a frozen
            # reference lets the palm sag and the closing sweep tips the can.
            if close_tgt is None:
                close_tgt = obj_f - rot_t @ offset + trim_w
            tgt, v = close_tgt, 0.03
            alpha = min(1.0, alpha + dt / CLOSE_RAMP_S)
            arm.hands[side] = closed * alpha
            # Latch contact only when SUSTAINED: a single-tick graze latched
            # contact_alpha early, capped the squeeze low, and the under-
            # gripped can dropped out during the lift.
            touch_ticks = (touch_ticks + 1
                           if side in s["objects"][obj_key]["touching"] else 0)
            if contact_alpha is None and touch_ticks >= 3:
                contact_alpha = alpha
            # Past contact, squeeze another half — the travel grip that
            # survives 2.4 m of gait bounce (the tighten-for-the-road step,
            # folded into the same ramp).
            if contact_alpha is not None and alpha >= min(
                    1.0, contact_alpha + 0.6):
                phase = "lift"
                lift_target = palm + np.array([0.0, 0.0, LIFT_HEIGHT])
            elif alpha >= 1.0 and contact_alpha is None:
                return False, "no contact after closing"
        else:                                          # lift
            tgt, v = lift_target, V_LIFT
            if float(np.linalg.norm(ref - lift_target)) < 0.02:
                settle_until = settle_until or (now + 0.8)
                if now >= settle_until:
                    lifted = float(p.object_pos(obj_key, s)[2] - z0)
                    if lifted < LIFT_MIN:
                        return False, (f"{obj_key} rose only {lifted:.3f} m, "
                                       f"need > {LIFT_MIN}")
                    p.log(f"  stream: lifted {lifted * 100:.1f} cm")
                    return True, ""

        ref = ref + _clipn(np.asarray(tgt, dtype=float) - ref, v * dt)
        lead = ref - palm
        if float(np.linalg.norm(lead)) > LEAD_MAX:
            # The policy is behind; wait for it instead of running away.
            ref = palm + _clipn(lead, LEAD_MAX)
            lead = ref - palm
        cmd = ref + np.clip(KP_TRACK * lead, -0.05, 0.05)
        aim_pf = p.world_to_root_full(cmd, s)
        rot_pf = Primitives._rot(s["pelvis"]["quat"]).T @ rot_t
        q_sol, res, _ = arm.ik.solve(side, aim_pf, rot_pf, q_mj29=arm.q,
                                     use_waist=True, iters=10)
        if res < 0.10:
            arm.q = q_sol
        arm._push(arm.q)

        if now - last_log > 2.0:
            extra = ""
            if phase == "in" and full_f is not None:
                extra = (f"  lat {float(np.linalg.norm(full_f[[0, 2]])) * 1000:.0f}mm"
                         f" depth {full_f[1] * 100:.1f}cm")
            p.log(f"    stream cont[{phase}] palm->ref "
                  f"{float(np.linalg.norm(lead)) * 100:.1f} cm{extra}")
            last_log = now
    return False, f"continuous servo timed out in the {phase} phase"


def _raise_palm(p: Primitives, arm: StreamedArm, target_w, side: str,
                timeout: float = 2.5) -> None:
    """Servo the palm to target_w before anything else moves the torso.

    Runs ahead of the squat: the squat drops the whole reference ~13 cm with
    the hand still at its rest pose beside the table edge, which can dip the
    hand below the tabletop and drag it up along the rim on the next travel
    phase.  Raising first keeps the whole approach above the table.
    """
    s = p.obs()
    rot_t = Primitives._rot(s["palms"][side]["quat"])
    ref = np.array(s["palms"][side]["pos"], dtype=float)
    tgt = np.asarray(target_w, dtype=float)
    t_prev = time.time()
    deadline = t_prev + timeout
    while time.time() < deadline:
        now = time.time()
        dt = min(max(now - t_prev, 1e-3), 0.1)
        t_prev = now
        s = p.obs()
        palm = np.array(s["palms"][side]["pos"], dtype=float)
        if float(np.linalg.norm(palm - tgt)) < 0.05:
            return
        ref = ref + _clipn(tgt - ref, V_TRAVEL * dt)
        lead = ref - palm
        if float(np.linalg.norm(lead)) > LEAD_MAX:
            ref = palm + _clipn(lead, LEAD_MAX)
            lead = ref - palm
        cmd = ref + np.clip(KP_TRACK * lead, -0.05, 0.05)
        aim_pf = p.world_to_root_full(cmd, s)
        rot_pf = Primitives._rot(s["pelvis"]["quat"]).T @ rot_t
        q_sol, res, _ = arm.ik.solve(side, aim_pf, rot_pf, q_mj29=arm.q,
                                     use_waist=True, iters=10)
        if res < 0.10:
            arm.q = q_sol
        arm._push(arm.q)


def grasp_streamed(p: Primitives, obj_key: str, grip_world, plan,
                   side: str = "right") -> tuple[bool, str]:
    """The whole grasp: pregrasp, align, advance, close, lift. -> (ok, why)."""
    _nudge_closer(p, grip_world, side)
    arm = StreamedArm(p, side)
    arm.start()
    try:
        s = p.obs()
        z0 = float(p.object_pos(obj_key, s)[2])
        grip_pf = p.world_to_root_full(grip_world, s)
        if grip_pf[2] < SQUAT_TRIGGER:
            # Hand up first, THEN squat — see _raise_palm.
            safe = np.array(s["palms"][side]["pos"], dtype=float)
            safe[2] = max(safe[2], float(grip_world[2]) + 0.18)
            _raise_palm(p, arm, safe, side)
            got = arm.squat(SQUAT_TRIGGER - grip_pf[2])
            p.log(f"  stream: raised palm, then squat dropped pelvis "
                  f"{got * 100:.1f} cm")

        # One continuous 50 Hz servo does the rest: up over the pregrasp,
        # down, jaw-gated slide onto the can, close, lift — no stop-and-
        # settle anywhere. The route (above -> pregrasp -> in along the
        # palm's +y) is the same one the discrete servo proved out; only the
        # cadence changed.
        return _servo_continuous(p, arm, obj_key, plan, side)
    finally:
        arm.stop()
