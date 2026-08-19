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

# Fraction of servo ticks with the object inside the ego camera's central 90%
# (set by _servo_continuous; None until a servo has run). The episode gate
# reads it: RGB frames that never show the task are worthless to a VLA.
LAST_EGO_VIS = None
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
        # The waist may HELP the reach but only within a human lean. With the
        # stock limits the IK bridged any height/side gap by folding the
        # torso flat onto the table (head down, palm never reaching its
        # lifted reference); with the waist excluded entirely, side-offset
        # pregrasps went unreachable (res 0.13). Indices 12/13/14 = waist
        # yaw/roll/pitch in MuJoCo order.
        for j, (lo, hi) in {12: (-0.35, 0.35), 13: (-0.15, 0.15),
                            14: (-0.05, 0.20)}.items():
            self.ik.lo[j], self.ik.hi[j] = lo, hi
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
        # Release the planner's upper-body pin before streaming: the deploy
        # OVERWRITES the upper-body joints of any reference motion while
        # `has_upper_body_control_` is latched (g1_deploy_onnx_ref.cpp:780),
        # and the latch updates per planner message — so one frame WITHOUT
        # the field clears it. Skipping this froze the arm mid-grasp with
        # IK res 0.000 and a motionless palm.
        self.p.cmd.upper_body = None
        end = time.time() + 0.4
        while time.time() < end:      # one frame raced the deploy's reader;
            self.p.cmd.frame(facing=self.p._facing(s))
            time.sleep(DT)
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
        from primitives import upper_body_from_mj
        s = self.p.obs()
        palm_pf = self.p.world_to_root_full(s["palms"][self.side]["pos"], s)
        self.p.cmd.set_wrists(**{f"{self.side}_pos":
                                 np.asarray(palm_pf, dtype=np.float32)})
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


GRIP_X_MAX = 0.30         # the reach map's outer band: [0.22, 0.30] is the
                          # only overlap of the graspable region with stands
                          # whose hip does not press the table edge
GRIP_X_MIN = 0.22         # closer, and the hip leans on the table edge
                          # and shoves it (FURN[R.hip_roll] forensics)

RAISE_CLEAR = 0.10        # palm above the tabletop while walking the last bit


def _raise_hand(p: Primitives, side: str, surface_top: float | None,
                grip_world=None) -> None:
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
        return                       # ready height already clears everything
    ready[2] = z_pf
    p.cmd.set_wrists(**{f"{side}_pos": ready})
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.log(f"  stream: ready hand trimmed up to clear the surface "
          f"(z_pf {z_pf:+.2f})")


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
    for _pass in range(3):
        s = p.obs()
        x0 = p.world_to_root_full(grip_world, s)[0]
        if GRIP_X_MIN <= x0 <= GRIP_X_MAX:
            return
        p.log(f"  stream: grip {x0:.2f} m out, "
              f"{'walking in' if x0 > GRIP_X_MAX else 'backing off'}")
        _nudge_pass(p, grip_world, side)
    s = p.obs()
    p.log(f"  stream: grip still {p.world_to_root_full(grip_world, s)[0]:.2f} "
          f"m out after 3 nudge passes")


def _nudge_pass(p: Primitives, grip_world, side: str) -> None:
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
        bearing_can = float(np.arctan2(d[1], d[0]))
        if p.hold_yaw is not None and abs(wrap(float(p.hold_yaw)
                                               - bearing_can)) < 0.8:
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
        backing = grip_pf[0] <= GRIP_X_MAX
        mv = (-fv[0], -fv[1], 0.0) if backing else fv
        # Backing off slower than walking in: the stop takes ~0.2 m of gait
        # momentum either way, but overshooting outward puts the grip beyond
        # the palm envelope while overshooting inward just costs a re-pass.
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK,
                    speed=0.22 if backing else 0.35,
                    movement=mv, facing=fv)
        time.sleep(DT)
    stop = time.time() + 1.5
    while time.time() < stop:
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(s))
        time.sleep(DT)
    s = p.obs()
    p.log(f"  stream: grip now {p.world_to_root_full(grip_world, s)[0]:.2f} m out")


def toppled(s, obj_key: str) -> bool:
    """Has the (upright-cylinder) object fallen over? Its body z-axis more
    than ~60 deg off vertical. Failure ATTRIBUTION: a toppled can later
    surfaces as 'grip height unreachable' (the grasp line follows the body
    origin down to lying height), which blames the arm for a knock that
    happened during the approach."""
    rot = Primitives._rot(s["objects"][obj_key]["quat"])
    return float(rot[2, 2]) < 0.5


def final_approach(p: Primitives, obj_key: str, rm, edge_dist_fn,
                   z_pf: float, side_pref: str = "right",
                   timeout: float = 25.0):
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
    X_STOP = 0.26             # object depth at which grasping is possible
    EDGE_STOP = 0.16          # hip-to-edge, incl. ~3 cm of reach-lean sway
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
            return False, side_pref, (
                f"object knocked {moved * 100:.0f} cm away during the final "
                f"approach")
        yaw = p.pelvis_yaw(s)
        d = can[:2] - p.pelvis_xy(s)
        c, sn = np.cos(yaw), np.sin(yaw)
        x_pf = c * d[0] + sn * d[1]
        if moved > 0.02 and backoffs < 3:
            # Reactive guard: something (us, most likely) disturbed it.
            # Back straight off, let it settle, re-anchor, resume.
            backoffs += 1
            p.log(f"  approach2: object stirred {moved * 100:.1f} cm, "
                  f"backing off ({backoffs}/3)")
            stop_t = time.time() + 0.8
            fv = (c, sn, 0.0)
            while time.time() < stop_t:
                p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.22,
                            movement=(-fv[0], -fv[1], 0.0), facing=fv)
                time.sleep(DT)
            p.cmd.hold(0.8, mode=LocomotionMode.IDLE, facing=fv)
            can0 = p.object_pos(obj_key)
            continue
        if x_pf <= X_STOP or edge_dist_fn(p.pelvis_xy(s)) <= EDGE_STOP:
            break
        bearing = float(np.arctan2(d[1], d[0]))
        face = yaw + float(np.clip(wrap(bearing - yaw), -0.6, 0.6))
        fv = (np.cos(face), np.sin(face), 0.0)
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.25,
                    movement=fv, facing=fv)
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
        p.log(f"  approach2: depth {x_now:.2f}, pulsing "
              f"{'in' if sign > 0 else 'out'} {dur:.2f} s")
        c2, s2 = np.cos(yaw), np.sin(yaw)
        end_t = time.time() + dur
        while time.time() < end_t:
            p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.22,
                        movement=(sign * c2, sign * s2, 0.0),
                        facing=(c2, s2, 0.0))
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
        for sgn in ((0,) if delta_deg == 0 else (1, -1)):
            delta = np.deg2rad(delta_deg) * (sgn or 1)
            rel = np.array([dist * np.cos(delta), -dist * np.sin(delta),
                            z_pf])
            if abs(rel[1]) < 0.12:
                # Kinematically the map allows a centred object; the SERVO
                # does not — the policy under-delivers in the chest zone
                # (every centred attempt stalls or shoves the can). "As
                # centred as possible" bottoms out at |y| = 0.12.
                continue
            for side in sides:
                if rm.graspable(rel, side):
                    psi = bearing + delta
                    p.turn_to(psi)
                    p.hold_yaw = p.pelvis_yaw()
                    p.log(f"  approach2: stopped at depth {dist:.2f} m, "
                          f"turned {np.degrees(delta):+.0f} deg -> {side} "
                          f"arm, object at rel {np.round(rel, 2)}")
                    return True, side, ""
    return False, side_pref, (
        f"object at {dist:.2f} m is outside the reach map at every facing "
        f"the body can take without touching the table")


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
# Pre-narrow the jaw while sliding in. Calibrated kinematically: fingertip
# gap ~ 6.1 cm at alpha 0, closing ~0.6 cm per 0.1 alpha, so 0.10 leaves
# ~5.6 cm over the 4.8 cm can — 8 mm of total margin instead of 13, but the
# first contact now lands on the closing envelope (which wraps) instead of a
# flat finger (which shoves): the graze-and-roll that killed 8 of 15
# episodes happens at the OPEN aperture's finger tips.
PRE_NARROW = 0.10
SAG_COMP = 0.018          # free-air sag is ~1.6-2 cm; 0.035 was measured
                          # against a WEDGED hand (the table held it down) and
                          # overcompensated: signed telemetry showed the palm
                          # riding 15-19 mm HIGH, closing the jaw at the rim
V_LIFT = 0.22
LEAD_MAX = 0.08           # m the reference may run ahead of the real palm
KP_TRACK = 0.7            # extra command past the reference, fights the sag
CLOSE_RAMP_S = 2.0       # 1.0 s fingers punched the can off the table


def _clipn(v, m):
    n = float(np.linalg.norm(v))

    return v if n <= m else v * (m / n)


FREEZE_TRAVEL = 0.10      # residual travel below which the plan is frozen
# Deliverable staging box, pelvis frame. The scene-model IK reaches further
# than the policy delivers (res 0.000 while the palm stalls): a pregrasp
# 0.5 m to the side solves fine and then never arrives — and so does one at
# the chest (x 0.10, z +0.08), where retracting the palm is equally
# undelivered. The box is the region the planner-era reach demonstrably
# worked in; candidates whose staging point leaves it are rejected before
# the arm ever chases them.
PRE_X_MIN, PRE_X_MAX = 0.17, 0.38
PRE_Z_MIN, PRE_Z_MAX = -0.24, 0.04
PRE_XY_MAX = 0.46         # horizontal reach from the pelvis


def _servo_continuous(p: Primitives, arm: StreamedArm, obj_key: str, plan,
                      side: str, replan=None, ghost=None,
                      pin=None, surface_top=None) -> tuple[bool, str]:
    """Travel, align, slide in, close, lift — one uninterrupted 50 Hz loop.

    `replan(wrist_now, palm_rot) -> Grasp` re-derives the candidate pose from
    the REAL wrist, so the approach is always the current shortest line; it is
    called at ~5 Hz during the travel phases and frozen once the residual
    travel drops under FREEZE_TRAVEL (or the slide-in begins) — past that
    point a moving target only feeds the push-away chatter. `ghost` mirrors
    every accepted replan into the video's translucent hand.
    """
    rot_t = np.asarray(plan.rot)
    entry = rot_t[:, 1]
    pre_w = np.asarray(plan.wrist) - entry * SAFE_GAP
    above_w = pre_w + np.array([0.0, 0.0, 0.10])
    frozen = replan is None
    last_replan = 0.0
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
    deadline = time.time() + 60.0   # wedge class fixed; a 90 s hover is a
                                     # real failure, not a slow success
    t_prev = time.time()
    last_log = 0.0
    res = 0.0
    in_since = None
    ent_mark = None
    stall_palm, stall_t = None, time.time()
    vis_n = tot_n = 0        # ego-camera visibility, for the VLA data gate
    in_lift = 0.0            # slide-in extra height, set in the in phase
    dbg_z = ""               # z-chain telemetry (line/cmd/palm/floor)
    touch_seen: set = set()  # object-contact links since the last status line

    while time.time() < deadline:
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
        obj = p.object_pos(obj_key, s)

        tl0 = s["objects"][obj_key].get("touch_links")
        if tl0:
            touch_seen.update(tl0)
        if "cam" in s["objects"][obj_key]:
            global LAST_EGO_VIS
            cam = s["objects"][obj_key]["cam"]
            tot_n += 1
            if cam is not None and abs(cam[0]) < 0.9 and abs(cam[1]) < 0.9:
                vis_n += 1
            LAST_EGO_VIS = vis_n / tot_n

        if not frozen and phase in ("up", "down") and now - last_replan > 0.2:
            last_replan = now
            npl = replan(palm, rot_p)
            if npl.ok and npl.travel > FREEZE_TRAVEL:
                cand_rot = np.asarray(npl.rot)
                cand_pre = np.asarray(npl.wrist) - cand_rot[:, 1] * SAFE_GAP
                pre_pf = p.world_to_root_full(cand_pre, s)
                if (float(np.hypot(pre_pf[0], pre_pf[1])) <= PRE_XY_MAX
                        and PRE_X_MIN <= pre_pf[0] <= PRE_X_MAX
                        and PRE_Z_MIN <= pre_pf[2] <= PRE_Z_MAX):
                    rot_t = cand_rot
                    entry = rot_t[:, 1]
                    pre_w = cand_pre
                    above_w = pre_w + np.array([0.0, 0.0, 0.10])
                    if ghost is not None:
                        ghost(npl)
            elif npl.ok:
                frozen = True           # committed: inside the freeze radius
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
                return False, (f"{obj_key} moved {moved * 100:.0f} cm in the "
                               f"{phase} phase (touched {', '.join(hit)}"
                               f"{'; furniture: ' + ', '.join(ft) if ft else ''})")

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
                arm.hands[side] = closed * alpha
            if float(np.linalg.norm(obj - obj_in0)) > 0.015:
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
                near = float(np.linalg.norm(
                    (obj_f - rot_t @ offset + trim_w) - palm)) < 0.06
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
                zf2 = (surface_top + 0.012 - low_c2
                       - float((rot_t @ _ag.OFFSET)[2]))
                need = zf2 + 0.003 - float(base[2])
                if need > 0.02:
                    return False, (
                        "grip height unreachable over the table: the hand's "
                        f"low point forces the palm {need * 100:.1f} cm "
                        f"above the grasp line (obj z {float(obj_f[2]):.3f}, "
                        f"low_c {low_c2:.3f})")
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
                return False, ("grip target ended up at the body/behind it "
                               "— the stand is wrong, not the arm")
            # 18 mm, was 12: the delivery floor wobbles at 11-16 mm, and a
            # 12 mm hard gate kept zeroing the advance — hands hovered a
            # full 14 cm out until the deadline. The pre-narrowed jaw wraps
            # from 18 mm of lateral error.
            # The gate keys on the HORIZONTAL lateral error only: the
            # vertical offset during the slide-in is the deliberate lift
            # (managed by SAG_COMP / the z-floor / the ramp below), and the
            # thumb-graze the gate exists to prevent is a horizontal miss.
            gate = 0.0 if lat_h > 0.018 else float(
                np.clip(1.0 - lat_h / 0.018, 0.0, 1.0))
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
            in_lift = 0.03 * float(np.clip((e_ent - 0.08) / 0.12, 0.0, 1.0))
            v = V_APPROACH if e_ent > 0.05 else V_APPROACH * 0.5
            # Close on the palm-frame TRUTH (`full` = measured object minus
            # the wanted offset), not on distance to the planned line: with
            # the pinned arm the hover sits 15-21 mm off the LINE while the
            # can is already inside the jaw envelope, and every extra hover
            # second is another fingertip graze inching the can away (the
            # 15 cm-cumulative aborts). If the can is within the envelope,
            # closing funnels it to the palm — that is what fingers curl FOR.
            g_lat = float(np.hypot(full[0], full[2]))
            good = g_lat < 0.015 and abs(full[1]) < 0.012
            patient = (now - in_since > 8.0
                       and g_lat < 0.020 and abs(full[1]) < 0.018)
            if good or patient:
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
        if phase in ("down", "in", "close"):
            # The policy arm SAGS ~2 cm below its reference; geometry that
            # clears the table on paper drags the middle fingertip across it
            # in the flesh (FURN[R:middle_1] wedged, lat/ent frozen for a
            # minute). Bias the commanded height so the DELIVERED height is
            # the planned one.
            cmd = cmd + np.array([0.0, 0.0, SAG_COMP
                                  + (in_lift if phase == "in" else 0.0)])
        if surface_top is not None:
            # Hard geometric floor: with the pinned arm executing IK heights
            # EXACTLY, any command whose HAND (not wrist — the fingertips
            # hang below) would dip under tabletop+1.2 cm turns into a
            # scrape that bulldozes the can. Computed from the measured
            # hand point cloud at the CURRENT planned rotation.
            import adaptive_grasp as _ag
            low_c = float((_ag.hand_points() @ rot_t.T)[:, 2].min())
            z_floor = (surface_top + 0.012 - low_c
                       - float((rot_t @ _ag.OFFSET)[2]))
            if cmd[2] < z_floor:
                cmd = cmd.copy()
                cmd[2] = z_floor
            if phase == "in" and obj_f is not None:
                line_z = float((obj_f - rot_t @ offset)[2])
                dbg_z = (f"  z(pal{palm[2] - line_z:+.3f}"
                         f" cmd{float(cmd[2]) - line_z:+.3f}"
                         f" fl{z_floor - line_z:+.3f})")
        aim_pf = p.world_to_root_full(cmd, s)
        rot_pf = Primitives._rot(s["pelvis"]["quat"]).T @ rot_t
        q_sol, res, _ = arm.ik.solve(side, aim_pf, rot_pf, q_mj29=arm.q,
                                     use_waist=True, iters=10)
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
            return False, (f"arm not tracking in the {phase} phase "
                           f"(palm stalled 8 s, IK res {res:.3f}, "
                           f"target {float(np.linalg.norm(tgt - palm)):.2f} m "
                           f"away)")

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
                extra += (f"  lat {full_f[0] * 1000:.0f}mm"
                          f"(h{full_f[2] * 1000:.0f}"
                          f" v{full_f[3] * 1000:+.0f})"
                          f" ent {full_f[1] * 100:.1f}cm"
                          + (dbg_z if dbg_z else ""))
            tpf = p.world_to_root_full(np.asarray(tgt, dtype=float), s)
            p.log(f"    stream cont[{phase}] palm->ref "
                  f"{float(np.linalg.norm(lead)) * 100:.1f} cm  res {res:.3f}"
                  f"  tgt_pf [{tpf[0]:+.2f} {tpf[1]:+.2f} {tpf[2]:+.2f}]{extra}")
            last_log = now
    return False, f"continuous servo timed out in the {phase} phase"


def grasp_streamed(p: Primitives, obj_key: str, grip_world, plan,
                   side: str = "right", surface_top: float | None = None,
                   replan=None, ghost=None,
                   nudge: bool = True, pin=None,
                   squat_hint: float | None = None) -> tuple[bool, str]:
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
    if nudge:
        _nudge_closer(p, grip_world, side)
    arm = StreamedArm(p, side)
    arm.start()
    try:
        # Tuck the IDLE arm before crouching. The crouch lowers the whole
        # body, and the idle arm — frozen at the level-forearm READY joint
        # angles — sinks 13 cm with it, straight into the can's height band
        # and onto the tabletop (up-phase topples, FURN[wrist_yaw] on the
        # non-servo side). The SERVO arm is safe: its IK closes on world
        # targets each tick and re-raises itself. Elbow NEGATIVE folds the
        # forearm UP (elbow zero = level).
        other = "left" if side == "right" else "right"
        q_t = arm.q.copy()
        roll = 0.35 if other == "left" else -0.35
        q_t[ARM_MJ[other]] = [-0.4, roll, 0.0, -0.6, 0.0, 0.0, 0.0]
        # ...and pre-lift the SERVO arm the same way: during the crouch its
        # joints are frozen too (the world-frame IK loop only starts with
        # the servo), so a level READY forearm sinks with the body straight
        # through the can's height band — that was the topple right at
        # servo start, before any status line.
        roll_s = 0.25 if side == "left" else -0.25
        q_t[ARM_MJ[side]] = [-0.5, roll_s, 0.0, -0.5, 0.0, 0.0, 0.0]
        arm.move_to(q_t, speed=1.0)
        # PLANNED crouch before reaching: bring the body down until the
        # wrist's natural height is near the grip, so the (limit-clamped)
        # waist only fine-tunes instead of folding to bridge the gap. The
        # wrist rides ~7.5 cm above the grip point, so the drop is the
        # palm's current height minus (grip + that margin).
        s0 = p.obs()
        palm_z = float(s0["palms"][side]["pos"][2])
        drop = float(np.clip(palm_z - (float(grip_world[2]) + 0.06),
                             0.0, 0.13))
        if squat_hint is not None:
            # Navigation planned this crouch so the sweet core's THICK band
            # lands on the object — never crouch less than the plan.
            drop = float(np.clip(max(drop, squat_hint), 0.0, 0.13))
        if drop > 0.03:
            got = arm.squat(drop)
            p.log(f"  stream: planned crouch {drop * 100:.0f} cm "
                  f"wanted, {got * 100:.0f} cm delivered")
        # One continuous 50 Hz servo does everything: down the live
        # replanned line to the pregrasp, jaw-gated slide onto the can,
        # close, lift — no stop-and-settle anywhere.
        ok, why = _servo_continuous(p, arm, obj_key, plan, side,
                                    replan=replan, ghost=ghost, pin=pin,
                                    surface_top=surface_top)
        if not ok and ("timed out" in why or "not tracking" in why):
            # Diagnose before medicating: "cannot reach" is usually the grip
            # sitting outside the depth band (a nudge overshoot), and walking
            # back fixes that; the squat only helps when the grip is LOW, and
            # squatting at the wrong depth just adds a knee into the table.
            s = p.obs()
            grip_pf = p.world_to_root_full(grip_world, s)
            if not (GRIP_X_MIN <= grip_pf[0] <= GRIP_X_MAX):
                arm.stop()
                _nudge_closer(p, grip_world, side)
                arm = StreamedArm(p, side)
                arm.start()
                s = p.obs()
                grip_pf = p.world_to_root_full(grip_world, s)
            # No squat here. It was tried as the can't-reach fallback and made
            # things strictly worse: a 13 cm drop leaves the palm targets
            # ABOVE the pelvis, which is the policy's worst delivery zone,
            # and the episode still stalled 0.27 m short. The raised hand
            # plus the tilted approach covers this table; a genuinely lower
            # surface needs a planned crouch, not a panic one.
            p.log("  stream: retrying the servo after re-nudging")
            ok, why = _servo_continuous(p, arm, obj_key, plan, side,
                                        replan=replan, ghost=ghost, pin=pin,
                                        surface_top=surface_top)
        return ok, why
    finally:
        if pin is not None:
            pin()          # release the arm motors back to the policy
        arm.stop()
