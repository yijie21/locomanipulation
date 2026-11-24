"""Seed: the 65-object line's closed-loop hand program on the motor layer (TODO 27.2.2, plan D4; user 2026-10-16: port
the closed loop, grasp from the object's box). Ported from scripts/experiments/wbc_compare_20261012/bfm_grasp.py
`Round.run` step 4 -- settle at the pre-grasp, entry with the gate and the lateral correction, retreat and retry,
close with the contact timeout, lift, hold -- with its thresholds (the simple expert's `reach.simple` block and
grasp_planned's constants), its jaw law and its translation feedback (SupportedMotion's, `Tracking`), line for line.
What differs: the wrist keypoint goes through `Motor.hand` (the motor's own servo is off for that hand), the object
pose is the scene's privileged one, the hand is either side, and the grasp comes from `grasp_box` instead of the
cache. The open-loop approach before it is `Motor.reach` to the pre-grasp (whole-body IK + servo)."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import SIDES, Outcome, expmap

# the simple expert's thresholds (configs/experiments/bottle_simple_200.yaml reach.simple, as bfm_grasp copies them)
SIMPLE = dict(
    settle_tolerance_m=0.02,
    settle_timeout_s=1.5,
    lateral_correction_speed_m_s=0.03,
    lateral_correction_max_m=0.04,
    entry_overshoot_m=-0.01,  # 0.02 (bfm_grasp) until the pick_single loop's set, as the three below
    entry_capture_depth_m=0.025,
    entry_timeout_s=9.0,
    entry_attempts=2,
    retreat_m=0.08,
    retreat_seconds=1.0,
    close_seconds=1.0,
    contact_timeout_s=0.8,
    lift_seconds=2.0,
    disturb_limit_m=0.05,
    settle_abort_m=None,  # a settle ending farther than this from the pre-grasp retreats and fails (None: never)
    lateral_abort_m=None,  # an entry whose grasp point sits more than this across the jaw (wrist y) fails (None: never)
)
TRACKING = dict(deadband=0.002, speed=0.04, acceleration=0.2, clip=0.08)  # SupportedMotion's translation feedback
# grasp_planned.ENTRY_GATE (lateral 0.015 m, hold 2.0 s until the pick_single loop's set)
ENTRY_GATE = {"on": True, "lateral_m": 0.006, "hold_s": 3.0}
DASH_SPEED = 0.06  # grasp_planned
PREGRASP_BACK = 0.14  # grasp_planned 0.10 until the pick_single loop's set
PAD_Y, PAD_Z = 0.0475, 0.014  # grasp_planned: open-pad half span across, pad half height
LIFT_M = 0.25  # (0.10 until gso_single_grasp's set) grasp_planned
HOLD_S = 4  # (2.0 until gso_single_grasp's set) bfm_grasp (decision 3 of the controller comparison)
RISE_MIN = 0.05
JAW_OPEN, JAW_CLOSED = 0.0245, -0.02  # bfm_object_scene
CONTINUOUS_CLOSE = 1  # 1: a close whose jaw is more than CONTINUOUS_MIN_M from the entry's last target keeps that
CONTINUOUS_MIN_M = 0.01  # target and lifts from it (0: bfm_grasp's reset of the target to the measured jaw at the
# close and again at the lift; on the 1.12 m shelf the hand lags the entry, and the two resets moved the commanded
# wrist +2.2 and +3.3 cm up: the top-shelf candle lost its pads in the lift). A jaw at its target keeps the reset.


def minjerk(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * u * (10 - 15 * u + 6 * u * u)


def _clip_norm(v, limit):
    n = float(np.linalg.norm(v))
    return v * (limit / n) if n > limit else v


class Tracking:
    """SupportedMotion's translation feedback (primitives/supported_motion.py), one hand (bfm_grasp.Tracking)."""

    def __init__(self):
        self.origin = None
        self.feedback = np.zeros(3)
        self.rate = np.zeros(3)

    def step(self, reference, measured, dt):
        error = reference - measured
        if self.origin is None:
            self.origin = error.copy()
            self.feedback = error.copy()
            return self.feedback
        error = self.origin + np.clip(error - self.origin, -TRACKING["clip"], TRACKING["clip"])
        resid = error - self.origin
        n = float(np.linalg.norm(resid))
        dead = resid * (n - TRACKING["deadband"]) / n if n > TRACKING["deadband"] else np.zeros(3)
        target = self.origin + dead
        w = TRACKING["acceleration"] / TRACKING["speed"]
        acc = w * w * (target - self.feedback) - 2 * w * self.rate
        self.rate += _clip_norm(acc, TRACKING["acceleration"]) * dt
        self.rate = _clip_norm(self.rate, TRACKING["speed"])
        self.feedback = self.feedback + self.rate * dt
        return self.feedback


def entry_captured(tool, grip, direction, capture_depth_m):
    return float((grip - tool) @ direction) <= capture_depth_m


def entry_command(command, grip, direction, speed, dt, overshoot_m):
    limit = float((grip - command) @ direction) + overshoot_m
    return command + min(speed * dt, max(0.0, limit)) * direction


def object_in_palm(palm_R, tool, obj_pos):
    return palm_R.T @ (obj_pos - tool)


def run(
    ctx,
    key: str,
    side: str,
    R_c,
    tool_c,
    width: float,
    entry_frame=None,
    simple: dict | None = None,
    gate: dict | None = None,
    jaw: tuple | None = None,
    rise_min: float | None = None,
) -> Outcome:
    """The closed loop from the pre-grasp: `R_c` = the grasp's wrist rotation (x = approach), `tool_c` = its jaw
    centre (world, at the object's pose now), `width` = the object's span across the jaw. ctx.note logs the phases.
    `entry_frame` (a rotation in the wrist frame, None: the wrist's own): the entry runs along R_c @ entry_frame's x and
    the gate, the lateral correction and the alignment read the offsets in that planned frame, not the measured wrist's
    (the pinch's level entry with a pitched wrist, pick.pick_pinch: a wrist pitched 40 deg instead of 20 leaked the
    12 cm still to go into a 4-10 cm "vertical" offset, and the correction drove the jaw the wrong way).
    `simple` and `gate` (entries over SIMPLE and ENTRY_GATE), `jaw` ((open, closed) for JAW_OPEN, JAW_CLOSED) and
    `rise_min` (for RISE_MIN): one grasp's own values (pick.NARROW, the covered book), the module's where None."""
    simple, gate = {**SIMPLE, **(simple or {})}, {**ENTRY_GATE, **(gate or {})}
    jaw_open, jaw_closed = (JAW_OPEN, JAW_CLOSED) if jaw is None else jaw
    rise_min = RISE_MIN if rise_min is None else rise_min
    world, m = ctx.world, ctx.motor
    sim, body = m.sim, world.free[key]
    i = SIDES.index(side)
    dt = R.DT
    entry_R = None if entry_frame is None else R_c @ entry_frame
    direction = R_c[:, 0] if entry_R is None else entry_R[:, 0]
    obj0 = sim.d.xpos[body].copy()
    grip_offset = tool_c - obj0
    m.servo_on[side] = False
    rec: dict = dict(phases=[])
    tracking = Tracking()
    phase, began, t0 = "settle", m.t, m.t
    rec["phases"].append(dict(phase=phase, t=round(began, 2)))
    target = m.jaw_centre(side)[0].copy()
    # bfm_grasp: the feedback's first error is the approach's last command minus the measured tool (the delivery error)
    # and becomes its origin. Here the approach was the motor's servoed reach, so its last command carries the servo's
    # lead; starting from the pre-grasp target instead dropped that lead in one step and every entry knocked the
    # canister (0/8, 2026-10-16)
    corrected_prev = m.commanded_jaw(side)
    # the same for the attitude: the reach's servo turned the wrist command to make ScaleBFM deliver R_c; the program
    # has no rotation feedback (bfm_grasp), so that turn is kept as a constant bias -- dropping it moved the commanded
    # wrist 4.3 cm in one tick (2026-10-16)
    R_cmd = expmap(m.servo.get(("turn", side), np.zeros(3))) @ R_c
    entry_cmd = lift_from = retreat_from = retreat_to = None
    continuous = False
    shift = np.zeros(3)
    gate_held = closure = 0.0
    z_before = hold_offset0 = None
    prev_target = target.copy()
    entries: list = []

    def enter(name, stamp):
        nonlocal phase, began
        rec["phases"].append(dict(phase=name, t=round(stamp, 2)))
        phase, began = name, stamp

    def done(ok, why):
        if ok:
            m.adopt_hand(side)  # the body carries the hand from here (a walk with the object)
        else:
            m.hand(side)
            m.servo_on[side] = True
        return ctx.note("grasp", Outcome(ok, why, t0, m.t), **{k: v for k, v in rec.items() if k != "phases"},
                        phases=[p["phase"] for p in rec["phases"]])  # fmt: skip

    while True:
        stamp = m.t
        tool_m, palm = m.jaw_centre(side)
        frame = palm if entry_R is None else entry_R
        obj_p = sim.d.xpos[body].copy()
        touch = {f for f in m.fingers[side] if body in _touching(sim, f)}
        bilateral = len(touch) == 2
        elapsed = stamp - began
        grip = obj_p + grip_offset
        pregrasp = grip - PREGRASP_BACK * direction
        if phase in ("settle", "entry", "retreat"):
            knocked = float(np.linalg.norm(obj_p[:2] - obj0[:2]))
            if knocked > simple["disturb_limit_m"]:
                return done(False, f"target knocked {knocked * 100:.0f} cm in {phase}")
        if phase == "settle":
            target = pregrasp
            err = float(np.linalg.norm(tool_m - pregrasp))
            if (err <= simple["settle_tolerance_m"] and elapsed > 0.3) or elapsed >= simple["settle_timeout_s"]:
                rec["settle_error_m"] = round(err, 4)
                if simple["settle_abort_m"] is not None and err > simple["settle_abort_m"]:
                    # back out from where the hand is, the feedback zeroed: a hand stalled on a neighbour winds the
                    # feedback up to its clip, and a retreat from the unreached target then swept the vase off its shelf
                    target = retreat_from = tool_m.copy()
                    retreat_to = retreat_from - simple["retreat_m"] * direction
                    tracking, corrected_prev, prev_target = Tracking(), tool_m.copy(), tool_m.copy()
                    enter("abort", stamp)
                else:
                    entry_cmd = target.copy()
                    enter("entry", stamp)
        elif phase == "abort":  # a missed pre-grasp backs out before any entry (a miss then knocked the target)
            target = retreat_from + minjerk(elapsed / simple["retreat_seconds"]) * (retreat_to - retreat_from)
            if elapsed >= simple["retreat_seconds"]:
                return done(False, f"pregrasp missed by {rec['settle_error_m'] * 100:.1f} cm")
        elif phase == "entry":
            off = object_in_palm(frame, tool_m, grip)
            if simple["lateral_abort_m"] is not None and abs(off[1]) > simple["lateral_abort_m"]:
                return done(False, f"entry {abs(off[1]) * 100:.1f} cm across the jaw")
            if gate["on"] and max(abs(off[1]), abs(off[2])) > gate["lateral_m"] and gate_held < gate["hold_s"]:
                gate_held += dt
            else:
                entry_cmd = entry_command(entry_cmd, grip, direction, DASH_SPEED, dt, simple["entry_overshoot_m"])
            wanted = shift + frame[:, 1] * off[1] + frame[:, 2] * off[2]
            delta = wanted - shift
            norm = float(np.linalg.norm(delta))
            if norm > 1e-9:
                shift = shift + delta * min(1.0, simple["lateral_correction_speed_m_s"] * dt / norm)
            total = float(np.linalg.norm(shift))
            if total > simple["lateral_correction_max_m"]:
                shift *= simple["lateral_correction_max_m"] / total
            target = entry_cmd + shift
            arrived = entry_captured(tool_m, grip, direction, simple["entry_capture_depth_m"])
            if arrived or bilateral:
                off = object_in_palm(frame, tool_m, grip)
                lateral_tol = max(0.005, (2 * PAD_Y - width) / 2 + 0.005)
                aligned = abs(off[1]) <= lateral_tol and abs(off[2]) <= PAD_Z
                entries.append(dict(arrived=bool(arrived), bilateral=bool(bilateral), aligned=bool(aligned),
                                    offset_in_palm=[round(float(v), 4) for v in off]))  # fmt: skip
                if aligned or bilateral or len(entries) >= max(2, int(simple["entry_attempts"])):
                    rec.update(
                        entries=entries, gate_held_s=round(gate_held, 2), shift_m=round(float(np.linalg.norm(shift)), 4)
                    )
                    jump = float(np.linalg.norm(tool_m - target))
                    continuous = bool(CONTINUOUS_CLOSE) and jump > CONTINUOUS_MIN_M
                    rec.update(close_jump_m=round(jump, 4), continuous_close=continuous)
                    if not continuous:
                        target = tool_m.copy()
                    enter("close", stamp)
                else:
                    retreat_from = target.copy()
                    retreat_to = retreat_from - simple["retreat_m"] * direction
                    shift = shift + frame[:, 1] * off[1] + np.array([0.0, 0.0, off[2]])
                    enter("retreat", stamp)
            elif elapsed > simple["entry_timeout_s"]:
                return done(False, f"entry timed out {np.linalg.norm(tool_m - grip) * 100:.1f} cm from the grasp")
        elif phase == "retreat":
            target = retreat_from + minjerk(elapsed / simple["retreat_seconds"]) * (retreat_to - retreat_from)
            if elapsed >= simple["retreat_seconds"]:
                entry_cmd = target.copy() - shift
                enter("entry", stamp)
        elif phase == "close":
            closure = min(1.0, elapsed / simple["close_seconds"])
            if elapsed >= simple["close_seconds"]:
                if bilateral:
                    lift_from = target.copy() if continuous else tool_m.copy()
                    z_before = float(obj_p[2])
                    rec["close_offset_in_palm"] = [round(float(v), 4) for v in object_in_palm(frame, tool_m, grip)]
                    enter("lift", stamp)
                elif elapsed >= simple["close_seconds"] + simple["contact_timeout_s"]:
                    return done(False, "no bilateral pad contact after the close")
        elif phase == "lift":
            target = lift_from + [0.0, 0.0, LIFT_M * minjerk(elapsed / simple["lift_seconds"])]
            if elapsed >= simple["lift_seconds"]:
                hold_offset0 = object_in_palm(palm, tool_m, obj_p)
                enter("recover", stamp)
        elif phase == "recover":
            target = lift_from + [0.0, 0.0, LIFT_M]
            if elapsed >= HOLD_S:
                rise = float(obj_p[2]) - z_before
                slip = object_in_palm(palm, tool_m, obj_p) - hold_offset0
                rec.update(rise_m=round(rise, 4), hold_slip_m=round(float(np.linalg.norm(slip)), 4))
                if not bilateral:
                    return done(False, f"lost pad contact during the lift (rise {rise * 100:.1f} cm)")
                if rise < rise_min:
                    return done(False, f"not lifted (rise {rise * 100:.1f} cm)")
                return done(True, "")
        fb = tracking.step(corrected_prev, tool_m, dt)
        corrected = target + fb
        v_cmd = (target - prev_target) / dt
        prev_target = target.copy()
        corrected_prev = corrected.copy()
        m.hand(side, corrected, R_cmd, v_cmd)
        m.jaw[i] = jaw_open + closure * (jaw_closed - jaw_open)
        if not m.tick():
            return done(False, f"fell in {phase}")


def _touching(sim, finger_body: int) -> set[int]:
    m, d = sim.m, sim.d
    out = set()
    for k in range(d.ncon):
        b1, b2 = int(m.geom_bodyid[d.contact[k].geom1]), int(m.geom_bodyid[d.contact[k].geom2])
        if b1 == finger_body:
            out.add(b2)
        elif b2 == finger_body:
            out.add(b1)
    return out
