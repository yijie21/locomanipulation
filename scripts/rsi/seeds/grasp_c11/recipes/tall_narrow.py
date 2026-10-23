"""A tall free object whose hull section at the jaw height fits the open jaw with under a centimetre a side (the 28 cm
dish soap bottle on the kitchen counter when its 8.7 cm side faces the hand): a closed-loop hand entry in place of
the IK-leg entry, then grip, lift and carry-out on the same hand program (grasp_atomic iteration 8, H2).

Why: Motor.reach's entry leg starts from a fresh whole-body IK solution, and the min-jerk joint blend from the
pregrasp pose into it swings the jaw (s410022: 12 cm sideways and 11 cm down within 0.15 s), after which the jaw
rushes back diagonally, overshoots the line by 9 cm with the wrist yawed 20-28 deg, and a finger meets the bottle's
side before the pads straddle it: the bottle tips 77 deg and the grip closes on nothing. With 0.4 cm a side there is
no room for that.

The flow (Motor.hand every tick; the body's reference stays at the pregrasp pose, so there is no joint blend):
1. Take over from the reach's last command (Motor.commanded_jaw and the last commanded wrist rotation): the servo's
   lead and turn are carried over, so the command does not step.
2. The desired jaw centre runs along the approach line through the section's middle (the object's live pose) at
   HAND_SPEED; an integral of the measured jaw centre's error (KI) and of the wrist attitude's (KI_ROT) adds to the
   carried lead.
3. Where the fingertips are GATE_CLEAR_M short of the section's near face it holds until the jaw is within GATE_LAT_M
   of the line sideways, GATE_Z_M vertically and GATE_DEG in attitude (at most GATE_S), then advances, pausing while
   the sideways error exceeds PAUSE_LAT_M or the wrist yaw PAUSE_YAW_DEG, and while exactly one finger touches the
   object, when the line itself shifts toward that finger's side at CONTACT_MPS (at most SHIFT_MAX_M): first smoke on
   s410022, a 0.6-0.9 cm sideways drift put the left finger on the bottle and turned it 8 deg. It stops where the
   housing front (HOUSING_FRONT_M behind the jaw centre, robot model) is HOUSING_CLEAR_M short of the near face within
   the housing's height, or ENTRY_PAST_M past the middle if that is nearer: going on to the middle pushed the bottle
   5 cm with the housing in that smoke (the r2_on_single library cue).
4. If the object moves more than PUSH_M, it backs off BACK_M, waits for the object to settle, re-centres on its new
   pose and gates again; at most BACKOFFS times.
5. Grip (Motor.grip with the hand command held), lift LIFT_M and, under a cover, carry out, all on the hand program:
   an IK leg re-solves the arm and swings it under the held bottle (iteration 7, draft 1). The plan ends with the hand
   still commanded, so the episode's hold keeps it where it is.
Every other object (and this one when its section does not fit the open jaw) keeps pick.side_grasp unchanged."""

from __future__ import annotations

import numpy as np

from locomani.backends.scalebfm import motor as M
from locomani.backends.scalebfm import runtime as R
from locomani.backends.scalebfm.motor import Outcome, expmap
from locomani.primitives.whole_body_ik import rotvec

from .. import pick as P

ON = 1  # 0: these objects take the generic entry again
# what it is for: resting this tall or taller, neither flat nor a container, and a hull section across the jaw at the
# grasp height (the widest within the pads' height) that fits the open 9.5 cm jaw with margin_min..margin_max a side
SIGNATURE = dict(height_min_m=0.18, flat=False, container=False, margin_min_m=0.0, margin_max_m=0.010)
# the hand program
HAND_SPEED = 0.05  # m/s of the desired jaw centre (pick.ENTRY_SPEED)
KI = 2.0  # 1/s jaw-centre integral (the motor servo's)
KI_ROT = 2.0  # 1/s wrist-attitude integral
LEAD_CAP_M = 0.08  # the integral per axis, on top of the carried lead
TURN_CAP = 0.5  # rad, the attitude integral on top of the carried turn
GATE_CLEAR_M = 0.01  # fingertips this far short of the section's near face: the gate
GATE_LAT_M = 0.005
GATE_Z_M = 0.01
GATE_DEG = 3.0
GATE_S = 3.0  # the longest wait at the gate; then it advances anyway
PAUSE_LAT_M = 0.003  # past the gate the advance pauses while the jaw is this far off the line ...
PAUSE_YAW_DEG = 3.0  # ... or the wrist x this far off the approach in yaw
CONTACT_MPS = 0.02  # one finger on the object: the line shifts toward its side at this speed ...
SHIFT_MAX_M = 0.01  # ... up to this far
HOUSING_FRONT_M = 0.041  # the housing's front behind the jaw centre (robot model: x -0.172..-0.041)
HOUSING_CLEAR_M = 0.005
ENTRY_PAST_M = 0.01  # past the section's middle, when the housing allows it
PUSH_M = 0.01  # the object moved this far (xy) since the last centring: back off
BACK_M = 0.03
BACKOFFS = 2
STILL_MPS = 0.01  # the object counts as settled below this speed ...
WAIT_MIN_S = 0.3  # ... after at least this long ...
WAIT_MAX_S = 1.5  # ... or after this long
SETTLE_S = 1.5  # at the end of a move: up to this long, ...
SETTLE_TOL_M = 0.01  # ... or until the jaw is this close (after 0.3 s)
ENTRY_MAX_S = 20.0
LIFT_M = 0.10
LIFT_SPEED = 0.08


def match(obj) -> float:
    """1 for a tall object whose section fits the open jaw with under SIGNATURE's margin a side, else 0."""
    sig = SIGNATURE
    if not ON or obj.margin is None or obj.flat != sig["flat"] or obj.container != sig["container"]:
        return 0.0
    if obj.height < sig["height_min_m"] or not sig["margin_min_m"] <= obj.margin < sig["margin_max_m"]:
        return 0.0
    return 1.0


def _tilt_deg(q) -> float:
    return float(np.degrees(np.arccos(np.clip(R.mat(q)[2, 2], -1.0, 1.0))))


def finger_contacts(m, side: str, body: int) -> tuple[bool, bool]:
    """Whether `side`'s finger 1 (on the wrist's -y side) and finger 2 (+y) touch `body` now."""
    mm, dd = m.sim.m, m.sim.d
    f1, f2 = m.fingers[side]
    hit = [False, False]
    for i in range(dd.ncon):
        b1, b2 = int(mm.geom_bodyid[dd.contact[i].geom1]), int(mm.geom_bodyid[dd.contact[i].geom2])
        if body in (b1, b2):
            other = b2 if b1 == body else b1
            hit[0] |= other == f1
            hit[1] |= other == f2
    return hit[0], hit[1]


class Hand:
    """One side's jaw command: the desired jaw centre + the lead carried from the reach + an integral of the measured
    error; the wrist rotation expmap(turn) @ Rg, turn carried from the reach plus an attitude integral."""

    def __init__(self, m, side: str, des0, Rg):
        self.m, self.side, self.Rg = m, side, np.asarray(Rg, float)
        Rc0 = R.mat(m.last_fut[M.WRIST[side]][1][0])
        self.lead0 = m.commanded_jaw(side) - np.asarray(des0, float)
        self.acc = np.zeros(3)
        self.turn0 = rotvec(Rc0 @ self.Rg.T)
        self.turn = self.turn0.copy()
        self.des = np.asarray(des0, float)

    def error(self, des):
        """(desired - measured jaw centre, attitude error in deg)."""
        jaw, Rw = self.m.jaw_centre(self.side)
        return np.asarray(des, float) - jaw, float(np.degrees(np.linalg.norm(rotvec(self.Rg @ Rw.T))))

    def step(self, des, vel=None) -> bool:
        """Integrate the error to `des`, command it and tick; False once the robot has fallen."""
        m = self.m
        jaw, Rw = m.jaw_centre(self.side)
        self.des = np.asarray(des, float)
        self.acc = np.clip(self.acc + KI * R.DT * (self.des - jaw), -LEAD_CAP_M, LEAD_CAP_M)
        dt = self.turn + KI_ROT * R.DT * rotvec(self.Rg @ Rw.T) - self.turn0
        n = float(np.linalg.norm(dt))
        self.turn = self.turn0 + (dt * (TURN_CAP / n) if n > TURN_CAP else dt)
        m.hand(self.side, self.des + self.lead0 + self.acc, expmap(self.turn) @ self.Rg, vel)
        return m.tick()

    def move(self, ctx, name: str, goal, speed: float) -> Outcome:
        """Straight from the last desired point to `goal` at `speed`, then settle (SETTLE_S, SETTLE_TOL_M)."""
        m = self.m
        m.primitive, t0 = name, m.t
        start = self.des.copy()
        goal = np.asarray(goal, float)
        length = float(np.linalg.norm(goal - start))
        u = (goal - start) / length if length > 1e-9 else np.zeros(3)
        s = 0.0
        while s < length:
            s = min(length, s + speed * R.DT)
            if not self.step(start + s * u, speed * u):
                return Outcome(False, f"fell at {m.fell_at:.2f} s", t0, m.t)
        t1 = m.t
        while m.t - t1 < SETTLE_S:
            e, _ = self.error(goal)
            if m.t - t1 >= 0.3 and float(np.linalg.norm(e)) <= SETTLE_TOL_M:
                break
            if not self.step(goal):
                return Outcome(False, f"fell at {m.fell_at:.2f} s", t0, m.t)
        e, att = self.error(goal)
        return Outcome(True, t0=t0, t1=m.t, info=dict(jaw_err_cm=round(100 * float(np.linalg.norm(e)), 2),
                                                      attitude_err_deg=round(att, 1)))  # fmt: skip


def entry(ctx, key: str, obj, pre, Rg, ux) -> Outcome:
    """Steps 1-4: the jaw from the pregrasp to ENTRY_PAST_M past the section's middle, on the line through it."""
    w, m = ctx.world, ctx.motor
    ux = np.asarray(ux, float)
    uy = np.array([-ux[1], ux[0], 0.0])
    p0, q0 = w.object_pose(key)
    off = R.mat(q0).T @ (np.asarray(obj.aim, float) - p0)  # the aim point in the object's frame

    def aim():
        p, q = w.object_pose(key)
        return p + R.mat(q) @ off

    body = w.free[key]
    d_gate = float(obj.near) + P.FINGERTIP_M + GATE_CLEAR_M
    d_end = max(-ENTRY_PAST_M, float(obj.near_housing) - HOUSING_FRONT_M + HOUSING_CLEAR_M)
    d = float((aim() - np.asarray(pre, float)) @ ux)
    shift = 0.0  # the line's sideways shift (along uy) from finger contacts

    def line(dd):
        return aim() + shift * uy - dd * ux

    hand = Hand(m, P.SIDE, line(d), Rg)
    phase = "approach" if d > d_gate else "gate"
    m.primitive, t0 = "hand_entry", m.t
    t_phase, ref_xy, prev_xy = m.t, p0[:2].copy(), p0[:2].copy()
    backoffs, d_back, swing, contact, gates, tilt = 0, d, 0.0, None, [], 0.0
    touches, pauses = [], 0
    why = ""
    while True:
        if m.t - t0 >= ENTRY_MAX_S:
            why = "hand entry timed out"
            break
        e, att = hand.error(line(d))
        lat, vz = float(e @ uy), float(e[2])
        xw = m.jaw_centre(P.SIDE)[1][:, 0]
        yaw = float(np.degrees(np.arctan2(xw @ uy, xw @ ux)))
        f1, f2 = finger_contacts(m, P.SIDE, body)
        if f1 != f2 and phase in ("gate", "advance", "settle"):
            shift = float(np.clip(shift + (CONTACT_MPS if f2 else -CONTACT_MPS) * R.DT, -SHIFT_MAX_M, SHIFT_MAX_M))
            if not touches or touches[-1][0] != (1 if f1 else 2):
                touches.append((1 if f1 else 2, round(m.t, 2), round(100 * d, 2), round(-100 * lat, 2)))
        po, qo = w.object_pose(key)
        moved = float(np.linalg.norm(po[:2] - ref_xy))
        tilt = max(tilt, _tilt_deg(qo))
        if m.t - t0 <= 0.5:
            swing = max(swing, float(np.hypot(lat, vz)))
        if contact is None and float(np.linalg.norm(po[:2] - p0[:2])) > 0.002:
            contact = dict(t=round(m.t, 2), d_cm=round(100 * d, 2), lat_cm=round(-100 * lat, 2),
                           dz_cm=round(-100 * vz, 2), along_cm=round(-100 * float(e @ ux), 2), phase=phase)  # fmt: skip
        if phase in ("approach", "gate", "advance") and moved > PUSH_M and backoffs < BACKOFFS:
            backoffs += 1
            phase, d_back = "back", d + BACK_M
        vel = None
        if phase == "approach":
            d = max(d_gate, d - HAND_SPEED * R.DT)
            vel = -HAND_SPEED * ux
            if d <= d_gate:
                phase, t_phase = "gate", m.t
        elif phase == "gate":
            if (abs(lat) < GATE_LAT_M and abs(vz) < GATE_Z_M and att < GATE_DEG) or m.t - t_phase >= GATE_S:
                gates.append(dict(wait_s=round(m.t - t_phase, 2), lat_cm=round(-100 * lat, 2),
                                  dz_cm=round(-100 * vz, 2), att_deg=round(att, 1)))  # fmt: skip
                phase = "advance"
        elif phase == "advance":
            if abs(lat) <= PAUSE_LAT_M and abs(yaw) <= PAUSE_YAW_DEG and f1 == f2:
                d = max(d_end, d - HAND_SPEED * R.DT)
                vel = -HAND_SPEED * ux
            else:
                pauses += 1
            if d <= d_end:
                phase, t_phase = "settle", m.t
        elif phase == "back":
            d = min(d_back, d + HAND_SPEED * R.DT)
            vel = HAND_SPEED * ux
            if d >= d_back:
                phase, t_phase = "wait", m.t
        elif phase == "wait":
            speed = float(np.linalg.norm(po[:2] - prev_xy)) / R.DT
            if (m.t - t_phase >= WAIT_MIN_S and speed < STILL_MPS) or m.t - t_phase >= WAIT_MAX_S:
                ref_xy = po[:2].copy()
                phase, t_phase = "gate", m.t
        elif phase == "settle":
            if m.t - t_phase >= SETTLE_S or (m.t - t_phase >= 0.3 and float(np.linalg.norm(e)) <= SETTLE_TOL_M):
                break
        prev_xy = po[:2].copy()
        if not hand.step(line(d), vel):
            return Outcome(False, f"fell at {m.fell_at:.2f} s", t0, m.t), hand
    e, att = hand.error(line(d))
    po, qo = w.object_pose(key)
    info = dict(jaw_err_cm=round(100 * float(np.linalg.norm(e)), 2), lat_cm=round(-100 * float(e @ uy), 2),
                dz_cm=round(-100 * float(e[2]), 2), attitude_err_deg=round(att, 1), swing_cm=round(100 * swing, 2),
                contact=contact, gates=gates, backoffs=backoffs, tilt_max_deg=round(tilt, 1),
                tilt_deg=round(_tilt_deg(qo), 1), moved_cm=round(100 * float(np.linalg.norm(po[:2] - p0[:2])), 2),
                d_gate_cm=round(100 * d_gate, 2), d_end_cm=round(100 * d_end, 2), shift_cm=round(100 * shift, 2),
                touches=touches[:8], pause_ticks=pauses)  # fmt: skip
    return Outcome(not why, why, t0, m.t, info), hand


def grasp(ctx, key: str, obj, pre, Rg, ux) -> Outcome:
    """The whole flow after the pregrasp: entry, grip, lift and carry-out. Returns pick.grip_and_lift's outcome shape
    (info lift_to) with info carried=True, so pick.pick does not run its IK carry-out after it."""
    w, m = ctx.world, ctx.motor
    out, hand = entry(ctx, key, obj, pre, Rg, ux)
    ctx.note("hand_entry", out)
    if m.fell_at is not None:
        return out
    out = ctx.note("grip", m.grip(P.SIDE), tilt_deg=round(_tilt_deg(w.object_pose(key)[1]), 1))
    if not out.ok or m.fell_at is not None:
        return out
    lift = hand.des + np.array([0.0, 0.0, LIFT_M])
    out = ctx.note("lift", hand.move(ctx, "hand_lift", lift, LIFT_SPEED))
    if m.fell_at is not None:
        return out
    support = w.meta["objects"][key]["surface"]
    po, _ = w.object_pose(key)
    bottom = float(po[2]) - float(w.meta["objects"][key]["box"]["half_extent"][2])  # as pick.carry_out
    over = P.overhead(w, po[:2], bottom, support)
    need = 0.0
    if over:
        need = P.CARRY_MAX_M
        for s in np.arange(0.0, P.CARRY_MAX_M + 1e-9, P.CARRY_STEP_M):
            if not P.overhead(w, po[:2] - s * ux[:2], bottom, support):
                need = float(s) + P.CARRY_MARGIN_M
                break
        back = ctx.note("carry_out", hand.move(ctx, "hand_carry", lift - need * np.r_[ux[:2], 0.0], LIFT_SPEED),
                        need=round(need, 3), under=over)  # fmt: skip
        if m.fell_at is not None:
            return back
    out.ok = m.fell_at is None
    out.info.update(lift_to=lift, carried=True)
    return out
