"""`_servo_continuous`'s six phase functions, split out of stream_reach.py
(refactor step 4b family 2, pure extraction; decision 9's file-size cap).
See servo.py's module docstring for the three-way split this belongs to and
the `None` / `"retick"` / `(ok, why)` return-value convention every function
below follows -- it mirrors exactly what the original branch of the
phase-dispatch if/elif chain did at that point in the loop.
"""

from __future__ import annotations

import numpy as np

from locomani.backends.sonic.primitives import HAND_OPEN, LIFT_HEIGHT, wrap
from locomani.backends.sonic.primitives_const import HAND as _HAND
from locomani.backends.sonic.servo import (
    CLOSE_ENT_MAX,
    CLOSE_RAMP_S,
    CLOSE_SEATED,
    LIFT_MIN,
    PRE_NARROW,
    V_APPROACH,
    V_DESCEND,
    V_LIFT,
    V_TRAVEL,
    ServoState,
)


def _phase_up(state: ServoState) -> None:
    """`phase == "up"`: travel to the staging point above the pregrasp."""
    state.tgt, state.v = state.above_w, V_TRAVEL
    if float(np.linalg.norm(state.ref - state.above_w)) < 0.03:
        state.phase = "down"
    return None


def _phase_down(state: ServoState) -> None:
    """`phase == "down"`: descend from the staging point to the pregrasp."""
    state.tgt, state.v = state.pre_w, V_DESCEND
    if (
        float(np.linalg.norm(state.ref - state.pre_w)) < 0.015
        and float(np.linalg.norm(state.palm - state.pre_w)) < 0.045
    ):
        state.phase = "in"
        state.p.log("  stream: at pregrasp, sliding in")
    return None


def _phase_in_backoff(state: ServoState) -> None:
    """`phase == "in"` while a contact-flinch backoff is still running.

    Contact flinch: the thumb has started pushing the can, so retreat a
    few centimetres along the entry axis, let it settle, and re-approach
    the place it ACTUALLY is now. Without this the servo chases the can
    it is pushing until the guard kills the episode — seven of ten in
    batch13."""
    state.tgt, state.v = state.palm - state.entry * 0.05 + np.array([0, 0, 0.03]), 0.15
    if state.now >= state.backoff_until - 0.02:
        state.obj_in0 = None
    return None


def _phase_in(state: ServoState) -> str | tuple[bool, str] | None:
    """`phase == "in"`, no backoff running: slide-in alignment, freeze/wedge
    detection, then the close gate."""
    if state.obj_in0 is None:
        state.obj_in0 = state.obj.copy()
        state.obj_f = state.obj.copy()
        if state.in_since is None:
            # Once per servo call, NOT per flinch re-entry: resetting
            # it every backoff meant the patience close-gate needed
            # 12 uninterrupted seconds and never fired on episodes
            # that flinched every few seconds — they hovered at the
            # can until the 75 s deadline instead.
            state.in_since = state.now
        state.alpha = PRE_NARROW
        state.arm.hands[state.side] = HAND_OPEN + state.alpha * (state.closed - HAND_OPEN)
    if float(np.linalg.norm(state.obj - state.obj_in0)) > 0.015:
        if _HAND == "dex1" and state.full_f is not None and state.full_f[1] < CLOSE_ENT_MAX:
            # Parallel jaw, can already between the pads (entry
            # nearly spent): contact means CLOSE NOW and funnel it
            # to the palm. The dex3-era flinch re-dives and pokes
            # the can over, once per cycle, until it topples.
            state.p.log(
                f"  stream: contact inside the jaw — closing "
                f"(ent {state.full_f[1] * 100:.1f}cm "
                f"lat {state.full_f[0] * 1000:.0f}mm)"
            )
            state.phase = "close"
            return "retick"
        state.backoff_until = state.now + 1.0
        state.p.log("  stream: contact flinch, backing off to re-centre")
        return "retick"
    if state.ent_mark is None or abs(state.full_f[1] - state.ent_mark[0]) > 0.005:
        state.ent_mark = (state.full_f[1] if state.full_f is not None else 1.0, state.now)
    elif state.now - state.ent_mark[1] > 6.0:
        # Wedged: no entry progress for 6 s with the fingertip on
        # the table — lift-and-retreat to break the friction lock.
        state.backoff_until = state.now + 1.2
        state.ent_mark = None
        state.p.log("  stream: wedged (no entry progress 6 s), lifting off")
        return "retick"
    # ABSOLUTE target, built from the (low-passed) can and the
    # PLANNED palm rotation — never from the current palm. A target
    # anchored on the live palm chases its own sway (telemetry: lat
    # wobbling 2-6 cm, self-excited), and a target chasing the raw
    # can position bulldozes a nudged can. rot_t is constant, obj_f
    # moves at EMA speed: the target is calm.
    state.obj_f = state.obj if state.obj_f is None else 0.85 * state.obj_f + 0.15 * state.obj
    q_obj = state.rot_p.T @ (state.obj - state.palm)
    full = q_obj - state.offset
    # The trim integrates the MEASURED palm-frame error (which is
    # rotation-truth) into the absolute target at 0.5/s — absolute
    # geometry keeps the loop calm, the trim keeps it honest.
    can_still = float(np.linalg.norm(state.obj - state.obj_in0)) < 0.008
    if can_still:
        # Trim only while the can is stationary: integrating the
        # error of a can the hand is pushing turns the trim into a
        # positive-feedback bulldozer.
        err_w = state.rot_p @ full
        near = float(np.linalg.norm((state.obj_f - state.rot_t @ state.offset + state.trim_w) - state.palm)) < 0.06
        if near:
            d_err = err_w
        else:
            # Far out, the entry component of the error is just the
            # un-traveled distance — project it out. The VERTICAL
            # error is genuine delivery bias (the policy under-
            # delivers the waist lean by ~2/3, pin covers only the
            # arm joints — measured pal 4-5 cm above cmd) except for
            # the deliberate slide-in lift, which is added back.
            d_err = err_w - float(np.dot(err_w, state.entry)) * state.entry
            d_err = d_err.copy()
            d_err[2] = err_w[2] + state.in_lift
        state.trim_w = state.trim_w + 0.5 * state.dt * d_err
        state.trim_w[:2] = np.clip(state.trim_w[:2], -0.04, 0.04)
        # z gets more headroom: the waist-lean shortfall alone is
        # 4-5 cm.
        state.trim_w[2] = float(np.clip(state.trim_w[2], -0.08, 0.08))
    base = state.obj_f - state.rot_t @ state.offset
    if state.surface_top is not None:
        # Grip-height feasibility. The z-floor (hand lowest point
        # over the table) moves with the PLAN's rotation, and some
        # plans put it above the grasp line — the palm then hovers
        # at the floor while the thumb sweeps the can at rim height
        # (TOUCH[R:thumb_1] + can_moved telemetry). Raise the grip
        # on the can by up to 2 cm to clear the floor; a grip that
        # needs more is unreachable — fail instead of bulldozing.
        from locomani.tasks.bottle import adaptive_grasp as _ag

        low_c2 = float((_ag.hand_points() @ state.rot_t.T)[:, 2].min())
        zf2 = state.surface_top + (0.030 if _HAND == "dex1" else 0.012) - low_c2 - float((state.rot_t @ _ag.OFFSET)[2])
        need = zf2 + 0.003 - float(base[2])
        if need > 0.02:
            return False, (
                "grip height unreachable over the table: the hand's "
                f"low point forces the palm {need * 100:.1f} cm "
                f"above the grasp line (obj z {float(state.obj_f[2]):.3f}, "
                f"low_c {low_c2:.3f})"
            )
        if need > 0:
            base = base + np.array([0.0, 0.0, need])
    final_w = base + state.trim_w
    # The gate measures the DELIVERED palm against the TRUE grip
    # line (trim excluded): the trim deliberately biases the COMMAND
    # to cancel the policy's delivery offset, so measuring against
    # the biased line would hold the gate shut by that offset
    # forever.
    to_true = base - state.palm
    e_ent = float(np.dot(to_true, state.entry))
    perp = to_true - e_ent * state.entry
    lat = float(np.linalg.norm(perp))
    lat_h = float(np.hypot(perp[0], perp[1]))
    raw_f = np.array([lat, e_ent, lat_h, perp[2]])
    state.full_f = raw_f if state.full_f is None else 0.8 * state.full_f + 0.2 * raw_f
    # HARD align precondition, like the discrete servo's separate
    # align phase: zero forward motion until laterally within 12 mm.
    # The soft 18 mm gate let the hand creep in 17%-speed while still
    # 15 mm off line, and that is exactly the thumb-graze band —
    # four episodes per batch died re-flinching their way to the
    # 10 cm guard.
    if float(state.p.world_to_root_full(final_w, state.s)[0]) < 0.05:
        _pf = state.p.world_to_root_full(final_w, state.s)
        _opf = state.p.world_to_root_full(state.obj, state.s)
        _pel0 = state.arm.servo_dbg.get("pel0")
        _drift = (
            (
                float(np.linalg.norm(np.array(state.s["pelvis"]["pos"][:2]) - _pel0[0])),
                float(np.degrees(wrap(state.s["pelvis"]["yaw"] - _pel0[1]))),
            )
            if _pel0
            else (0, 0)
        )
        return False, (
            "grip target ended up at the body/behind it "
            "— the stand is wrong, not the arm "
            f"(final_pf {np.round(_pf, 2).tolist()} obj_pf {np.round(_opf, 2).tolist()} "
            f"trim {np.round(state.trim_w, 2).tolist()} rot_off {np.round(state.rot_t @ state.offset, 2).tolist()} "
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
    state.tgt = final_w - (1.0 - gate) * max(e_ent, 0.0) * state.entry
    # Slide in HIGH, drop late: full 3 cm of extra height while the
    # entry residual is >20 cm, blending to zero by 8 cm — the
    # fingertips clear the tabletop for the whole travel instead of
    # dragging across it (the wedge that froze lat at 21-30 mm).
    # dex1: no slide-in lift. Its fingertips hang only 1.45 cm
    # under the grasp centre (vs dex3's table-dragging wedge), and
    # any extra height puts the leading finger's UNDERSIDE at the
    # can's rim — it rides the rim and knocks the can over
    # (TOUCH[dex1_finger_2] at ent 5-12 cm with pal 3-7 cm high).
    state.in_lift = 0.0 if _HAND == "dex1" else 0.03 * float(np.clip((e_ent - 0.08) / 0.12, 0.0, 1.0))
    state.v = V_APPROACH if e_ent > 0.05 else V_APPROACH * 0.5
    # Close on the palm-frame TRUTH (`full` = measured object minus
    # the wanted offset), not on distance to the planned line: with
    # the pinned arm the hover sits 15-21 mm off the LINE while the
    # can is already inside the jaw envelope, and every extra hover
    # second is another fingertip graze inching the can away (the
    # 15 cm-cumulative aborts). If the can is within the envelope,
    # closing funnels it to the palm — that is what fingers curl FOR.
    # Palm-frame gate axes follow the entry column: depth along
    # the entry axis, lateral = the other two.
    la, lb = (1, 2) if state.ENTRY_COL == 0 else (0, 2)
    g_lat = float(np.hypot(full[la], full[lb]))
    good = g_lat < 0.015 and abs(full[state.ENTRY_COL]) < 0.012
    if e_ent < (0.06 if _HAND == "dex1" else 0.02) and state.side in state.s["objects"][state.obj_key]["touching"]:
        # The hand is already riding the can with the entry spent:
        # the can is inside the jaw envelope but the wrist presses
        # its top, the moving can freezes the trim (anti-bulldozer),
        # and every extra hover second pushes it further (4x
        # "toppled in the in phase"/batch, lat hovering 8-18 mm).
        # Closing now funnels it to the palm — take the patient
        # tolerance immediately instead of after 8 s.
        good = g_lat < 0.020 and abs(full[state.ENTRY_COL]) < CLOSE_SEATED
    patient = state.now - state.in_since > 8.0 and g_lat < 0.020 and abs(full[state.ENTRY_COL]) < CLOSE_SEATED
    if good or patient:
        state.ok_ticks += 1
        if state.ok_ticks > 20:  # ~0.4 s in tolerance
            state.phase = "close"
            state.p.log(f"  stream: pre-close object@palm {np.round(q_obj, 3)} want {np.round(state.offset, 3)}")
    else:
        state.ok_ticks = max(0, state.ok_ticks - 2)
    return None


def _phase_close(state: ServoState) -> tuple[bool, str] | None:
    """`phase == "close"`: keep station actively while the fingers close --
    a frozen reference lets the palm sag and the closing sweep tips the
    can -- latch a sustained contact, then hand off to `lift`."""
    if state.close_tgt is None:
        state.close_tgt = state.obj_f - state.rot_t @ state.offset + state.trim_w
    state.tgt, state.v = state.close_tgt, 0.03
    advance = True
    if state.capture_feedback is not None:
        state.tgt, advance, lost = state.capture_feedback.update(state.s, state.dt)
        if state.execution is not None:
            stats = state.execution.metrics.setdefault("feedback_close", dict(active_ticks=0, paused_ticks=0))
            stats["active_ticks"] += 1
            stats["paused_ticks"] += int(not advance)
            counts = stats.setdefault("pause_reasons", {})
            for reason in state.capture_feedback.blocked_reasons:
                counts[reason] = counts.get(reason, 0) + 1
        if lost:
            return False, "capture geometry or relative motion lost during closure"
    if advance:
        state.alpha = min(1.0, state.alpha + state.dt / CLOSE_RAMP_S)
    state.arm.hands[state.side] = HAND_OPEN + state.alpha * (state.closed - HAND_OPEN)
    # Latch contact only when SUSTAINED: a single-tick graze latched
    # contact_alpha early, capped the squeeze low, and the under-
    # gripped can dropped out during the lift.
    from locomani.tasks.bottle.grasp_capture import bilateral_dex1_contact

    paired = state.require_bilateral and _HAND == "dex1"
    contact = (
        bilateral_dex1_contact(state.s["objects"][state.obj_key], state.side)
        if paired
        else state.side in state.s["objects"][state.obj_key]["touching"]
    )
    state.touch_ticks = state.touch_ticks + 1 if contact else 0
    if state.contact_alpha is None and state.touch_ticks >= 3:
        state.contact_alpha = state.alpha
    # Past contact, squeeze another half — the travel grip that
    # survives 2.4 m of gait bounce (the tighten-for-the-road step,
    # folded into the same ramp).
    lift_ready = (
        state.alpha >= 1.0 and state.touch_ticks >= 3
        if paired
        else state.contact_alpha is not None and state.alpha >= min(1.0, state.contact_alpha + 0.6)
    )
    if lift_ready:
        state.phase = "lift"
        state.lift_target = state.controlled + np.array([0.0, 0.0, LIFT_HEIGHT])
    elif paired and state.alpha >= 1.0 and state.touch_ticks < 3:
        return False, "no sustained bilateral finger contact after closing"
    elif state.alpha >= 1.0 and state.contact_alpha is None:
        return False, "no contact after closing"
    return None


def _phase_lift(state: ServoState) -> tuple[bool, str] | None:
    """`phase == "lift"` (the dispatcher's fallback branch): rise, settle,
    then the terminal success/failure check."""
    state.tgt, state.v = state.lift_target, V_LIFT
    if float(np.linalg.norm(state.ref - state.lift_target)) < 0.02:
        state.settle_until = state.settle_until or (state.now + 0.8)
        if state.now >= state.settle_until:
            lifted = float(state.p.object_pos(state.obj_key, state.s)[2] - state.z0)
            if lifted < LIFT_MIN:
                return False, (f"{state.obj_key} rose only {lifted:.3f} m, need > {LIFT_MIN}")
            state.p.log(f"  stream: lifted {lifted * 100:.1f} cm")
            return True, ""
    return None
