"""`_servo_continuous`'s setup, per-tick machinery and entry point, split out
of stream_reach.py (refactor step 4b family 2, pure extraction; decision 9's
file-size cap). See servo.py's module docstring for the three-way split this
belongs to and the `None` / `"retick"` / `(ok, why)` return-value convention
every function below follows.
"""

from __future__ import annotations

import time

import numpy as np

from locomani.backends.sonic.primitives import (
    GRASP_CENTRE_OFFSET,
    HAND_CLOSED_LEFT,
    HAND_CLOSED_RIGHT,
    Primitives,
    wrap,
)
from locomani.backends.sonic.primitives_const import HAND as _HAND
from locomani.backends.sonic.streamed_arm import NUDGE_MAX
from locomani.primitives.g1_joints import ARM_MJ
from locomani.tasks.bottle import servo
from locomani.tasks.bottle.servo import (
    FREEZE_TRAVEL,
    KP_TRACK,
    LEAD_MAX,
    PRE_X_MAX,
    PRE_X_MIN,
    PRE_XY_MAX,
    PRE_Z_MAX,
    PRE_Z_MIN,
    SAFE_GAP,
    SAG_COMP,
    STAGE_UP,
    ServoState,
    _clipn,
)
from locomani.tasks.bottle.servo_phases import (
    _phase_close,
    _phase_down,
    _phase_in,
    _phase_in_backoff,
    _phase_lift,
    _phase_up,
)


def _servo_setup(state: ServoState) -> tuple[bool, str] | None:
    """Everything `_servo_continuous` used to do before its `while` loop.

    Populates `state`'s derived fields in place; a `(False, why)` return
    here is the original function's own early `return` (before the loop
    ever started)."""
    state.gap = SAFE_GAP if state.pregrasp_gap is None else float(state.pregrasp_gap)
    if state.preserve_capture_reference and (
        not state.close_from_current or state.grasp_point_control or state.capture_feedback is not None
    ):
        raise ValueError("Reference handover requires guarded measured wrist closure")
    if state.grasp_point_control and (not state.close_from_current or state.capture_feedback is not None):
        raise ValueError("Grasp-point servo requires measured closure without legacy wrist feedback")
    if not np.isfinite(state.gap) or not 0.07 <= state.gap <= 0.25:
        raise ValueError("pregrasp gap must be between .07 and .25 m")
    state.rot_t = np.asarray(state.plan.rot)
    # Entry axis: dex3 slides along palm +y (the finger cage sweeps the can
    # in sideways); dex1 is a parallel jaw and MUST come in along palm +x,
    # between the fingers — sliding along +y leads with one finger and rams
    # the can (TOUCH[dex1_finger_2] on every approach).
    state.ENTRY_COL = 0 if _HAND == "dex1" else 1
    state.entry = state.rot_t[:, state.ENTRY_COL]
    state.pre_w = np.asarray(state.plan.wrist) - state.entry * state.gap
    state.above_w = state.pre_w + np.array([0.0, 0.0, STAGE_UP])
    state.frozen = state.replan is None
    state.last_replan = 0.0
    state.offset = np.asarray(GRASP_CENTRE_OFFSET, dtype=float)
    state.closed = (HAND_CLOSED_RIGHT if state.side == "right" else HAND_CLOSED_LEFT).astype(np.float32)

    state.s = state.p.obs()
    state.obj0 = state.p.object_pos(state.obj_key, state.s)
    state.z0 = float(state.obj0[2])
    state.ref = np.array(state.s["palms"][state.side]["pos"], dtype=float)
    if state.start_at_pregrasp:
        # Diagnostic joint transport may hand over here only after actual
        # pregrasp delivery. Never skip transport based on a planned pose.
        rot_now = Primitives._rot(state.s["palms"][state.side]["quat"])
        angle = float(np.arccos(np.clip((np.trace(state.rot_t @ rot_now.T) - 1) / 2, -1.0, 1.0)))
        if np.linalg.norm(state.ref - state.pre_w) >= 0.045 or angle >= 0.25:
            return False, "pregrasp handover geometry not delivered"
    state.phase = "in" if state.start_at_pregrasp else "up"
    state.ok_ticks, state.alpha, state.contact_alpha = 0, 0.0, None
    state.capture_bias = np.zeros(3)
    if state.close_from_current:
        from locomani.tasks.bottle.measured_grasp import closure_geometry

        if state.start_at_pregrasp or _HAND != "dex1" or state.reference_guard is None:
            raise ValueError("measured closure requires guarded Dex1 control")
        if state.grip_target is None:
            raise ValueError("measured closure requires the shared live grasp target")
        if not closure_geometry(state.s, state.side, state.grip_target(state.s))["ready"]:
            return False, "measured closure handover geometry not delivered"
        state.phase = "close"
        if state.grasp_point_control:
            state.ref += Primitives._rot(state.s["palms"][state.side]["quat"]) @ state.offset
    state.full_f = None
    state.obj_f = None
    state.close_tgt = state.ref.copy() if state.close_from_current else None
    if state.close_from_current:
        state.obj_f = state.obj0.copy()
    state.obj_in0 = None
    state.backoff_until = 0.0
    state.touch_ticks = 0
    state.trim_w = np.zeros(3)
    state.lift_target, state.settle_until = None, None
    state.deadline = time.time() + 60.0  # wedge class fixed; a 90 s hover is a
    # real failure, not a slow success
    state.t_prev = time.time()
    state.last_log = 0.0
    state.res = 0.0
    state.in_since = None
    state.ent_mark = None
    state.stall_palm, state.stall_t = None, time.time()
    state.vis_n = state.tot_n = 0
    state.in_lift = 0.0  # slide-in extra height, set in the in phase
    state.dbg_z = ""  # z-chain telemetry (line/cmd/palm/floor)
    state.touch_seen = set()  # object-contact links since the last status line

    state.recorded_phase = None
    return None


def _tick_prelude(state: ServoState) -> tuple[bool, str] | None:
    """Telemetry, replan and the moved-too-much guard -- ran at the top of
    every tick, before the phase dispatch below. `None` to proceed to the
    phase dispatch, `(False, why)` to end the servo now."""
    state.transport_phase = state.phase if state.phase in ("up", "down") else None
    if state.execution is not None and state.phase != state.recorded_phase:
        state.execution.enter("grasp." + state.phase)
        state.recorded_phase = state.phase
    state.s = state.p.state.latest(5) or state.s
    state.now = time.time()
    state.dt = float(np.clip(state.now - state.t_prev, 0.005, 0.06))
    state.t_prev = state.now
    if state.s["fallen"]:
        return False, "fell during the continuous servo"
    # Deferred + module-qualified, not `from ... import toppled`: several
    # tests monkeypatch `stream_reach.toppled` and expect THIS check (the
    # original in-loop `toppled(s, obj_key)` call) to see the patched
    # version -- a plain imported name here would bind to the real
    # function at import time and never see the patch. See stream_reach.py's
    # own docstring for the rest of this monkeypatch inventory.
    from locomani.tasks.bottle import execution as stream_reach

    if stream_reach.toppled(state.s, state.obj_key):
        return False, f"{state.obj_key} toppled in the {state.phase} phase"
    state.palm = np.array(state.s["palms"][state.side]["pos"])
    state.rot_p = Primitives._rot(state.s["palms"][state.side]["quat"])
    state.controlled = state.palm + state.rot_p @ state.offset if state.grasp_point_control else state.palm
    state.obj = state.p.object_pos(state.obj_key, state.s)
    # servo telemetry (2026-09-03 topple study): palm-vs-reference
    # tracking error and torso motion per phase, printed by arm.stop()
    _d = state.arm.__dict__.setdefault("servo_dbg", {})
    _ph = _d.setdefault(state.phase, {"n": 0, "err": 0.0, "errmax": 0.0, "z": [], "pitch": []})
    _e = float(np.linalg.norm(state.controlled - state.ref))
    _ph["n"] += 1
    _ph["err"] += _e
    _ph["errmax"] = max(_ph["errmax"], _e)
    _pq = state.s["pelvis"]["quat"]
    _ph["z"].append(float(state.s["pelvis"]["pos"][2]))
    _ph["pitch"].append(float(np.arcsin(np.clip(2.0 * (_pq[0] * _pq[2] - _pq[3] * _pq[1]), -1, 1))))
    _d["last"] = (state.phase, _e, float(np.linalg.norm(state.obj - state.obj0)))
    if "pel0" not in _d:
        _d["pel0"] = (np.array(state.s["pelvis"]["pos"][:2]), float(state.s["pelvis"]["yaw"]))
    _ph.setdefault("drift", 0.0)
    _ph["drift"] = max(_ph["drift"], float(np.linalg.norm(np.array(state.s["pelvis"]["pos"][:2]) - _d["pel0"][0])))
    _ph.setdefault("dyaw", 0.0)
    _ph["dyaw"] = max(_ph["dyaw"], abs(float(wrap(state.s["pelvis"]["yaw"] - _d["pel0"][1]))))

    tl0 = state.s["objects"][state.obj_key].get("touch_links")
    if tl0:
        state.touch_seen.update(tl0)
    if "cam" in state.s["objects"][state.obj_key]:
        cam = state.s["objects"][state.obj_key]["cam"]
        state.tot_n += 1
        if cam is not None and abs(cam[0]) < 0.9 and abs(cam[1]) < 0.9:
            state.vis_n += 1
        if state.execution is None:
            # legacy direct servo callers -- see servo.py for why this lives
            # on that module object rather than being a plain global here.
            servo.LAST_EGO_VIS = state.vis_n / state.tot_n
        else:
            state.execution.metrics["ego_visibility"] = state.vis_n / state.tot_n

    if not state.frozen and state.phase in ("up", "down") and state.now - state.last_replan > 0.2:
        state.last_replan = state.now
        npl = state.replan(state.palm, state.rot_p)
        if npl.ok and npl.travel > FREEZE_TRAVEL:
            cand_rot = np.asarray(npl.rot)
            cand_pre = np.asarray(npl.wrist) - cand_rot[:, state.ENTRY_COL] * state.gap
            pre_pf = state.p.world_to_root_full(cand_pre, state.s)
            if (
                float(np.hypot(pre_pf[0], pre_pf[1])) <= PRE_XY_MAX
                and PRE_X_MIN <= pre_pf[0] <= PRE_X_MAX
                and PRE_Z_MIN <= pre_pf[2] <= PRE_Z_MAX
            ):
                state.rot_t = cand_rot
                state.entry = state.rot_t[:, state.ENTRY_COL]
                state.pre_w = cand_pre
                state.above_w = state.pre_w + np.array([0.0, 0.0, STAGE_UP])
                if state.ghost is not None:
                    state.ghost(npl)
        elif npl.ok:
            state.frozen = True  # committed: inside the freeze radius
            state.p.log(f"  stream: plan frozen {npl.travel * 100:.0f} cm out")
    if state.phase == "in":
        state.frozen = True
    if state.phase in ("up", "down", "in"):
        moved = float(np.linalg.norm(state.obj - state.obj0))
        # The in-phase tolerance is double: the servo target tracks the
        # LIVE can, so a thumb-graze that scoots it 5 cm is recoverable —
        # the hand simply re-centres on where it now is. Five episodes
        # died at exactly "5 cm in the in phase" for no lasting reason.
        # Up/down tolerate a table-shove: the close stands lean the hip
        # on the edge and slide the can a few cm with no robot contact;
        # the live replan re-anchors on it, so only a big displacement
        # (roll-away) is fatal before the slide-in begins.
        limit = NUDGE_MAX * 3 if state.phase == "in" else NUDGE_MAX * 2.4
        if moved > limit:
            hit = state.s["objects"][state.obj_key].get("touch_links") or ["nothing"]
            ft = state.s.get("furniture_touch") or []
            return False, (
                f"{state.obj_key} moved {moved * 100:.0f} cm in the "
                f"{state.phase} phase (touched {', '.join(hit)}"
                f"{'; furniture: ' + ', '.join(ft) if ft else ''})"
            )
    return None


def _dispatch_phase(state: ServoState) -> str | tuple[bool, str] | None:
    """Route to this tick's phase handler -- the original `if phase == ...`
    chain, unchanged in order and conditions. See servo.py's module docstring
    for what `None` / `"retick"` / `(ok, why)` mean here."""
    if state.phase == "up":
        return _phase_up(state)
    if state.phase == "down":
        return _phase_down(state)
    if state.phase == "in" and state.now < state.backoff_until:
        return _phase_in_backoff(state)
    if state.phase == "in":
        return _phase_in(state)
    if state.phase == "close":
        return _phase_close(state)
    return _phase_lift(state)  # else: lift


def _tick_advance(state: ServoState) -> tuple[bool, str] | None:
    """Ref update, IK solve, guards, push, pin, stall check, logging -- ran
    once per tick after the phase dispatch above, using whatever
    `state.tgt` / `state.v` that phase set."""
    state.ref = state.ref + _clipn(np.asarray(state.tgt, dtype=float) - state.ref, state.v * state.dt)
    lead = state.ref - state.controlled
    if float(np.linalg.norm(lead)) > LEAD_MAX:
        # The policy is behind; wait for it instead of running away.
        state.ref = state.controlled + _clipn(lead, LEAD_MAX)
        lead = state.ref - state.controlled
    cmd = state.ref + np.clip(KP_TRACK * lead, -0.05, 0.05) + state.capture_bias
    if state.phase in ("down", "in", "close") and not (state.close_from_current and state.phase == "close"):
        # The policy arm SAGS ~2 cm below its reference; geometry that
        # clears the table on paper drags the middle fingertip across it
        # in the flesh (FURN[R:middle_1] wedged, lat/ent frozen for a
        # minute). Bias the commanded height so the DELIVERED height is
        # the planned one.
        cmd = cmd + np.array([0.0, 0.0, SAG_COMP + (state.in_lift if state.phase == "in" else 0.0)])
    if state.surface_top is not None:
        # Hard geometric floor: with the pinned arm executing IK heights
        # EXACTLY, any command whose HAND (not wrist — the fingertips
        # hang below) would dip under tabletop+1.2 cm turns into a
        # scrape that bulldozes the can. Computed from the measured
        # hand point cloud at the CURRENT planned rotation.
        from locomani.tasks.bottle import adaptive_grasp as _ag

        low_c = float((_ag.hand_points() @ state.rot_t.T)[:, 2].min())
        z_floor = (
            state.surface_top + (0.030 if _HAND == "dex1" else 0.012) - low_c - float((state.rot_t @ _ag.OFFSET)[2])
        )
        if state.grasp_point_control:
            z_floor += float((state.rot_t @ _ag.OFFSET)[2])
        if state.rotation_bias is not None:
            from scipy.spatial.transform import Rotation

            biased_rot = Rotation.from_rotvec(state.rotation_bias).as_matrix() @ state.rot_t
            biased_low = float((_ag.hand_points() @ biased_rot.T)[:, 2].min())
            z_floor = max(z_floor, state.surface_top + 0.030 - biased_low - float((biased_rot @ _ag.OFFSET)[2]))
        if cmd[2] < z_floor:
            cmd = cmd.copy()
            cmd[2] = z_floor
        if state.phase == "in" and state.obj_f is not None:
            line_z = float((state.obj_f - state.rot_t @ state.offset)[2])
            state.dbg_z = (
                f"  z(pal{state.palm[2] - line_z:+.3f} cmd{float(cmd[2]) - line_z:+.3f} fl{z_floor - line_z:+.3f})"
            )
    aim_pf = state.p.world_to_root_full(cmd, state.s)
    rot_use = state.rot_t
    if _HAND == "dex1" and state.phase in ("up", "down"):
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
        d_pre = float(np.linalg.norm(np.asarray(state.pre_w) - state.palm))
        k_up = float(np.clip((d_pre - 0.02) / 0.12, 0.0, 1.0))
        th = np.deg2rad(55.0) * k_up
        if th > 1e-3:
            c, sn = np.cos(th), np.sin(th)
            for sgn in (1.0, -1.0):
                ry = np.array([[c, 0.0, sgn * sn], [0.0, 1.0, 0.0], [-sgn * sn, 0.0, c]])
                cand = state.rot_t @ ry
                if (cand @ np.array([1.0, 0.0, 0.0]))[2] > 0:
                    rot_use = cand
                    break
    if state.rotation_bias is not None:
        from scipy.spatial.transform import Rotation

        rot_use = Rotation.from_rotvec(state.rotation_bias).as_matrix() @ rot_use
    rot_pf = Primitives._rot(state.s["pelvis"]["quat"]).T @ rot_use
    q_sol, state.res, _ = state.arm.ik.solve(
        state.side,
        aim_pf,
        rot_pf,
        q_mj29=state.arm.q,
        use_waist=True,
        iters=10,
        **(dict(point_offset=state.offset, rot_weight=0.1) if state.grasp_point_control else {}),
    )
    if state.rotation_bias is not None or state.reference_guard is not None:
        q_sol = state.arm.q + np.clip(q_sol - state.arm.q, -np.deg2rad(2.0), np.deg2rad(2.0))
    if state.transport_guard is not None and state.transport_phase is not None:
        candidate = q_sol if state.res < 0.10 else state.arm.q
        checked = np.asarray(candidate).copy()
        checked[:12] = np.asarray(state.s["joints"])[:12]  # legs are controlled by WBC
        guard_start = time.perf_counter()
        swept = state.transport_guard.check(state.s, checked, state.side)
        if state.execution is not None:
            stats = state.execution.metrics.setdefault("transport_guard", dict(checks=0, max_ms=0.0, rejected=None))
            stats["checks"] += 1
            stats["max_ms"] = max(stats["max_ms"], (time.perf_counter() - guard_start) * 1000)
            if not swept.clear:
                stats["rejected"] = swept.to_dict()
        if not swept.clear:
            return False, (
                f"transport clearance in {state.transport_phase}: {swept.link} vs "
                f"{swept.obstacle}, {swept.distance:.4f} m"
            )
    if state.reference_guard is not None:
        rejection = state.reference_guard(state.s, q_sol if state.res < 0.10 else state.arm.q, state.phase)
        if rejection:
            return False, rejection
    if state.res < 0.10:
        state.arm.q = q_sol
    state.arm._push(state.arm.q)
    if state.pin is not None:
        # Position-mode arm: the sim-side per-motor PD tracks the IK
        # solution exactly (+-0.03 rad measured), instead of the policy's
        # ~1 cm-and-worse delivery. The streamed reference above still
        # runs so the policy's balance model sees the arm where it is;
        # its arm torques are overridden at the actuator.
        # Waist stays with the policy: pinning it stiff fought the
        # balance controller (torso jitter, 9/15 toppled) — the arm pin
        # plus the vertical trim absorb the lean shortfall instead.
        state.pin(**{state.side: [float(v) for v in state.arm.q[ARM_MJ[state.side]]]})

    # A palm that has not moved 2 cm in 8 s during the travel phases is
    # not slow, it is stuck — either the IK is rejecting every solution
    # (res stays high) or the policy is not delivering the reference.
    # Fail fast and SAY WHICH: three episodes burned their whole 60 s
    # deadline pinned at the lead cap with no diagnosis in the log.
    if state.stall_palm is None or float(np.linalg.norm(state.palm - state.stall_palm)) > 0.02:
        state.stall_palm, state.stall_t = state.palm.copy(), state.now
    elif state.phase in ("up", "down") and state.now - state.stall_t > 8.0:
        return False, (
            f"arm not tracking in the {state.phase} phase "
            f"(palm stalled 8 s, IK res {state.res:.3f}, "
            f"target {float(np.linalg.norm(state.tgt - state.palm)):.2f} m "
            f"away)"
        )

    if state.now - state.last_log > 2.0:
        extra = ""
        ft = state.s.get("furniture_touch")
        if ft:
            extra += f"  FURN[{','.join(ft)}]"
        if state.touch_seen:
            extra += f"  TOUCH[{','.join(sorted(state.touch_seen))}]"
            state.touch_seen.clear()
        dcan = float(np.linalg.norm(state.obj - state.obj0))
        if dcan > 0.01:
            extra += f"  can_moved {dcan * 100:.1f}cm"
        if state.phase == "in" and state.full_f is not None:
            extra += (
                f"  lat {state.full_f[0] * 1000:.0f}mm"
                f"(h{state.full_f[2] * 1000:.0f}"
                f" v{state.full_f[3] * 1000:+.0f})"
                f" ent {state.full_f[1] * 100:.1f}cm" + (state.dbg_z if state.dbg_z else "")
            )
            # jaw-axis vertical components: planned rot vs MEASURED palm
            extra += f"  jawz(t{state.rot_t[2, 1]:+.2f} m{state.rot_p[2, 1]:+.2f})"
            hq = state.s["palms"][state.side].get("hand_q")
            if hq:
                extra += "  hq[" + ",".join(f"{v:+.3f}" for v in hq) + "]"
        tpf = state.p.world_to_root_full(np.asarray(state.tgt, dtype=float), state.s)
        point_name = "grasp_center" if state.grasp_point_control else "palm"
        state.p.log(
            f"    stream cont[{state.phase}] {point_name}->ref "
            f"{float(np.linalg.norm(lead)) * 100:.1f} cm  res {state.res:.3f}"
            f"  tgt_pf [{tpf[0]:+.2f} {tpf[1]:+.2f} {tpf[2]:+.2f}]{extra}"
        )
        state.last_log = state.now
    return None


def _servo_continuous(
    p,
    arm,
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
    state = ServoState(
        p=p,
        arm=arm,
        obj_key=obj_key,
        plan=plan,
        side=side,
        replan=replan,
        ghost=ghost,
        pin=pin,
        surface_top=surface_top,
        execution=execution,
        transport_guard=transport_guard,
        start_at_pregrasp=start_at_pregrasp,
        rotation_bias=rotation_bias,
        reference_guard=reference_guard,
        pregrasp_gap=pregrasp_gap,
        close_from_current=close_from_current,
        grip_target=grip_target,
        require_bilateral=require_bilateral,
        capture_feedback=capture_feedback,
        grasp_point_control=grasp_point_control,
        preserve_capture_reference=preserve_capture_reference,
    )
    outcome = _servo_setup(state)
    if outcome is not None:
        return outcome

    while time.time() < state.deadline:
        outcome = _tick_prelude(state)
        if outcome is not None:
            return outcome
        outcome = _dispatch_phase(state)
        if outcome == "retick":
            continue
        if outcome is not None:
            return outcome
        outcome = _tick_advance(state)
        if outcome is not None:
            return outcome
    return False, f"continuous servo timed out in the {state.phase} phase"
