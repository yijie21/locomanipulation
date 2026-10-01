"""`run_round`'s descend/close/lift phases, split out of wall_phases.py
(refactor step 4b family 5, pure extraction; decision 9's file-size cap) --
both files together would not fit wall_phases.py's own 600-line cap. Pure
extraction, same as wall_phases.py's own split from bimanual_wall.py: no
statement reordered, no constant changed.

Same circular-import note as wall_phases.py: this module must not import
`locomani.tasks.crate.bimanual_wall` at its own top level (nothing here
needs it at import time); every module constant and helper (`wall_point`,
`errs_of`, `hand_txt`, `crate_tilt_deg`, `rot_z`, `rot_y`, `OBJ`,
...) is a deferred, in-function import instead.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from locomani.backends.sonic.primitives import DT
from locomani.tasks.crate.wall_phases import WallRoundContext, servo_tick


def _phase_descend(ctx: WallRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_wall import (
        ACROSS_BIAS,
        ACROSS_TOL,
        CONTACT_STALL_MM,
        CONTACT_STALL_S,
        DESCEND_SPEED,
        DESCEND_TIMEOUT,
        FLOOR_DWELL,
        GRIP_BELOW,
        HOVER_ABOVE,
        OBJ,
        SEAT_BELOW,
        SEAT_DWELL,
        errs_of,
        hand_txt,
        wall_frame,
        wall_point,
    )

    p, log, rnd = ctx.p, ctx.log, ctx.rnd
    arms, depth = ctx.arms, ctx.depth

    # --- B: descend until seated ----------------------------------------------
    t0 = time.time()
    dz = HOVER_ABOVE
    seated = {"left": False, "right": False}
    seated_since = None
    floor_since = None
    last_log = 0.0
    close_trigger = None
    hz_hist = []  # (t, mean measured pad height error)
    while time.time() - t0 < DESCEND_TIMEOUT:
        s = p.obs()
        if s["fallen"]:
            ctx.m["why"] = "[descend] fell"
            arms.stop()
            return ctx.m
        c, n, y_hat, psi, sign, R = wall_frame(s)
        ctx.c, ctx.n, ctx.y_hat, ctx.psi, ctx.sign, ctx.R = c, n, y_hat, psi, sign, R
        if float(np.linalg.norm(c[:2] - ctx.crate_start[:2])) > 0.05:
            ctx.m["why"] = "[descend] crate pushed > 5 cm before closing"
            arms.stop()
            return ctx.m
        dz = max(dz - DESCEND_SPEED * DT, -GRIP_BELOW)
        ctx.ref = {sd: wall_point(s, sd, depth, dz, ACROSS_BIAS) for sd in ("left", "right")}
        ref = ctx.ref
        servo_tick(ctx, s, ref)
        e = {sd: errs_of(s, arms, sd, depth) for sd in ("left", "right")}
        for sd in e:
            seated[sd] = bool(abs(e[sd][1]) <= ACROSS_TOL and e[sd][2] <= -SEAT_BELOW)
        if time.time() - last_log > 0.5:
            last_log = time.time()
            log(
                f"[descend] t={time.time() - t0:.1f} ref dz {dz * 100:+.1f} cm errs L {np.round(e['left'] * 1000, 0).tolist()} "
                f"R {np.round(e['right'] * 1000, 0).tolist()} seated {seated} touch {s['objects'][OBJ].get('touch_links')}  {hand_txt(p, s)}"
            )
        # primary: the reference is still descending but the hands are not --
        # they are resting on the bar (contact), and across is within tolerance
        hz_hist.append((time.time(), float(np.mean([e[sd][2] for sd in e]))))
        hz_hist = [(tt, h) for tt, h in hz_hist if time.time() - tt <= CONTACT_STALL_S]
        descending = dz > -GRIP_BELOW + 1e-6
        if (
            descending
            and hz_hist[0][0] <= time.time() - CONTACT_STALL_S + DT
            and abs(hz_hist[-1][1] - hz_hist[0][1]) * 1000 < CONTACT_STALL_MM
            and all(abs(e[sd][1]) <= ACROSS_TOL + 0.005 for sd in e)
            and all(e[sd][2] <= 0.0 for sd in e)
        ):
            close_trigger = "contact"
            break
        if seated["left"] and seated["right"]:
            seated_since = seated_since or time.time()
            if time.time() - seated_since >= SEAT_DWELL:
                close_trigger = "seated"
                break
        else:
            seated_since = None
        if dz <= -GRIP_BELOW + 1e-6:
            floor_since = floor_since or time.time()
            if time.time() - floor_since >= FLOOR_DWELL:
                # the hands must actually be at the bar top: pilot 23 r5 closed
                # on air with both pads 26-33 mm above it from a 0.30 m stand
                if all(abs(e[sd][1]) <= ACROSS_TOL + 0.005 and e[sd][2] <= 0.005 for sd in e):
                    close_trigger = "floor"
                    break
        time.sleep(DT)
    ctx.s = s
    ctx.m["capture_err_mm"] = {sd: [round(v * 1000, 1) for v in e[sd]] for sd in e}
    ctx.m["close_trigger"] = close_trigger
    ctx.m["descend_s"] = round(time.time() - t0, 1)
    if ctx.dump_dir:
        with open(os.path.join(ctx.dump_dir, f"wall_end_round{rnd}.json"), "w") as f:
            json.dump(
                {
                    "round": rnd,
                    "joints_meas": [float(v) for v in s["joints"]],
                    "q_cmd": [float(v) for v in arms.arm.q],
                    "pelvis": s["pelvis"],
                    "palms": s["palms"],
                    "crate": s["objects"][OBJ],
                    "depth": depth,
                    "errs_mm": ctx.m["capture_err_mm"],
                },
                f,
                indent=1,
            )
    if close_trigger is None:
        ctx.m["why"] = f"[descend] never seated: {ctx.m['capture_err_mm']}"
        arms.stop()
        return ctx.m
    log(f"[grasp] closing ({close_trigger}) after {ctx.m['descend_s']} s: {ctx.m['capture_err_mm']}")
    return None


def _phase_close(ctx: WallRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_carry import CLOSE_RAMP
    from locomani.tasks.crate.bimanual_wall import OBJ

    p, arms = ctx.p, ctx.arms
    # --- C: close -------------------------------------------------------------
    nq = int(round(CLOSE_RAMP / DT))
    for k in range(nq + 1):
        arms.set_hands(k / nq)
        arms.push(arms.arm.q)
    for _ in range(int(0.6 / DT)):
        arms.push(arms.arm.q)
    s = p.obs()
    ctx.s = s
    touching = list(s["objects"][OBJ]["touching"])
    ctx.m["touching"] = touching
    if not ("left" in touching and "right" in touching):
        ctx.m["why"] = f"[grasp] closed without bilateral contact: {touching}"
        arms.stop()
        return ctx.m
    return None


def _phase_lift(ctx: WallRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_carry import HOLD_S, LEVEL_CAP, LEVEL_GAIN, LEVEL_LP, LIFT_HEIGHT, LIFT_MIN
    from locomani.tasks.crate.bimanual_wall import (
        DEPTH_AXIS,
        OBJ,
        PALM_PITCH_DEG,
        PITCH_FOLLOW_CAP_DEG,
        PITCH_FOLLOW_GAIN,
        PITCH_FOLLOW_LP,
        TILT_OK_DEG,
        crate_tilt_deg,
        rot_y,
        rot_z,
        wall_frame,
        wall_point,
    )

    p, log = ctx.p, ctx.log
    arms, depth, psi = ctx.arms, ctx.depth, ctx.psi
    s = ctx.s

    # --- D: lift with levelling, hold -----------------------------------------
    z0 = float(s["objects"][OBJ]["pos"][2])
    lift_ref = {sd: arms.grip_point(s, sd)[0] for sd in ("left", "right")}
    nl = int(round(2.0 / DT))
    nh = int(round(HOLD_S / DT))
    level, max_dz = 0.0, 0.0
    pitch_cmd, max_pitch = 0.0, 0.0
    for k in range(1, nl + nh + 1):
        s = p.obs()
        u = min(k / nl, 1.0)
        gl, gr = wall_point(s, "left", depth, 0.0), wall_point(s, "right", depth, 0.0)
        dzb = float(gl[2] - gr[2])
        # crate pitch about the bar axis: + when the far end is low (nose down),
        # the same sense as the palm pitching its fingers down
        _, _, _, _, sgn, Rc = wall_frame(s)
        e_depth = np.zeros(3)
        e_depth[DEPTH_AXIS] = sgn
        d_far = Rc @ e_depth
        pitch_down = float(np.degrees(np.arcsin(np.clip(-d_far[2], -1.0, 1.0))))
        max_pitch = max(max_pitch, abs(pitch_down))
        pitch_cmd += PITCH_FOLLOW_LP * (
            float(np.clip(PITCH_FOLLOW_GAIN * pitch_down, -PITCH_FOLLOW_CAP_DEG, PITCH_FOLLOW_CAP_DEG)) - pitch_cmd
        )
        rot_w = {sd: rot_z(psi) @ rot_y(np.radians(PALM_PITCH_DEG - pitch_cmd)) for sd in ("left", "right")}
        max_dz = max(max_dz, abs(dzb))
        level += LEVEL_LP * (float(np.clip(LEVEL_GAIN * dzb, -LEVEL_CAP, LEVEL_CAP)) - level)
        z_lift = LIFT_HEIGHT * (3 * u * u - 2 * u**3)
        tgt = {
            "left": lift_ref["left"] + np.array([0.0, 0.0, z_lift - 0.5 * level]),
            "right": lift_ref["right"] + np.array([0.0, 0.0, z_lift + 0.5 * level]),
        }
        # Open IK target on purpose: the closed-loop lift (servo_tick) capped
        # the rise at ~6.5 cm (lead cap + the policy's height bias, rail pilot
        # 15), while the open target rises 9-14 cm. The pitch follow above is
        # what keeps the crate level from a close stand.
        arms.push(arms.solve(s, tgt, rot_w))
        if k % int(round(0.5 / DT)) == 0:
            log(
                f"[lift] t={k * DT:.1f} rise {float(s['objects'][OBJ]['pos'][2]) - z0:+.3f} m tilt {crate_tilt_deg(s):.1f} deg "
                f"pitch_down {pitch_down:+.1f} palm {PALM_PITCH_DEG - pitch_cmd:.0f} "
                f"wall dz {dzb * 1000:+.0f} mm level {level * 1000:+.0f} mm touch {list(s['objects'][OBJ]['touching'])}"
            )
        if crate_tilt_deg(s) > TILT_OK_DEG:
            ctx.m["why"] = f"[{'lift' if k <= nl else 'hold'}] crate tilted {crate_tilt_deg(s):.1f} deg"
            ctx.m.update(rise=float(s["objects"][OBJ]["pos"][2]) - z0, max_wall_dz=max_dz)
            arms.stop()
            return ctx.m
    s = p.obs()
    rise = float(s["objects"][OBJ]["pos"][2]) - z0
    tilt = crate_tilt_deg(s)
    touching = list(s["objects"][OBJ]["touching"])
    ctx.m.update(
        rise=rise,
        tilt_deg=tilt,
        touching_after=touching,
        max_wall_dz=max_dz,
        max_pitch_deg=max_pitch,
        video_s=time.time() - ctx.t_video,
    )
    if rise < LIFT_MIN:
        ctx.m["why"] = f"[lift] rose {rise * 100:.1f} cm (< {LIFT_MIN * 100:.0f})"
    elif not ("left" in touching and "right" in touching):
        ctx.m["why"] = f"[hold] lost contact: {touching}"
    else:
        ctx.m["ok"] = True
    arms.stop()
    return ctx.m
