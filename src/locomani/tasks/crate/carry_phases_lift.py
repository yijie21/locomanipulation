"""Bilateral close and lift acceptance for the crate carry round.

Task constants are imported locally to avoid a cycle with the entry module.
"""

from __future__ import annotations

import time

import numpy as np

from locomani.backends.sonic.primitives import DT
from locomani.tasks.crate.carry_phases import CarryRoundContext


def _phase_close(ctx: CarryRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_carry import CLOSE_RAMP, OBJ

    p, arms = ctx.p, ctx.arms
    # --- C: same-frame double close ------------------------------------------
    hold_target = arms.arm.q.copy()
    nq = int(round(CLOSE_RAMP / DT))
    for k in range(nq + 1):
        arms.set_hands(k / nq)
        arms.push(hold_target)
    for _ in range(int(0.6 / DT)):
        arms.push(hold_target)
    s = p.obs()
    ctx.s = s
    touching = list(s["objects"][OBJ]["touching"])
    ctx.m["touching"] = touching
    if not ("left" in touching and "right" in touching):
        ctx.m["why"] = f"[grasp] closed without bilateral contact: {touching}"
        arms.stop()
        return ctx.m
    return None


def _phase_lift(ctx: CarryRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_carry import (
        HOLD_S,
        LEVEL_CAP,
        LEVEL_GAIN,
        LEVEL_LP,
        LIFT_HEIGHT,
        LIFT_MIN,
        OBJ,
        TILT_MAX_DEG,
        crate_frame,
        crate_tilt_deg,
    )

    p, log, rnd = ctx.p, ctx.log, ctx.rnd
    arms, rot_w = ctx.arms, ctx.rot_w
    s = ctx.s

    # --- D: lift, level check, hold -------------------------------------------
    z0 = float(s["objects"][OBJ]["pos"][2])
    lift_ref = {sd: arms.grip_point(s, sd)[0] for sd in ("left", "right")}
    nl = int(round(2.0 / DT))
    nh = int(round(HOLD_S / DT))
    body_lift = None
    if ctx.config.reach_control == "whole_body":
        from locomani.tasks.crate.carry_whole_body import PairedLiftReference

        body_lift = PairedLiftReference(arms, s, LIFT_HEIGHT, ctx.config)
        ctx.m["lift_reference_errors_m"] = body_lift.errors.tolist()
        if np.max(body_lift.errors) > 0.02:
            ctx.m["why"] = "[lift] no continuous paired lift reference"
            return ctx.m
        nl = int(np.ceil(body_lift.duration / DT))
    level = 0.0  # + means the left hand is lowered
    max_dz = 0.0
    for k in range(1, nl + nh + 1):
        s = p.obs()
        u = min(k / nl, 1.0)
        _, bars, _, _, _ = crate_frame(p, s)
        dz = float(bars["left"][2] - bars["right"][2])  # left bar end higher: +
        max_dz = max(max_dz, abs(dz))
        level += LEVEL_LP * (float(np.clip(LEVEL_GAIN * dz, -LEVEL_CAP, LEVEL_CAP)) - level)
        z_lift = LIFT_HEIGHT * (3 * u * u - 2 * u**3)
        tgt = {
            "left": lift_ref["left"] + np.array([0.0, 0.0, z_lift - 0.5 * level]),
            "right": lift_ref["right"] + np.array([0.0, 0.0, z_lift + 0.5 * level]),
        }
        arms.push(body_lift.sample(k * DT) if body_lift is not None else arms.solve(s, tgt, rot_w))
        if k % int(round(0.5 / DT)) == 0:
            log(
                f"[lift] t={k * DT:.1f} rise {float(s['objects'][OBJ]['pos'][2]) - z0:+.3f} m "
                f"tilt {crate_tilt_deg(s):.1f} deg bar dz {dz * 1000:+.0f} mm level {level * 1000:+.0f} mm "
                f"touch {list(s['objects'][OBJ]['touching'])}"
            )
        if crate_tilt_deg(s) > TILT_MAX_DEG:
            ctx.m["why"] = f"[{'lift' if k <= nl else 'hold'}] crate tilted {crate_tilt_deg(s):.1f} deg"
            ctx.m.update(rise=float(s["objects"][OBJ]["pos"][2]) - z0, max_bar_dz=max_dz)
            arms.stop()
            return ctx.m
    ctx.m["max_bar_dz"] = max_dz
    s = p.obs()
    rise = float(s["objects"][OBJ]["pos"][2]) - z0
    tilt = crate_tilt_deg(s)
    touching = list(s["objects"][OBJ]["touching"])
    ctx.m.update(rise=rise, tilt_deg=tilt, touching_after=touching, video_s=time.time() - ctx.t_video)
    if rise < LIFT_MIN:
        ctx.m["why"] = f"[lift] rose {rise * 100:.1f} cm (< {LIFT_MIN * 100:.0f})"
    elif tilt > TILT_MAX_DEG:
        ctx.m["why"] = f"[hold] crate tilted {tilt:.1f} deg"
    elif not ("left" in touching and "right" in touching):
        ctx.m["why"] = f"[hold] lost contact: {touching}"
    else:
        ctx.m["ok"] = True
    arms.stop()
    return ctx.m
