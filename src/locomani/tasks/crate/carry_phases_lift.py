"""`run_round`'s close/lift phases, split out of carry_phases.py (refactor
step 4b family 5, pure extraction; decision 9's file-size cap) -- both files
together would not fit carry_phases.py's own 600-line cap. Pure extraction,
same as carry_phases.py's own split from bimanual_carry.py: no statement
reordered, no constant changed.

Same circular-import note as carry_phases.py: this module must not import
`locomani.tasks.crate.bimanual_carry` at its own top level (nothing here
needs it at import time); `CLOSE_RAMP`, `OBJ`, `HOLD_S`, `LEVEL_*`,
`LIFT_HEIGHT`, `LIFT_MIN`, `TILT_MAX_DEG`, `crate_frame`, `crate_tilt_deg`
are deferred, in-function imports instead.
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
        arms.push(arms.solve(s, tgt, rot_w))
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
