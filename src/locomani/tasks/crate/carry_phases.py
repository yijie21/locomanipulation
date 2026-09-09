"""`run_round`'s phases, split out of bimanual_carry.py (refactor step 4b
family 5, pure extraction; decision 9's file-size cap). Every statement
here is unchanged from the original inline body -- only the packaging
moved: each closure's captured locals became a field on `CarryRoundContext`,
and the four helpers used from more than one place (`on_tick`, `brake`,
`stand_metrics`, `pitched`) became plain functions taking `ctx` instead of
nested closures. The last two phases (`_phase_close`, `_phase_lift`) live in
`carry_phases_lift.py` instead -- all five here would not fit this file's
own 600-line cap.

This module must not import `locomani.tasks.crate.bimanual_carry` at its own
top level: `bimanual_carry.py` imports names from here (`run_round`'s new
home is `bimanual_carry.py` itself, calling these phases in sequence), so a
module-level import in the other direction would make the two files a
load-order-sensitive circular import. Nothing here needs anything of
`bimanual_carry.py`'s at import time anyway -- every module constant
(`STAND_OFF`, `APPR_SPEED`, ...) and helper (`crate_frame`, `hand_txt`,
`crate_tilt_deg`, `rot_z`, `rot_y`) it needs is a deferred, in-function
import instead (none of them is ever overridden by assignment anywhere in
the repo, so a plain import -- not a module-qualified read -- is correct
and sufficient here, unlike vla_round.py's family-3 knobs).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from locomani.backends.sonic import upper_ref
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode, StreamMode
from locomani.backends.sonic.primitives import DT, Primitives, upper_body_from_mj, wrap
from locomani.primitives.g1_joints import ARM_MJ

# Every phase function below returns `dict | None`: `None` to continue to the
# next phase, a (failure-populated) `m` dict to end the round with that
# result.


@dataclass
class CarryRoundContext:
    """Everything one `run_round` call threads through its phases.

    The first block is `run_round`'s own parameters, unchanged (`m` is
    pre-populated by the caller before phase 1 runs); the rest is state a
    phase computes for a LATER phase (or a converted closure) to use -- each
    field names the closure/phase that used to capture it.
    """

    # -- run_round's own parameters ---------------------------------------
    p: Primitives
    reset: Any
    meta: dict
    rnd: int
    rng: np.random.Generator
    log: Callable
    video_label: str
    m: dict

    # -- filled in by _phase_reset, read by later phases -------------------
    surf: dict = None  # meta["surfaces"]["A"], read by _phase_approach
    hx: float = None  # surf's flat half-extent, read by _phase_approach
    s: dict = None  # latest p.obs(), read/reassigned by every later phase
    t_video: float = None  # video start wall-clock, read by _phase_lift

    # -- filled in by _phase_approach, read by _phase_grasp / on_tick /
    #    brake / pitched -----------------------------------------------------
    centre: np.ndarray = None  # crate_frame's centre; reassigned every _phase_grasp tick
    bars: dict = None  # crate_frame's bar points; reassigned every
    # _phase_grasp tick, read by pitched()
    n: np.ndarray = None  # crate_frame's outward normal; reassigned every _phase_grasp tick
    y_hat: np.ndarray = None  # crate_frame's left-hand axis; reassigned every _phase_grasp tick
    psi: float = None  # crate_frame's heading; reassigned every
    # _phase_grasp tick, read by brake()
    stand_xy: np.ndarray = None  # the walk-in's target pelvis xy
    edge_x: float = None  # table edge x, read by stand_metrics()
    q_free: np.ndarray = None  # the free-standing upper-body pose, read by on_tick()
    ramp: Any = None  # upper_ref.UpperRamp, read by on_tick()/brake()
    q_pre17: np.ndarray = None  # the pre-grasp 17-joint upper body pose,
    # read by on_tick()/brake()
    arms: Any = None  # BimanualArms, read by every later phase
    R_stand: np.ndarray = None  # rot_z(psi) at the walk-in's start; reassigned
    # once more at _phase_grasp's own start (from the
    # post-brake psi) -- NOT re-derived per tick even
    # though psi itself is; read by pitched()
    rot_w: dict = None  # per-side palm rotation; set once in _phase_approach
    # (fingers level for the walk-in), reassigned every
    # _phase_grasp tick from pitched(), read by _phase_lift

    # -- filled in by _phase_grasp, read by pitched() -----------------------
    ref: dict = None  # per-side Cartesian reference, advanced every tick


def on_tick(ctx: CarryRoundContext, _s, dist) -> None:
    u = float(np.clip(1.0 - (dist - 0.10) / 0.90, 0.0, 1.0))
    ctx.ramp.push(ctx.q_free + (3 * u * u - 2 * u**3) * (ctx.q_pre17 - ctx.q_free))


def brake(ctx: CarryRoundContext) -> None:
    # pelvis speed under 0.06 m/s for 0.25 s, arms finishing their rise
    p = ctx.p
    t0, still = time.time(), None
    prev = p.pelvis_xy(p.obs())
    while time.time() - t0 < 6.0:
        ctx.ramp.push(ctx.q_pre17)
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(ctx.psi)), float(np.sin(ctx.psi)), 0.0))
        time.sleep(DT)
        cur = p.pelvis_xy(p.obs())
        v = float(np.linalg.norm(cur - prev)) / DT
        prev = cur
        still = time.time() if v > 0.06 else (still if still is not None else time.time())
        if still is not None and time.time() - still > 0.25 and time.time() - t0 > 1.0:
            break


def stand_metrics(ctx: CarryRoundContext, tag):
    from locomani.tasks.crate.bimanual_carry import STAND_LAT_BIAS, STAND_OFF, crate_frame, hand_txt

    p = ctx.p
    s = p.obs()
    centre, bars, n, y_hat, psi = crate_frame(p, s)
    pel = np.asarray(p.pelvis_xy(s))
    yaw_err = float(np.degrees(wrap(psi - p.pelvis_yaw(s))))
    stand_err = float(np.dot(pel - centre[:2], n)) + STAND_OFF
    lateral = float(np.dot(pel - centre[:2], y_hat)) - STAND_LAT_BIAS  # + = left of the biased line
    edge_clear = ctx.edge_x - float(pel[0])
    ctx.log(
        f"[stand{tag}] yaw residual {yaw_err:+.1f} deg, along-track {stand_err:+.3f} m, "
        f"lateral {lateral:+.3f} m, pelvis {edge_clear:.2f} m outside the edge  {hand_txt(p, s)}"
    )
    return s, yaw_err, stand_err, lateral, edge_clear


def pitched(ctx: CarryRoundContext, sd):
    from locomani.tasks.crate.bimanual_carry import PALM_PITCH_DEG, UP_ABOVE, rot_y

    # palm pitch follows the descent: level at the staging height, the
    # full PALM_PITCH_DEG at the bar, so the fingertips never lead the jaw
    # into the rim.
    u = float(np.clip(1.0 - (ctx.ref[sd][2] - ctx.bars[sd][2]) / UP_ABOVE, 0.0, 1.0))
    return ctx.R_stand @ rot_y(np.radians(PALM_PITCH_DEG * u))


def _phase_reset(ctx: CarryRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_carry import OBJ, STAND_OFF, hand_txt

    p, reset, meta, rng = ctx.p, ctx.reset, ctx.meta, ctx.rng
    ctx.surf = meta["surfaces"]["A"]
    ctx.hx = float(ctx.surf["flat_half_extent"][0])
    crate0 = np.asarray(meta["objects"][OBJ]["init_pos"], dtype=float)

    # --- reset: crate yaw jitter, robot spawn 1.0-1.3 m behind the stand ----
    crate_dyaw = 0.0  # bars parallel to the table edge (was +-15 deg)
    crate_dy = float(rng.uniform(-0.10, 0.10))
    n0 = np.array([1.0, 0.0])  # crate near wall faces -x
    stand0 = crate0[:2] - n0 * STAND_OFF
    spawn_back = float(rng.uniform(1.0, 1.3))
    spawn_lat = float(rng.uniform(-0.15, 0.15))
    spawn_xy = stand0 - n0 * spawn_back + np.array([0.0, spawn_lat])
    robot_nom = np.asarray(meta["robot_start"]["xy"], dtype=float)
    psi0 = float(np.arctan2(*(crate0[:2] - spawn_xy)[::-1])) + float(rng.uniform(-0.17, 0.17))
    ctx.m.update(crate_dyaw_deg=np.degrees(crate_dyaw), crate_dy=crate_dy, spawn_back=spawn_back, spawn_lat=spawn_lat)

    # reset() takes an ABSOLUTE yaw (privileged_pub: qpos quat := yaw_quat),
    # so the crate's scene yaw (90 deg, handles at +-y) is the base, not zero.
    # The first pilot passed the jitter alone: the crate came up handles-fore,
    # the stand point landed beside the table and every walk-in timed out.
    w, x, y, z = meta["objects"][OBJ]["init_quat"]
    crate_yaw0 = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)  # never carry a closed jaw into the walk-in
    reset.pin_arms()
    reset.set_ghost()
    reset.reset(
        objects={OBJ: {"dxy": [0.0, crate_dy], "yaw": crate_yaw0 + crate_dyaw}},
        robot={"dxy": [float(v) for v in (spawn_xy - robot_nom)], "yaw": psi0},
    )
    time.sleep(0.3)
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.start_control()
    p._gym_started = True
    upper_ref.shared().wait(3.0)
    p.cmd.heading_offset = float(p.pelvis_yaw(p.obs()))
    fv0 = (float(np.cos(psi0)), float(np.sin(psi0)), 0.0)
    p.cmd.hold(3.0, mode=LocomotionMode.IDLE, facing=fv0)
    ctx.s = p.obs()
    if ctx.s["fallen"]:
        ctx.m["why"] = "[stand] did not stabilise after reset"
        return ctx.m
    reset.start_video(ctx.video_label)
    ctx.log(f"[reset] {hand_txt(p, p.obs())}")
    ctx.t_video = time.time()
    return None


def _phase_approach(ctx: CarryRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_carry import (
        APPR_SPEED,
        EDGE_FLOOR,
        LAT_TOL,
        OFFSET,
        STAND_LAT_BIAS,
        STAND_OFF,
        STAND_TOL,
        UP_ABOVE,
        UP_SHORT,
        WAYPOINT_BACK,
        WAYPOINT_TOL,
        BimanualArms,
        crate_frame,
        hand_txt,
        rot_z,
    )

    p, reset, log = ctx.p, ctx.reset, ctx.log
    s = ctx.s

    # --- A+B: walk in facing the crate, both arms rising to the pre-grasp ----
    centre, bars, n, y_hat, psi = crate_frame(p, s)
    ctx.centre, ctx.bars, ctx.n, ctx.y_hat, ctx.psi = centre, bars, n, y_hat, psi
    ctx.stand_xy = centre[:2] - n * STAND_OFF + y_hat * STAND_LAT_BIAS
    stand_xy = ctx.stand_xy
    ctx.edge_x = float(ctx.surf["xy"][0]) - ctx.hx
    edge_x = ctx.edge_x
    ctx.m["stand_target"] = [float(v) for v in stand_xy]
    q_meas = upper_body_from_mj(s["joints"])
    ctx.q_free = upper_ref.shared().upper17()
    ctx.q_free = q_meas if ctx.q_free is None else ctx.q_free
    q_free = ctx.q_free
    ctx.ramp = upper_ref.UpperRamp(p.cmd, q_free)
    ramp = ctx.ramp
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    # pre-grasp: both palms at the staging points, expressed for the FINAL stand
    ctx.arms = BimanualArms(p)
    arms = ctx.arms
    ctx.R_stand = rot_z(psi)
    R_stand = ctx.R_stand
    ctx.rot_w = {sd: R_stand for sd in ("left", "right")}  # fingers level while walking in
    up_w = {sd: bars[sd] + np.array([*(-n * UP_SHORT), UP_ABOVE]) for sd in ("left", "right")}
    # IK in the pelvis frame the robot WILL have at the stand: same yaw as psi,
    # pelvis over stand_xy at the standing height it has now.
    s_stand = dict(s)
    s_stand["pelvis"] = dict(
        s["pelvis"],
        pos=[float(stand_xy[0]), float(stand_xy[1]), float(s["pelvis"]["pos"][2])],
        quat=[float(np.cos(psi / 2)), 0.0, 0.0, float(np.sin(psi / 2))],
    )
    # Seed the pre-grasp IK from the zero posture, NOT the measured arms:
    # from the hanging arms the DLS lands in a straight-elbow, wrist-rolled
    # branch (elbow -6 deg, wrist roll 80) that the policy tracks with a 20 deg
    # gap, and the servo inherits it (pilot 6: right hand stuck 2.3 cm out).
    # From zero it finds the comfortable branch (elbow 43, wrist roll 7,
    # 61 deg from the nearest limit), and every later solve warm-starts there.
    arms.arm.q = np.zeros(29)
    q_pre = arms.solve(s_stand, up_w, ctx.rot_w, iters=200)
    ctx.q_pre17 = upper_body_from_mj(q_pre)
    q_pre17 = ctx.q_pre17
    for sd in ("left", "right"):
        pos, R = arms.arm.ik.fk(q_pre, sd)
        g = np.asarray(s_stand["pelvis"]["pos"]) + Primitives._rot(s_stand["pelvis"]["quat"]) @ (pos + R @ OFFSET)
        log(
            f"[pre-grasp] {sd}: ik residual {np.linalg.norm(g - up_w[sd]) * 100:.1f} cm, "
            f"arm q {np.round(np.degrees(q_pre[ARM_MJ[sd]]), 0).tolist()}"
        )

    # Walk in facing the crate; speed by distance remaining; stop on the edge
    # floor or at the stand. walk_to alone scattered the stand by -18..+14 cm
    # (pilot 3): its stop_dist has no notion of the table, and the carry after
    # the last frame put the feet under the edge.
    t0, why = time.time(), "timeout"
    best_along, best_t = None, time.time()
    way_xy = stand_xy - n * WAYPOINT_BACK
    leg = "waypoint"
    ctx.m["waypoint"] = [float(v) for v in way_xy]
    while time.time() - t0 < 40.0:
        s = p.obs()
        if s["fallen"]:
            ctx.m["why"] = "[stand] fell during the walk-in"
            return ctx.m
        pel = np.asarray(p.pelvis_xy(s))
        to_stand = stand_xy - pel
        along = float(np.dot(to_stand, n))  # + = still short of the stand
        dist = float(np.linalg.norm(to_stand))
        on_tick(ctx, s, dist)
        if leg == "waypoint":
            to_way = way_xy - pel
            if float(np.dot(to_way, n)) <= 0.03 or float(np.linalg.norm(to_way)) < WAYPOINT_TOL:
                leg = "in"
                log(
                    f"[walk] waypoint reached: lateral {float(np.dot(pel - stand_xy, y_hat)) * 100:+.1f} cm, "
                    f"{float(np.dot(to_stand, n)) * 100:.0f} cm to the stand"
                )
        edge_clear = edge_x - float(pel[0])
        if along <= 0.04 or dist < STAND_TOL:
            why = "at the stand"
            break
        if edge_clear <= EDGE_FLOOR:
            why = "edge floor"
            break
        # no progress for 3 s: the gait has parked; take the stand it gives
        # (the band check below decides whether it is usable)
        if best_along is None or along < best_along - 0.015:
            best_along, best_t = along, time.time()
        elif time.time() - best_t > 3.0:
            why = "stalled"
            break
        centre_now, _, n_now, _, _ = crate_frame(p, s)
        look = centre_now[:2] - pel
        b = float(np.arctan2(look[1], look[0]))
        speed = APPR_SPEED[0] if along > 0.35 else APPR_SPEED[1] if along > 0.15 else APPR_SPEED[2]
        if leg == "waypoint":
            mv = to_way / max(float(np.linalg.norm(to_way)), 1e-3)
        else:
            mv = to_stand / max(dist, 1e-3)
            b = psi  # straight in, square to the bars
        if int((time.time() - t0) / 2.0) != int((time.time() - t0 - DT) / 2.0):
            log(
                f"[walk] t={time.time() - t0:4.1f} pelvis {np.round(pel, 2).tolist()} dist {dist:.2f} "
                f"along {along:+.2f} edge {edge_clear:.2f} speed {speed} facing_err "
                f"{np.degrees(wrap(b - p.pelvis_yaw(s))):+.0f} deg ramp_gap {np.degrees(np.abs(ramp.q - q_pre17).max()):.0f} deg  {hand_txt(p, s)}"
            )
        p.cmd.frame(
            mode=LocomotionMode.SLOW_WALK,
            facing=(float(np.cos(b)), float(np.sin(b)), 0.0),
            movement=(float(mv[0]), float(mv[1]), 0.0),
            speed=speed,
        )
        time.sleep(DT)
    ctx.m["approach_exit"] = why
    if why == "timeout":
        ctx.m["why"] = "[stand] walk-in timed out"
        return ctx.m

    brake(ctx)
    s, yaw_err, stand_err, lateral, edge_clear = stand_metrics(ctx, "")
    ctx.m.update(yaw_err_deg=yaw_err, stand_err=stand_err, lateral=lateral, edge_clear=edge_clear)
    ramp.settle()
    s = p.obs()
    centre, bars, n, y_hat, psi = crate_frame(p, s)
    ctx.s, ctx.centre, ctx.bars, ctx.n, ctx.y_hat, ctx.psi = s, centre, bars, n, y_hat, psi
    if abs(yaw_err) > 20 or stand_err < -0.08 or stand_err > 0.10:
        ctx.m["why"] = "[stand] outside the reach band"
        return ctx.m
    if abs(lateral) > LAT_TOL:
        ctx.m["why"] = f"[stand] lateral {lateral * 100:+.1f} cm off the centre line"
        return ctx.m
    return None


def _phase_grasp(ctx: CarryRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_carry import (
        ACROSS_INT_CAP,
        ACROSS_INT_RATE,
        ACROSS_TOL,
        ADVANCE_PER_TICK,
        ALONG_TOL,
        DUMP_DIR,
        HEIGHT_TOL,
        HEIGHT_TOL_BELOW,
        LEAD_CAP,
        OBJ,
        OVERDRIVE,
        OVERDRIVE_CAP,
        SERVO_TIMEOUT,
        STALL_HEIGHT_TOL,
        STALL_MM,
        STALL_S,
        UP_ABOVE,
        UP_SHORT,
        WRIST_PITCH_COMP,
        WRIST_PITCH_COMP_CAP,
        crate_frame,
        hand_txt,
        rot_z,
    )

    p, log, rnd = ctx.p, ctx.log, ctx.rnd
    arms = ctx.arms
    centre, bars, n, y_hat, psi = ctx.centre, ctx.bars, ctx.n, ctx.y_hat, ctx.psi

    # --- B+C: one servo: lean + reach + pitch, both hands, to the bars -------
    arms.start(q17=ctx.ramp.q)
    R_stand = rot_z(psi)
    ctx.R_stand = R_stand
    ctx.ref = {sd: bars[sd] + np.array([*(-n * UP_SHORT), UP_ABOVE]) for sd in ("left", "right")}
    ref = ctx.ref

    crate_start = centre.copy()
    t0 = time.time()
    ready_since = None
    err_hist = []  # (t, errs) for the stall test
    close_trigger = None
    q_last = None
    across_int = {"left": 0.0, "right": 0.0}
    phase = "servo"
    while time.time() - t0 < SERVO_TIMEOUT:
        s = p.obs()
        if s["fallen"]:
            ctx.m["why"] = "[grasp] fell during the reach"
            arms.stop()
            return ctx.m
        centre, bars, n, y_hat, psi = crate_frame(p, s)
        ctx.centre, ctx.bars, ctx.n, ctx.y_hat, ctx.psi = centre, bars, n, y_hat, psi
        if float(np.linalg.norm(centre[:2] - crate_start[:2])) > 0.05:
            ctx.m["why"] = "[grasp] crate pushed > 5 cm before closing"
            arms.stop()
            return ctx.m
        errs, ready = {}, True
        for sd in ("left", "right"):
            g, _ = arms.grip_point(s, sd)
            e = g - bars[sd]
            along, across, height = float(e[:2] @ n), float(e[:2] @ y_hat), float(e[2])
            errs[sd] = (along, across, height)  # logged as [along, across, height]
            ready &= abs(along) <= ALONG_TOL and abs(across) <= ACROSS_TOL and -HEIGHT_TOL_BELOW <= height <= HEIGHT_TOL
            # reference: advance toward the bar, capped ahead of the measured point
            want = bars[sd]
            step = want - ref[sd]
            d = float(np.linalg.norm(step))
            if d > ADVANCE_PER_TICK:
                step *= ADVANCE_PER_TICK / d
            nxt = ref[sd] + step
            lead = nxt - g
            ld = float(np.linalg.norm(lead))
            if ld > LEAD_CAP:
                nxt = g + lead * (LEAD_CAP / ld)
            ref[sd] = nxt
        # The planner channel tracks the reference with a 1-3 cm residual and
        # the reference alone never closes it (pilot 3: 9-50 mm left over after
        # 25 s). Same remedy as the single-hand servo: overdrive the command by
        # a fraction of the CURRENT residual, from the current measurement only
        # (accumulating corrections diverges on transients), capped.
        cmd = {}
        for sd in ("left", "right"):
            g, _ = arms.grip_point(s, sd)
            corr = np.clip(OVERDRIVE * (ref[sd] - g), -OVERDRIVE_CAP, OVERDRIVE_CAP)
            # slow integrator across the bar: the proportional overdrive leaves the
            # policy's constant outward offset in place, an integrator does not
            across_int[sd] = float(
                np.clip(across_int[sd] - ACROSS_INT_RATE * errs[sd][1], -ACROSS_INT_CAP, ACROSS_INT_CAP)
            )
            cmd[sd] = ref[sd] + corr + np.array([*(y_hat * across_int[sd]), 0.0])
        rot_w = {sd: pitched(ctx, sd) for sd in ("left", "right")}
        ctx.rot_w = rot_w
        q = arms.solve(s, cmd, rot_w, from_measured=True, rot_weight=0.25)
        # Joint-space compensation on the wrist pitch alone: the policy delivers
        # the right wrist pitch 20 deg short of the command (diagnostic dump
        # 2026-09-07: cmd 49 / meas 29, every other joint within 10), which is
        # 5 cm at the pad and the Cartesian overdrive cannot buy it back.
        # Pure proportional on the tracking error of the IK solution itself.
        # Using the previous (already compensated) command as the reference
        # made this an integrator that sat on its +-15 deg cap; pilots 25-28
        # then showed the right wrist 50-60 deg past the command and the hand
        # 13 cm high.
        for sd in ("left", "right"):
            j = ARM_MJ[sd][5]
            gap = float(q[j] - s["joints"][j])
            q[j] += float(np.clip(WRIST_PITCH_COMP * gap, -WRIST_PITCH_COMP_CAP, WRIST_PITCH_COMP_CAP))
        q_last = q.copy()
        arms.push(q)
        if int((time.time() - t0) / 1.0) != int((time.time() - t0 - DT) / 1.0):
            from locomani.primitives.g1_joints import MUJOCO_JOINT_NAMES

            dj = np.degrees(np.asarray(s["joints"])[15:29] - q[15:29])
            worst = np.argsort(-np.abs(dj))[:3]
            gap = float(np.abs(dj).max())
            gap_txt = " ".join(f"{MUJOCO_JOINT_NAMES[15 + i].replace('_joint', '')}{dj[i]:+.0f}" for i in worst)
            log(
                f"[servo] t={time.time() - t0:4.1f} err_mm L {np.round(np.array(errs['left']) * 1000, 0).tolist()} "
                f"R {np.round(np.array(errs['right']) * 1000, 0).tolist()} ref-lead L "
                f"{np.linalg.norm(ref['left'] - arms.grip_point(s, 'left')[0]) * 100:.1f} R "
                f"{np.linalg.norm(ref['right'] - arms.grip_point(s, 'right')[0]) * 100:.1f} cm  meas-cmd {gap_txt}"
                f"  touch {s['objects'][OBJ].get('touch_links')} across_int L {across_int['left'] * 100:+.1f} R {across_int['right'] * 100:+.1f} cm  {hand_txt(p, s)}"
            )
        now = time.time()
        err_hist.append((now, {sd: np.array(errs[sd]) for sd in errs}))
        err_hist = [(tt, ee) for tt, ee in err_hist if now - tt <= STALL_S]
        stalled = False
        if now - err_hist[0][0] >= STALL_S - DT and all(
            abs(errs[sd][0]) <= ALONG_TOL
            and abs(errs[sd][1]) <= ACROSS_TOL
            and -HEIGHT_TOL_BELOW <= errs[sd][2] <= STALL_HEIGHT_TOL
            for sd in errs
        ):
            span = max(float(np.max(np.abs(ee[sd] - err_hist[-1][1][sd]))) for _, ee in err_hist for sd in errs)
            stalled = span <= STALL_MM
        if ready or stalled:
            ready_since = ready_since or now
            if now - ready_since >= 0.1:
                phase = "close"
                close_trigger = "region" if ready else "stall"
                break
        else:
            ready_since = None
    ctx.s = s
    ctx.m["capture_err_mm"] = {sd: [round(v * 1000, 1) for v in errs[sd]] for sd in errs}
    # diagnostic dump of the servo end state: measured joints + last command +
    # crate pose + contact links, for an offline FK / contact reconstruction
    if DUMP_DIR:
        dump = {
            "round": rnd,
            "joints_meas": [float(v) for v in s["joints"]],
            "q_cmd": [float(v) for v in arms.arm.q],
            "pelvis": s["pelvis"],
            "palms": s["palms"],
            "crate": s["objects"][OBJ],
            "bars": {sd: [float(v) for v in bars[sd]] for sd in bars},
            "ref": {sd: [float(v) for v in ref[sd]] for sd in ref},
            "n": [float(v) for v in n],
            "y_hat": [float(v) for v in y_hat],
            "errs_mm": ctx.m["capture_err_mm"],
        }
        path = os.path.join(DUMP_DIR, f"servo_end_round{rnd}.json")
        with open(path, "w") as f:
            json.dump(dump, f, indent=1)
        log(f"[dump] {path}  touch_links {s['objects'][OBJ].get('touch_links')}")
    ctx.m["waist_pitch_cmd_deg"] = float(np.degrees(arms.arm.q[14]))
    ctx.m["waist_pitch_meas_deg"] = float(np.degrees(s["joints"][14]))
    if phase != "close":
        ctx.m["why"] = f"[grasp] capture regions never both satisfied: {ctx.m['capture_err_mm']}"
        arms.stop()
        return ctx.m
    ctx.m["close_trigger"] = close_trigger
    log(
        f"[grasp] both in region ({close_trigger}) after {time.time() - t0:.1f} s: {ctx.m['capture_err_mm']} "
        f"waist pitch cmd {ctx.m['waist_pitch_cmd_deg']:.0f} / meas {ctx.m['waist_pitch_meas_deg']:.0f} deg"
    )
    return None
