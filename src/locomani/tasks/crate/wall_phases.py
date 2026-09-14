"""`run_round`'s reset/stand/hover phases, split out of bimanual_wall.py
(refactor step 4b family 5, pure extraction; decision 9's file-size cap).
Every statement here is unchanged from the original inline body -- only the
packaging moved: each closure's captured locals became a field on
`WallRoundContext`, and the five helpers used from more than one place
(`solve_for`, `mirror29`, `measure_stand`, `stand_fault`, `servo_tick`)
became plain functions taking `ctx` instead of nested closures (`mirror29`
and `stand_fault` capture nothing from the enclosing scope, so they stay
plain functions with no `ctx` parameter at all). The last three phases
(`_phase_descend`, `_phase_close`, `_phase_lift`) live in
`wall_phases_descend.py` instead -- all six here would not fit this file's
own 600-line cap.

This module must not import `locomani.tasks.crate.bimanual_wall` at its own
top level: `bimanual_wall.py` imports names from here (`run_round`'s new
home is `bimanual_wall.py` itself, calling these phases in sequence), so a
module-level import in the other direction would make the two files a
load-order-sensitive circular import. Nothing here needs anything of
`bimanual_wall.py`'s at import time anyway -- every module constant
(`STAND_OFF_RANGE`, `WALK`, ...) and helper (`wall_frame`, `wall_point`,
`arms_offset`, `hand_txt`, `rot_z`, `rot_y`) it needs is a deferred,
in-function import instead (none of them is ever overridden by assignment
anywhere in the repo, so a plain import -- not a module-qualified read -- is
correct and sufficient here, unlike vla_round.py's family-3 knobs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from locomani.backends.sonic import upper_ref
from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode, StreamMode
from locomani.backends.sonic.primitives import DT, Primitives, upper_body_from_mj, wrap
from locomani.backends.sonic.streamed_arm import reanchor
from locomani.primitives.g1_joints import ARM_MJ as ARM_IDX
from locomani.sim.privileged_pub import EpisodeControlClient
from locomani.tasks.crate.config import CrateTaskConfig

# Every phase function below returns `dict | None`: `None` to continue to the
# next phase, a (failure-populated) `m` dict to end the round with that
# result.


@dataclass
class WallRoundContext:
    """Everything one `run_round` call threads through its phases.

    The first block is `run_round`'s own parameters, unchanged (`m` is
    pre-populated by the caller before phase 1 runs); the rest is state a
    phase computes for a LATER phase (or a converted closure) to use -- each
    field names the closure/phase that used to capture it.
    """

    # -- run_round's own parameters ---------------------------------------
    p: Primitives
    reset: EpisodeControlClient
    meta: dict
    rnd: int
    rng: np.random.Generator
    log: Callable
    video_label: str
    m: dict
    config: CrateTaskConfig = field(default_factory=CrateTaskConfig)

    # -- filled in by _phase_reset, read by later phases -------------------
    surf: dict = None  # meta["surfaces"]["A"], read by _phase_stand
    hx: float = None  # surf's flat half-extent, read by _phase_stand
    so: float = None  # sampled stand-off, read by _phase_stand's non-WALK branch
    stand_xy: np.ndarray = None  # target pelvis xy (overwritten if WALK)
    s: dict = None  # latest p.obs(), read/reassigned by every later phase

    # -- filled in by _phase_stand, read by _phase_hover / solve_for /
    #    measure_stand / servo_tick -------------------------------------------
    c: np.ndarray = None  # wall_frame's centre; reassigned by _phase_hover
    n: np.ndarray = None  # wall_frame's outward normal; reassigned by _phase_hover
    y_hat: np.ndarray = None  # wall_frame's left-hand axis; reassigned by _phase_hover
    psi: float = None  # wall_frame's heading; reassigned by _phase_hover,
    # read by solve_for()/servo_tick()
    sign: float = None  # wall_frame's near-face sign; reassigned by _phase_hover
    R: np.ndarray = None  # wall_frame's crate rotation; reassigned by _phase_hover
    q_free: np.ndarray = None  # the free-standing upper-body pose
    ramp: Any = None  # upper_ref.UpperRamp
    arms: Any = None  # BimanualArms, read by every later phase, and by
    # solve_for()/servo_tick()
    t_video: float = None  # video start wall-clock, read by _phase_lift

    # -- filled in by _phase_hover, read by _phase_descend / _phase_lift ----
    depth: float = None  # grasp depth along the wall
    crate_start: np.ndarray = None  # crate centre at the hover's end, read by _phase_descend
    ref: dict = None  # per-side Cartesian reference, read/advanced by _phase_descend
    rot_w: dict = None  # per-side palm rotation, read by servo_tick() / _phase_lift


def solve_for(ctx: WallRoundContext, stand, dz, pitch_deg, depth_):
    """Arm pose for the pelvis over `stand` facing psi, pads `dz` above the bar tops."""
    from locomani.tasks.crate.bimanual_wall import ACROSS_BIAS, rot_y, rot_z, wall_point

    s, psi, arms = ctx.s, ctx.psi, ctx.arms
    rot = {sd: rot_z(psi) @ rot_y(np.radians(pitch_deg)) for sd in ("left", "right")}
    tg = {sd: wall_point(s, sd, depth_, dz, ACROSS_BIAS) for sd in ("left", "right")}
    s_stand = dict(s)
    s_stand["pelvis"] = dict(
        s["pelvis"],
        pos=[float(stand[0]), float(stand[1]), float(s["pelvis"]["pos"][2])],
        quat=[float(np.cos(psi / 2)), 0.0, 0.0, float(np.sin(psi / 2))],
    )
    arms.arm.q = np.zeros(29)
    return arms.solve(s_stand, tg, rot, iters=200)


def mirror29(left_deg):
    q = np.zeros(29)
    q[ARM_IDX["left"]] = np.radians(left_deg)
    q[ARM_IDX["right"]] = np.radians(left_deg) * np.array([1, -1, -1, 1, -1, 1, -1], dtype=float)
    return q


def measure_stand(ctx: WallRoundContext, s):
    from locomani.tasks.crate.bimanual_wall import wall_frame, wall_point

    p = ctx.p
    c_, n_, y_, psi_, _, _ = wall_frame(s)
    pel_ = np.asarray(p.pelvis_xy(s))
    near_ = wall_point(s, "left", 0.0, 0.0)
    so_ = float(np.dot(near_[:2] - pel_, n_))
    lat_ = float(np.dot(pel_ - c_[:2], y_))
    yaw_ = float(np.degrees(wrap(psi_ - p.pelvis_yaw(s))))
    return so_, lat_, yaw_


def stand_fault(so_, lat_, yaw_):
    from locomani.tasks.crate.bimanual_wall import STAND_YAW_MAX_DEG, WALK_LAT_TOL, WALK_STAND_OFF_BAND

    if abs(yaw_) > STAND_YAW_MAX_DEG:
        return f"yaw residual {yaw_:+.1f} deg"
    if not (WALK_STAND_OFF_BAND[0] <= so_ <= WALK_STAND_OFF_BAND[1]):
        return f"stand-off {so_:.2f} m outside the reach band"
    if abs(lat_) > WALK_LAT_TOL:
        return f"lateral {lat_ * 100:+.1f} cm off the centre line"
    return ""


def servo_tick(ctx: WallRoundContext, s, ref):
    from locomani.tasks.crate.bimanual_wall import (
        ACROSS_OD,
        ACROSS_OD_CAP,
        LEAD_CAP,
        OVERDRIVE,
        OVERDRIVE_CAP,
        wall_frame,
    )

    arms, rot_w = ctx.arms, ctx.rot_w
    cmd = {}
    _, n_, y_, _, _, _ = wall_frame(s)
    yh = np.array([y_[0], y_[1], 0.0])
    for sd in ("left", "right"):
        g, _ = arms.grip_point(s, sd)
        e = ref[sd] - g
        ld = float(np.linalg.norm(e))
        if ld > LEAD_CAP:  # rate-limit: lead capped, not the target
            e = e * (LEAD_CAP / ld)
        e_across = float(e @ yh)
        corr = np.clip(OVERDRIVE * (e - e_across * yh), -OVERDRIVE_CAP, OVERDRIVE_CAP)
        corr += yh * float(np.clip(ACROSS_OD * e_across, -ACROSS_OD_CAP, ACROSS_OD_CAP))
        cmd[sd] = g + e + corr
    arms.push(arms.solve(s, cmd, rot_w, from_measured=True, ghost_targets=ref))


def _phase_reset(ctx: WallRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_wall import (
        CRATE_SETBACK_LEVELS,
        CRATE_YAW_DELTA,
        CRATE_YAW_LEVELS_DEG,
        NEAR_X,
        OBJ,
        SPAWN_BACK,
        STAND_LAT_RANGE,
        STAND_OFF_RANGE,
        STAND_OFF_WALK,
        STAND_YAW_DEG,
        WALK,
        WALK_LAT_BIAS,
        WALK_SPAWN_BACK,
        WALK_SPAWN_LAT,
    )

    p, reset, meta, rng = ctx.p, ctx.reset, ctx.meta, ctx.rng
    ctx.surf = meta["surfaces"]["A"]
    ctx.hx = float(ctx.surf["flat_half_extent"][0])
    edge_x = float(ctx.surf["xy"][0]) - ctx.hx
    crate0 = np.asarray(meta["objects"][OBJ]["init_pos"], dtype=float)
    w, x, y, z = meta["objects"][OBJ]["init_quat"]
    crate_yaw0 = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    # --- reset: crate rotated, near face on the table edge; robot on a sampled stand
    crate_dy = float(rng.uniform(*ctx.config.crate_dy_range_m))
    # Placement design (2026-09-10): the crate can sit `setback` in from the
    # edge and be yawed about its centre. The levels' product is walked in a
    # fixed shuffled order so a short pilot covers every corner; nothing
    # below assumes the crate is square to the table -- the stand is laid
    # out along the crate's own normal, and the walk / hover phases already
    # read the measured pose (wall_frame).
    setback, crate_yaw = 0.0, 0.0
    if CRATE_SETBACK_LEVELS or CRATE_YAW_LEVELS_DEG:
        combos = [(sb, yw) for yw in (CRATE_YAW_LEVELS_DEG or (0.0,)) for sb in (CRATE_SETBACK_LEVELS or (0.0,))]
        order = np.random.default_rng(1234).permutation(len(combos))
        setback, crate_yaw = combos[int(order[ctx.rnd % len(combos)])]
        crate_yaw = float(np.radians(crate_yaw))
    crate_x = edge_x + NEAR_X + float(setback)
    n_c = np.array([np.cos(crate_yaw), np.sin(crate_yaw)])  # robot -> crate, through the near face
    y_c = np.array([-n_c[1], n_c[0]])
    ctx.so = float(rng.uniform(*STAND_OFF_RANGE))
    so = ctx.so
    lat = float(rng.uniform(*STAND_LAT_RANGE))
    yaw = crate_yaw + float(np.radians(rng.uniform(-STAND_YAW_DEG, STAND_YAW_DEG)))
    ctx.stand_xy = np.array([crate_x, crate_dy]) - n_c * (NEAR_X + so) + y_c * lat
    stand_xy = ctx.stand_xy
    robot_nom = np.asarray(meta["robot_start"]["xy"], dtype=float)
    ctx.m.update(
        crate_dy=crate_dy,
        crate_setback=float(setback),
        crate_yaw_deg=float(np.degrees(crate_yaw)),
        stand_off=so,
        stand_lat=lat,
        stand_yaw_deg=float(np.degrees(yaw)),
    )
    p.cmd.reset_pose()
    p.cmd.set_hands(left=HAND_OPEN, right=HAND_OPEN)
    reset.pin_arms()
    reset.set_ghost()
    # The robot comes up SPAWN_BACK behind the stand: spawned on the stand, the
    # policy's start-up arm swing knocked the crate 84 deg round (wall pilot 1).
    # It is teleported onto the stand once the arms are up at the hover pose.
    if WALK:
        ctx.stand_xy = np.array([crate_x, crate_dy]) - n_c * (NEAR_X + STAND_OFF_WALK) + y_c * WALK_LAT_BIAS
        stand_xy = ctx.stand_xy
        spawn_back = float(rng.uniform(*WALK_SPAWN_BACK))
        spawn_lat = float(rng.uniform(-WALK_SPAWN_LAT, WALK_SPAWN_LAT))
        spawn_xy = stand_xy - n_c * spawn_back + y_c * spawn_lat
        yaw = float(np.arctan2(*(np.array([crate_x, crate_dy]) - spawn_xy)[::-1])) + float(rng.uniform(-0.17, 0.17))
        ctx.m.update(spawn_back=spawn_back, spawn_lat=spawn_lat, spawn_yaw_deg=float(np.degrees(yaw)))
    else:
        spawn_xy = stand_xy - n_c * SPAWN_BACK
    reset.reset(
        objects={OBJ: {"dxy": [float(crate_x - crate0[0]), crate_dy], "yaw": crate_yaw0 + CRATE_YAW_DELTA + crate_yaw}},
        robot={"dxy": [float(v) for v in (spawn_xy - robot_nom)], "yaw": yaw},
    )
    time.sleep(0.3)
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.start_control()
    p._gym_started = True
    upper_ref.shared().wait(3.0)
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
    # The deploy keeps its heading anchor across resets; setting heading_offset
    # from the pelvis yaw alone left a residual that the IDLE facing command
    # then turned into a real rotation, growing ~5 deg per round with
    # alternating sign (wall pilot 8: +2.6 -> -84 deg by round 19). A streamed
    # toggle re-anchors the deploy at the current yaw and records it.
    reanchor(p)
    fv0 = (float(np.cos(yaw)), float(np.sin(yaw)), 0.0)
    p.cmd.hold(1.5, mode=LocomotionMode.IDLE, facing=fv0)
    ctx.s = p.obs()
    if ctx.s["fallen"]:
        ctx.m["why"] = "[stand] did not stabilise after reset"
        return ctx.m
    return None


def _phase_stand(ctx: WallRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_wall import (
        APPR_SPEED,
        DEPTH_MAX,
        DEPTH_MIN,
        EDGE_FLOOR,
        HOVER_ABOVE,
        PALM_PITCH_DEG,
        REACH_SUM,
        SHOULDER_PITCH_MIN_DEG,
        STAND_TOL,
        WALK,
        WALK_ATTEMPTS,
        WALK_SIDE_LEFT_DEG,
        WAYPOINT_BACK,
        WAYPOINT_TOL,
        BimanualArms,
        hand_txt,
        wall_frame,
    )

    p, reset, log = ctx.p, ctx.reset, ctx.log
    s = ctx.s
    stand_xy, so = ctx.stand_xy, ctx.so

    # --- A: arms up, robot onto the stand (walk-in or teleport) ---------------
    c, n, y_hat, psi, sign, R = wall_frame(s)
    ctx.c, ctx.n, ctx.y_hat, ctx.psi, ctx.sign, ctx.R = c, n, y_hat, psi, sign, R
    edge_x = float(ctx.surf["xy"][0]) - ctx.hx
    ctx.q_free = upper_ref.shared().upper17()
    ctx.q_free = upper_body_from_mj(s["joints"]) if ctx.q_free is None else ctx.q_free
    q_free = ctx.q_free
    ctx.ramp = upper_ref.UpperRamp(p.cmd, q_free)
    ramp = ctx.ramp
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    ctx.arms = BimanualArms(p)
    from locomani.tasks.crate import bimanual_wall as _bw

    ctx.arms.ghost = _bw.GHOST_TRACE
    arms = ctx.arms
    for sd in ("left", "right"):  # no overhead branch (wall pilot 5: +65 cm)
        a = ARM_IDX[sd]
        arms.arm.ik.lo[a[0]] = np.radians(SHOULDER_PITCH_MIN_DEG)

    if WALK:
        reset.start_video(ctx.video_label)
        ctx.t_video = time.time()
        q_walk17 = upper_body_from_mj(mirror29(WALK_SIDE_LEFT_DEG))
        way_xy = stand_xy - n * WAYPOINT_BACK
        ctx.m["stand_target"] = [float(v) for v in stand_xy]
        ctx.m["stand_attempts"] = []
        for attempt in range(WALK_ATTEMPTS):
            t0, why = time.time(), "timeout"
            best_along, best_t = None, time.time()
            leg = "waypoint"
            while time.time() - t0 < 40.0:
                s = p.obs()
                if s["fallen"]:
                    ctx.m["why"] = "[stand] fell during the walk-in"
                    return ctx.m
                pel = np.asarray(p.pelvis_xy(s))
                to_stand = stand_xy - pel
                along = float(np.dot(to_stand, n))
                dist = float(np.linalg.norm(to_stand))
                ramp.push(q_walk17)
                if leg == "waypoint":
                    to_way = way_xy - pel
                    if float(np.dot(to_way, n)) <= 0.03 or float(np.linalg.norm(to_way)) < WAYPOINT_TOL:
                        leg = "in"
                        log(
                            f"[walk] waypoint reached: lateral {float(np.dot(pel - stand_xy, y_hat)) * 100:+.1f} cm, "
                            f"{along * 100:.0f} cm to the stand"
                        )
                edge_clear = edge_x - float(pel[0])
                if along <= 0.04 or dist < STAND_TOL:
                    why = "at the stand"
                    break
                if edge_clear <= EDGE_FLOOR:
                    why = "edge floor"
                    break
                if best_along is None or along < best_along - 0.015:
                    best_along, best_t = along, time.time()
                elif time.time() - best_t > 3.0:
                    why = "stalled"
                    break
                look = c[:2] - pel
                b = float(np.arctan2(look[1], look[0]))
                speed = APPR_SPEED[0] if along > 0.35 else APPR_SPEED[1] if along > 0.15 else APPR_SPEED[2]
                if leg == "waypoint":
                    mv = to_way / max(float(np.linalg.norm(to_way)), 1e-3)
                else:
                    mv = to_stand / max(dist, 1e-3)
                    b = psi
                if int((time.time() - t0) / 2.0) != int((time.time() - t0 - DT) / 2.0):
                    log(
                        f"[walk] t={time.time() - t0:4.1f} pelvis {np.round(pel, 2).tolist()} dist {dist:.2f} along {along:+.2f} "
                        f"edge {edge_clear:.2f} speed {speed} facing_err {np.degrees(wrap(b - p.pelvis_yaw(s))):+.0f} deg  {hand_txt(p, s)}"
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
            # brake: pelvis under 0.06 m/s for 0.25 s
            t0, still = time.time(), None
            prev = p.pelvis_xy(p.obs())
            while time.time() - t0 < 6.0:
                ramp.push(q_walk17)
                p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
                time.sleep(DT)
                cur = p.pelvis_xy(p.obs())
                v = float(np.linalg.norm(cur - prev)) / DT
                prev = cur
                still = time.time() if v > 0.06 else (still if still is not None else time.time())
                if still is not None and time.time() - still > 0.25 and time.time() - t0 > 1.0:
                    break
            ramp.settle()
            s = p.obs()
            if s["fallen"]:
                ctx.m["why"] = "[stand] fell at the stand"
                return ctx.m
            so_a, lat_a, yaw_a = measure_stand(ctx, s)
            fault = stand_fault(so_a, lat_a, yaw_a)
            ctx.m["stand_attempts"].append(dict(stand_off=so_a, lateral=lat_a, yaw_deg=yaw_a, fault=fault))
            if not fault:
                break
            log(
                f"[stand] attempt {attempt + 1}: {fault} (stand-off {so_a:.3f}, lateral {lat_a:+.3f}, yaw {yaw_a:+.1f})"
            )
            if attempt == WALK_ATTEMPTS - 1:
                ctx.m["why"] = f"[stand] {fault} after {WALK_ATTEMPTS} walk-ins"
                return ctx.m
            # back up to the waypoint, facing the crate, and walk in again
            t0 = time.time()
            while time.time() - t0 < 8.0:
                s = p.obs()
                if s["fallen"]:
                    ctx.m["why"] = "[stand] fell backing up"
                    return ctx.m
                pel = np.asarray(p.pelvis_xy(s))
                if float(np.dot(stand_xy - pel, n)) >= WAYPOINT_BACK:
                    break
                ramp.push(q_walk17)
                p.cmd.frame(
                    mode=LocomotionMode.SLOW_WALK,
                    facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0),
                    movement=(float(-n[0]), float(-n[1]), 0.0),
                    speed=APPR_SPEED[2],
                )
                time.sleep(DT)
            log(
                f"[walk] backed up to {float(np.dot(stand_xy - np.asarray(p.pelvis_xy(p.obs())), n)):.2f} m for attempt {attempt + 2}"
            )
        ctx.m["stand_attempt_n"] = len(ctx.m["stand_attempts"])
    else:
        depth = float(np.clip(REACH_SUM - so, DEPTH_MIN, DEPTH_MAX))  # from the sampled stand; re-measured below
        q_hover = solve_for(ctx, stand_xy, HOVER_ABOVE, PALM_PITCH_DEG, depth)
        q_h17 = upper_body_from_mj(q_hover)
        n_ramp = int(2.0 / DT)
        for k in range(1, n_ramp + 1):
            u = k / n_ramp
            ramp.push(q_free + (3 * u * u - 2 * u**3) * (q_h17 - q_free))
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
            time.sleep(DT)
        ramp.settle()
        # teleport the standing robot onto the stand, arms already up
        reset.align_robot_xy(stand_xy)
        for _ in range(int(1.0 / DT)):
            ramp.push(q_h17)
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
            time.sleep(DT)
        ramp.settle()
        s = p.obs()
        if s["fallen"]:
            ctx.m["why"] = "[stand] fell after the teleport"
            return ctx.m
        reset.start_video(ctx.video_label)
        ctx.t_video = time.time()
    ctx.s = s
    return None


def _phase_hover(ctx: WallRoundContext) -> dict | None:
    from locomani.tasks.crate.bimanual_wall import (
        ACROSS_BIAS,
        DEPTH_MAX,
        DEPTH_MIN,
        HOVER_ABOVE,
        HOVER_S,
        PALM_PITCH_DEG,
        REACH_OUT,
        REACH_OVER,
        REACH_PITCH0_DEG,
        REACH_POSTURE_W,
        REACH_S,
        REACH_SUM,
        REACH_UP,
        STAND_YAW_MAX_DEG,
        WALK,
        arms_offset,
        hand_txt,
        rot_y,
        rot_z,
        wall_frame,
        wall_point,
    )

    p, log = ctx.p, ctx.log
    s = ctx.s
    arms, ramp = ctx.arms, ctx.ramp

    # measured stand -> depth along the wall
    c, n, y_hat, psi, sign, R = wall_frame(s)
    ctx.c, ctx.n, ctx.y_hat, ctx.psi, ctx.sign, ctx.R = c, n, y_hat, psi, sign, R
    pel = np.asarray(p.pelvis_xy(s))
    near = wall_point(s, "left", 0.0, 0.0)
    so_meas = float(np.dot(near[:2] - pel, n))
    lat_meas = float(np.dot(pel - c[:2], y_hat))
    yaw_err = float(np.degrees(wrap(psi - p.pelvis_yaw(s))))
    ctx.depth = float(np.clip(REACH_SUM - so_meas, DEPTH_MIN, DEPTH_MAX))
    depth = ctx.depth
    ctx.m.update(stand_off_meas=so_meas, stand_lat_meas=lat_meas, yaw_err_deg=yaw_err, depth=depth)
    log(
        f"[stand] stand-off {so_meas:.3f} m, lateral {lat_meas:+.3f} m, yaw {yaw_err:+.1f} deg -> depth {depth:.2f}  {hand_txt(p, s)}"
    )
    if abs(yaw_err) > STAND_YAW_MAX_DEG:
        ctx.m["why"] = f"[stand] yaw residual {yaw_err:+.1f} deg after the {'walk-in' if WALK else 'teleport'}"
        return ctx.m
    # hover pose for the stand we actually have (from zeros: the validated branch)
    q_hover = solve_for(ctx, pel, HOVER_ABOVE, PALM_PITCH_DEG, depth)
    for sd in ("left", "right"):
        pos, Rk = arms.arm.ik.fk(q_hover, sd)
        g = np.asarray(s["pelvis"]["pos"]) + Primitives._rot(s["pelvis"]["quat"]) @ (pos + Rk @ arms_offset())
        log(
            f"[hover] {sd}: ik residual {np.linalg.norm(g - wall_point(s, sd, depth, HOVER_ABOVE, ACROSS_BIAS)) * 100:.1f} cm, "
            f"arm q {np.round(np.degrees(q_hover[ARM_IDX[sd]]), 0).tolist()}"
        )
    ctx.rot_w = {sd: rot_z(psi) @ rot_y(np.radians(PALM_PITCH_DEG)) for sd in ("left", "right")}
    rot_w = ctx.rot_w
    if not WALK:
        # teleport path keeps the old joint ramp onto the hover pose
        q_h17 = upper_body_from_mj(q_hover)
        q_from = ramp.q.copy()
        n_ramp = int(1.0 / DT)
        for k in range(1, n_ramp + 1):
            u = k / n_ramp
            ramp.push(q_from + (3 * u * u - 2 * u**3) * (q_h17 - q_from))
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
            time.sleep(DT)
        ramp.settle()
        arms.start(q17=ramp.q)
        arms.arm.q = q_hover.copy()
    else:
        # one continuous reach: side pads -> Bezier -> hover, streamed as IK
        # joint targets, warm-started from the previous command with the
        # posture prior toward q_hover
        arms.start(q17=ramp.q)
        arms.arm.q = np.asarray(s["joints"], dtype=float).copy()
        c, n, y_hat, psi, sign, R = wall_frame(s)
        ctx.c, ctx.n, ctx.y_hat, ctx.psi, ctx.sign, ctx.R = c, n, y_hat, psi, sign, R
        p0 = {sd: arms.grip_point(s, sd)[0].copy() for sd in ("left", "right")}
        p3 = {sd: wall_point(s, sd, depth, HOVER_ABOVE, ACROSS_BIAS) for sd in ("left", "right")}
        out_w = {"left": np.array([y_hat[0], y_hat[1], 0.0]), "right": -np.array([y_hat[0], y_hat[1], 0.0])}
        p1 = {sd: p0[sd] + out_w[sd] * REACH_OUT + np.array([0.0, 0.0, REACH_UP]) for sd in p0}
        p2 = {sd: p3[sd] + np.array([0.0, 0.0, REACH_OVER]) for sd in p3}
        n_reach = int(REACH_S / DT)
        t0 = time.time()
        last_log = 0.0
        for k in range(1, n_reach + 1):
            u = k / n_reach
            s = p.obs()
            if s["fallen"]:
                ctx.m["why"] = "[reach] fell"
                arms.stop()
                return ctx.m
            tg = {
                sd: (1 - u) ** 3 * p0[sd] + 3 * (1 - u) ** 2 * u * p1[sd] + 3 * (1 - u) * u * u * p2[sd] + u**3 * p3[sd]
                for sd in p0
            }
            pitch = REACH_PITCH0_DEG + (PALM_PITCH_DEG - REACH_PITCH0_DEG) * u
            rot_u = {sd: rot_z(psi) @ rot_y(np.radians(pitch)) for sd in ("left", "right")}
            arms.push(arms.solve(s, tg, rot_u, iters=60, posture=q_hover, posture_weight=REACH_POSTURE_W))
            if time.time() - last_log > 0.5:
                last_log = time.time()
                res = {sd: float(np.linalg.norm(arms.grip_point(s, sd)[0] - tg[sd])) for sd in tg}
                log(
                    f"[reach] u={u:.2f} lag L {res['left'] * 100:.1f} R {res['right'] * 100:.1f} cm  "
                    f"arm q L {np.round(np.degrees(arms.arm.q[ARM_IDX['left']]), 0).tolist()}"
                )
            time.sleep(DT)
        ctx.m["reach_s"] = round(time.time() - t0, 1)

    # No hover hold: the descent starts as the reach ends (the descend servo
    # takes up whatever lag the policy still carries). The teleport path
    # keeps a short settle so the old A/B numbers stay comparable.
    ctx.crate_start = c.copy()
    s = p.obs()
    ctx.ref = {sd: wall_point(s, sd, depth, HOVER_ABOVE, ACROSS_BIAS) for sd in ("left", "right")}
    ref = ctx.ref
    if not WALK:
        t0 = time.time()
        while time.time() - t0 < HOVER_S:
            s = p.obs()
            arms.push(q_hover)
            time.sleep(DT)
    res = {sd: float(np.linalg.norm(arms.grip_point(s, sd)[0] - ref[sd])) for sd in ref}
    from locomani.tasks.crate.bimanual_wall import errs_of

    log(
        f"[hover] residual L {res['left'] * 100:.1f} R {res['right'] * 100:.1f} cm "
        f"errs L {np.round(errs_of(s, arms, 'left', depth) * 1000, 0).tolist()} R {np.round(errs_of(s, arms, 'right', depth) * 1000, 0).tolist()}"
    )
    ctx.m["hover_residual_mm"] = {sd: round(v * 1000, 1) for sd, v in res.items()}
    ctx.s = s
    return None
