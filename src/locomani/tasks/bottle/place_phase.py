"""The place phase of the pick-and-place task (docs/handoff/pick_place_plan.md §2, steps 4-9).

Runs after `_phase_grasp_and_verify`'s lift when `RoundContext.goal_key` names a
container: raise the held object above the container's rim, turn on the spot
until the container sits in the active hand's placing band, carry the object
over the container's centre, descend, open, retreat, and verify. The grasp
policy's own `PlannerArm` is reused for the arm (same IK, same comfort limits,
same frozen-upper-body channel), and the legs are driven the way
`stream_reach._creep_to_band` drives them -- so the arm and the base never
move in the same phase, which is also what the bottle data already shows.

Design rules this phase keeps (plan §0): the container is checked against the
ego frame at every tick it matters, every move is a bounded-speed reference
the palm follows (no jumps), there is no hover-and-wait before the release
(open on arrival or on the descent timeout), and the object is controlled --
the servo integrates on the OBJECT's measured position, so the grip offset
never enters the placement error.

The same phase puts an object down on a surface (`put_down_single`, docs/handoff/put-down-single-20261003.md)
when the context carries `surface` and `spot` instead of a container: the geometry object (`_Geometry` /
`_SurfaceGeometry`) owns everything that differs -- heights, a squat for a low spot, when the descent has
landed, and the verdict -- and the sub-phases call only that.

Pure geometry helpers (`in_container`, `resting_on`, `squat_drop`, `turn_command`, `step_reference`) are
kept free of the robot so tests/test_place_phase.py can exercise them.
"""

from __future__ import annotations

import time

import numpy as np

from locomani.backends.sonic.agent_server import HAND_OPEN, LocomotionMode
from locomani.backends.sonic.primitives import DT, Primitives

VIEW_X, VIEW_Y = 0.80, 0.90  # the same ego-frame gates the spawn/stand checks use
# Per-tick joint step for the place servo: 0.5 rad/s. The upper-body slew the
# grasp ramp allows (upper_ref.MAX_STEP, 2 deg/tick = 1.7 rad/s) lets a
# redundant IK solution (waist + 7 arm joints) jump between postures tick to
# tick, and the held can slipped 3.8 cm during the carry-over on the first
# pilot round that used it (2026-09-10). A 0.15 m/s reference on a 0.35 m
# lever needs ~0.45 rad/s, so this cap costs nothing on the intended motion.
JOINT_STEP = 0.010
# Surface put-down (docs/handoff/put-down-single-20261003.md). Offline IK 2026-10-03 (ArmIK with PlannerArm's waist
# limits, palm rotation held): standing, the right palm reaches 0.30-0.35 m ahead down to about the pelvis height
# (2 cm short at 5 cm below it), so a lower release needs the planner squat.
REACH_BELOW_PELVIS = 0.0  # m under the pelvis the standing palm still reaches at the placing band's depth
SQUAT_MARGIN = 0.03  # m of extra squat beyond the computed shortfall
SQUAT_MAX = 0.20  # m; streamed_arm.SQUAT_DROP_MAX, the deepest calibrated planner squat
SURFACE_TOL = 0.01  # m: descent error that opens the hand over a surface (the container's place_tol is 3 cm)
LANDED_BAND = 0.04  # m above the resting height inside which a descent that stopped sinking has landed
LANDED_STALL_S = 0.3  # s without sinking by more than LANDED_SINK that counts as stopped
LANDED_SINK = 0.002  # m
# Surface band correction (put_down_single v3): the container turn walks at SLOW_WALK 0.20, inside the v1.1 noisy
# zone (dead <= 0.15), and stood still for 8 s with the spot 0.16 m too far (2026-10-03, acceptance v2 round 4).
# The nudge primitive's measured pulse instead: 0.35 for 0.5 s (0-12 cm), brake, re-measure (skills/catalogue/locomotion.Nudge).
# Surface put-down: the held object tilts in the simple grasp's hold (acceptance v5: 12 -> 27 deg from the carry to the
# landing in a round that then tipped over, 6-8 deg in one that stood). The carry-over and the descent turn the palm
# by the object's measured tilt (at their start, capped) so it comes down upright.
LEVEL_MAX = 0.6  # rad of palm correction at most
BAND_PULSE_SPEED = 0.35
BAND_PULSE_S = 0.5  # every pulse: 0.3 s pulses moved the base 0-1.5 cm, 4 of 4 (acceptance v3 round 6)
BAND_PULSES = 6  # nudge's default
# The canister is held ~0.17-0.2 m ahead of the palm: with the spot at the band's near edge (x 0.27) the palm had to
# come back to 0.08 m and the carry-over timed out (acceptance v2 round 12), so a surface spot stays this far out.
# The far side keeps most of the band: the pulses stop anywhere inside, and 0.41 is where five of them left a spot
# that the narrower 0.30-0.40 of v3 refused (acceptance v3 round 6).
SURFACE_BAND_X = (0.30, 0.42)


# --- pure helpers -----------------------------------------------------------


def yaw_of_quat(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def in_container(obj_pos, goal_pos, goal_quat, inner_half, floor_z, rim_z) -> tuple[bool, str]:
    """Is the object's centre inside the container's inner box (container's
    own frame, so a yawed container is judged in its own axes)?"""
    d = np.asarray(obj_pos[:2], float) - np.asarray(goal_pos[:2], float)
    yaw = yaw_of_quat(goal_quat)
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx, ly = c * d[0] - s * d[1], s * d[0] + c * d[1]
    hx, hy = inner_half
    if abs(lx) > hx or abs(ly) > hy:
        return False, f"outside the container footprint by ({abs(lx) - hx:+.2f}, {abs(ly) - hy:+.2f}) m"
    z = float(obj_pos[2])
    if z < floor_z - 0.01:
        return False, "below the container floor"
    if z > rim_z:
        return False, f"centre {z - rim_z:.2f} m above the rim (resting on it?)"
    return True, ""


def quat_matrix(quat_wxyz) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def tilt_from(quat_wxyz, up_body) -> float:
    """Degrees between the body axis `up_body` (unit, body frame) turned by the quaternion and world +z."""
    r = quat_matrix(quat_wxyz)
    return float(np.degrees(np.arccos(np.clip((r @ np.asarray(up_body, float))[2], -1.0, 1.0))))


def level_rotation(up_world, max_angle: float = LEVEL_MAX) -> np.ndarray:
    """World rotation that turns the vector `up_world` toward +z, by at most `max_angle` rad (Rodrigues)."""
    u = np.asarray(up_world, float) / max(float(np.linalg.norm(up_world)), 1e-9)
    axis = np.cross(u, [0.0, 0.0, 1.0])
    sin = float(np.linalg.norm(axis))
    if sin < 1e-9:
        return np.eye(3)
    angle = min(float(np.arctan2(sin, u[2])), max_angle)
    k = axis / sin
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * kx + (1.0 - np.cos(angle)) * (kx @ kx)


def resting_on(obj_pos, rest_z, tilt_deg, surf_xy, surf_half, surf_yaw, *, below=0.02, above=0.03, tilt_max=20.0):
    """Does the object stand on the surface: its origin at the resting height (underside on the top, `below`..
    `above`), its centre over the footprint (the surface's own frame), no more than `tilt_max` degrees off upright?"""
    dz = float(obj_pos[2]) - float(rest_z)
    if dz < -below:
        return False, f"{-dz * 100:.1f} cm below its resting height on the surface (fell off?)"
    if dz > above:
        return False, f"{dz * 100:.1f} cm above its resting height (hanging, or resting on something else)"
    d = np.asarray(obj_pos[:2], float) - np.asarray(surf_xy[:2], float)
    c, s = np.cos(-surf_yaw), np.sin(-surf_yaw)
    over = np.abs([c * d[0] - s * d[1], s * d[0] + c * d[1]]) - np.asarray(surf_half[:2], float)
    if np.any(over > 0):
        return False, f"centre {over.max() * 100:.1f} cm outside the surface footprint"
    if tilt_deg > tilt_max:
        return False, f"tipped {tilt_deg:.0f} deg off upright"
    return True, ""


def squat_drop(palm_z, obj_z, release_z, pelvis_z, *, reach_below=REACH_BELOW_PELVIS, margin=SQUAT_MARGIN) -> float:
    """Pelvis drop a release at `release_z` (object origin, world) needs. The palm keeps its height above the object
    while it is carried down, and standing it reaches only `reach_below` under the pelvis; 0 when no squat is needed."""
    need = (float(pelvis_z) - reach_below) - (float(palm_z) + float(release_z) - float(obj_z))
    return float(np.clip(need + margin, 0.0, SQUAT_MAX)) if need > 0 else 0.0


class Landing:
    """`until` for the surface descent: the object is at its resting height, or it stopped sinking near it
    (resting on the top a little higher than the box says) -- either way the hand should open now."""

    def __init__(self, key: str, rest_z: float, tol: float = SURFACE_TOL):
        self.key, self.rest_z, self.tol = key, float(rest_z), tol
        self.low, self.since = np.inf, None
        self.how = ""

    def __call__(self, s) -> bool:
        z, t = float(s["objects"][self.key]["pos"][2]), float(s["t"])
        if z <= self.rest_z + self.tol:
            self.how = "at_rest_height"
            return True
        if z < self.low - LANDED_SINK:
            self.low, self.since = z, t
            return False
        if self.since is not None and z <= self.rest_z + LANDED_BAND and t - self.since >= LANDED_STALL_S:
            self.how = "stopped_sinking"
            return True
        return False


def turn_command(gx: float, gy: float, yaw: float, band_x, band_y, speed: float = 0.20):
    """Legs-only command that brings a pelvis-frame goal point (gx, gy) into
    the placing band: (in_band, movement, facing). Yaw steers the goal's
    bearing toward the band's lateral centre; depth is closed by walking
    along the (new) facing. Same shape as `_creep_to_band`'s lateral branch."""
    in_x = band_x[0] <= gx <= band_x[1]
    in_y = band_y[0] <= gy <= band_y[1]
    if in_x and in_y:
        return True, (0.0, 0.0, 0.0), (float(np.cos(yaw)), float(np.sin(yaw)), 0.0), 0.0
    y_mid = 0.5 * (band_y[0] + band_y[1])
    az = float(np.arctan2(gy, gx))
    desired = float(np.arctan2(y_mid, max(gx, 0.10)))
    face = yaw + float(np.clip(az - desired, -0.35, 0.35))
    direction = 0.0 if in_x else (1.0 if gx > band_x[1] else -1.0)
    mv = (direction * float(np.cos(face)), direction * float(np.sin(face)), 0.0)
    return False, mv, (float(np.cos(face)), float(np.sin(face)), 0.0), speed


def step_reference(ref, target, v: float, dt: float) -> np.ndarray:
    """Move a reference toward a target at most `v * dt`: the visible speed profile."""
    ref, target = np.asarray(ref, float), np.asarray(target, float)
    d = target - ref
    n = float(np.linalg.norm(d))
    if n <= v * dt or n < 1e-9:
        return target.copy()
    return ref + d * (v * dt / n)


def ego_visible(cam) -> bool:
    return cam is not None and abs(cam[0]) <= VIEW_X and abs(cam[1]) <= VIEW_Y


# --- robot-side pieces --------------------------------------------------------


def _reseed(p, arm, s) -> None:
    """Take the grasp policy's arm back after `finish_cleanup` stopped it."""
    arm.q = np.asarray(s["joints"], dtype=float).copy()
    arm._q_prev = arm.q.copy()
    arm.next_t = time.time()
    arm.mode = LocomotionMode.IDLE
    arm.height = -1.0
    arm._face = p._facing(s)


class _Watch:
    """Per-tick checks shared by every sub-phase: fall, slip, goal visibility."""

    def __init__(self, ctx, hold_ref, goal_key):
        from locomani.tasks.bottle.grasp_gym import palm_frame_offset

        self.ctx, self.hold_ref, self.goal = ctx, hold_ref, goal_key
        self.offset = palm_frame_offset
        self.seen = self.total = 0
        # 10 Hz drift trace (stage, t, palm-frame drift) into the round's metrics,
        # so a slip can be read as creep vs a step and placed in its sub-phase.
        self.trace = ctx.execution.metrics.setdefault("place_trace", [])
        self.t0 = time.monotonic()
        self.next_sample = 0.0

    def __call__(self, s, holding: bool = True) -> str | None:
        ctx = self.ctx
        if s["fallen"]:
            return "fell"
        self.total += 1
        if self.goal is not None:  # a surface put-down has no goal object in the state
            self.seen += int(ego_visible(s["objects"][self.goal].get("cam")))
        if holding:
            drift = float(np.linalg.norm(self.offset(s, ctx.obj_key, ctx.side) - self.hold_ref))
            now = time.monotonic() - self.t0
            if now >= self.next_sample:
                self.next_sample = now + 0.1
                stage = ctx.execution.events[-1].get("stage") if ctx.execution.events else None
                self.trace.append([stage, round(now, 2), round(drift, 4)])
            if drift >= ctx.palm_drift_max:
                return f"[carry] object slipped {drift * 100:.1f} cm in the palm frame"
        return None

    @property
    def visibility(self) -> float | None:
        """Share of ticks the container was in the ego view; None for a surface put-down (no goal object)."""
        return None if self.goal is None else self.seen / max(self.total, 1)


def _servo(
    ctx,
    arm,
    watch,
    target_fn,
    *,
    control: str,
    v: float,
    tol: float,
    timeout: float,
    holding=True,
    until=None,
    level=None,
):
    """Drive the controlled point (object centre or palm) to a world target
    at bounded speed, IK on the grasp policy's arm, base in IDLE. `until(state)`
    true ends it early as arrived (a put-down that has landed). `level` (the held
    object's resting up axis, body frame) turns the held palm rotation by the
    object's tilt at the start, so the object is carried upright. Every call
    appends a summary to the round's `place_servo` metrics (`_ServoLog`).
    Returns (ok, why, xy_error)."""
    p, side, key = ctx.p, ctx.side, ctx.obj_key
    s = p.obs()
    rot_hold = Primitives._rot(s["palms"][side]["quat"])  # keep the object upright: hold the palm rotation
    log = _ServoLog(ctx, s, control, target_fn)
    if level is not None and "quat" in s["objects"][key]:
        corr = level_rotation(quat_matrix(s["objects"][key]["quat"]) @ np.asarray(level, float))
        rot_hold = corr @ rot_hold
        log.row["level_deg"] = round(float(np.degrees(np.arccos(np.clip((np.trace(corr) - 1) / 2, -1, 1)))), 1)
    ref = None
    stall = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = p.obs()
        why = watch(s, holding)
        if why:
            return log.done(s, False, why, np.inf)
        palm = np.asarray(s["palms"][side]["pos"], float)
        ctrl = np.asarray(s["objects"][key]["pos"], float) if control == "object" else palm
        tgt = np.asarray(target_fn(s), float)
        ref = ctrl.copy() if ref is None else step_reference(ref, tgt, v, DT)
        err = tgt - ctrl
        if float(np.linalg.norm(err)) < tol or (until is not None and until(s)):
            return log.done(s, True, "", float(np.linalg.norm(err[:2])))
        # The palm carries the controlled point: ask for the reference's
        # offset, capped so a lagging arm cannot wind the request up.
        lead = ref - ctrl
        n = float(np.linalg.norm(lead))
        if n > 0.06:
            lead *= 0.06 / n
        aim_pf = p.world_to_root_full(palm + lead, s)
        rot_pf = Primitives._rot(s["pelvis"]["quat"]).T @ rot_hold
        q, res, _ = arm.ik.solve(side, aim_pf, rot_pf, q_mj29=arm.q, use_waist=True, iters=30)
        log.tick(res)
        stall = stall + 1 if res > 0.04 else 0
        if stall > 25:
            return log.done(s, False, f"target unreachable by {res * 100:.1f} cm", float(np.linalg.norm(err[:2])))
        q = arm.q + np.clip(np.asarray(q) - arm.q, -JOINT_STEP, JOINT_STEP)
        arm.q = q
        arm._push(q)
    s = p.obs()
    ctrl = np.asarray(s["objects"][key]["pos"], float) if control == "object" else np.asarray(s["palms"][side]["pos"])
    err = np.asarray(target_fn(s), float) - ctrl
    return log.done(
        s, False, f"timed out {float(np.linalg.norm(err)) * 100:.1f} cm short", float(np.linalg.norm(err[:2]))
    )


class _ServoLog:
    """One `_servo` call in the round's `place_servo` metrics: the controlled point and the target in the pelvis
    frame at the start and the end, the IK residual, how it ended. A carry-over that times out leaves no other
    trace of where the arm went (2026-10-03, put_down_single acceptance rounds 1 and 6)."""

    def __init__(self, ctx, s, control: str, target_fn):
        self.ctx, self.control, self.target_fn = ctx, control, target_fn
        ex = ctx.execution
        self.row = {"stage": ex.events[-1].get("stage") if ex.events else None, "control": control}
        self.row["start"] = self._frame(s)
        self.ticks, self.res_max, self.res_last = 0, 0.0, None
        self.t0 = time.monotonic()

    def _frame(self, s) -> dict:
        p, side, key = self.ctx.p, self.ctx.side, self.ctx.obj_key
        point = s["objects"][key]["pos"] if self.control == "object" else s["palms"][side]["pos"]
        pf = lambda w: [round(float(v), 3) for v in p.world_to_root_full(np.asarray(w, float), s)]  # noqa: E731
        up = _up_axis((self.ctx.meta["objects"].get(key) or {}).get("init_quat") or (1.0, 0.0, 0.0, 0.0))
        return {
            "ctrl_pf": pf(point),
            "target_pf": pf(self.target_fn(s)),
            "palm_pf": pf(s["palms"][side]["pos"]),
            "pelvis_z": round(float(s["pelvis"]["pos"][2]), 3),
            # the held object's tilt from the way it stood at the start: a canister hanging 9 cm under the palm
            # tipped over after the release (acceptance v4 round 6)
            "obj_tilt_deg": round(tilt_from(s["objects"][key]["quat"], up), 1) if "quat" in s["objects"][key] else None,
        }

    def tick(self, res: float) -> None:
        self.ticks += 1
        self.res_last = round(float(res), 4)
        self.res_max = max(self.res_max, float(res))

    def done(self, s, ok: bool, why: str, xy_error: float):
        self.row.update(
            ok=bool(ok),
            why=why,
            end=self._frame(s),
            ticks=self.ticks,
            seconds=round(time.monotonic() - self.t0, 2),
            res_max=round(self.res_max, 4),
            res_last=self.res_last,
        )
        self.ctx.execution.metrics.setdefault("place_servo", []).append(self.row)
        return ok, why, xy_error


def _turn_to_band(ctx, watch, goal_pos, cfg) -> tuple[bool, str]:
    """Legs only: rotate/shuffle until the container is in the placing band, then brake."""
    p = ctx.p
    t0 = time.monotonic()
    ready_since = None
    while time.monotonic() - t0 < cfg.turn_timeout:
        s = p.obs()
        why = watch(s)
        if why:
            return False, why
        g = np.asarray(p.world_to_root(goal_pos, s), float)
        yaw = p.pelvis_yaw(s)
        in_band, mv, face, speed = turn_command(float(g[0]), float(g[1]), yaw, cfg.band_x, cfg.band_y)
        if in_band:
            ready_since = ready_since or time.monotonic()
            if time.monotonic() - ready_since >= 0.2:
                break
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=face)
        else:
            ready_since = None
            p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=speed, movement=mv, facing=face)
        time.sleep(DT)
    else:
        s = p.obs()
        g = p.world_to_root(goal_pos, s)
        return (
            False,
            f"[turn] container not in the placing band after {cfg.turn_timeout:.0f} s (x {g[0]:.2f} y {g[1]:+.2f})",
        )
    # Brake: the same stop the grasp policy uses before it squats.
    t0 = time.monotonic()
    still = None
    while time.monotonic() - t0 < 3.0:
        s = p.obs()
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(s))
        speed = float(np.linalg.norm(s["pelvis"]["vel"][:2]))
        still = (still or time.monotonic()) if speed < 0.06 else None
        if still and time.monotonic() - still >= 0.25:
            g = p.world_to_root(goal_pos, s)
            p.log(f"[place] turn done: container at pelvis-frame x {g[0]:.2f} y {g[1]:+.2f}")
            return True, ""
        time.sleep(DT)
    return False, "[turn] base did not stop after the turn"


def surface_band(band_x, band_y, surface_x=SURFACE_BAND_X):
    """The placing band for a surface spot: depth clipped to `surface_x`, lateral unchanged."""
    return (max(band_x[0], surface_x[0]), min(band_x[1], surface_x[1])), tuple(band_y)


def band_residual(gx: float, gy: float, band_x, band_y) -> np.ndarray:
    """Pelvis-frame (dx, dy) that would bring the point (gx, gy) to the band's centre; zero inside the band."""
    if band_x[0] <= gx <= band_x[1] and band_y[0] <= gy <= band_y[1]:
        return np.zeros(2)
    return np.array([gx - 0.5 * (band_x[0] + band_x[1]), gy - 0.5 * (band_y[0] + band_y[1])])


def _pulse_to_band(ctx, arm, watch, goal_pos, cfg):
    """Surface put-down: bring the spot into the placing band. After the base stands still the spot is re-aimed at
    the free point on the surface nearest the band's centre (`ctx.respot(state, aim_xy)`, when given); only when that
    point is still outside the band, timed SLOW_WALK pulses on the arm's own frames (facing held, the frozen
    reference re-sent every tick) walk toward it, each followed by a brake and a fresh measurement and re-aim.
    -> (ok, why, goal_pos): the spot the rest of the put-down aims at."""
    from locomani.backends.sonic.agent_server import LocomotionMode

    respot = getattr(ctx, "respot", None)
    goal_pos = np.asarray(goal_pos, float).copy()

    p, ex = ctx.p, ctx.execution
    pulses = ex.metrics.setdefault("place_pulses", [])
    band_x, band_y = surface_band(cfg.band_x, cfg.band_y)
    # Measure from a standstill: the loaded approach's last walk correction may still be coasting (the spot moved
    # 0.23 m between its stance and the first pulse, acceptance v3 round 6).
    why = _brake_on_arm(p, arm, watch)
    if why:
        return False, why, goal_pos
    centre = np.array([0.5 * (band_x[0] + band_x[1]), 0.5 * (band_y[0] + band_y[1])])
    for _k in range(BAND_PULSES + 1):
        s = p.obs()
        if respot is not None:
            yaw0 = float(p.pelvis_yaw(s))
            c0, s0 = np.cos(yaw0), np.sin(yaw0)
            aim = np.asarray(p.pelvis_xy(s), float)[:2] + np.array(
                [c0 * centre[0] - s0 * centre[1], s0 * centre[0] + c0 * centre[1]]
            )
            new = respot(s, aim)
            if new is not None:
                goal_pos[:2] = new
        g = np.asarray(p.world_to_root(goal_pos, s), float)
        res = band_residual(float(g[0]), float(g[1]), band_x, band_y)
        d = float(np.linalg.norm(res))
        if d == 0.0:
            p.log(f"[place] spot in the band: pelvis-frame x {g[0]:.2f} y {g[1]:+.2f} after {len(pulses)} pulse(s)")
            ex.metrics["place_spot_final"] = [round(float(v), 4) for v in goal_pos[:2]]
            return True, "", goal_pos
        if len(pulses) == BAND_PULSES:
            break
        yaw = float(p.pelvis_yaw(s))
        c, sn = np.cos(yaw), np.sin(yaw)
        direction = np.array([c * res[0] - sn * res[1], sn * res[0] + c * res[1]]) / d
        dur = BAND_PULSE_S
        arm.mode = LocomotionMode.SLOW_WALK
        try:
            for _ in range(max(int(round(dur / DT)), 1)):
                why = watch(p.obs())
                if why:
                    return False, why, goal_pos
                arm._push(arm.q, movement=(float(direction[0]), float(direction[1]), 0.0), speed=BAND_PULSE_SPEED)
        finally:
            arm.mode = LocomotionMode.IDLE
        why = _brake_on_arm(p, arm, watch)
        if why:
            return False, why, goal_pos
        g_after = np.asarray(p.world_to_root(goal_pos, p.obs()), float)
        pulses.append(
            [round(float(g[0]), 3), round(float(g[1]), 3), round(dur, 2), [round(float(v), 3) for v in g_after[:2]]]
        )
    return (
        False,
        f"[turn] spot not in the placing band after {len(pulses)} pulses (x {g[0]:.2f} y {g[1]:+.2f})",
        goal_pos,
    )


def _brake_on_arm(p, arm, watch, timeout: float = 3.0) -> str | None:
    """Hold the arm's frames in IDLE until the pelvis is still for 0.25 s (or `timeout`); the watch's verdict if it trips."""
    still = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = p.obs()
        why = watch(s)
        if why:
            return why
        still = (still or time.monotonic()) if float(np.linalg.norm(s["pelvis"]["vel"][:2])) < 0.06 else None
        if still is not None and time.monotonic() - still >= 0.25:
            return None
        arm._push(arm.q)
    return None


def _release(arm, side: str, seconds: float) -> None:
    start = np.asarray(arm.hands[side], dtype=np.float32).copy()
    n = max(int(round(seconds / DT)), 1)
    for k in range(1, n + 1):
        arm.hands[side] = start + (k / n) * (HAND_OPEN - start)
        arm._push(arm.q)


class _Geometry:
    """World-frame numbers the sub-phases share, taken once from the lift-end snapshot: into a container."""

    landed = None  # the container descent ends on its tolerance or its timeout
    target_name, spot_name = "container", "container centre"  # in the failure / success log lines

    def __init__(self, ctx, s):
        cfg, key, goal = ctx.place_config, ctx.obj_key, ctx.goal_key
        self.goal_key = goal
        self.goal_pos = np.asarray(s["objects"][goal]["pos"], float)
        self.rim_z = float(ctx.surf_a["top"] + ctx.meta["objects"][goal]["height"])
        self.floor_z = float(ctx.surf_a["top"] + 0.01)
        half_h = 0.5 * float(ctx.meta["objects"][key]["height"])
        self.hover_z = self.rim_z + half_h + cfg.hover_clearance
        self.release_z = self.rim_z + half_h + cfg.release_clearance
        self.obj_xy0 = np.asarray(s["objects"][key]["pos"][:2], float)
        self.raise_z = self.hover_z

    def prepare(self, ctx, arm) -> dict:
        return {}

    def descend_tol(self, cfg) -> float:
        return cfg.place_tol

    def descend_timeout(self, cfg, distance: float) -> float:
        return cfg.descend_timeout

    def centre(self, s):
        return np.asarray(s["objects"][self.goal_key]["pos"], float)

    def verdict(self, ctx, s, obj) -> tuple[bool, str]:
        cfg = ctx.place_config
        goal_now = self.centre(s)
        moved = float(np.linalg.norm(goal_now[:2] - self.goal_pos[:2]))
        if moved > cfg.goal_move_max:
            return False, f"[place] container moved {moved * 100:.0f} cm"
        quat = s["objects"][self.goal_key]["quat"]
        inside, why = in_container(obj, goal_now, quat, cfg.goal_inner_half, self.floor_z, self.rim_z)
        return (True, "") if inside else (False, f"[place] missed: {why}")


def _up_axis(quat_wxyz) -> np.ndarray:
    """The body-frame axis the quaternion turns onto world +z (the third row of its rotation matrix)."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])


class _SurfaceGeometry:
    """The same numbers for a put-down on `ctx.surface` at `ctx.spot`. The resting height comes from the object's
    box along the axis that was up while it rested at the start (scene `init_quat`); the release sits `ctx.drop_m`
    above it. The carry-over keeps the height the object is carried at (never lower than `hover_clearance` above
    the rest), so the order is: over the spot standing, squat with the palm held, then straight down. The first
    version hovered low and squatted before the carry-over; that diagonal move from a folded carry pose after the
    squat timed out 2 of 2 times (2026-10-03, acceptance rounds 1 and 6), and the round that passed had carried
    the object with the arm stretched forward, where the same move is nearly vertical."""

    goal_key = None
    target_name = spot_name = "spot"

    def __init__(self, ctx, s):
        cfg, key, surf = ctx.place_config, ctx.obj_key, ctx.surface
        o = ctx.meta["objects"][key]
        box = o.get("box") or {}
        self.up_body = _up_axis(o.get("init_quat") or (1.0, 0.0, 0.0, 0.0))
        if box.get("half_extent") is not None:
            half_up = float(np.abs(self.up_body) @ np.asarray(box["half_extent"], float))
            centre_up = float(self.up_body @ np.asarray(box.get("centre") or (0.0, 0.0, 0.0), float))
        else:
            half_up, centre_up = 0.5 * float(o["height"]), 0.0
        self.surface = surf
        self.rest_z = float(surf["top"]) + half_up - centre_up
        self.goal_pos = np.array([float(ctx.spot[0]), float(ctx.spot[1]), float(surf["top"])])
        self.release_z = self.rest_z + float(ctx.drop_m)
        self.obj_xy0 = np.asarray(s["objects"][key]["pos"][:2], float)
        carried = float(s["objects"][key]["pos"][2])
        self.hover_z = max(self.rest_z + cfg.hover_clearance, carried)
        # already carried above the hover: no raise (it only spent ~1 s of planner IDLE, which drifts the base 8-24 cm/s)
        self.raise_z = None if carried >= self.rest_z + cfg.hover_clearance else self.hover_z
        self.landed = Landing(key, self.rest_z)

    def descend_timeout(self, cfg, distance: float) -> float:
        """Time for the whole move at half `v_descend` plus 2 s. `distance` is measured when the descent starts: the
        squat sets the base back 0.12-0.135 m, so the descent is a diagonal of about 0.4 m, and the object follows the
        reference at 0.05-0.07 m/s, not 0.10 (the 6 cm lead cap and the joint-step clip; 2026-10-03, acceptance v2:
        rounds 8 and 18 covered 0.30-0.32 m in a 6 s timeout, round 1 landed after 0.36 m in 5.0 s)."""
        return max(cfg.descend_timeout, distance / (0.5 * cfg.v_descend) + 2.0)

    def prepare(self, ctx, arm) -> dict:
        """Squat over the spot (after the carry-over, before the descent) when the release is lower than the
        standing hand reaches; `squat` re-solves the arm so the palm stays where it is while the body sinks."""
        s = ctx.p.obs()
        drop = squat_drop(
            s["palms"][ctx.side]["pos"][2], s["objects"][ctx.obj_key]["pos"][2], self.release_z, s["pelvis"]["pos"][2]
        )
        if drop <= 0.0:
            return {"squat_m": 0.0}
        ctx.execution.enter("place_squat", drop_m=drop)
        return {"squat_asked_m": drop, "squat_m": float(arm.squat(drop))}

    def descend_tol(self, cfg) -> float:
        return SURFACE_TOL

    def centre(self, s):
        return self.goal_pos

    def verdict(self, ctx, s, obj) -> tuple[bool, str]:
        surf = self.surface
        tilt = tilt_from(s["objects"][ctx.obj_key]["quat"], self.up_body)
        ctx.execution.metrics.setdefault("place", {})["tilt_deg"] = tilt
        ok, why = resting_on(obj, self.rest_z, tilt, surf["xy"], surf["half_extent"], float(surf.get("yaw", 0.0)))
        return (True, "") if ok else (False, f"[place] not standing on {surf['key']}: {why}")


def _place_move(ctx, arm, watch, g: _Geometry):
    """Raise, turn, carry over, descend. -> (ok, why, xy_error at the release point)."""
    p, ex, cfg = ctx.p, ctx.execution, ctx.place_config
    servo = dict(control="object", v=cfg.v_travel, timeout=cfg.carry_timeout)
    if g.raise_z is not None:
        ex.enter("place_raise")
        ok, why, _ = _servo(ctx, arm, watch, lambda _s: [*g.obj_xy0, g.raise_z], tol=0.02, **servo)
        if not ok:
            return False, why if why.startswith("[") else f"[carry] raise: {why}", np.inf
    ex.enter("place_turn")
    if g.landed is None:
        ok, why = _turn_to_band(ctx, watch, g.goal_pos, cfg)
    else:
        _reseed(p, arm, p.obs())
        ok, why, g.goal_pos = _pulse_to_band(ctx, arm, watch, g.goal_pos, cfg)
    if not ok:
        return False, why if why.startswith("[") else f"[turn] {why}", np.inf
    _reseed(p, arm, p.obs())
    ex.enter("place_carry")
    level = getattr(g, "up_body", None) if g.landed is not None else None
    ok, why, _ = _servo(
        ctx, arm, watch, lambda _s: [*g.goal_pos[:2], g.hover_z], tol=cfg.place_tol, level=level, **servo
    )
    if not ok:
        return False, why if why.startswith("[") else f"[place] carry-over: {why}", np.inf
    prep = ex.metrics["place_prep"] = g.prepare(ctx, arm)  # kept when the descent fails
    ex.enter("place_descend")
    left = float(
        np.linalg.norm(np.asarray(p.obs()["objects"][ctx.obj_key]["pos"], float) - [*g.goal_pos[:2], g.release_z])
    )
    servo.update(v=cfg.v_descend, timeout=g.descend_timeout(cfg, left))
    ok, why, xy_err = _servo(
        ctx,
        arm,
        watch,
        lambda _s: [*g.goal_pos[:2], g.release_z],
        tol=g.descend_tol(cfg),
        until=g.landed,
        level=level,
        **servo,
    )
    if not ok and why.startswith("["):
        return False, why, xy_err
    if not ok and xy_err > cfg.place_tol_late:
        return False, f"[place] not over the {g.target_name} at the descent timeout (xy {xy_err * 100:.1f} cm)", xy_err
    if not ok and g.landed is not None:  # opening above a surface drops the object: only a landed descent releases
        z = float(p.obs()["objects"][ctx.obj_key]["pos"][2])
        return False, f"[place] descent timed out {(z - g.release_z) * 100:.1f} cm above the release height", xy_err
    if not ok:
        p.log(f"[place] descent timed out with xy {xy_err * 100:.1f} cm -- releasing anyway")
    ex.metrics["place"] = dict(
        xy_error=float(xy_err), released_on_time=bool(ok), goal_visibility=watch.visibility, **prep
    )
    if g.landed is not None:
        ex.metrics["place"]["landed"] = g.landed.how
    return True, "", xy_err


def _place_finish(ctx, arm, watch, g: _Geometry):
    """Release, retreat (lift, then back off), terminal hold and the verdict."""
    p, ex, cfg, key, side = ctx.p, ctx.execution, ctx.place_config, ctx.obj_key, ctx.side
    ex.enter("place_release")
    _release(arm, side, cfg.release_s)
    arm.hold(0.4)
    s = p.obs()
    ex.enter("place_retreat")
    palm0 = np.asarray(s["palms"][side]["pos"], float)
    lift = lambda _s: palm0 + [0.0, 0.0, cfg.retreat_up]  # noqa: E731
    ok, why, _ = _servo(ctx, arm, watch, lift, control="palm", v=cfg.v_descend, tol=0.02, timeout=3.0, holding=False)
    if not ok and why.startswith("["):
        return False, why
    if not ok:
        p.log(f"[place] retreat lift {why}")
    arm.stop()
    s = p.obs()
    # Reference for the sweep check is taken AFTER the hand has risen clear: a can
    # that tips as it lands is the release's doing, not the retreat's.
    obj_at_release = np.asarray(s["objects"][key]["pos"], float)
    back_tgt = np.asarray(p.pelvis_xy(s)) - cfg.back_off * np.array(p._facing(s)[:2])
    if not p.walk_to(back_tgt, stop_dist=0.08, timeout=8.0, face_target=False):
        return False, "[retreat] fell while backing away from the table"
    ex.metrics["training_tail"] = dict(end_seq=int(p.obs()["seq"]), reason="post_place_hold_verification")
    ex.enter("verify_terminal_hold")
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=p._facing(p.obs()))
    s = p.obs()
    obj = np.asarray(s["objects"][key]["pos"], float)
    swept = float(np.linalg.norm(obj[:2] - obj_at_release[:2]))
    if swept > 0.05:
        return False, f"[retreat] object moved {swept * 100:.0f} cm after the release"
    if s["objects"][key]["touching"]:
        return False, f"[place] hand still touching the object ({', '.join(s['objects'][key]['touching'])})"
    ok, why = g.verdict(ctx, s, obj)
    if not ok:
        return False, why
    off = float(np.linalg.norm(obj[:2] - g.centre(s)[:2]))
    ex.metrics["place"]["final_offset"] = off
    ex.metrics["goal_visibility"] = watch.visibility
    seen = "" if watch.visibility is None else f", goal in view {watch.visibility * 100:.0f}%"
    p.log(f"[place] OK: object {off * 100:.1f} cm from the {g.spot_name}{seen}")
    return True, ""


def _phase_place(ctx):
    """Steps 4-9 of the plan, into `ctx.goal_key` or onto `ctx.surface` at `ctx.spot`. Returns (ok, why)."""
    from locomani.tasks.bottle.grasp_gym import palm_frame_offset

    arm = getattr(ctx.coordinated, "arm", None)
    if arm is None:
        return False, "[gym] place needs the grasp policy's arm (ready_grasp_baseline)"
    s = ctx.p.obs()
    _reseed(ctx.p, arm, s)
    watch = _Watch(ctx, palm_frame_offset(s, ctx.obj_key, ctx.side), ctx.goal_key)
    g = _SurfaceGeometry(ctx, s) if getattr(ctx, "surface", None) is not None else _Geometry(ctx, s)
    ok, why, _ = _place_move(ctx, arm, watch, g)
    if not ok:
        return False, why
    return _place_finish(ctx, arm, watch, g)
