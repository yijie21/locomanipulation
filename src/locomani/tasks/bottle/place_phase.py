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

Pure geometry helpers (`in_container`, `turn_command`, `step_reference`) are
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
    def visibility(self) -> float:
        return self.seen / max(self.total, 1)


def _servo(ctx, arm, watch, target_fn, *, control: str, v: float, tol: float, timeout: float, holding=True):
    """Drive the controlled point (object centre or palm) to a world target
    at bounded speed, IK on the grasp policy's arm, base in IDLE.
    Returns (ok, why, xy_error)."""
    p, side, key = ctx.p, ctx.side, ctx.obj_key
    s = p.obs()
    rot_hold = Primitives._rot(s["palms"][side]["quat"])  # keep the object upright: hold the palm rotation
    ref = None
    stall = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        s = p.obs()
        why = watch(s, holding)
        if why:
            return False, why, np.inf
        palm = np.asarray(s["palms"][side]["pos"], float)
        ctrl = np.asarray(s["objects"][key]["pos"], float) if control == "object" else palm
        tgt = np.asarray(target_fn(s), float)
        ref = ctrl.copy() if ref is None else step_reference(ref, tgt, v, DT)
        err = tgt - ctrl
        if float(np.linalg.norm(err)) < tol:
            return True, "", float(np.linalg.norm(err[:2]))
        # The palm carries the controlled point: ask for the reference's
        # offset, capped so a lagging arm cannot wind the request up.
        lead = ref - ctrl
        n = float(np.linalg.norm(lead))
        if n > 0.06:
            lead *= 0.06 / n
        aim_pf = p.world_to_root_full(palm + lead, s)
        rot_pf = Primitives._rot(s["pelvis"]["quat"]).T @ rot_hold
        q, res, _ = arm.ik.solve(side, aim_pf, rot_pf, q_mj29=arm.q, use_waist=True, iters=30)
        stall = stall + 1 if res > 0.04 else 0
        if stall > 25:
            return False, f"target unreachable by {res * 100:.1f} cm", float(np.linalg.norm(err[:2]))
        q = arm.q + np.clip(np.asarray(q) - arm.q, -JOINT_STEP, JOINT_STEP)
        arm.q = q
        arm._push(q)
    s = p.obs()
    ctrl = np.asarray(s["objects"][key]["pos"], float) if control == "object" else np.asarray(s["palms"][side]["pos"])
    err = np.asarray(target_fn(s), float) - ctrl
    return False, f"timed out {float(np.linalg.norm(err)) * 100:.1f} cm short", float(np.linalg.norm(err[:2]))


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


def _release(arm, side: str, seconds: float) -> None:
    start = np.asarray(arm.hands[side], dtype=np.float32).copy()
    n = max(int(round(seconds / DT)), 1)
    for k in range(1, n + 1):
        arm.hands[side] = start + (k / n) * (HAND_OPEN - start)
        arm._push(arm.q)


class _Geometry:
    """World-frame numbers the sub-phases share, taken once from the lift-end snapshot."""

    def __init__(self, ctx, s):
        cfg, key, goal = ctx.place_config, ctx.obj_key, ctx.goal_key
        self.goal_pos = np.asarray(s["objects"][goal]["pos"], float)
        self.rim_z = float(ctx.surf_a["top"] + ctx.meta["objects"][goal]["height"])
        self.floor_z = float(ctx.surf_a["top"] + 0.01)
        half_h = 0.5 * float(ctx.meta["objects"][key]["height"])
        self.hover_z = self.rim_z + half_h + cfg.hover_clearance
        self.release_z = self.rim_z + half_h + cfg.release_clearance
        self.obj_xy0 = np.asarray(s["objects"][key]["pos"][:2], float)


def _place_move(ctx, arm, watch, g: _Geometry):
    """Raise, turn, carry over, descend. -> (ok, why, xy_error at the release point)."""
    p, ex, cfg = ctx.p, ctx.execution, ctx.place_config
    servo = dict(control="object", v=cfg.v_travel, timeout=cfg.carry_timeout)
    ex.enter("place_raise")
    ok, why, _ = _servo(ctx, arm, watch, lambda _s: [*g.obj_xy0, g.hover_z], tol=0.02, **servo)
    if not ok:
        return False, why if why.startswith("[") else f"[carry] raise: {why}", np.inf
    ex.enter("place_turn")
    ok, why = _turn_to_band(ctx, watch, g.goal_pos, cfg)
    if not ok:
        return False, why if why.startswith("[") else f"[turn] {why}", np.inf
    _reseed(p, arm, p.obs())
    ex.enter("place_carry")
    ok, why, _ = _servo(ctx, arm, watch, lambda _s: [*g.goal_pos[:2], g.hover_z], tol=cfg.place_tol, **servo)
    if not ok:
        return False, why if why.startswith("[") else f"[place] carry-over: {why}", np.inf
    ex.enter("place_descend")
    servo.update(v=cfg.v_descend, timeout=cfg.descend_timeout)
    ok, why, xy_err = _servo(ctx, arm, watch, lambda _s: [*g.goal_pos[:2], g.release_z], tol=cfg.place_tol, **servo)
    if not ok and why.startswith("["):
        return False, why, xy_err
    if not ok and xy_err > cfg.place_tol_late:
        return False, f"[place] not over the container at the descent timeout (xy {xy_err * 100:.1f} cm)", xy_err
    if not ok:
        p.log(f"[place] descent timed out with xy {xy_err * 100:.1f} cm -- releasing anyway")
    ex.metrics["place"] = dict(xy_error=float(xy_err), released_on_time=bool(ok), goal_visibility=watch.visibility)
    return True, "", xy_err


def _place_finish(ctx, arm, watch, g: _Geometry):
    """Release, retreat (lift, then back off), terminal hold and the verdict."""
    p, ex, cfg, key, goal, side = ctx.p, ctx.execution, ctx.place_config, ctx.obj_key, ctx.goal_key, ctx.side
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
    goal_now = np.asarray(s["objects"][goal]["pos"], float)
    moved = float(np.linalg.norm(goal_now[:2] - g.goal_pos[:2]))
    if moved > cfg.goal_move_max:
        return False, f"[place] container moved {moved * 100:.0f} cm"
    inside, why = in_container(obj, goal_now, s["objects"][goal]["quat"], cfg.goal_inner_half, g.floor_z, g.rim_z)
    if not inside:
        return False, f"[place] missed: {why}"
    off = float(np.linalg.norm(obj[:2] - goal_now[:2]))
    ex.metrics["place"]["final_offset"] = off
    ex.metrics["goal_visibility"] = watch.visibility
    p.log(
        f"[place] OK: object {off * 100:.1f} cm from the container centre, goal in view {watch.visibility * 100:.0f}%"
    )
    return True, ""


def _phase_place(ctx):
    """Steps 4-9 of the plan. Returns the round's (ok, why)."""
    from locomani.tasks.bottle.grasp_gym import palm_frame_offset

    arm = getattr(ctx.coordinated, "arm", None)
    if arm is None:
        return False, "[gym] place needs the grasp policy's arm (ready_grasp_baseline)"
    s = ctx.p.obs()
    _reseed(ctx.p, arm, s)
    watch = _Watch(ctx, palm_frame_offset(s, ctx.obj_key, ctx.side), ctx.goal_key)
    g = _Geometry(ctx, s)
    ok, why, _ = _place_move(ctx, arm, watch, g)
    if not ok:
        return False, why
    return _place_finish(ctx, arm, watch, g)
