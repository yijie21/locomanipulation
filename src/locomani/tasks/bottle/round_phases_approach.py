"""`_phase_approach` and `_phase_grasp_and_verify`, split out of round_phases.py
(refactor step 4b, pure extraction; decision 9's file-size cap -- the five
phases of `_run_round` did not fit the 600-line file cap in one module).
See round_phases.py's own docstring for the split's rationale and the
no-top-level-grasp_gym-import rule; the same reasoning applies here.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from locomani.backends.sonic import gait_speeds
from locomani.backends.sonic.primitives import READY_UPPER, LocomotionMode
from locomani.sim import sim_vis
from locomani.tasks.bottle import execution as stream_reach
from locomani.tasks.bottle.round_phases import RoundContext, _rel_dbg, _rel_now


def _phase_approach(ctx: RoundContext):
    p, rm = ctx.p, ctx.rm
    if ctx.reach_config.image_servo:
        return ctx.coordinated.approach()

    def raise_arm(_s, dist):
        if ctx.coordinated is not None:
            ctx.coordinated.walk_tick(_s)
            return
        u = float(np.clip(1.0 - (dist - 0.06) / ctx.raise_lead, 0.0, 1.0))
        sm = 3 * u * u - 2 * u**3

        return ctx.arm_ramp.push(ctx.q_free + sm * (READY_UPPER - ctx.q_free))

    # ONE monotone approach, stopped by a measurement instead of a plan.
    #
    # The old arrival walked to a point 0.30 m short of the planned stand,
    # waited 2 s for the deceleration creep, then crawled the remainder
    # against that plan. For VLA data the setpoint is the problem: it is a
    # hidden variable, so two near-identical ego frames carry opposite
    # actions whenever the correction reverses -- the visible pacing, and
    # an action the student cannot learn because nothing in RGB says which
    # way to go. It also left 35-57 cm to shuffle once the spawn moved out
    # to 1.8 m.
    #
    # Here the plan only sets the LINE (which is what puts the can on the
    # right-hand side); the STOP is rm.margin() on the live can, a property
    # of the scene the camera can see. Measured profile along the approach
    # at this can height (y_pf -0.06..-0.18): margin is 0 beyond x 0.45,
    # 0.025 at 0.40, 0.075 at 0.35, 0.100-0.125 by 0.25 -- monotone while
    # walking in, so the first crossing is the stop and reversing is not
    # representable. (The CORE margin is 0 everywhere at this height: the
    # sweet zone is too thin below the pelvis, so the test is against the
    # full envelope.)
    # ONE monotone approach, stopped by a MEASUREMENT instead of a plan,
    # and re-entrant. The old arrival walked to a point 0.30 m short of the
    # planned stand, waited out the creep, then crawled against that plan;
    # for VLA data the setpoint is the problem, because it is a hidden
    # variable and two near-identical ego frames then carry opposite
    # actions. Here the plan only sets the LINE -- that is what puts the
    # can on the right-hand side -- while the STOP is rm.margin() on the
    # live can, a property of the scene the camera can see.
    #
    # Re-entrant matters as much as monotone. When the arrival left the can
    # a little too far, the old corrector commanded an open-loop "depth
    # trim" -- and SONIC micro-steps overshoot 2-3x, so a 15 cm trim
    # measured 27 cm, put the robot inside the edge floor, and the
    # corrector then backed 0.45 m out and rewalked. Stop-a-little-far ->
    # trim -> overshoot -> retreat -> stop-a-little-far IS the pacing.
    # Running this gate-stopped walk again cannot overshoot the gate.
    # 0.075 -> 0.04 (2026-09-03): brake ~4 cm earlier so the gait's momentum
    # slide (5-18 cm after the brake, measured with the settle loop that
    # this replaces) lands the post-crouch grip x in the middle of the
    # [0.22, 0.36] band instead of on its near edge (5/14 too-close discards
    # without a settle, seed 4250).
    # back to 0.075 (2026-09-03 pm): the last centimetres are now closed-loop
    # (stream_reach._creep_to_band), so the brake only has to fire somewhere
    # before the band; braking early costs a second of shuffle, braking late
    # costs a discard.
    MARGIN_STOP = 0.075
    EDGE_LEAD = 0.06  # brake lead on the edge floor
    # Three speed bands. The COMMAND is not m/s and its meaning belongs to the
    # checkpoint, so the triple comes from the measured map in gait_speeds.py
    # (ledger bottle.md 21) instead of being written here:
    #   release   0.30 / 0.215 / 0.20  -> 0.203 / 0.016 / 0.018 m/s
    #   v1.1      0.40 / 0.215 / 0.20  -> 0.187 / 0.014 / 0.009 m/s
    # The gate fires at x 0.37-0.42 (matching the offline margin profile),
    # so the braking has to be DONE by then. With a single 0.12 m shuffle
    # band the robot only slowed at x 0.43 and an 11 cm coarse stride was
    # still in flight when the gate fired: measured carry 0.08-0.24 m, and
    # every round whose carry passed ~0.15 m ended inside the edge floor.
    # COARSE is above the checkpoint's own gait STEP, not at 0.26 (2026-08-25):
    # below the step the gait initiation minces -- the first seconds of every
    # episode showed a step taken and taken back before the walk committed,
    # recorded hesitation a VLA then learns as "dither first". On release that
    # meant 0.30 (17 cm/step); on v1.1 the step sits 0.065 higher, so 0.30
    # delivers 0.044 m/s and 0.40 is the equivalent choice (ledger 21.2).
    # Only the far band changes per checkpoint; the MID/SHUFFLE deceleration
    # bands keep release's commands because v1.1's step jumps straight over the
    # 0.016-0.018 m/s they deliver there (gait_speeds.py says why).
    APPR_COARSE, APPR_MID, APPR_SHUFFLE = gait_speeds.approach_bands()
    # The expert's own 1-2 cm/s shuffle-imitating crawl let the learned policy
    # set its own stop point independent of where the expert stopped; removing
    # it (formerly GYM_SHUFFLE_BAND=0) moved 07d 30% -> 48%, McNemar p=0.052
    # (docs/experiments.md:234). MID speed now holds all the way to the brake.
    MID_BAND = ctx.mid_band

    # Coverage-first (2026-09-04): steer the walk at a point offset laterally
    # from the planned stand, so the last 0.3 m has to correct sideways --
    # the correction the policy needs is then IN the data instead of the
    # expert always arriving aligned. The creep closes both axes.
    # GYM_STAND_COVER retired (refactor step 2, family 2, 2026-09-09): it only
    # applied to the flow200 sampler (coordinated is None), which is gone --
    # every surviving profile builds a coordinated controller, so this is 0.0.
    appr_lat = 0.0
    aim_xy = np.asarray(ctx.spawn_xy) + appr_lat * np.array([-ctx.fv[1], ctx.fv[0]])
    advance = float(np.clip(ctx.reach_config.arrival_advance, 0.0, 0.10))
    VISIBLE = bool(getattr(ctx.reach_config, "visible_expert", False))
    if VISIBLE:
        # Visible expert: no latch, no advance. The walk stops the tick the
        # MEASURED object depth reaches visible_brake_depth and decelerates
        # over visible_decel_band above it -- both functions of the relative
        # pose the cameras see, nothing per-episode.
        advance = 0.0
    aim_xy = aim_xy + advance * np.asarray(ctx.fv[:2])
    if appr_lat:
        p.log(f"[gym] approach aim offset {appr_lat * 100:+.0f} cm lateral")

    def do_approach(seconds: float, ramp: bool) -> str:
        appr_end = time.time() + seconds
        why, stall_x, stall_t = "timeout", None, time.time()
        brake_x, brake_e = float("nan"), float("nan")
        from locomani.tasks.bottle.arrival_validation import ArrivalAdvance

        advance_gate = ArrivalAdvance(advance)
        while time.time() < appr_end:
            s = p.obs()
            if s["fallen"]:
                return "fell"
            r = _rel_now(ctx, s)
            if ramp or ctx.coordinated is not None:
                raise_arm(s, float(np.linalg.norm(np.asarray(ctx.spawn_xy) - p.pelvis_xy(s))))
            if ctx.coordinated is not None and ctx.coordinated.error:
                return ctx.coordinated.error
            reachable = rm.margin(r, "right", cap=MARGIN_STOP + 0.01) >= MARGIN_STOP
            if VISIBLE:
                stop_here = r[0] <= ctx.reach_config.visible_brake_depth
            else:
                stop_here = advance_gate.stop(r[0], reachable) if advance else reachable
            if stop_here:
                why = "visible brake depth" if VISIBLE else "advanced reachable stop" if advance else "can reachable"
                brake_x, brake_e = r[0], ctx.edge_dist(p.pelvis_xy(s))
                break
            if ctx.edge_dist(p.pelvis_xy(s)) <= ctx.edge_min + EDGE_LEAD:
                why = "edge floor"
                brake_x, brake_e = r[0], ctx.edge_dist(p.pelvis_xy(s))
                break
            if stall_x is None or stall_x - r[0] > 0.015:
                stall_x, stall_t = r[0], time.time()
            elif time.time() - stall_t > 5.0:
                why = "stalled"
                break
            # STEER at the planned stand, do not just push along the
            # heading. Pushing along fv was the first version of this loop
            # and it lost the lateral geometry: sideways error persisted,
            # the can drifted onto the centreline (measured rel y +0.14
            # where the plan wanted -0.12), the right arm's envelope is
            # narrower there so the gate fired late, and the walk ran on to
            # the edge floor -- which is what the corrector then backed out
            # of.
            to_stand = np.asarray(aim_xy) - p.pelvis_xy(s)
            n_ts = float(np.linalg.norm(to_stand))
            mv = to_stand / n_ts if n_ts > 1e-3 else np.array(ctx.fv[:2])
            fv_now = ctx.fv
            speed = APPR_MID if r[0] - ctx.plan_r <= MID_BAND else APPR_COARSE
            if VISIBLE and ctx.reach_config.visible_decel_band > 0:
                # Monotone in the measured depth: full speed above the band,
                # ramping to the gait's slow floor at the brake depth.
                u = float(
                    np.clip(
                        (r[0] - ctx.reach_config.visible_brake_depth) / ctx.reach_config.visible_decel_band, 0.0, 1.0
                    )
                )
                speed = min(speed, APPR_SHUFFLE + (speed - APPR_SHUFFLE) * u)
            p.cmd.frame(
                mode=LocomotionMode.SLOW_WALK, facing=fv_now, movement=(float(mv[0]), float(mv[1]), 0.0), speed=speed
            )
            time.sleep(0.02)
        if ramp and ctx.coordinated is None:
            # Finish whatever is left of the rise: the stop can fire before
            # the slew reaches READY, and a snap there would be a step in the
            # reference. The origin is the RAMP's own commanded state, not the
            # measured pose -- re-anchoring on the measurement here (which is
            # what this used to do) put a step of the whole tracking error
            # back in, every time it ran. Costs nothing when the rise already
            # finished during the walk: the first push returns converged and
            # no frame is sent, so no standing is recorded.
            t0 = time.time()
            while time.time() - t0 < ctx.raise_finish_max_s:
                if ctx.arm_ramp.push(READY_UPPER) < 1e-3:
                    break
                p.cmd.frame(mode=LocomotionMode.IDLE, facing=ctx.fv)
                time.sleep(0.02)
            ctx.arm_ramp.settle()
            if time.time() - t0 > 0.10:
                p.log(f"[gym] arm rise finished at the table: {time.time() - t0:.2f} s of standing")
        # No settle hold at all any more (was 1.0 s, then 0.2 s). The grasp
        # is a 50 Hz servo that re-aims on the measured object pose every
        # tick, so a settled base buys nothing -- and every recorded standing
        # frame here teaches a VLA that standing is what one does at the
        # table (the round-1/2 stall).
        x_end = _rel_now(ctx, p.obs())[0]
        p.log(
            f"[gym] approach: exit={why} braked at x {brake_x:.2f} "
            f"edge {brake_e:.2f} -> stopped x {x_end:.2f} "
            f"edge {ctx.edge_dist(p.pelvis_xy(p.obs())):.2f} "
            f"(carry {brake_x - x_end:+.2f}, plan_r {ctx.plan_r:.2f})"
        )
        ctx.execution.metrics.setdefault("arrival_measurements", []).append(
            dict(
                requested_advance_m=advance,
                first_reachable_depth_m=advance_gate.first_reachable,
                brake_depth_m=brake_x,
                brake_edge_m=brake_e,
                end_depth_m=x_end,
                end_edge_m=ctx.edge_dist(p.pelvis_xy(p.obs())),
                reason=why,
            )
        )
        _rel_dbg(ctx, "post-approach", p.obs())

        return why

    approach_result = do_approach(45.0, ramp=True)
    if ctx.coordinated is not None and ctx.coordinated.error:
        return False, ctx.coordinated.error
    if approach_result == "fell":
        return False, "[stand] fell during the approach"
    _r0 = _rel_now(ctx, p.obs())
    ctx.execution.metrics["first_stop"] = round(float(np.hypot(_r0[0], _r0[1])), 3)

    # FORWARD-ONLY repair. Too far -> walk in again on the same gate. Too
    # close, or laterally off -> ACCEPT: 12 rounds that stopped anywhere
    # from x 0.15 to 0.46 (edge 0.09 to 0.35) went 12/12 with ZERO
    # furniture contacts, so the arm absorbs the arrival scatter, which is
    # where that precision belongs. Nothing in an episode walks backwards
    # any more except the deliberate post-grasp back-off.
    # --one-shot (flow200): no repair walk at all. The round either parked
    # inside the graspable envelope on its single approach or it is
    # discarded by the gates below; the re-walk's brake/stand/re-accelerate
    # is exactly the recorded hesitation this dataset exists to exclude.
    for _ in range(0 if ctx.one_shot or ctx.coordinated is not None else 2):
        s = p.obs()
        rel_a = _rel_now(ctx, s)
        too_far = rel_a[0] > ctx.plan_r + 0.06 or not rm.graspable(rel_a, "right")
        if not too_far:
            break
        p.log(f"[gym] still far: rel [{rel_a[0]:.2f} {rel_a[1]:+.2f}] — walking in again")
        if do_approach(12.0, ramp=False) == "fell":
            return False, "[stand] fell walking in again"
    s = p.obs()
    rel_a = _rel_now(ctx, s)
    ctx.execution.metrics["park"] = round(float(np.hypot(rel_a[0], rel_a[1])), 3)
    # A real safety floor, not the planning floor. EDGE_MIN (0.17) is where
    # the PLANNER aims; rounds that actually stopped at edge 0.09-0.17 went
    # 12/12 with zero furniture contacts in the whole batch, so failing them
    # here would only discard good episodes now that nothing backs up to
    # rescue them.
    if ctx.edge_dist(p.pelvis_xy(s)) < 0.06:
        return False, "[stand] pelvis ended against the table edge"
    if not rm.graspable(rel_a, "right"):
        return False, (f"[stand] arrival left the object out of reach (rel [{rel_a[0]:.2f} {rel_a[1]:+.2f}])")

    # -- the grasp, same path as collect_episode ---------------------------
    st = p.obs()
    if stream_reach.toppled(st, ctx.obj_key):
        return False, "[gym] object toppled during the settle"
    cam = st["objects"][ctx.obj_key].get("cam")
    # Vertical bound view_gate_y, horizontal view_gate_x: the 52-deg-pitched
    # camera puts a close can low in the frame (fy -0.82..-0.94 killed 4
    # otherwise-good rounds/batch). fy 0.9 is still inside the physical image
    # (edge 1.0), so on real hardware the object stays detectable -- the
    # horizontal bound keeps the stricter margin because yaw scatter moves it
    # fast.
    if "cam" in st["objects"][ctx.obj_key] and (
        cam is None or abs(cam[0]) > ctx.view_gate_x or abs(cam[1]) > ctx.view_gate_y
    ):
        return False, f"[view] object outside ego view at stand (cam={cam})"
    return None


def _phase_grasp_and_verify(ctx: RoundContext):
    p, reset = ctx.p, ctx.reset
    st = p.obs()
    ctx.execution.enter("grasp_plan")
    grip = ctx.task_grasp.grip(st)

    def show_ghost(pl):
        # The ghost is baked into the scene at build time and shares geom
        # group 4 with the reach cloud, so it cannot be hidden from the video
        # side alone. Leaving it parked at its spawn (below the floor) is the
        # lever that works with an already-built scene.
        if not pl.ok or not sim_vis.enabled("ghost"):
            return
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, np.asarray(pl.rot, dtype=float).flatten())
        reset.set_ghost(pos=np.asarray(pl.wrist, dtype=float).tolist(), quat=q.tolist())

    # The flow200-style plan (adaptive_grasp initial()/replan(), used only
    # when ready_grasp_baseline was False) is retired -- refactor step 2,
    # family 2 (2026-09-09). Every surviving profile is baseline=True, so
    # plan is always None here; show_ghost still travels into execute_grasp
    # as the `ghost` callback below.
    plan = None

    # Nothing to set up for the arm here any more. The wrist targets this
    # used to write are only read in PLANNER_VR_3PT, and PlannerArm.start()
    # switches to PLANNER_FROZEN_UPPER_BODY on its first frame -- a leftover
    # from the streamed channel, which also needed the pin released before
    # its handoff. The planner channel has no handoff.
    reset.pin_arms()
    # The pin is OFF by default now (GYM_ARM_PIN=1 restores it). It makes the
    # executed motion precise (3.0 cm / 5 deg) but the recorded latent is
    # then not the latent that produced it: the same reference without the
    # pin delivers 4.1 cm / 16 deg (diag_channel.py, 216 samples). For VLA
    # data the label has to match the outcome, and on the planner channel
    # the policy alone now grasps 12/12, so the pin is gone.
    grasp_result = stream_reach.execute_grasp(
        p,
        ctx.obj_key,
        grip,
        plan,
        side=ctx.side,
        surface_top=ctx.surf_a["top"],
        replan=ctx.task_grasp.replan,
        ghost=show_ghost,
        nudge=False,
        execution=ctx.execution,
        config=ctx.reach_config,
        coordinated=ctx.coordinated,
        transport_guard_factory=lambda ik: ctx.task_grasp.transport_guard(ik, ctx.reach_config),
        pin=(reset.pin_arms if ctx.arm_pin else None),
        squat_hint=ctx.squat_hint,
        plan_r=float(np.hypot(ctx.rel[0], ctx.rel[1])),
        stand_rel=None,
        envelope=None,
    )
    if not grasp_result.ok:
        return False, f"[grasp] {grasp_result.why}"

    ctx.execution.enter("verify_visibility")
    vis = grasp_result.visibility
    if vis is not None:
        p.log(f"  ego visibility through servo: {vis * 100:.0f}%")
        if vis < 0.7:
            return False, (f"[view] object in ego view only {vis * 100:.0f}% of the servo")

    if ctx.goal_key:
        # Pick-and-place: the lifted object goes into the container instead of
        # the back-off-and-hold tail below (place_phase.py, plan §2 steps 4-9).
        from locomani.tasks.bottle.place_phase import _phase_place

        return _phase_place(ctx)

    ctx.execution.enter("carry_and_verify")
    s = p.obs()
    from locomani.tasks.bottle.grasp_gym import palm_frame_offset

    hold_ref = palm_frame_offset(s, ctx.obj_key, ctx.side)
    hold_side = "left" if ctx.side == "right" else "right"
    p.stand_up(hold_side)
    p.carry_pose(ctx.side)
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=ctx.fv)
    # With the can secured in the carry pose, back off two steps (~0.35 m,
    # heading held) before the hold check: clears the table edge so
    # whatever comes next (turn, navigation) cannot clip it, and a grip
    # that cannot survive two steps should fail HERE, not mid-carry.
    s = p.obs()
    back_tgt = np.asarray(p.pelvis_xy(s)) - 0.35 * np.array(ctx.fv[:2])
    if not p.walk_to(back_tgt, stop_dist=0.08, timeout=8.0, face_target=False):
        return False, "[hold] fell while backing away from the table"
    # Keep validation in raw data; conversion can take a contiguous prefix
    # using the same sequence key as the exporter, without wall-clock guesses.
    ctx.execution.metrics["training_tail"] = dict(
        end_seq=int(p.obs()["seq"]), reason="post_transport_hold_verification"
    )
    ctx.execution.enter("verify_terminal_hold")
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=ctx.fv)
    s = p.obs()
    drift = float(np.linalg.norm(palm_frame_offset(s, ctx.obj_key, ctx.side) - hold_ref))
    if drift >= ctx.palm_drift_max:
        return False, (f"[hold] object slipped {drift * 100:.1f} cm in the palm frame during the hold")
    if float(p.object_pos(ctx.obj_key, s)[2]) <= ctx.surf_a["top"] + 0.03:
        return False, "[hold] object is back on the table after the hold"
    return True, ""
