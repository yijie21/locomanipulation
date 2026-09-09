"""`_run_round`'s five phases, split out of grasp_gym.py (refactor step 4b,
pure extraction; decision 9's file-size cap). Every statement here is
unchanged from the original inline body -- only the packaging moved:
each closure's captured locals became a field on `RoundContext`, and the two
helpers used from more than one phase (`_rel_dbg`, `_rel_now`) became plain
functions taking `ctx` instead of nested closures. The last two phases
(`_phase_approach`, `_phase_grasp_and_verify`) live in round_phases_approach.py
instead -- all five here would not fit this file's own 600-line cap.

This module must not import `locomani.tasks.bottle.grasp_gym` at its own top
level: grasp_gym.py imports names from here (`_run_round`'s new home is
grasp_gym.py itself, calling these phases in sequence -- see the docstring on
`_run_round`), so a module-level import in the other direction would make the
two files a load-order-sensitive circular import. Nothing here needs anything
of grasp_gym.py's at import time anyway; `RoundContext` already carries the
one-time snapshot of grasp_gym's monkeypatchable globals (`vla_policy`,
`arm_pin`, `video_include_init`, `edge_min`, `raise_lead`,
`raise_finish_max_s`, `palm_drift_max`) that `_run_round` takes before
calling phase 1, and `_phase_spawn_and_settle`'s one direct reference to
`grasp_gym.obj_top` is a deferred, in-function import for the same reason.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from locomani.backends.sonic import initial_pose, upper_ref
from locomani.backends.sonic.primitives import (
    UPPER_BODY_NAMES,
    LocomotionMode,
    StreamMode,
    upper_body_from_mj,
    wrap,
)
from locomani.pipeline.execution_record import ExecutionRecord
from locomani.tasks.bottle import grasping
from locomani.tasks.bottle.plan import sample_object_yaw
from locomani.tasks.bottle.reach_config import DEFAULT_REACH
from locomani.tasks.bottle.tabletop_grasp import TabletopGrasp, TabletopTarget

# Every phase function below returns `tuple[bool, str] | None`: `None` to
# continue to the next phase, a tuple to end the round with that result.


@dataclass
class RoundContext:
    """Everything one `_run_round` call threads through its phases.

    The first block is `_run_round`'s own parameters, unchanged; the second
    is grasp_gym.py's monkeypatchable globals, snapshotted once per round by
    `_run_round` itself (so a caller can still override any of them by
    assignment -- eval_closed_loop.py sets grasp_gym.VLA_POLICY, for one);
    the third is state a phase computes for a LATER phase to use -- each
    field names the closure that used to capture it.
    """

    # -- _run_round's own parameters -----------------------------------
    p: Any
    reset: Any
    meta: dict
    obj_key: str
    surf_a: dict
    edge_dist: Callable
    d_can: np.ndarray
    d_robot: np.ndarray
    psi: float
    side: str
    rel: np.ndarray
    rm: Any
    spawn_xy: np.ndarray
    squat_hint: float | None = None
    spawn_off: float = 0.30
    spawn_lat: float = 0.0
    dpsi: float = 0.0
    appearance: int | None = None
    record: bool = False
    rec: dict | None = None
    view_gate_x: float = 0.80
    view_gate_y: float = 0.90
    one_shot: bool = False
    mid_band: float = 0.30
    execution: ExecutionRecord = None
    reach_config: Any = DEFAULT_REACH
    object_yaw: float | None = None

    # -- grasp_gym.py globals, snapshotted once per round --------------
    vla_policy: Callable | None = None
    arm_pin: bool = False
    video_include_init: bool = False
    edge_min: float = 0.17
    raise_lead: float = 1.00
    raise_finish_max_s: float = 0.30
    palm_drift_max: float = 0.03
    tabletop_grasp_cls: type = None  # grasp_gym.TabletopGrasp; a test replaces it wholesale via patch.object

    # -- filled in by _phase_spawn_and_settle ---------------------------
    fv: tuple = None  # the heading held through settle/turn/approach/hold; replaces every phase's own `fv`
    reset_objects: dict = None  # this round's reset() object args; replaces handover_to_vla's `replace()` closure
    reset_robot: dict = None  # this round's reset() robot args; ditto
    task_grasp: TabletopGrasp = None  # replaces rel_now/rel_dbg's + grasp_and_verify's captured `task_grasp`
    grip_za: float = None  # replaces rel_now's captured `grip_za`

    # -- filled in by _phase_navigate ------------------------------------
    plan_r: float = None  # replaces approach's captured `plan_r`
    arm_ramp: Any = None  # replaces raise_arm/do_approach's captured `arm_ramp`
    coordinated: Any = None  # replaces raise_arm/do_approach/grasp_and_verify's captured `coordinated`
    q_free: np.ndarray = field(default=None, repr=False)  # replaces raise_arm's captured `q_free`


def _rel_dbg(ctx: RoundContext, tag, s2) -> None:
    y2 = ctx.p.pelvis_yaw(s2)
    c2, s2n = np.cos(y2), np.sin(y2)
    dv2 = ctx.p.object_pos(ctx.obj_key, s2)[:2] - ctx.p.pelvis_xy(s2)
    rx = c2 * dv2[0] + s2n * dv2[1]
    ry = -s2n * dv2[0] + c2 * dv2[1]
    yerr = np.degrees((y2 - ctx.psi + np.pi) % (2 * np.pi) - np.pi)
    ctx.p.log(f"[gym] {tag}: rel [{rx:.2f} {ry:+.2f}] yaw_err {yerr:+.0f} deg")


def _rel_now(ctx: RoundContext, s2):
    y2 = ctx.p.pelvis_yaw(s2)
    c2, s2n = np.cos(y2), np.sin(y2)
    dv2 = ctx.p.object_pos(ctx.obj_key, s2)[:2] - ctx.p.pelvis_xy(s2)
    return [
        c2 * dv2[0] + s2n * dv2[1],
        -s2n * dv2[0] + c2 * dv2[1],
        float(ctx.task_grasp.grip(s2)[2]) - float(s2["pelvis"]["pos"][2]),
    ]


def _phase_spawn_and_settle(ctx: RoundContext):
    from locomani.tasks.bottle.grasp_gym import obj_top

    p, reset = ctx.p, ctx.reset
    # Spawn 0.30 m behind the stand. Directly ON the stand the STAND-UP
    # fails (13/24 never rose): the nominal pose is a deep crouch whose
    # knees ride ~0.25 m forward, and the rise needs more clearance than
    # standing does. 0.30 m back gives the rise ~0.5 m of room and leaves
    # only a two-step walk to the stand.
    fv2 = np.array([np.cos(ctx.psi), np.sin(ctx.psi)])
    perp = np.array([-fv2[1], fv2[0]])
    ctx.d_robot = np.asarray(ctx.d_robot, dtype=float) - ctx.spawn_off * fv2 + ctx.spawn_lat * perp
    # The heading the robot WAKES UP with, which is not the heading it walks
    # in on. psi is the approach heading; dpsi is the error it has to turn
    # out at the spawn, before the walk, metres from the table.
    # A VLA round spawns ALREADY FACING the approach heading and skips the
    # scripted spawn turn below. Not a convenience -- a necessity twice over:
    #   1. The turn runs on the deploy's locomotion policy, whose token_state
    #      is ZEROED in the token-only deployment until external tokens
    #      arrive (g1_deploy_onnx_ref.cpp:3066 "Robot will use zero tokens").
    #      Standing survives that; active turning does not (measured: turn
    #      timeouts and 0.43 m of walk-off, where the encoder-loaded
    #      collection runs turned cleanly at 95%).
    #   2. The training episodes START after the turn anyway -- recording
    #      begins at pre-walk, residual yaw error <= ~9 deg -- so a spawn
    #      that faces psi directly is what the policy's t=0 actually looked
    #      like in every episode it learned from.
    if ctx.vla_policy is not None:
        ctx.dpsi = 0.0
    psi_spawn = float(ctx.psi + ctx.dpsi)
    # QUIESCE the deploy before the teleport: it keeps executing the
    # previous round's locomotion intent through the reset (fresh-deploy
    # round 1 was always clean, later rounds marched off mid-settle,
    # toppled the can, or 'failed to land'). A second of IDLE in default
    # mode makes 'standing still' the intent the reset inherits.
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    if getattr(p, "_gym_started", False):
        # Quiesce only a RUNNING deploy (holding frames before the first
        # start_control broke every fresh-deploy round). Hold the heading the
        # robot currently HAS -- the facing-less default (world +x) was the
        # third contributor to the end-of-round pirouette.
        yaw_now = float(p.pelvis_yaw(p.obs()))
        p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=(float(np.cos(yaw_now)), float(np.sin(yaw_now)), 0.0))
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.set_ghost()
    # Captured, not inline: the VLA replace() below re-sends the IDENTICAL
    # reset (same drawn object yaw, same appearance) as a mid-prelude
    # re-teleport. A redraw here would both re-orient the object and shift
    # the RNG stream, breaking cross-run pairing.
    ctx.object_yaw = sample_object_yaw(ctx.object_yaw)
    ctx.execution.metrics["object_yaw"] = ctx.object_yaw
    ctx.reset_objects = {ctx.obj_key: {"dxy": [float(ctx.d_can[0]), float(ctx.d_can[1])], "yaw": ctx.object_yaw}}
    ctx.reset_robot = {"dxy": [float(ctx.d_robot[0]), float(ctx.d_robot[1])], "yaw": psi_spawn}
    reset.reset(objects=ctx.reset_objects, robot=ctx.reset_robot, appearance=ctx.appearance)
    time.sleep(0.3)
    landed_xy = np.asarray(p.pelvis_xy(p.obs()))
    want_xy = np.asarray(ctx.spawn_xy) - ctx.spawn_off * fv2 + ctx.spawn_lat * perp
    tp_err = float(np.linalg.norm(landed_xy - want_xy))
    if tp_err > 0.10:
        return False, (
            f"[stand] teleport landed {tp_err:.2f} m off ({np.round(landed_xy, 2)} vs {np.round(want_xy, 2)})"
        )
    # Round 2+ contamination: the commander KEEPS the previous round's
    # FROZEN_UPPER_BODY mode and its stale 17-value arm target across the
    # reset — stand up in the default mode with no target, like a fresh
    # process does.
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.start_control()
    p._gym_started = True
    # Warm the reference tap here, not at the takeover: the deploy publishes
    # g1_debug only once control is running (nothing at all arrives before
    # start_control), so a tap started at the moment it is first read is
    # always cold and would silently fall back to the measured pose.
    if not upper_ref.shared().wait(3.0):
        p.log(
            "[gym] WARNING: no body_q_target 3 s after start_control -- "
            "the upper-body takeover will fall back to the measured pose"
        )
    # Every IDLE frame carries a facing, and the default is the ABSOLUTE
    # world +x — the settle was quietly rotating the freshly-teleported
    # robot 50-70 deg off its sampled yaw (pre-walk yaw_err telemetry).
    # Hold the sampled heading explicitly through every settle.
    ctx.fv = (float(np.cos(ctx.psi)), float(np.sin(ctx.psi)), 0.0)
    fv = ctx.fv
    # The settle holds the heading the robot WOKE UP with. Commanding the
    # approach heading here instead would make the stand-up fight the turn,
    # and an IDLE facing more than ~60 deg away does not turn at all -- it
    # stalls (Primitives.turn_to: 68 deg timed out where 30 deg finished in
    # a second). The turn gets its own phase below, after the robot is up.
    fv_spawn = (float(np.cos(psi_spawn)), float(np.sin(psi_spawn)), 0.0)
    # Tell the commander where the deploy's heading anchor IS, rather than
    # forcing a new one. The deploy captures its anchor at start_control,
    # from the yaw it sees THEN -- which is the post-teleport yaw, because
    # the reset above already happened. So the anchor is simply the landing
    # yaw, and frame() can undo the rotation from that.
    #
    # This replaces a stream_reach.reanchor(p) call, which was there to
    # force the anchor to a yaw this process knew. Measured side by side
    # from a -63 deg spawn, holding that heading for 5 s:
    #
    #     offset := landing yaw   ->  yaw -61.1 (err +1.9 deg), moved 0.10 m
    #     reanchor() toggle       ->  yaw -71.5 (err -8.5 deg), moved 0.05 m
    #
    # and reanchor is worse than those numbers suggest: its streamed toggle
    # SPINS the robot ~40 deg while it runs, then records that spun yaw as
    # the anchor. With a spawn heading error to turn out afterwards, that
    # left the robot chasing a 40 deg phantom through the settle -- 0.55 m
    # of stand drift, over the 0.35 m budget, before the deliberate turn
    # even started.
    p.cmd.heading_offset = float(p.pelvis_yaw(p.obs()))
    # VLA rounds settle SHORT: in the drift-prone spawn configs the gait
    # limit cycle walks ~0.15 m/s through any planner hold (IDLE, squat
    # brake, even walk_to back -- all measured), so the only working lever
    # is窗口长度: every second of settle is up to 15 cm of unwanted
    # approach. 1.5 s is enough for the fallen-check; the scripted branch
    # keeps the full 4 s (its distribution is the training data's).
    p.cmd.hold(1.5 if ctx.vla_policy is not None else 4.0, mode=LocomotionMode.IDLE, facing=fv_spawn)
    s = p.obs()
    if s["fallen"]:
        return False, (f"[stand] robot did not stabilise after reset (pelvis z {s['pelvis']['pos'][2]:.3f})")
    # Separate check, separate meaning: the teleport itself was verified
    # above; this one is how far the STAND-UP walked the robot off its
    # spawn, which is a policy behaviour and gets its own budget.
    drift = float(np.linalg.norm(np.asarray(p.pelvis_xy(s)) - landed_xy))
    p.log(f"[gym] spawn {np.round(landed_xy, 2)} -> settled {np.round(p.pelvis_xy(s), 2)} (stand drift {drift:.2f} m)")
    if drift > 0.35:
        return False, (f"[stand] walked {drift:.2f} m off the spawn while standing up")
    # TURN AT THE SPAWN. This is the whole reason the spawn heading and the
    # approach heading are separate: a heading correction near the object is
    # a 110-160 deg pirouette that walks the pelvis into the table edge,
    # whereas out here there is nothing to hit. turn_to walks the commanded
    # facing across in <=0.5 rad steps because a larger jump stalls the
    # controller outright.
    # tol 0.16, not tighter: the controller holds a 7-8 deg heading dead
    # zone (diag_turn.py measures a 6.5-8.1 deg residual on every probe in
    # both modes at every angle), and a tolerance inside it does not turn
    # harder, it just burns the timeout -- which is exactly how the first
    # attempt at this failed. SLOW_WALK, not IDLE: it converged on every
    # probe and roughly three times faster at the angles that matter, while
    # IDLE failed outright at +60 and +120. A stepping turn is also what a
    # person does.
    if abs(wrap(ctx.dpsi)) > 0.10 and ctx.vla_policy is None:
        turned_from = np.asarray(p.pelvis_xy(p.obs()))
        if not p.turn_to(ctx.psi, tol=0.16, timeout=20.0, mode=LocomotionMode.SLOW_WALK):
            return False, (
                f"[stand] could not turn {np.degrees(ctx.dpsi):+.0f} deg to the approach heading at the spawn ({p.aborted})"
            )
        s = p.obs()
        if s["fallen"]:
            return False, "[stand] fell while turning at the spawn"
        # The turn is allowed to shuffle, not to walk away: it happens
        # before the sampler's clearance guarantees mean anything, since
        # those were computed for the spawn POINT.
        walked = float(np.linalg.norm(np.asarray(p.pelvis_xy(s)) - turned_from))
        p.log(
            f"[gym] spawn turn {np.degrees(ctx.dpsi):+.0f} deg -> "
            f"yaw_err {np.degrees(wrap(p.pelvis_yaw(s) - ctx.psi)):+.0f} deg, "
            f"moved {walked:.2f} m"
        )
        if walked > 0.40:
            return False, (f"[stand] the spawn turn walked {walked:.2f} m off the sampled spawn")

    # Walk with the arms FREE. In PLANNER_VR_3PT the walking policy ignores
    # the wrist targets and swings the arms with the gait (agent_server:151),
    # which is what a person does; the old pre-walk switch to
    # FROZEN_UPPER_BODY held both forearms out level for the whole approach
    # and is where the stiff, un-human walk came from. Nothing is pinned
    # either -- the hands only need to be up by the time the reach starts,
    # and that is what the RAISE_LEAD ramp below does.
    def _wmark(tag):
        # settled->fork forensics: 0.34 m (median) lands somewhere in this
        # window (blocking90, 85 rounds); these marks name the segment.
        if ctx.vla_policy is not None:
            try:
                sw = p.obs()
                p.log(
                    f"[gym] win {tag}: pelvis {np.round(p.pelvis_xy(sw), 3)} "
                    f"yaw {np.degrees(p.pelvis_yaw(sw)):.1f} "
                    f"anchor_off {np.degrees(p.cmd.heading_offset):.1f}"
                )
            except Exception as e:  # a missing mark must LOOK missing
                p.log(f"[gym] win {tag}: MARK FAILED ({e})")

    _wmark("pre-switch")
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    reset.pin_arms()
    _wmark("pinned")
    if ctx.vla_policy is None:
        p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=fv)
    else:
        # No quiesce hold and NO corrective walking for VLA rounds. In the
        # drift-prone spawn configs the gait limit cycle walks ~0.15 m/s
        # through every planner countermeasure tried (IDLE hold, IDEL_SQUAT
        # brake, walk_to back -- the last one IGNITES the runaway: 0.14 m
        # off after the short settle, 0.6 m off after "walking back").
        # The only lever that works is window length: settle cut to 1.5 s
        # above, nothing else commanded here, hand over to the VLA at once.
        pass
    _wmark("held")
    # FIRST-FRAME VISIBILITY GATE. The student only ever sees ego_view, so a
    # spawn where the bottle is not in that frame teaches an approach that
    # nothing in the image explains. It matters at the spawn distances this
    # batch uses and not at 0.7 m: the head camera is pitched 52 deg down, so
    # a tabletop object leaves the TOP of the frame past a stand-off, and the
    # forward-lean budget before it does shrinks fast.
    #
    # THE UNITS BELOW ARE --spawn-edge (pelvis to TABLE EDGE), because that is
    # the knob a caller actually sets. The table here used to be in
    # pelvis-to-BOTTLE metres, which reads 0.128 m (the bottle's inset from
    # the near edge) more generous than reality: it said --spawn-edge 1.50
    # still had 2.3 deg of lean to spare when the true figure is 0.66 deg.
    # Recomputed with privileged_pub.py's own normalisation (ty = tan(fovy/2),
    # u_y divided by ty), upright, against gate |u_y| <= 0.90:
    #
    #     --spawn-edge 0.70 -> u_y 0.379,  21.18 deg of lean to spare
    #     --spawn-edge 1.20 -> u_y 0.738,   5.35 deg
    #     --spawn-edge 1.50 -> u_y 0.878,   0.66 deg   <- luck, not sampling
    #     --spawn-edge 1.55 -> u_y 0.900              <- THE GATE CUTS HERE
    #     --spawn-edge 1.60 -> u_y 0.918,  gated while perfectly upright
    #     --spawn-edge 1.87 -> u_y 1.010,  off the frame entirely
    #
    # THOSE ROWS ARE AN UPRIGHT-POSE UPPER BOUND, NOT WHAT YOU WILL SEE. They
    # are computed from the scene's keyframe 0; the real robot has settled
    # under SONIC by the time this gate runs and carries extra torso pitch.
    # Measured against live rejections at --spawn-edge 1.1 1.3, where the
    # table above predicts u_y 0.72-0.79: actual u_y came in at 0.91, 1.02,
    # 1.06 and 1.17. The empirical rejection rate at that envelope is the
    # number to plan with -- walk200's 101-round eval logged pre-vla 12/101
    # (~12%) -- not the 1.55 m cutoff, which the settled posture never reaches.
    #
    # A gated round is NOT retried -- the retry loop at the bottom of run_gym
    # fires only on '[stand]'. It is written to the manifest as a failed round
    # and aggregates under the 'pre-vla' stage. So --spawn-edge much past 1.5
    # does not sample a harder distribution, it manufactures failures.
    #
    # The gate itself is camera geometry (fovy, mount pitch), NOT a function
    # of spawn distance -- widening it to "compensate" for a farther spawn
    # would just let an invisible object through. Same signal and same
    # thresholds as the arrival gate further down, so a round that passes
    # here is being judged by the rule that judges it later.
    # Diagnostic video can include readiness/visibility failures without
    # changing the training recorder's acceptance window.
    include_init_video = ctx.video_include_init
    if include_init_video:
        stamp = time.time()
        reset.start_video(f"gym_{int(stamp)}")
        ctx.execution.metrics["video_start_wall"] = stamp
    if ctx.reach_config.ready_grasp_baseline and ctx.vla_policy is None:
        from locomani.tasks.bottle.ready_grasp_policy import prepare_ready

        # This drives the expert encoder's upper-body reference. Token-only
        # deployment initializes through its own VLA prelude; asking it to
        # execute this reference can time out before the first model request.
        if not prepare_ready(p):
            return False, "[stand] baseline ready posture not delivered"
    elif ctx.reach_config.ready_grasp_baseline and ctx.vla_policy is not None and initial_pose.INIT_TOKEN_FILE:
        from locomani.backends.sonic.initial_pose import prepare_token_pose
        from locomani.tasks.bottle.ready_grasp_policy import ready_q

        if not prepare_token_pose(p, ready_q()[15:]):
            return False, "[stand] token-only ready posture not delivered"
    _o0 = p.obs()["objects"][ctx.obj_key]
    cam0 = _o0.get("cam")
    # `cam is None` means BEHIND the camera and must fail too -- the arrival
    # gate below always treated it that way, this one used to let it through.
    # Guarded on the key's presence, not its value, so a publisher built
    # without a camera still runs (the key is absent then, not None).
    if "cam" in _o0 and (cam0 is None or abs(cam0[0]) > ctx.view_gate_x or abs(cam0[1]) > ctx.view_gate_y):
        return False, (
            "[view] object not in ego view at spawn (cam="
            + ("behind camera)" if cam0 is None else f"[{cam0[0]:+.2f} {cam0[1]:+.2f}])")
        )
    if not include_init_video:
        stamp = time.time()
        reset.start_video(f"gym_{int(stamp)}")
        ctx.execution.metrics["video_start_wall"] = stamp
    _wmark("video")
    if ctx.record:
        # 'c' toggles the exporter IDLE -> RECORDING; a second 'c' saves and
        # 'x' discards (zmq_driver.SonicCommander.key). Recording starts HERE,
        # after the settle, so the first frame of the episode is the first
        # frame of the task rather than a robot standing still.
        p.cmd.key("c")
        if ctx.rec is not None:
            ctx.rec["on"] = True
    ctx.task_grasp = ctx.tabletop_grasp_cls(
        p,
        TabletopTarget(
            key=ctx.obj_key,
            side=ctx.side,
            below_center=ctx.reach_config.grip_below_center if ctx.reach_config.ready_grasp_recovery else None,
            grip_z=obj_top(ctx.meta, ctx.obj_key) - grasping.GRIP_BELOW_RIM,
            surface_top=ctx.surf_a["top"],
            box=ctx.meta["objects"][ctx.obj_key]["box"],
            body=ctx.meta["objects"][ctx.obj_key]["body"],
            surface_bodies=tuple(s["body"] for s in ctx.meta["surfaces"].values()),
        ),
    )
    ctx.grip_za = float(ctx.task_grasp.grip(p.obs())[2])

    _rel_dbg(ctx, "pre-walk ", p.obs())
    return None


def _phase_handover_to_vla(ctx: RoundContext):
    p, reset = ctx.p, ctx.reset
    # ---- policy fork: everything above is shared, everything below is the
    # scripted policy. A VLA round hands over here facing the object.
    #
    # NOT "settled at its spawn": measured on every eval round set (R2 62%,
    # R3 76%, R4 69%, ens90 84%), by the time the VLA's stream starts the
    # robot is ALREADY AT THE STAND POSE in ~2/3 of rounds -- something
    # moves it ~0.35 m table-ward between the settle drift-check and the
    # first streamed frame (the 2 s IDLE hold below the drift check plus
    # the VLA client's prelude). The fork log line and the trace's
    # entry_pelvis/tstart_pelvis fields exist to pin down which segment;
    # until that is fixed, closed-loop numbers describe a mixed task
    # (~1/3 walk+reach, ~2/3 stand-start reach).
    if ctx.vla_policy is not None:
        s_fork = p.obs()
        d_fork = float(np.linalg.norm(np.asarray(p.object_pos(ctx.obj_key, s_fork)[:2]) - p.pelvis_xy(s_fork)))
        p.log(f"[gym] fork->vla: pelvis {np.round(p.pelvis_xy(s_fork), 3)} obj-dist {d_fork:.3f} m")

        def replace():
            # Mid-prelude re-teleport: the gait limit cycle walks drift-prone
            # spawn configs ~0.15 m/s through every planner countermeasure
            # (hold/squat/walk_to -- all measured), so the position is
            # RESTORED physically instead. Identical args -> identical
            # scene; label "replace" so frame_rate_guard's marker rule can
            # exclude it. qvel is zeroed by the reset, which also kills the
            # cycle's momentum.
            reset.reset(objects=ctx.reset_objects, robot=ctx.reset_robot, appearance=ctx.appearance, label="replace")
            time.sleep(0.8)  # stance settle; deploys tolerate teleports

        # spawn_xy is the ACCEPTANCE REFERENCE for tstart_spawn_off. It
        # used to be landed_xy (the raw teleport landing), which folded the
        # planner-mode settle (~0.13 m median, walk200) into the metric --
        # a segment collection rounds share, so the 0.1 m acceptance
        # complained every round about a displacement training also saw.
        # s_fork is the post-settle pose at handover: from here the metric
        # measures only what the VLA prelude itself adds (hold + blend).
        ctx.execution.enter("vla")
        return ctx.vla_policy(
            p=p,
            reset=reset,
            meta=ctx.meta,
            obj_key=ctx.obj_key,
            psi=ctx.psi,
            fv=ctx.fv,
            grip_za=ctx.grip_za,
            record=ctx.record,
            replace=replace,
            spawn_xy=[float(v) for v in p.pelvis_xy(s_fork)],
        )
    return None


def _phase_navigate(ctx: RoundContext):
    p = ctx.p
    ctx.execution.enter("navigate")

    plan_r = float(np.hypot(ctx.rel[0], ctx.rel[1]))
    ctx.plan_r = plan_r
    # Two-stage arrival. The WALK is precise (telemetry: walk_to returns
    # within 0.05 m of its target) — what scatters is the DECELERATION:
    # 0.03-0.34 m of momentum creep lands AFTER walk_to returns. So stop
    # the momentum walk 0.30 m short (heading held: face_target near a
    # close target spun the robot 110-160 deg at the stop), wait 2 s for
    # the creep to finish, then CRAWL the remainder inside the gait's
    # deadband: a 0.30 speed command measures ~0.05 m/s, i.e. near-zero
    # momentum, closed 50 Hz on the LIVE can depth — stop the tick it
    # enters the plan band.
    # The reach begins ON THE WAY IN, not after arriving. Leg 1 walks with
    # the arms swinging; leg 2 hands the 17 upper-body joints over and drives
    # them to READY by distance remaining, so the arm is up exactly when the
    # body arrives.
    #
    # The handover used to be seeded from the arm's CURRENT MEASURED angles
    # and was believed to have no step in it. It has two, both of them in the
    # REFERENCE the encoder observes rather than in the motion:
    #
    #  1. the measured pose is not the quantity the deploy replaces. It
    #     substitutes the 17 values into `current_motion_`'s joints verbatim
    #     (g1_deploy_onnx_ref.cpp:781), so what is being displaced is the
    #     k-planner's own arm reference -- and the two differ by the whole
    #     tracking error of the frozen decoder. Seed from `body_q_target`,
    #     which IS that quantity (upper_ref.RefTap).
    #  2. the rise is parameterised by DISTANCE REMAINING, which is not a
    #     rate. A round that skips leg 1 starts its approach already inside
    #     RAISE_LEAD, evaluates the smoothstep at u=0.3-0.6 on the first tick
    #     and commands a third of the whole rise in 20 ms. UpperRamp caps the
    #     per-tick change and sends the matching `upper_body_velocity`, which
    #     nothing on this path used to send at all.
    lead_tgt = np.asarray(ctx.spawn_xy) - (0.30 + ctx.raise_lead) * np.array(ctx.fv[:2])
    # ALONG-TRACK, not distance. The lead point is fixed 1.30 m behind the
    # stand while the spawn is now sampled anywhere from 1.0 to 3.0 m back,
    # so for a close spawn the lead point sits BEHIND the robot -- and a
    # plain distance test still fires, because a lateral offset keeps the
    # distance above the old 0.20 m threshold. Leg 1 would then walk the
    # robot backwards away from the table. Project onto the approach
    # direction and skip leg 1 when there is nothing forward to walk to;
    # the deadband approach closes the whole gap on its own, lateral error
    # included, because it steers at the stand rather than pushing along
    # the heading.
    along = float(np.dot(lead_tgt - np.asarray(p.pelvis_xy(p.obs())), np.array(ctx.fv[:2])))
    if along > 0.20:
        # face_yaw, not face_target=False: hold the LINE, not the current
        # heading. See Primitives.walk_to -- holding the heading locks in the
        # spawn turn's dead-zone residual, which a long walk then magnifies
        # into a lateral error the deadband approach cannot erase.
        if not p.walk_to(
            lead_tgt, stop_dist=0.10, timeout=10.0 + 8.0 * ctx.spawn_off, face_target=False, face_yaw=ctx.psi
        ):
            return False, "[stand] fell during the walk-up"
    else:
        p.log(f"[gym] leg 1 skipped (lead point {along:+.2f} m along track)")
    q_meas = upper_body_from_mj(p.obs()["joints"])
    q_free = upper_ref.shared().upper17()
    if q_free is None:  # stale, not merely un-started
        # A cold tap must LOOK cold -- seeding from the measurement silently
        # is exactly the bug this replaced.
        q_free = q_meas
        p.log(
            "[gym] upper takeover: no body_q_target (tap cold) -- seeded "
            "from the measured pose, the reference step is back"
        )
    else:
        gap = np.abs(q_free - q_meas)
        worst = int(np.argmax(gap))
        p.log(
            f"[gym] upper takeover: reference-vs-measured "
            f"{np.degrees(gap[worst]):.1f} deg on "
            f"{UPPER_BODY_NAMES[worst]} (mean {np.degrees(gap.mean()):.1f})"
        )
    if ctx.reach_config.ready_grasp_baseline:
        from locomani.tasks.bottle.ready_grasp_policy import ready_q

        q_free = upper_body_from_mj(ready_q())
    ctx.q_free = q_free
    ctx.arm_ramp = upper_ref.UpperRamp(p.cmd, q_free)
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY

    coordinated = None
    if ctx.reach_config.ready_grasp_baseline:
        from locomani.tasks.bottle.ready_grasp_policy import ReadyGraspPolicy

        arrival_task = ctx.task_grasp
        controller = ReadyGraspPolicy
        if ctx.reach_config.ready_grasp_recovery:
            from locomani.tasks.bottle.recoverable_grasp_policy import RecoverableGraspPolicy

            controller = RecoverableGraspPolicy
        if ctx.reach_config.coupled_squat:
            from locomani.tasks.bottle.coupled_grasp_policy import CoupledGraspPolicy

            controller = CoupledGraspPolicy
        coordinated = controller(p, arrival_task, ctx.reach_config, ctx.arm_ramp, ctx.execution)
    ctx.coordinated = coordinated
    return None
