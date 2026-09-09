"""The grasp, on SONIC's streamed-motion channel.

Planner-mode wrist targets pass through the kinematic planner and arrive with
a 25 cm warp and a second of lag; three weeks of tuning got the closed loop to
3.7 cm, which is not a grasp. Pico teleop never touches that channel — it
streams full 29-DoF joint references on the `pose` topic and the policy
imitates them. diag_stream.py measured this channel at ~1 cm palm error with
one or two IK refinement steps, standing rock-solid the whole time.

This module replaces the planner-era crouch/track/servo_grasp/grasp/lift chain
for the grasp stage only: walking and carrying stay in planner mode, and the
deploy is switched back the moment the object is lifted.

The servo logic is servo_grasp's (align across the jaw at a safe depth, then
advance gated on alignment) — what changes is the actuation underneath it:
IK on the scene model streamed as joint references, instead of wrist targets
nudged through the planner.

Split across three files (refactor step 4b family 2, pure extraction;
decision 9's file-size cap): this file keeps `make_arm`, `_raise_hand`,
`_creep_to_band`, `toppled`, `reanchor`, `execute_grasp`/`_execute_grasp` and
the `ApproachStatus`/`ApproachResult` types; `streamed_arm.py` has
`StreamedArm`/`PlannerArm`; `servo.py` has `_servo_continuous` (631 lines,
split there into a `ServoState` dataclass plus one function per phase).
Every name that used to live here and moved is re-exported below, so
`from locomani.backends.sonic.stream_reach import X` still works -- this
matters beyond the usual import compatibility: several tests
`patch("locomani.backends.sonic.stream_reach.X", ...)` or
`patch.object(stream_reach, "X", ...)` (`toppled`, `_servo_continuous`,
`make_arm`, `PlannerArm`, `_creep_to_band`), and diagnostic tools read
`sr.SAFE_GAP` / `sr._HAND` / `sr.SQUAT_SETTLE` off the module directly. Every
one of those names is kept ON THIS MODULE OBJECT (by staying defined here, or
by a plain top-level import that binds it into this module's own namespace),
which is what makes `patch.object(this module, name)` and bare-attribute
reads see the same thing a caller inside this file would. The one place that
is not enough: `_servo_continuous`'s own internal `toppled(...)` check (in
servo.py's `_tick_prelude`) reads `stream_reach.toppled` back through a
deferred, module-qualified reference instead of a plain imported name --
see the comment there for why.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from locomani.backends.sonic.primitives import DT, LocomotionMode, Primitives
from locomani.backends.sonic.primitives_const import HAND as _HAND
from locomani.backends.sonic.servo import (
    CLOSE_ENT_MAX,
    CLOSE_RAMP_S,
    CLOSE_SEATED,
    FREEZE_TRAVEL,
    KP_TRACK,
    LAST_EGO_VIS,
    LEAD_MAX,
    LIFT_MIN,
    PRE_NARROW,
    PRE_X_MAX,
    PRE_X_MIN,
    PRE_XY_MAX,
    PRE_Z_MAX,
    PRE_Z_MIN,
    SAFE_GAP,
    SAG_COMP,
    STAGE_UP,
    V_APPROACH,
    V_DESCEND,
    V_LIFT,
    V_TRAVEL,
)
from locomani.backends.sonic.servo_ticks import _servo_continuous
from locomani.backends.sonic.streamed_arm import (
    CHUNK,
    FINE_SPEED,
    JOINT_SPEED,
    LEFT_GUARD,
    LEFT_GUARD_RATE,
    LEFT_RETREAT_M,
    LEFT_SAFE_M,
    NUDGE_MAX,
    REFINE_MAX,
    REFINE_TOL,
    SETTLE,
    SQUAT_DROP_MAX,
    SQUAT_GAIN,
    SQUAT_HEIGHT_BIAS,
    SQUAT_MAX,
    SQUAT_SETTLE,
    PlannerArm,
    StreamedArm,
    yaw_only,
)
from locomani.pipeline.execution_record import ExecutionRecord, GraspResult
from locomani.tasks.bottle.config import DEFAULT_BOTTLE_TASK
from locomani.tasks.bottle.reach_config import DEFAULT_REACH, ReachConfig

# BottleTaskConfig.{arm_channel,arm_vel} defaults (decision 8, were
# GYM_ARM_CHANNEL/GYM_ARM_VEL); module-level so a caller can override by
# assignment, same pattern as grasp_gym.RESTART_DEPLOY. (left_guard moved to
# streamed_arm.py with StreamedArm/PlannerArm, and is re-exported above.)
ARM_CHANNEL = DEFAULT_BOTTLE_TASK.arm_channel
ARM_VEL = DEFAULT_BOTTLE_TASK.arm_vel


class ApproachStatus(str, Enum):
    READY = "ready"
    TIMEOUT = "timeout"
    FALLEN = "fallen"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ApproachResult:
    status: ApproachStatus
    depth: float
    elapsed: float
    azimuth: float


# Shared depth limit, independent of the sampled navigation waypoint. The
# bottle pilot physically grasped at yaw-frame depths 0.294..0.381 m. This
# 0.40 m gate is a screening bound, not a collision-free IK certificate;
# post-crouch replanning and actual grasp verification remain mandatory.
TRANSITION_POST_MAX = DEFAULT_REACH.post_max

# Below: constants left over from the retired discrete servo (servo_grasp),
# unused by anything in this file or elsewhere (checked at the 4b/family-2
# split) -- left in place rather than deleted, since no ruling has retired
# them (docs/reference/standards.md's "delete over keep" needs a ruling).
DEPTH_TOL = 0.03
# Looser than servo_grasp's 6 mm: that loop nudged a 50 Hz drifting command,
# this one hops between IK solutions with a ~1 cm delivery floor, so an 8 mm
# gate just oscillates. The advance gate keeps re-centring on the way in.
ALIGN_TOL = 0.013
GATE_TOL = 0.018
GRASP_TOL = 0.020
STEP_MAX = 0.03  # m, largest palm correction per servo step
CROUCH_BACK = 0.09  # m the planner crouch moves the pelvis BACK (measured 4-13 cm, 2026-09-03)
SQUAT_TRIGGER = 0.03


def make_arm(p: Primitives, side: str):
    """Build the arm driver for the channel this run is testing.

    The default is now `planner`: it keeps the deploy in planner mode for
    the whole grasp, so there is no mode switch, no latch-clear window and
    no stand twitch. Measured 12/12 with the sim pin OFF against 2/12 for
    the streamed channel with its pin ON, same seed (see the Phase 0
    commit). BottleTaskConfig.arm_channel="streamed" restores the old path.
    """
    if ARM_CHANNEL == "planner":
        return PlannerArm(p, side, send_vel=ARM_VEL)

    return StreamedArm(p, side)


RAISE_CLEAR = 0.10  # palm above the tabletop while walking the last bit


def _raise_hand(p: Primitives, side: str, surface_top: float | None, grip_world=None) -> None:
    """Lift the palm above the tabletop BEFORE walking in past the edge.

    The squat-then-raise order used to put the hand below the rim and drag it
    up into the table. Raising first has to happen out at the approach ring —
    inside the ring the swing itself can clip the edge — and it has to happen
    on the PLANNER channel, because the walk that follows is planner-mode and
    the wrist target persists through it (the carry leg has walked 2.4 m on
    exactly this mechanism for every successful episode so far). Position-only
    and ~5 cm sloppy, but "above the table" needs no better.
    """
    if surface_top is None:
        return
    from locomani.backends.sonic.agent_server import LocomotionMode
    from locomani.backends.sonic.primitives import READY_WRISTS

    s = p.obs()
    # The arms already travel in the READY pose (forearms level, hands above
    # tabletop height) for the whole episode, so this is a trim, not a move:
    # keep the ready x/y and only push the height up if THIS surface or can
    # rim demands more than the ready pose gives.
    top = surface_top + RAISE_CLEAR
    if grip_world is not None:
        top = max(top, float(grip_world[2]) + 0.12)
    ready = np.asarray(READY_WRISTS[side], dtype=np.float32).copy()
    z_pf = float(top - s["pelvis"]["pos"][2])
    if z_pf <= ready[2] + 0.005:
        return  # ready height already clears everything
    ready[2] = z_pf
    p.cmd.set_wrists(**{f"{side}_pos": ready})
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.log(f"  stream: ready hand trimmed up to clear the surface (z_pf {z_pf:+.2f})")


def _creep_to_band(
    p,
    grip_world,
    side: str,
    x_min: float = 0.18,
    x_max: float = 0.27,
    max_s: float = 8.0,
    target_rel=None,
    envelope=None,
    update_upper=None,
    target_lateral=None,
) -> ApproachResult:
    """Reach an observable depth/cone gate; timeout and falling are failures.

    target_rel is accepted for old callers but never controls termination.
    Random approach waypoints may vary; identical live geometry has the same
    transition rule. Both this depth and the post-crouch depth are yaw-only.
    """
    t0 = time.time()
    ready_since = None
    status = ApproachStatus.TIMEOUT
    aim_max = float(envelope[2]) if envelope is not None else np.deg2rad(36.5)
    while time.time() - t0 < max_s:
        s = p.obs()
        if s["fallen"]:
            status = ApproachStatus.FALLEN
            break
        if update_upper is not None and not update_upper(s):
            status = ApproachStatus.BLOCKED
            break
        grip = np.asarray(p.world_to_root(grip_world, s))
        gx = float(grip[0])
        az = float(np.arctan2(grip[1], grip[0]))
        in_band = x_min <= gx <= x_max
        in_cone = abs(az) <= aim_max
        if target_lateral is not None:
            in_cone = in_cone and abs(grip[1] - target_lateral) <= 0.025
        yaw = p.pelvis_yaw(s)
        # Stop translating while validating the band. The old branch kept
        # walking forward even while accumulating its ready counter.
        if in_band and in_cone:
            if ready_since is None:
                ready_since = time.time()
            if time.time() - ready_since >= 0.2:
                status = ApproachStatus.READY
                break
            face = yaw
            mv = (0.0, 0.0, 0.0)
            speed = 0.20
        else:
            ready_since = None
            face = yaw + float(np.clip(az, -0.5, 0.5)) if not in_cone else yaw
            fv = (np.cos(face), np.sin(face), 0.0)
            if target_lateral is not None:
                # Align the body to put the object ahead of the active hand.
                # The released gait tracks forward/turn commands more reliably
                # than tiny lateral shuffles with the arms held in front.
                desired_az = float(np.arctan2(target_lateral, max(gx, 0.10)))
                face = yaw + float(np.clip(az - desired_az, -0.35, 0.35))
                direction = 0.0 if in_band else (-1.0 if gx < x_min else 1.0)
                mv = (direction * np.cos(face), direction * np.sin(face), 0.0)
                speed = 0.215
            elif not in_cone:
                mv, speed = (0.0, 0.0, 0.0), 0.215
            else:
                backing = gx < x_min
                err = x_min - gx if backing else gx - x_max
                mv = (-fv[0], -fv[1], 0.0) if backing else fv
                speed = 0.20 if err < 0.05 else 0.215
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=speed, movement=mv, facing=(np.cos(face), np.sin(face), 0.0))
        time.sleep(DT)
    s = p.obs()
    if s["fallen"]:
        status = ApproachStatus.FALLEN
    grip = np.asarray(p.world_to_root(grip_world, s))
    az = float(np.arctan2(grip[1], grip[0]))
    if status == ApproachStatus.READY and not (
        x_min <= grip[0] <= x_max
        and abs(az) <= aim_max
        and (target_lateral is None or abs(grip[1] - target_lateral) <= 0.025)
    ):
        status = ApproachStatus.TIMEOUT
    result = ApproachResult(status, float(grip[0]), time.time() - t0, az)
    # Never leave a walking command active on failure.
    if status != ApproachStatus.READY:
        p.cmd.frame(
            mode=LocomotionMode.IDLE,
            speed=0.0,
            movement=(0.0, 0.0, 0.0),
            facing=(np.cos(p.pelvis_yaw(s)), np.sin(p.pelvis_yaw(s)), 0.0),
        )
    p.log(
        f"  stream: approach {status.value}, depth {result.depth:.3f} m "
        f"in [{x_min:.3f}, {x_max:.3f}], az {np.degrees(az):+.1f} deg, "
        f"{result.elapsed:.2f} s"
    )
    return result


def toppled(s, obj_key: str) -> bool:
    """Has the (upright-cylinder) object fallen over? Its body z-axis more
    than ~60 deg off vertical. Failure ATTRIBUTION: a toppled can later
    surfaces as 'grip height unreachable' (the grasp line follows the body
    origin down to lying height), which blames the arm for a knock that
    happened during the approach."""
    rot = Primitives._rot(s["objects"][obj_key]["quat"])
    return float(rot[2, 2]) < 0.5


def reanchor(p: Primitives, side: str = "right") -> None:
    """Force a planner heading re-anchor at a yaw this process knows.

    The deploy may carry a stale anchor from a previous process's streamed
    session (the anchor lives in the deploy, the compensation in the
    commander). A two-second streamed toggle re-anchors it at the robot's
    current yaw and records that yaw in cmd.heading_offset.

    This one is ALWAYS a StreamedArm, never make_arm(): the re-anchor IS the
    mode switch, so a PlannerArm here is a no-op and the deploy keeps the
    stale anchor -- the robot then walks off closing a standing heading
    error (measured 1.01 m of drift through the 4 s settle, which the gym
    reports as "teleport did not land"). The twitch this toggle costs
    happens before the recording starts, so it is free.
    """
    arm = StreamedArm(p, side)
    arm.start()
    arm.hold(0.3)
    arm.stop()


def execute_grasp(p, obj_key, grip_world, plan, *, execution=None, **options):
    """Approach, posture delivery, measured replan, servo; return owned results.

    Optional execution joins a containing task's journal. No caller reads globals.
    Failure stage is the first control failure, not a post-hoc physical taxonomy.
    """
    execution = execution if execution is not None else ExecutionRecord()
    execution.transition.clear()
    execution.metrics.pop("ego_visibility", None)
    try:
        ok, why = _execute_grasp(p, obj_key, grip_world, plan, execution=execution, **options)
    except Exception as exc:
        execution.fail(f"{type(exc).__name__}: {exc}")
        raise
    if not ok:
        execution.fail(why)
    return GraspResult(ok, why, execution.snapshot())


def _execute_grasp(
    p: Primitives,
    obj_key: str,
    grip_world,
    plan,
    side: str = "right",
    surface_top: float | None = None,
    replan=None,
    ghost=None,
    nudge: bool = True,
    pin=None,
    squat_hint: float | None = None,
    stand_rel=None,
    envelope=None,
    plan_r: float | None = None,
    *,
    execution,
    config=DEFAULT_REACH,
    transport_guard_factory=None,
    coordinated=None,
) -> tuple[bool, str]:
    """Dispatch the grasp to the recovery controller. -> (ok, why).

    Refactor step 2, family 2 (2026-09-09): the old non-recovery approach/
    crouch/servo tail (_creep_to_band, GYM_ONE_SHOT, _nudge_pass/_nudge_closer,
    reach retry, GYM_IDLE_SWEEP) is retired -- every surviving collection
    profile sets ready_grasp_recovery=True and grasp_gym always builds a
    recovery controller when ready_grasp_baseline is True. See
    docs/handoff/refactor-step2-candidates.md.
    """
    execution.metrics["reach_config"] = config.to_dict()
    if not config.ready_grasp_recovery:
        return False, "ready_grasp_recovery=False is retired; every collection profile now uses a recovery controller"
    if coordinated is None or not hasattr(coordinated, "execute"):
        return False, "recovery configuration requires a recovery controller"
    return coordinated.execute(grip_world=grip_world, squat_hint=squat_hint, ghost=ghost)
