"""Grasp-only experiment environment: teleport to a sampled stand, grasp.

Navigation is solid; the grasp is what needs iterations, and a full episode
spends ~2/3 of its wall clock walking. Each gym round:

    1. sample a can placement (nominal +-10 cm, staying >=10 cm inboard) and
       a stand: the can's pelvis-frame offset drawn from the reach map
       (squat extrusion included, facing offset within +-30 deg — the band
       stage-2 navigation actually delivers), robot world yaw by rejection
       until the stand clears the table edge by the hip margin;
    2. reset: teleport robot + can there, settle, two short turn bursts to
       inject the gait transient a walked approach would have left;
    3. run the SAME pre-grasp + grasp + hold code path as collect_episode
       (READY pin, ego-view gates, streamed servo with the actuator pin,
       2 s hold) — nothing gym-specific downstream of the stand, so what
       works here transplants to the full episode unchanged.

Result lines match collect_episode ("EPISODE OK/FAILED [stage] why") so
par_batch's grep and the cross-batch statistics stay comparable.

Usage:  python src/locomani/tasks/bottle/grasp_gym.py [--rounds N] [--seed K]
Assumes the stack is up (start_stack.sh).

Split across six files (refactor step 4b, pure extraction; decision 9's
file-size cap): this file keeps `main`, `_build_scene_and_plan` (scene/target
setup) and the round loop's own entry points (`run_round`, `_run_round`);
`cli_args.py` has the argparse CLI; `round_loop.py` has the per-round
sampling + execution loop; `plan.py` has the per-target stand planner and
per-round sampling helpers; `round_phases.py` + `round_phases_approach.py`
have `_run_round`'s five phases behind a `RoundContext`. Every name that used
to live here and moved is re-exported below, so
`from locomani.tasks.bottle.grasp_gym import X` still works.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)  # reach src/ for `python src/locomani/tasks/bottle/grasp_gym.py`

from locomani import paths
from locomani.backends.sonic import initial_pose, upper_ref  # noqa: E402, F401 -- re-exported, see docstring
from locomani.backends.sonic.primitives import (  # noqa: E402
    LocomotionMode,
    Primitives,
    StreamMode,
    wrap,
)
from locomani.pipeline.execution_record import ExecutionRecord, RoundResult
from locomani.sim.privileged_pub import EpisodeControlClient  # noqa: E402
from locomani.tasks.bottle import (  # noqa: E402
    cli_args,
    reach_map,  # noqa: E402
    round_loop,
)
from locomani.tasks.bottle.config import DEFAULT_BOTTLE_TASK
from locomani.tasks.bottle.plan import (
    TargetPlan,  # noqa: F401 -- re-exported, see module docstring
    build_rel_candidates,  # noqa: F401
    plan_for_target,
    resolve_pool,
    sample_object_yaw,  # noqa: F401
)
from locomani.tasks.bottle.reach_config import DEFAULT_REACH, ReachConfig
from locomani.tasks.bottle.round_phases import RoundContext
from locomani.tasks.bottle.round_phases import _phase_handover_to_vla as _p_handover
from locomani.tasks.bottle.round_phases import _phase_navigate as _p_navigate
from locomani.tasks.bottle.round_phases import _phase_spawn_and_settle as _p_spawn
from locomani.tasks.bottle.round_phases_approach import _phase_approach as _p_approach
from locomani.tasks.bottle.round_phases_approach import _phase_grasp_and_verify as _p_grasp
from locomani.tasks.bottle.tabletop_grasp import TabletopGrasp  # noqa: F401 -- re-exported, see docstring

SCENE_META = paths.SCENE_META

# Moved verbatim from the retired src/locomani/collect_episode.py
# (refactor step 2, family 2, 2026-09-09) -- grasp_gym was the last live
# importer of these four names.
PALM_DRIFT_MAX = 0.03  # 3 cm of slip in the palm frame before it counts as dropped
STAND_SETTLE = 4.0  # seconds of IDLE after a reset before any walking


def obj_top(meta: dict, obj_key: str) -> float:
    """World z of the object's top, from its bounding box."""
    o = meta["objects"][obj_key]

    return float(o["init_pos"][2] + np.asarray(o["box"]["half_extent"])[2])


def palm_frame_offset(s: dict, obj_key: str, side: str) -> np.ndarray:
    """Object position expressed in the palm's frame."""
    palm = np.array(s["palms"][side]["pos"])
    w, x, y, z = s["palms"][side]["quat"]
    # Rotation matrix from a scalar-first quaternion.
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )

    return R.T @ (np.array(s["objects"][obj_key]["pos"]) - palm)


EDGE_MIN = 0.17  # pelvis-to-edge HARD floor, enforced at arrival.
# 0.10 -> 17 hip contacts, 0.12 -> 22: the crouch
# lean protrudes ~14 cm past the pelvis centre;
# 0.15 still brushed the hip 15x/batch once the
# planned crouch deepened to 13-15 cm delivered
# How far before the stand the arm starts coming up, and therefore where the
# momentum walk hands over to the deadband approach. Module scope because the
# sampler needs it too: the walk is two legs, spawn -> lead point -> stand,
# and both have to clear the furniture.
RAISE_LEAD = 1.00
# Ceiling on the arm rise finishing AT the table. The slew (2 deg/tick,
# upper_ref.MAX_STEP) needs ~0.6 s for the largest residual a skipped leg 1
# can leave; anything past that is recorded standing, so it is bounded and
# logged rather than waited out.
RAISE_FINISH_MAX_S = 0.30
EDGE_MAX = 0.32  # sampling ring upper bound: core-aim stands sit
# at rel depth 0.18-0.36 (the sweet slice), so
# they clear the edge by more than the old 0.20
CAN_INBOARD = (0.04, 0.08)  # can centre this far in from its nearest edge:
# close enough that a 15 cm stand-off still
# reaches it, far enough that it cannot roll off

# Ego-frame visibility gate: (u_x, u_y) is the object's position in the ego
# camera frame, normalised so the frame EDGE is 1.0 (diag_ego_geom.frame_pos).
# A camera-geometry constant (fovy, mount pitch), not spawn distance --
# --view-gate-x/-y (defaults below) thread as view_gate_x/view_gate_y through
# run_round/_run_round to the spawn check and the arrival check.

# Per-round deploy restart (BottleTaskConfig.restart_deploy). OFF by default
# since 2026-09-10: the deploy keeps its heading anchor across sim resets, which
# used to make rounds 2+ wander or fall; round_phases now re-anchors it after
# every reset instead (the crate line's streamed toggle), and 30 paired rounds
# showed no yield difference at 34 s a round instead of 64 (ledger bottle §11).
# True restores the restart (~33 s a round); eval_closed_loop pins it True.
# Module-level default (decision 8), same pattern as vla_round.GRIP_GAIN: a
# caller assigns grasp_gym.RESTART_DEPLOY to override.
RESTART_DEPLOY = DEFAULT_BOTTLE_TASK.restart_deploy
RESTART_SCRIPT = f"{paths.REPO}/scripts/restart_deploy.sh"

# BottleTaskConfig.{pose_sidecar,video_include_init,arm_pin} defaults (decision 8,
# were LOCOMANI_POSE_SIDECAR / SIM_VIDEO_INCLUDE_INIT / GYM_ARM_PIN); module-level
# so a caller can override by assignment, same pattern as RESTART_DEPLOY above.
POSE_SIDECAR = DEFAULT_BOTTLE_TASK.pose_sidecar
VIDEO_INCLUDE_INIT = DEFAULT_BOTTLE_TASK.video_include_init
ARM_PIN = DEFAULT_BOTTLE_TASK.arm_pin


def instruction(phrase: str) -> str:
    """The episode's task string. One place, so the dataset and any future
    eval harness cannot drift apart."""
    return f"pick up {phrase}"


def start_control_once(p) -> None:
    """start_control + remember that THIS deploy now holds a heading anchor.

    The deploy captures its heading anchor at its first start_control and keeps it until it is
    restarted. `p._gym_started` is the gym's record of that; round_phases reads it to decide
    between "anchor = landing yaw" (fresh deploy) and "hold + reanchor" (anchor is stale).
    The boot calibration below calls start_control at the NOMINAL pose (yaw 0), so without
    setting the flag here round 1 of every batch treated its -90 deg spawn as freshly anchored:
    17/17 first rounds of visible200_20260910 failed at the spawn (turn_to timeout / object
    out of the ego view), rounds 2+ were fine (docs/ledger/bottle.md §12).
    """
    p.cmd.start_control()
    p._gym_started = True


def fresh_deploy(p) -> bool:
    """Kill + relaunch this instance's deploy; returns True when it's up."""
    try:
        r = subprocess.run(["bash", RESTART_SCRIPT], capture_output=True, text=True, timeout=360)
    except subprocess.TimeoutExpired:
        print("[gym] deploy restart timed out")
        return False
    if "deploy restarted" not in r.stdout:
        print(f"[gym] deploy restart failed: {r.stdout[-300:]}")
        return False
    p._gym_started = False  # fresh process: nothing to quiesce
    # The heading compensation pairs with the DEAD deploy's anchor: the
    # commander keeps rotating facing/movement by the previous round's yaw
    # (~-80 deg) while the fresh deploy anchors clean — every IDLE frame
    # then steers sideways and the stand drifts 0.4-0.6 m ("teleport did
    # not land"). A fresh deploy starts like a fresh boot: offset zero.
    p.cmd.heading_offset = 0.0
    time.sleep(4.0)  # let the engines finish lazy-loading before
    # the round's start_control leans on them
    return True


@dataclass
class _SceneSetup:
    """Everything `_build_scene_and_plan` computes for `_round_loop` to use."""

    fixed_cases: list | None
    sampled_cases: list
    reach_config: ReachConfig
    edge_range: tuple | None
    off_lo: float
    off_hi: float
    sidecar: object | None
    rng: np.random.Generator
    meta: dict
    pool: list
    surf_a: dict
    half: np.ndarray
    cxy: np.ndarray
    surf_b: dict | None
    rects: list
    edge_dist: object
    rm: reach_map.ReachMap
    plans: dict
    p: object | None
    reset: object | None
    robot_nom: np.ndarray | None


def _build_scene_and_plan(args: argparse.Namespace) -> _SceneSetup:
    """Scene/target/sampling setup, split out of main() (refactor step 4b)."""
    from locomani.pipeline.experiment_cases import load_cases, save_cases

    fixed_cases = load_cases(args.cases, SCENE_META) if args.cases else None
    if fixed_cases is not None and args.rounds > len(fixed_cases):
        # Was ap.error(...) when this check lived in main() next to the
        # parser (pre refactor step 4b); ap isn't in scope split out into
        # its own function, so this loses argparse's usage-string preamble
        # and its exit code 2 (this is a plain exit 1) but still refuses to
        # run with a clear message on stderr.
        raise SystemExit("requested more rounds than fixed cases")
    sampled_cases = []
    reach_config = ReachConfig.load(args.reach_config) if args.reach_config else DEFAULT_REACH
    if args.manifest:
        # collect.sh mkdir -p's its own $OUT before this, but a bare
        # `--manifest outputs/logs/x.jsonl` run (the README quickstart) would
        # otherwise fail with FileNotFoundError on the FIRST round's write,
        # after already having paid for the reset + walk + grasp.
        manifest_dir = os.path.dirname(args.manifest)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
    off_lo, off_hi = (args.spawn_off * 2)[:2]
    edge_range = None
    if args.spawn_edge is not None:
        edge_range = tuple(sorted((args.spawn_edge * 2)[:2]))
        if edge_range[0] < EDGE_MIN:
            raise SystemExit(f"--spawn-edge {edge_range[0]} is inside the pelvis-to-edge floor EDGE_MIN={EDGE_MIN}")
    if off_lo > off_hi:
        off_lo, off_hi = off_hi, off_lo
    # World-pose sidecar: one jsonl per attempt, next to the manifest. Only
    # when recording -- an eval round has its own trace.
    # BottleTaskConfig.pose_sidecar=False (was LOCOMANI_POSE_SIDECAR=0) opts
    # out. Fail-safe: a dead sidecar prints and costs nothing.
    sidecar = None
    if args.record and args.manifest and POSE_SIDECAR:
        from locomani.sim.privileged_pub import PoseSidecar

        sidecar = PoseSidecar(os.path.join(os.path.dirname(os.path.abspath(args.manifest)), "pose_sidecar"))

    rng = np.random.default_rng(args.seed if args.seed is not None else (os.getpid() * 7919) % 2**31)

    meta = json.load(open(SCENE_META))
    pool = resolve_pool(meta, args.targets)
    print(f"[gym] target pool: {', '.join(pool)}")
    surf_a = meta["surfaces"]["A"]
    half = np.asarray(surf_a.get("half_extent", [0.6, 0.6])[:2], dtype=float)
    cxy = np.asarray(surf_a["xy"], dtype=float)
    surf_b = meta["surfaces"].get("B")
    rects = [(cxy, half)]
    if surf_b:
        rects.append(
            (np.asarray(surf_b["xy"], dtype=float), np.asarray(surf_b.get("half_extent", [0.6, 0.6])[:2], dtype=float))
        )

    def edge_dist(xy):
        """SIGNED clearance to the nearest table footprint: positive
        outside, NEGATIVE inside. (collect_episode's version returns the
        penetration depth as a positive number for inside points — its
        callers never sample inside a table; the gym does, and that sign
        error spawned robots standing in the tabletop.)"""
        best = np.inf
        for rc, rh in rects:
            dxy = np.abs(np.asarray(xy) - rc) - rh
            out = np.maximum(dxy, 0.0)
            d = float(np.hypot(out[0], out[1])) if (dxy > 0).any() else float(max(dxy))
            best = min(best, d)
        return best

    rm = reach_map.ReachMap()
    # One stand plan per target, solved once up front. Every target has its
    # own grip height, so every target has its own squat and its own set of
    # aim cells -- this used to be a single plan because there was a single
    # target.
    plans = {k: plan_for_target(meta, k, rm, reach_config) for k in pool}
    for k, pl in plans.items():
        print(
            f"[gym] {k:8s} grip_z {pl.grip_z:.3f}  {len(pl.rel_cands)} aim "
            f"cells  squat_hint "
            f"{'none' if pl.squat_hint is None else f'{pl.squat_hint * 100:.1f} cm'}"
            f"  edge {'+' if pl.sgn > 0 else '-'}{'xy'[pl.ax]}"
        )

    p = reset = robot_nom = None
    if not args.sample_only:
        p = Primitives(meta=meta)
        reset = EpisodeControlClient()

        # Nominal robot spawn: one clean reset, read where the pelvis lands.
        p.cmd.reset_pose()
        reset.pin_arms()
        reset.reset()
        time.sleep(0.3)
        # Read the nominal from the TELEPORT LANDING, before standing up. The
        # sim's reset is `qpos[:] = qpos0` then `+= dxy` (privileged_pub:386),
        # so every per-round teleport lands at qpos0 + dxy -- and the round's
        # dxy is built as `stand - robot_nom`, which is only correct if
        # robot_nom IS qpos0. Reading it after start_control + a 2.5 s stand
        # instead (what this used to do) folds the stand-up drift into the
        # nominal, and that drift is not repeatable: measured 0.12 m on a fresh
        # stack and 0.91 m once the deploy carried a stale intent, which then
        # biased EVERY teleport in that process by the same amount and showed
        # up as "teleport did not land" for the rest of the run.
        robot_nom = np.array(p.obs()["pelvis"]["pos"][:2])
        print(f"[gym] nominal robot xy {np.round(robot_nom, 3)} (teleport landing)")
        start_control_once(p)
        p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
    return _SceneSetup(
        fixed_cases=fixed_cases,
        sampled_cases=sampled_cases,
        reach_config=reach_config,
        edge_range=edge_range,
        off_lo=off_lo,
        off_hi=off_hi,
        sidecar=sidecar,
        rng=rng,
        meta=meta,
        pool=pool,
        surf_a=surf_a,
        half=half,
        cxy=cxy,
        surf_b=surf_b,
        rects=rects,
        edge_dist=edge_dist,
        rm=rm,
        plans=plans,
        p=p,
        reset=reset,
        robot_nom=robot_nom,
    )


def main() -> None:
    # Run as a script this module is `__main__`; the phase modules import it by its
    # package name. Alias so both names are the same module (module-level state such as
    # VLA_POLICY / RESTART_DEPLOY / ARM_PIN is then read from one place). No-op when
    # eval_closed_loop imports grasp_gym as a package and calls main() in-process.
    sys.modules.setdefault("locomani.tasks.bottle.grasp_gym", sys.modules[__name__])
    args = cli_args.parse_args()
    setup = _build_scene_and_plan(args)
    round_loop.round_loop(args, setup)


# Installed by tools/eval_closed_loop.py to swap the SCRIPTED policy for a
# VLA, leaving everything around it identical. The fork is placed after the
# settle and before the walk (see _run_round), so a VLA round shares the round
# sampling, the spawn bisection, the teleport, the re-anchor, the appearance
# seed, the video and the exporter toggle with a scripted round, and differs
# only in what drives the robot.
#
# That sharing is the whole point: the reference number a closed-loop result is
# read against is the scripted policy's 200/210 = 95.2% on THIS distribution,
# measured by the collection itself. Re-implementing the sampler in the eval
# would put the two policies on subtly different distributions and quietly
# invalidate the comparison.
VLA_POLICY = None


def run_round(
    p,
    reset,
    meta,
    obj_key,
    surf_a,
    edge_dist,
    d_can,
    d_robot,
    psi,
    side,
    rel,
    rm,
    spawn_xy,
    squat_hint=None,
    spawn_off=0.30,
    spawn_lat=0.0,
    dpsi=0.0,
    appearance=None,
    record: bool = False,
    view_gate_x: float = 0.80,
    view_gate_y: float = 0.90,
    one_shot: bool = False,
    mid_band: float = 0.30,
    *,
    reach_config=DEFAULT_REACH,
    object_yaw=None,
) -> RoundResult:
    """One round, and the exporter left in a clean state whatever happens.

    Returns a RoundResult (also unpacks as ok, why, produced_episode).
    The produced flag is what the
    dataset's per-episode instruction manifest is keyed on: an attempt that
    started recording produced an episode even if it then failed, because
    the exporter's 'x' path is save_episode_as_discarded(), not "forget it".

    The discard here is not tidiness. The exporter's 'c' is a three-state
    cycle -- IDLE -> RECORDING -> NEED_TO_SAVE -> IDLE -- and [stand]
    failures are RETRIED, so before this wrapper the retry's opening 'c'
    landed on a live recording and SAVED the attempt that had just fallen
    over. Every episode after it in the session was then one 'c' out of
    phase. It never fired in the recorded batches because none of them hit
    a stand retry; at collection scale it certainly would.
    """
    rec = {"on": False}
    execution = ExecutionRecord()
    execution.enter("initialize")
    ok, why = False, ""
    try:
        ok, why = _run_round(
            p,
            reset,
            meta,
            obj_key,
            surf_a,
            edge_dist,
            d_can,
            d_robot,
            psi,
            side,
            rel,
            rm,
            spawn_xy,
            squat_hint,
            spawn_off,
            spawn_lat,
            dpsi,
            appearance,
            record,
            rec,
            view_gate_x,
            view_gate_y,
            one_shot,
            mid_band,
            execution=execution,
            reach_config=reach_config,
            object_yaw=object_yaw,
        )
        if not ok:
            execution.fail(why)
    except Exception as exc:
        execution.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        recovery_hold = execution.metrics.get("terminal_hold") if reach_config.ready_grasp_recovery else None
        if recovery_hold is not None and not ok:
            p.cmd.set_upper_body_vel(np.zeros(17, dtype=np.float32))
            p.cmd.frame(**recovery_hold)
        elif reach_config.ready_grasp_baseline and not ok:
            p.cmd.set_upper_body_vel(np.zeros(17, dtype=np.float32))
            p.cmd.frame(
                mode=LocomotionMode.IDLE,
                speed=0.0,
                movement=(0.0, 0.0, 0.0),
                height=-1.0,
                facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0),
            )
        if record and rec["on"] and not ok:
            p.cmd.key("x")
            if recovery_hold is not None:
                p.cmd.hold(2.0, **recovery_hold)
            else:
                p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0))
    execution.enter("complete" if ok else "failed", ok=ok, why=why)
    return RoundResult(ok, why, rec["on"], execution.snapshot())


def _run_round(
    p,
    reset,
    meta,
    obj_key,
    surf_a,
    edge_dist,
    d_can,
    d_robot,
    psi,
    side,
    rel,
    rm,
    spawn_xy,
    squat_hint=None,
    spawn_off=0.30,
    spawn_lat=0.0,
    dpsi=0.0,
    appearance=None,
    record: bool = False,
    rec: dict | None = None,
    view_gate_x: float = 0.80,
    view_gate_y: float = 0.90,
    one_shot: bool = False,
    mid_band: float = 0.30,
    *,
    execution=None,
    reach_config=DEFAULT_REACH,
    object_yaw=None,
) -> tuple[bool, str]:
    """One round's five phases (round_phases.py / round_phases_approach.py),
    threaded through a RoundContext.

    Body moved there (refactor step 4b, pure extraction) -- this function
    stays HERE, thin, so that grasp_gym.VLA_POLICY / ARM_PIN /
    VIDEO_INCLUDE_INIT / EDGE_MIN / RAISE_LEAD / RAISE_FINISH_MAX_S /
    PALM_DRIFT_MAX are read as THIS module's own globals (bare-name lookup,
    resolved fresh every call) and snapshotted onto the RoundContext once per
    round -- exactly what the original inline body did, since a caller can
    still override any of them by assignment (grasp_gym.VLA_POLICY = ...,
    eval_closed_loop.py). diag_transition_live.py replaces
    `grasp_gym._run_round` itself with a wrapper around the original, which
    is why this function's own name/identity stays put here too -- `run_round`
    above calls it by its bare name, resolved against this module's globals
    at call time.
    """
    execution = execution if execution is not None else ExecutionRecord()
    ctx = RoundContext(
        p=p,
        reset=reset,
        meta=meta,
        obj_key=obj_key,
        surf_a=surf_a,
        edge_dist=edge_dist,
        d_can=d_can,
        d_robot=d_robot,
        psi=psi,
        side=side,
        rel=rel,
        rm=rm,
        spawn_xy=spawn_xy,
        squat_hint=squat_hint,
        spawn_off=spawn_off,
        spawn_lat=spawn_lat,
        dpsi=dpsi,
        appearance=appearance,
        record=record,
        rec=rec,
        view_gate_x=view_gate_x,
        view_gate_y=view_gate_y,
        one_shot=one_shot,
        mid_band=mid_band,
        execution=execution,
        reach_config=reach_config,
        object_yaw=object_yaw,
        vla_policy=VLA_POLICY,
        arm_pin=ARM_PIN,
        video_include_init=VIDEO_INCLUDE_INIT,
        edge_min=EDGE_MIN,
        raise_lead=RAISE_LEAD,
        raise_finish_max_s=RAISE_FINISH_MAX_S,
        palm_drift_max=PALM_DRIFT_MAX,
        tabletop_grasp_cls=TabletopGrasp,
    )
    for phase in (_p_spawn, _p_handover, _p_navigate, _p_approach, _p_grasp):
        result = phase(ctx)
        if result is not None:
            return result
    raise AssertionError("round_phases: no phase returned a result")


if __name__ == "__main__":
    main()
