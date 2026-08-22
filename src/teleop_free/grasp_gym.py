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

Usage:  python src/teleop_free/grasp_gym.py [--rounds N] [--seed K]
Assumes the stack is up (start_stack.sh).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_grasp  # noqa: E402
import grasping  # noqa: E402
import reach_map  # noqa: E402
import stream_reach  # noqa: E402
from collect_episode import (  # noqa: E402
    PALM_DRIFT_MAX,
    STAND_SETTLE,
    obj_top,
    palm_frame_offset,
)
from primitives import (  # noqa: E402
    READY_ARM,
    READY_UPPER,
    STANDING_PELVIS_Z,
    LocomotionMode,
    Primitives,
    StreamMode,
    upper_body_from_mj,
)
from privileged_pub import EpisodeControlClient  # noqa: E402
import paths
import sim_vis

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"

EDGE_MIN = 0.17            # pelvis-to-edge HARD floor, enforced at arrival.
                           # 0.10 -> 17 hip contacts, 0.12 -> 22: the crouch
                           # lean protrudes ~14 cm past the pelvis centre;
                           # 0.15 still brushed the hip 15x/batch once the
                           # planned crouch deepened to 13-15 cm delivered
EDGE_MAX = 0.32            # sampling ring upper bound: core-aim stands sit
                           # at rel depth 0.18-0.36 (the sweet slice), so
                           # they clear the edge by more than the old 0.20
CAN_INBOARD = (0.04, 0.08)  # can centre this far in from its nearest edge:
                            # close enough that a 15 cm stand-off still
                            # reaches it, far enough that it cannot roll off
FACING_MAX = np.deg2rad(30.0)

# Per-round deploy restart (GYM_RESTART_DEPLOY=0 disables). The deploy keeps
# locomotion/heading state across sim resets; round 1 of a fresh deploy is
# always clean, rounds 2+ wander, fall on the walk-up, or land metres off.
# ~30 s a round buys back the dominant failure class.
RESTART_DEPLOY = os.environ.get("GYM_RESTART_DEPLOY", "1") == "1"
RESTART_SCRIPT = f"{paths.REPO}/scripts/restart_deploy.sh"


def fresh_deploy(p) -> bool:
    """Kill + relaunch this instance's deploy; returns True when it's up."""
    try:
        r = subprocess.run(["bash", RESTART_SCRIPT], capture_output=True,
                           text=True, timeout=360)
    except subprocess.TimeoutExpired:
        print("[gym] deploy restart timed out")
        return False
    if "deploy restarted" not in r.stdout:
        print(f"[gym] deploy restart failed: {r.stdout[-300:]}")
        return False
    p._gym_started = False   # fresh process: nothing to quiesce
    # The heading compensation pairs with the DEAD deploy's anchor: the
    # commander keeps rotating facing/movement by the previous round's yaw
    # (~-80 deg) while the fresh deploy anchors clean — every IDLE frame
    # then steers sideways and the stand drifts 0.4-0.6 m ("teleport did
    # not land"). A fresh deploy starts like a fresh boot: offset zero.
    p.cmd.heading_offset = 0.0
    time.sleep(4.0)          # let the engines finish lazy-loading before
                             # the round's start_control leans on them
    return True


def build_rel_candidates(rm: reach_map.ReachMap, z_pf: float) -> np.ndarray:
    """Pelvis-frame (x, y) offsets of a graspable can, facing within 30 deg.

    No |y| >= 0.12 filter on purpose: dead-ahead placements are exactly what
    the side-grasp azimuth is being tested on."""
    cands = []
    # x capped at 0.32: stage-2 navigation delivers stands at depth
    # 0.22-0.30, and beyond ~0.32 the OUTBOARD staging point (pregrasp =
    # grip - 0.13*entry) leaves the waist-clamped arm's reach even where
    # the grip itself is in the map (batch: 12x 'arm not tracking',
    # IK res 0.06-0.09).
    for x in np.arange(0.16, 0.30, 0.025):
        for y in np.arange(-0.40, 0.401, 0.025):
            if abs(np.arctan2(y, x)) > FACING_MAX:
                continue
            # RIGHT arm only for now: GRASP_CENTRE_OFFSET / hand_points
            # are right-hand palm-frame constants that were never mirrored
            # — the all-left batch produced targets on the wrong side of the
            # body (9x arm-not-tracking). Left support is its own work item.
            if not rm.graspable([x, y, z_pf], "right"):
                continue
            cands.append((x, y))
    return np.asarray(cands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--dump-sample", default=None,
                    help="write this round's scene sample to a JSON file so a "
                         "replay can reproduce the identical initial condition "
                         "(diag_token_replay.py)")
    ap.add_argument("--record", action="store_true",
                    help="drive the data exporter too, and keep only the "
                         "episodes that pass every stage predicate")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--spawn-off", type=float, default=1.80,
                    help="spawn this far behind the stand and walk in. Long "
                         "enough that the approach is a real walk: the arm "
                         "rises over the last RAISE_LEAD metres instead of "
                         "being held out level the whole way, and the ego "
                         "camera picks the can up at 1.77 m (diag_ego_geom).")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed if args.seed is not None
                                else (os.getpid() * 7919) % 2 ** 31)

    meta = json.load(open(SCENE_META))
    obj_key = meta["target_object"]
    surf_a = meta["surfaces"]["A"]
    half = np.asarray(surf_a.get("half_extent", [0.6, 0.6])[:2], dtype=float)
    cxy = np.asarray(surf_a["xy"], dtype=float)
    can_nom = np.asarray(meta["objects"][obj_key]["init_pos"][:2], dtype=float)
    grip_z = obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM

    surf_b = meta["surfaces"].get("B")
    rects = [(cxy, half)]
    if surf_b:
        rects.append((np.asarray(surf_b["xy"], dtype=float),
                      np.asarray(surf_b.get("half_extent",
                                            [0.6, 0.6])[:2], dtype=float)))

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
            d = (float(np.hypot(out[0], out[1])) if (dxy > 0).any()
                 else float(max(dxy)))
            best = min(best, d)
        return best

    rm = reach_map.ReachMap()
    # Core-aim stand plan: the smallest squat that brings a THICK sweet-core
    # slice onto the grip height, plus that slice's ego-safe aim cells. The
    # sampled rel then points the CORE (arm-centred, offset right) at the
    # can instead of the torso, and the crouch is planned before arrival.
    plan_cs = rm.plan_core_stand(float(grip_z - STANDING_PELVIS_Z))
    if plan_cs is not None:
        rel_cands, want_drop, core_marg = plan_cs
        # plan_core_stand's drop is a DELIVERED pelvis drop; arm.squat takes
        # a command whose delivery runs ~1.5x (measured: cmd 5/9/13 cm ->
        # 4.4/12/19.5 cm) — invert that curve for the hint.
        squat_hint = float(np.interp(want_drop,
                                     [0.0, 0.044, 0.12, 0.195],
                                     [0.0, 0.05, 0.09, 0.13]))
        print(f"[gym] core stand: {len(rel_cands)} aim cells, "
              f"drop {want_drop * 100:.1f} cm (cmd {squat_hint * 100:.1f}), "
              f"scatter margin {core_marg * 100:.0f} cm")
    else:
        rel_cands = build_rel_candidates(rm, float(grip_z - STANDING_PELVIS_Z))
        squat_hint = None
        print(f"[gym] no core slice — {len(rel_cands)} envelope stands")

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
    p.cmd.start_control()
    p.cmd.hold(2.5, mode=LocomotionMode.IDLE)

    wins = 0
    for rnd in range(args.rounds):
        t0 = time.time()
        # -- sample --------------------------------------------------------
        # The can lives in a band hugging its nominal edge (radial 5-10 cm
        # inboard, +-10 cm along the edge): a pelvis 10 cm OUTSIDE the edge
        # then reaches it at rel depth 0.15-0.24 — comfortably inside the
        # map instead of at its rim, which is what lets the no-contact
        # stand-off exist at all.
        e0 = can_nom - cxy
        if abs(abs(e0[0]) - half[0]) < abs(abs(e0[1]) - half[1]):
            ax, sgn = 0, float(np.sign(e0[0]))
        else:
            ax, sgn = 1, float(np.sign(e0[1]))
        for try_i in range(500):
            can_xy = can_nom.copy()
            can_xy[ax] = cxy[ax] + sgn * (half[ax]
                                          - rng.uniform(*CAN_INBOARD))
            can_xy[1 - ax] = can_nom[1 - ax] + rng.uniform(-0.10, 0.10)
            rel = rel_cands[rng.integers(len(rel_cands))]
            # Face the table from outside, along the can's inward edge
            # normal, +-25 deg (a uniform world yaw almost never yields a
            # stand outside the table).
            normal = np.zeros(2)
            normal[ax] = -sgn
            psi = float(np.arctan2(normal[1], normal[0])
                        + rng.uniform(-0.44, 0.44))
            c, s = np.cos(psi), np.sin(psi)
            stand = can_xy - np.array([c * rel[0] - s * rel[1],
                                       s * rel[0] + c * rel[1]])
            # Prefer slack over the bare floor: a stand SAMPLED at edge
            # 0.17 exactly leaves the +-4 cm execution scatter nowhere to
            # land (the edge-hold retry loop — a yaw -90 draw spent six
            # re-approaches hovering at 0.11-0.15). First 300 tries demand
            # 3 cm of headroom; only then fall back to the bare band.
            edge_lo = EDGE_MIN + (0.03 if try_i < 300 else 0.0)
            if not (edge_lo <= edge_dist(stand) <= EDGE_MAX):
                continue
            # The WALK must clear the table too: the batch's "fell during
            # the walk-up" class was the robot brushing the table edge on
            # the way in (video: hip catches the rim, robot tips backwards
            # onto the tabletop). Check the whole spawn->stand line — for a
            # long approach it can cross the OTHER table.
            spawn_pt = stand - args.spawn_off * np.array([np.cos(psi),
                                                          np.sin(psi)])
            n_chk = max(2, int(np.ceil(args.spawn_off / 0.25)) + 1)
            line = np.linspace(spawn_pt, stand, n_chk, endpoint=False)
            if all(edge_dist(q) >= EDGE_MIN for q in line):
                break
        else:
            print("EPISODE FAILED [gym] no feasible sample found")
            continue
        side = "right"
        print(f"[gym] round {rnd + 1}/{args.rounds}: can {np.round(can_xy, 2)}"
              f" rel [{rel[0]:.2f} {rel[1]:+.2f}] yaw {np.degrees(psi):.0f}"
              f" side {side}")

        # -- reset + settle -----------------------------------------------
        # Stand-stage failures (fell on the walk-up / never stabilised) are
        # deploy-state contamination from the PREVIOUS round, not properties
        # of this sample — round 1 of every fresh deploy stands clean, and a
        # plain reset recovers (OK streaks follow falls). Retry those once.
        for attempt in range(2):
            if RESTART_DEPLOY and (rnd > 0 or attempt > 0):
                # A fresh deploy per round (and per stand retry): the
                # nominal-spawn calibration already ran on this one, so
                # round 1 right after a stack boot skips the restart.
                # (A warm-up stand after the restart was tried and REVERTED:
                # it made 9/24 rounds fail to rise at all — start_control
                # right after "Init Done" while the engines still lazy-load
                # leaves the deploy's stand controller broken. The walk-up
                # falls it targeted turned out to be table brushes, fixed by
                # the walk-line clearance in the sampler.)
                fresh_deploy(p)
            if args.dump_sample:
                json.dump(dict(d_can=list(map(float, can_xy - can_nom)),
                               d_robot=list(map(float, stand - robot_nom)),
                               psi=float(psi), side=side,
                               rel=list(map(float, rel)),
                               spawn_off=float(args.spawn_off),
                               squat_hint=(None if squat_hint is None
                                           else float(squat_hint)),
                               stand=list(map(float, stand)),
                               robot_nom=list(map(float, robot_nom))),
                          open(args.dump_sample, "w"), indent=1)
            ok, why = run_round(p, reset, meta, obj_key, surf_a, edge_dist,
                                can_xy - can_nom, stand - robot_nom, psi,
                                side, rel, rm, stand, squat_hint,
                                spawn_off=args.spawn_off, record=args.record)
            if ok or not why.startswith("[stand]"):
                break
            p.log(f"[gym] stand-stage retry ({why})")
        dt = time.time() - t0
        if args.record:
            # One 'c' finishes RECORDING -> NEED_TO_SAVE and the exporter's
            # own loop writes the episode; 'x' throws it away. A failed round
            # must be discarded, not saved: the stage predicates are what
            # turn a scripted sequence into a dataset.
            p.cmd.key("c" if ok else "x")
            p.cmd.hold(4.0, mode=LocomotionMode.IDLE)   # let the write finish
        if ok:
            wins += 1
            print(f"EPISODE OK in {dt:.0f}s")
        else:
            print(f"EPISODE FAILED {why} in {dt:.0f}s")
        reset.end_video()
    print(f"[gym] {wins}/{args.rounds} OK")


def run_round(p, reset, meta, obj_key, surf_a, edge_dist,
              d_can, d_robot, psi, side, rel, rm,
              spawn_xy, squat_hint=None,
              spawn_off=0.30, record: bool = False) -> tuple[bool, str]:
    # Spawn 0.30 m behind the stand. Directly ON the stand the STAND-UP
    # fails (13/24 never rose): the nominal pose is a deep crouch whose
    # knees ride ~0.25 m forward, and the rise needs more clearance than
    # standing does. 0.30 m back gives the rise ~0.5 m of room and leaves
    # only a two-step walk to the stand.
    d_robot = (np.asarray(d_robot, dtype=float)
               - spawn_off * np.array([np.cos(psi), np.sin(psi)]))
    # QUIESCE the deploy before the teleport: it keeps executing the
    # previous round's locomotion intent through the reset (fresh-deploy
    # round 1 was always clean, later rounds marched off mid-settle,
    # toppled the can, or 'failed to land'). A second of IDLE in default
    # mode makes 'standing still' the intent the reset inherits.
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    if getattr(p, "_gym_started", False):
        # Quiesce only a RUNNING deploy (holding frames before the first
        # start_control broke every fresh-deploy round).
        p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.set_ghost()
    reset.reset(objects={obj_key: {"dxy": [float(d_can[0]), float(d_can[1])],
                                   "yaw": float(np.random.uniform(-np.pi,
                                                                  np.pi))}},
                robot={"dxy": [float(d_robot[0]), float(d_robot[1])],
                       "yaw": float(psi)})
    time.sleep(0.3)
    landed_xy = np.asarray(p.pelvis_xy(p.obs()))
    want_xy = (np.asarray(spawn_xy)
               - spawn_off * np.array([np.cos(psi), np.sin(psi)]))
    tp_err = float(np.linalg.norm(landed_xy - want_xy))
    if tp_err > 0.10:
        return False, (f"[stand] teleport landed {tp_err:.2f} m off "
                       f"({np.round(landed_xy, 2)} vs {np.round(want_xy, 2)})")
    # Round 2+ contamination: the commander KEEPS the previous round's
    # FROZEN_UPPER_BODY mode and its stale 17-value arm target across the
    # reset — stand up in the default mode with no target, like a fresh
    # process does.
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    p.cmd.start_control()
    p._gym_started = True
    # Every IDLE frame carries a facing, and the default is the ABSOLUTE
    # world +x — the settle was quietly rotating the freshly-teleported
    # robot 50-70 deg off its sampled yaw (pre-walk yaw_err telemetry).
    # Hold the sampled heading explicitly through every settle.
    fv = (float(np.cos(psi)), float(np.sin(psi)), 0.0)
    # Re-anchor BEFORE the settle, not after it. The deploy captures its
    # heading anchor at start_control, i.e. at the pose the robot had
    # BEFORE this round's teleport, and reanchor() re-captures it here and
    # records the yaw in cmd.heading_offset — which frame() then uses to
    # express `facing` in the robot's own heading frame. The release policy
    # tolerates a stale anchor (its reference orientations are absolute),
    # but SONIC v1.1 normalizes its observations by the robot heading, so a
    # stale anchor is a standing heading error it keeps turning to close:
    # measured over a 5 s IDLE hold, 0.134 m of drift without this call and
    # 0.004 m with it (diag_idle_drift.py).
    stream_reach.reanchor(p, side=side)
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE, facing=fv)
    s = p.obs()
    if s["fallen"]:
        return False, ("[stand] robot did not stabilise after reset "
                       f"(pelvis z {s['pelvis']['pos'][2]:.3f})")
    # Separate check, separate meaning: the teleport itself was verified
    # above; this one is how far the STAND-UP walked the robot off its
    # spawn, which is a policy behaviour and gets its own budget.
    drift = float(np.linalg.norm(np.asarray(p.pelvis_xy(s)) - landed_xy))
    p.log(f"[gym] spawn {np.round(landed_xy, 2)} -> settled "
          f"{np.round(p.pelvis_xy(s), 2)} (stand drift {drift:.2f} m)")
    if drift > 0.35:
        return False, (f"[stand] walked {drift:.2f} m off the spawn "
                       f"while standing up")
    # Walk with the arms FREE. In PLANNER_VR_3PT the walking policy ignores
    # the wrist targets and swings the arms with the gait (agent_server:151),
    # which is what a person does; the old pre-walk switch to
    # FROZEN_UPPER_BODY held both forearms out level for the whole approach
    # and is where the stiff, un-human walk came from. Nothing is pinned
    # either -- the hands only need to be up by the time the reach starts,
    # and that is what the RAISE_LEAD ramp below does.
    p.cmd.stream_mode = StreamMode.PLANNER_VR_3PT
    p.cmd.upper_body = None
    reset.pin_arms()
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=fv)
    reset.start_video(f"gym_{int(time.time())}")
    if record:
        # 'c' toggles the exporter IDLE -> RECORDING; a second 'c' saves and
        # 'x' discards (zmq_driver.SonicCommander.key). Recording starts HERE,
        # after the settle, so the first frame of the episode is the first
        # frame of the task rather than a robot standing still.
        p.cmd.key("c")
    grip_za = obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM

    def rel_dbg(tag, s2):
        y2 = p.pelvis_yaw(s2)
        c2, s2n = np.cos(y2), np.sin(y2)
        dv2 = p.object_pos(obj_key, s2)[:2] - p.pelvis_xy(s2)
        rx = c2 * dv2[0] + s2n * dv2[1]
        ry = -s2n * dv2[0] + c2 * dv2[1]
        yerr = np.degrees((y2 - psi + np.pi) % (2 * np.pi) - np.pi)
        p.log(f"[gym] {tag}: rel [{rx:.2f} {ry:+.2f}] "
              f"yaw_err {yerr:+.0f} deg")

    rel_dbg("pre-walk ", p.obs())

    def rel_now(s2):
        y2 = p.pelvis_yaw(s2)
        c2, s2n = np.cos(y2), np.sin(y2)
        dv2 = p.object_pos(obj_key, s2)[:2] - p.pelvis_xy(s2)
        return [c2 * dv2[0] + s2n * dv2[1], -s2n * dv2[0] + c2 * dv2[1],
                grip_za - float(s2["pelvis"]["pos"][2])]

    plan_r = float(np.hypot(rel[0], rel[1]))
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
    # the arms swinging; leg 2 hands the 17 upper-body joints over -- seeded
    # from the arm's CURRENT angles, so the takeover has no step in it (the
    # deploy substitutes them into the reference verbatim, with no ramp of
    # its own, g1_deploy_onnx_ref.cpp:781) -- and drives them to READY by
    # distance remaining, so the arm is up exactly when the body arrives.
    RAISE_LEAD = 1.00
    lead_tgt = np.asarray(spawn_xy) - (0.30 + RAISE_LEAD) * np.array(fv[:2])
    if float(np.linalg.norm(lead_tgt - p.pelvis_xy(p.obs()))) > 0.20:
        if not p.walk_to(lead_tgt, stop_dist=0.10,
                         timeout=10.0 + 8.0 * spawn_off,
                         face_target=False):
            return False, "[stand] fell during the walk-up"
    q_free = upper_body_from_mj(p.obs()["joints"])
    p.cmd.set_upper_body(q_free)
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY

    def raise_arm(_s, dist):
        u = float(np.clip(1.0 - (dist - 0.06) / RAISE_LEAD, 0.0, 1.0))
        sm = 3 * u * u - 2 * u ** 3
        p.cmd.set_upper_body(q_free + sm * (READY_UPPER - q_free))

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
    MARGIN_STOP = 0.075
    EDGE_LEAD = 0.06           # brake lead on the edge floor
    # Three speed bands, from the measured gait quantum (diag_shuffle.py):
    #   0.26  -> 0.155 m/s, 11 cm per step
    #   0.215 -> 0.040 m/s,  5 cm per step
    #   0.20  -> 0.013 m/s,  1-2 cm per step
    # The gate fires at x 0.37-0.42 (matching the offline margin profile),
    # so the braking has to be DONE by then. With a single 0.12 m shuffle
    # band the robot only slowed at x 0.43 and an 11 cm coarse stride was
    # still in flight when the gate fired: measured carry 0.08-0.24 m, and
    # every round whose carry passed ~0.15 m ended inside the edge floor.
    APPR_COARSE, APPR_MID, APPR_SHUFFLE = 0.26, 0.215, 0.20
    MID_BAND, SHUFFLE_BAND = 0.30, 0.10

    def do_approach(seconds: float, ramp: bool) -> str:
        appr_end = time.time() + seconds
        why, stall_x, stall_t = "timeout", None, time.time()
        brake_x, brake_e = float("nan"), float("nan")
        while time.time() < appr_end:
            s = p.obs()
            if s["fallen"]:
                return "fell"
            r = rel_now(s)
            if ramp:
                raise_arm(s, float(np.linalg.norm(
                    np.asarray(spawn_xy) - p.pelvis_xy(s))))
            if rm.margin(r, "right", cap=MARGIN_STOP + 0.01) >= MARGIN_STOP:
                why = "can reachable"
                brake_x, brake_e = r[0], edge_dist(p.pelvis_xy(s))
                break
            if edge_dist(p.pelvis_xy(s)) <= EDGE_MIN + EDGE_LEAD:
                why = "edge floor"
                brake_x, brake_e = r[0], edge_dist(p.pelvis_xy(s))
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
            to_stand = np.asarray(spawn_xy) - p.pelvis_xy(s)
            n_ts = float(np.linalg.norm(to_stand))
            mv = to_stand / n_ts if n_ts > 1e-3 else np.array(fv[:2])
            p.cmd.frame(mode=LocomotionMode.SLOW_WALK, facing=fv,
                        movement=(float(mv[0]), float(mv[1]), 0.0),
                        speed=(APPR_SHUFFLE if r[0] - plan_r <= SHUFFLE_BAND
                               else APPR_MID if r[0] - plan_r <= MID_BAND
                               else APPR_COARSE))
            time.sleep(0.02)
        if ramp:
            # Finish whatever is left of the arm ramp smoothly: the stop can
            # fire before the ramp completes, and the deploy substitutes the
            # 17 joints verbatim, so a snap to READY would be a step.
            q_now = upper_body_from_mj(p.obs()["joints"])
            for k in range(1, 41):
                u = k / 40.0
                p.cmd.set_upper_body(q_now + (3 * u * u - 2 * u ** 3)
                                     * (READY_UPPER - q_now))
                p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
                time.sleep(0.02)
        p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=fv)
        x_end = rel_now(p.obs())[0]
        p.log(f"[gym] approach: exit={why} braked at x {brake_x:.2f} "
              f"edge {brake_e:.2f} -> stopped x {x_end:.2f} "
              f"edge {edge_dist(p.pelvis_xy(p.obs())):.2f} "
              f"(carry {brake_x - x_end:+.2f}, plan_r {plan_r:.2f})")
        rel_dbg("post-approach", p.obs())

        return why

    if do_approach(45.0, ramp=True) == "fell":
        return False, "[stand] fell during the approach"

    # FORWARD-ONLY repair. Too far -> walk in again on the same gate. Too
    # close, or laterally off -> ACCEPT: 12 rounds that stopped anywhere
    # from x 0.15 to 0.46 (edge 0.09 to 0.35) went 12/12 with ZERO
    # furniture contacts, so the arm absorbs the arrival scatter, which is
    # where that precision belongs. Nothing in an episode walks backwards
    # any more except the deliberate post-grasp back-off.
    for _ in range(2):
        s = p.obs()
        rel_a = rel_now(s)
        too_far = (rel_a[0] > plan_r + 0.06
                   or not rm.graspable(rel_a, "right"))
        if not too_far:
            break
        p.log(f"[gym] still far: rel [{rel_a[0]:.2f} {rel_a[1]:+.2f}] "
              f"— walking in again")
        if do_approach(12.0, ramp=False) == "fell":
            return False, "[stand] fell walking in again"
    s = p.obs()
    rel_a = rel_now(s)
    # A real safety floor, not the planning floor. EDGE_MIN (0.17) is where
    # the PLANNER aims; rounds that actually stopped at edge 0.09-0.17 went
    # 12/12 with zero furniture contacts in the whole batch, so failing them
    # here would only discard good episodes now that nothing backs up to
    # rescue them.
    if edge_dist(p.pelvis_xy(s)) < 0.06:
        return False, "[stand] pelvis ended against the table edge"
    if not rm.graspable(rel_a, "right"):
        return False, (f"[stand] arrival left the object out of reach "
                       f"(rel [{rel_a[0]:.2f} {rel_a[1]:+.2f}])")

    # -- the grasp, same path as collect_episode ---------------------------
    st = p.obs()
    if stream_reach.toppled(st, obj_key):
        return False, "[gym] object toppled during the settle"
    cam = st["objects"][obj_key].get("cam")
    # Vertical bound 0.9, horizontal 0.8: the 52-deg-pitched camera puts a
    # close can low in the frame (fy -0.82..-0.94 killed 4 otherwise-good
    # rounds/batch). fy 0.9 is still inside the physical image (edge 1.0),
    # so on real hardware the object stays detectable — the horizontal
    # bound keeps the stricter margin because yaw scatter moves it fast.
    if "cam" in st["objects"][obj_key] and (
            cam is None or abs(cam[0]) > 0.8 or abs(cam[1]) > 0.9):
        return False, f"[view] object outside ego view at stand (cam={cam})"

    grip = np.array([p.object_pos(obj_key, st)[0],
                     p.object_pos(obj_key, st)[1],
                     obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM])
    _d = grip[:2] - p.pelvis_xy(st)
    _d = _d / max(np.linalg.norm(_d), 1e-6)
    virtual_start = grip - 0.25 * np.array([_d[0], _d[1], 0.0]) + [0, 0, 0.08]
    obj_box = meta["objects"][obj_key]["box"]

    def azimuth_anchor(st2):
        yaw2 = p.pelvis_yaw(st2)
        lat = -0.31 if side == "right" else 0.31
        c2, s2 = np.cos(yaw2), np.sin(yaw2)
        pz = st2["pelvis"]["pos"]
        return np.array([pz[0] - s2 * lat, pz[1] + c2 * lat, pz[2] + 0.44])

    def bearing_h(st2, grip_pt):
        d2 = np.asarray(grip_pt[:2]) - p.pelvis_xy(st2)
        n2 = max(float(np.linalg.norm(d2)), 1e-6)
        return np.array([d2[0] / n2, d2[1] / n2, 0.0])

    avoid_sign = 1.0 if side == "right" else -1.0

    def show_ghost(pl):
        # The ghost is baked into the scene at build time and shares geom
        # group 4 with the reach cloud, so it cannot be hidden from the video
        # side alone. Leaving it parked at its spawn (below the floor) is the
        # lever that works with an already-built scene.
        if not pl.ok or not sim_vis.enabled("ghost"):
            return
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, np.asarray(pl.rot, dtype=float).flatten())
        reset.set_ghost(pos=np.asarray(pl.wrist, dtype=float).tolist(),
                        quat=q.tolist())

    plan = adaptive_grasp.plan(grip, virtual_start,
                               Primitives._rot(st["palms"][side]["quat"]),
                               surf_a["top"], box=obj_box,
                               obj_rot=Primitives._rot(
                                   st["objects"][obj_key]["quat"]),
                               azimuth_from=azimuth_anchor(st),
                               avoid_dir=bearing_h(st, grip),
                               avoid_sign=avoid_sign)
    p.log(adaptive_grasp.describe(plan))
    if not plan.ok:
        return False, f"[grasp] {plan.why}"
    show_ghost(plan)

    def replan(wrist_now, palm_rot):
        st2 = p.obs()
        c = p.object_pos(obj_key, st2)
        grip_live = np.array([c[0], c[1],
                              obj_top(meta, obj_key)
                              - grasping.GRIP_BELOW_RIM])
        return adaptive_grasp.plan(
            grip_live, wrist_now, palm_rot, surf_a["top"], box=obj_box,
            obj_rot=Primitives._rot(st2["objects"][obj_key]["quat"]),
            azimuth_from=azimuth_anchor(st2),
            avoid_dir=bearing_h(st2, grip_live),
            avoid_sign=avoid_sign)

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
    ok, why = stream_reach.grasp_streamed(
        p, obj_key, grip, plan, side=side, surface_top=surf_a["top"],
        replan=replan, ghost=show_ghost, nudge=False,
        pin=(reset.pin_arms if os.environ.get("GYM_ARM_PIN", "0") == "1"
             else None),
        squat_hint=squat_hint, plan_r=float(np.hypot(rel[0], rel[1])))
    if not ok:
        return False, f"[grasp] {why}"

    vis = stream_reach.LAST_EGO_VIS
    if vis is not None:
        p.log(f"  ego visibility through servo: {vis * 100:.0f}%")
        if vis < 0.7:
            return False, (f"[view] object in ego view only "
                           f"{vis * 100:.0f}% of the servo")

    s = p.obs()
    hold_ref = palm_frame_offset(s, obj_key, side)
    hold_side = "left" if side == "right" else "right"
    p.stand_up(hold_side)
    p.carry_pose(side)
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE, facing=fv)
    # With the can secured in the carry pose, back off two steps (~0.35 m,
    # heading held) before the hold check: clears the table edge so
    # whatever comes next (turn, navigation) cannot clip it, and a grip
    # that cannot survive two steps should fail HERE, not mid-carry.
    s = p.obs()
    back_tgt = np.asarray(p.pelvis_xy(s)) - 0.35 * np.array(fv[:2])
    if not p.walk_to(back_tgt, stop_dist=0.08, timeout=8.0,
                     face_target=False):
        return False, "[hold] fell while backing away from the table"
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=fv)
    s = p.obs()
    drift = float(np.linalg.norm(palm_frame_offset(s, obj_key, side)
                                 - hold_ref))
    if drift >= PALM_DRIFT_MAX:
        return False, (f"[hold] object slipped {drift * 100:.1f} cm in the "
                       "palm frame during the hold")
    if float(p.object_pos(obj_key, s)[2]) <= surf_a["top"] + 0.03:
        return False, "[hold] object is back on the table after the hold"
    return True, ""


if __name__ == "__main__":
    main()
