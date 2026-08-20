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
    READY_WRISTS,
    STANDING_PELVIS_Z,
    LocomotionMode,
    Primitives,
    StreamMode,
)
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = "/workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json"

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
RESTART_SCRIPT = "/workspace/code/locomanipulation/scripts/restart_deploy.sh"


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
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--spawn-off", type=float, default=0.30,
                    help="spawn this far behind the stand and walk in")
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
    p.cmd.start_control()
    p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
    robot_nom = np.array(p.obs()["pelvis"]["pos"][:2])
    print(f"[gym] nominal robot xy {np.round(robot_nom, 3)}")

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
            ok, why = run_round(p, reset, meta, obj_key, surf_a, edge_dist,
                                can_xy - can_nom, stand - robot_nom, psi,
                                side, rel, rm, stand, squat_hint,
                                spawn_off=args.spawn_off)
            if ok or not why.startswith("[stand]"):
                break
            p.log(f"[gym] stand-stage retry ({why})")
        dt = time.time() - t0
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
              spawn_off=0.30) -> tuple[bool, str]:
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
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE, facing=fv)
    s = p.obs()
    if s["fallen"]:
        return False, ("[stand] robot did not stabilise after reset "
                       f"(pelvis z {s['pelvis']['pos'][2]:.3f})")
    land_err = float(np.linalg.norm(
        np.asarray(p.pelvis_xy(s)) - np.asarray(spawn_xy)
        + spawn_off * np.array([np.cos(psi), np.sin(psi)])))
    if land_err > 0.30:
        return False, (f"[stand] teleport did not land "
                       f"({land_err:.2f} m from the spawn)")
    stream_reach.reanchor(p, side=side)
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    p.tuck_arms()
    reset.pin_arms(left=READY_ARM["left"], right=READY_ARM["right"])
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=fv)
    reset.start_video(f"gym_{int(time.time())}")
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
    # Two steps forward onto the stand. Aim 12 cm SHORT and bias every
    # arrival error toward TOO FAR: the stop + settle creeps 6-13 cm
    # forward, backward stepping barely moves (edge back-offs measured
    # 0.13 -> 0.11 -> 0.11), while the forward depth-trim step is reliable
    # — so land far and walk the remainder in.
    walk_tgt = np.asarray(spawn_xy) - 0.12 * np.array(fv[:2])
    if not p.walk_to(walk_tgt, stop_dist=0.06,
                     timeout=10.0 + 8.0 * spawn_off):
        return False, "[stand] fell during the walk-up"
    rel_dbg("post-walk", p.obs())
    for _ in range(int(1.0 / 0.02)):
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
        time.sleep(0.02)
    rel_dbg("post-face", p.obs())
    # No turn correction: the teleport already faces the sampled yaw, and a
    # turn_to on a contamination-stale heading anchor sent the robot
    # marching metres across the room (rel [6.66 +2.19] telemetry).
    # HARD no-contact floor on the actual pose (the settle can drift).
    s = p.obs()

    def rel_now(s2):
        y2 = p.pelvis_yaw(s2)
        c2, s2n = np.cos(y2), np.sin(y2)
        dv2 = p.object_pos(obj_key, s2)[:2] - p.pelvis_xy(s2)
        return [c2 * dv2[0] + s2n * dv2[1], -s2n * dv2[0] + c2 * dv2[1],
                grip_za - float(s2["pelvis"]["pos"][2])]

    # Unified stand correction. Edge clearance, map membership and depth
    # used to be three independent correctors on the same axis with
    # different setpoints — near the boundaries they triggered each other
    # (the visible pacing). Micro-steps are no fix either: 5-15 cm
    # commands are bimodal on SONIC (small ones stall, larger ones stride
    # 2-3x past — a 12 cm back command measured 29 cm), so iterating on
    # them oscillated. What IS accurate is a 0.4 m+ walk, i.e. the main
    # approach itself. So: ONE target (the stand that realizes the
    # SAMPLED rel from the actual can pose/yaw, clamped back along the
    # facing until the edge keeps EDGE_MIN), and per violation ONE of two
    # proven moves — pure overshoot past the plan gets the old one-step
    # forward trim; anything needing net backward/lateral motion gets a
    # full re-approach (retreat 0.45 m, walk back in with the same -0.12
    # far bias as the main walk). Trigger thresholds are the LENIENT
    # floors the 18/24 batches actually ran at (the old back-off could
    # not step 2 cm anyway and its near-misses were accepted): edge
    # EDGE_MIN-0.03, envelope membership (not core — 2/24 batch), depth
    # plan_r+0.06.
    plan_r = float(np.hypot(rel[0], rel[1]))

    def stand_from(s2):
        y2 = p.pelvis_yaw(s2)
        c2, s2n = np.cos(y2), np.sin(y2)
        st2 = (p.object_pos(obj_key, s2)[:2]
               - np.array([c2 * rel[0] - s2n * rel[1],
                           s2n * rel[0] + c2 * rel[1]]))
        # Clamp to 2 cm ABOVE the floor, not the floor itself: the accept
        # threshold is EDGE_MIN-0.03 and execution scatters +-4 cm, so a
        # target at the floor lands half its attempts below acceptance.
        # The extra depth this costs sits inside the +0.06 depth band.
        short = (EDGE_MIN + 0.02) - edge_dist(st2)
        if short > 0:
            st2 = st2 - np.array(fv[:2]) * short
        return st2

    for _ in range(3):
        rel_a = rel_now(s)
        ed = edge_dist(p.pelvis_xy(s))
        bad_edge = ed < EDGE_MIN - 0.03
        # Bearing: an arrival 60+ deg off-axis is still IN the envelope
        # (lateral cells exist) but the ego camera loses the can — that
        # was a [view] episode fail; catch it here and re-approach.
        bad_map = (not rm.graspable(rel_a, "right")
                   or abs(np.arctan2(rel_a[1], rel_a[0]))
                   > np.deg2rad(45.0))
        too_deep = rel_a[0] > plan_r + 0.06
        if not (bad_edge or bad_map or too_deep):
            break
        stand2 = stand_from(s)
        if too_deep and not (bad_edge or bad_map):
            step = min(rel_a[0] - plan_r,
                       edge_dist(p.pelvis_xy(s)) - EDGE_MIN)
            if step <= 0.02:
                break
            p.log(f"[gym] depth trim: rel x {rel_a[0]:.2f} — stepping in "
                  f"{step * 100:.0f} cm")
            tgt = np.asarray(p.pelvis_xy(s)) + np.array(fv[:2]) * step
            if not p.walk_to(tgt, stop_dist=0.03, timeout=6.0,
                             face_target=False):
                return False, "[stand] fell during the depth trim"
            p.cmd.hold(0.8, mode=LocomotionMode.IDLE, facing=fv)
        else:
            p.log(f"[gym] stand correct: rel [{rel_a[0]:.2f} "
                  f"{rel_a[1]:+.2f}] edge {ed:.2f} — re-approach")
            far_pt = stand2 - 0.45 * np.array(fv[:2])
            if not p.walk_to(far_pt, stop_dist=0.06, timeout=10.0,
                             face_target=False):
                return False, "[stand] fell during the stand correction"
            p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=fv)
            s = p.obs()
            stand2 = stand_from(s)
            # -0.18, not the main walk's -0.12: correction re-approaches
            # start table-adjacent where the stop creep measured ~20 cm
            # (three -0.12 re-approaches in a row landed 8 cm short).
            # Landing far is safe — the forward depth trim is the one
            # reliable micro-step and closes the remainder.
            if not p.walk_to(stand2 - 0.18 * np.array(fv[:2]),
                             stop_dist=0.06, timeout=10.0,
                             face_target=False):
                return False, "[stand] fell during the stand correction"
            for _ in range(int(1.0 / 0.02)):
                p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
                time.sleep(0.02)
        s = p.obs()
    rel_a = rel_now(s)
    if edge_dist(p.pelvis_xy(s)) < EDGE_MIN - 0.03:
        return False, "[stand] could not hold the edge clearance"
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
        if not pl.ok:
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

    p.cmd.set_wrists(left_pos=READY_WRISTS["left"],
                     right_pos=READY_WRISTS["right"])
    reset.pin_arms()
    ok, why = stream_reach.grasp_streamed(
        p, obj_key, grip, plan, side=side, surface_top=surf_a["top"],
        replan=replan, ghost=show_ghost, nudge=False, pin=reset.pin_arms,
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
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
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
