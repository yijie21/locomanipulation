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

EDGE_MIN = 0.16            # hip-to-table-edge floor (same as final_approach)
CAN_EDGE_MIN = 0.10        # keep the sampled can this far inboard
FACING_MAX = np.deg2rad(30.0)


def build_rel_candidates(rm: reach_map.ReachMap, z_pf: float) -> np.ndarray:
    """Pelvis-frame (x, y) offsets of a graspable can, facing within 30 deg.

    No |y| >= 0.12 filter on purpose: dead-ahead placements are exactly what
    the side-grasp azimuth is being tested on."""
    cands = []
    for x in np.arange(0.14, 0.45, 0.025):
        for y in np.arange(-0.40, 0.401, 0.025):
            if abs(np.arctan2(y, x)) > FACING_MAX:
                continue
            if not rm.graspable([x, y, z_pf]):
                continue
            cands.append((x, y))
    return np.asarray(cands)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
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

    def edge_dist(xy):
        dxy = np.abs(np.asarray(xy) - cxy) - half
        out = np.maximum(dxy, 0.0)
        return float(np.hypot(out[0], out[1])) if (dxy > 0).any() else \
            -float(max(dxy))

    rm = reach_map.ReachMap()
    rel_cands = build_rel_candidates(rm, float(grip_z - STANDING_PELVIS_Z))
    print(f"[gym] {len(rel_cands)} candidate stands in the reach map")

    p = Primitives(meta=meta)
    reset = EpisodeControlClient()

    # Nominal robot spawn: one clean reset, read where the pelvis lands.
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset()
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(STAND_SETTLE, mode=LocomotionMode.IDLE)
    robot_nom = np.array(p.obs()["pelvis"]["pos"][:2])
    print(f"[gym] nominal robot xy {np.round(robot_nom, 3)}")

    wins = 0
    for rnd in range(args.rounds):
        t0 = time.time()
        # -- sample --------------------------------------------------------
        for _ in range(200):
            d_can = rng.uniform(-0.10, 0.10, 2)
            can_xy = can_nom + d_can
            if (np.abs(can_xy - cxy) > half - CAN_EDGE_MIN).any():
                continue
            rel = rel_cands[rng.integers(len(rel_cands))]
            psi = float(rng.uniform(-np.pi, np.pi))
            c, s = np.cos(psi), np.sin(psi)
            stand = can_xy - np.array([c * rel[0] - s * rel[1],
                                       s * rel[0] + c * rel[1]])
            if edge_dist(stand) >= EDGE_MIN:
                break
        else:
            print("EPISODE FAILED [gym] no feasible sample found")
            continue
        side = "right" if rel[1] < 0 else "left"
        print(f"[gym] round {rnd + 1}/{args.rounds}: can {np.round(can_xy, 2)}"
              f" rel [{rel[0]:.2f} {rel[1]:+.2f}] yaw {np.degrees(psi):.0f}"
              f" side {side}")

        # -- reset + settle -----------------------------------------------
        ok, why = run_round(p, reset, meta, obj_key, surf_a, edge_dist,
                            d_can, stand - robot_nom, psi, side, rel)
        dt = time.time() - t0
        if ok:
            wins += 1
            print(f"EPISODE OK in {dt:.0f}s")
        else:
            print(f"EPISODE FAILED {why} in {dt:.0f}s")
        reset.end_video()
    print(f"[gym] {wins}/{args.rounds} OK")


def run_round(p, reset, meta, obj_key, surf_a, edge_dist,
              d_can, d_robot, psi, side, rel) -> tuple[bool, str]:
    # Spawn 0.55 m BEHIND the sampled stand: a teleport straight to the
    # stand puts the nominal pose's hanging hands inside the table volume
    # (the stand is only ~0.16 m off the edge) and the contact explosion
    # knocks the can over before the round starts. The short walk up also
    # IS the arrival noise a real approach would carry.
    back = 0.55
    spawn_d = np.asarray(d_robot, dtype=float) - back * np.array(
        [np.cos(psi), np.sin(psi)])
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.set_ghost()
    reset.reset(objects={obj_key: {"dxy": [float(d_can[0]), float(d_can[1])],
                                   "yaw": float(np.random.uniform(-np.pi,
                                                                  np.pi))}},
                robot={"dxy": [float(spawn_d[0]), float(spawn_d[1])],
                       "yaw": float(psi)})
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(STAND_SETTLE, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        return False, ("[stand] robot did not stabilise after reset "
                       f"(pelvis z {s['pelvis']['pos'][2]:.3f})")
    stream_reach.reanchor(p, side=side)
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    p.tuck_arms()
    reset.pin_arms(left=READY_ARM["left"], right=READY_ARM["right"])
    p.cmd.hold(2.5, mode=LocomotionMode.IDLE)
    reset.start_video(f"gym_{int(time.time())}")
    # Walk the last 0.55 m to the sampled stand (SLOW_WALK, facing locked
    # at psi), then settle.
    stand_w = p.obs()["pelvis"]["pos"][:2] + back * np.array(
        [np.cos(psi), np.sin(psi)])
    fv = (float(np.cos(psi)), float(np.sin(psi)), 0.0)
    end = time.time() + 14.0
    t_walk = time.time()
    while time.time() < end:
        s = p.obs()
        if s["fallen"]:
            return False, "[stand] fell during the walk-up"
        gap = stand_w - p.pelvis_xy(s)
        if float(np.linalg.norm(gap)) < 0.06:
            break
        g = gap / max(float(np.linalg.norm(gap)), 1e-6)
        # Ramp in: a full-speed SLOW_WALK command into the post-reset
        # stance transient is what tips the robot over.
        spd = 0.12 if time.time() - t_walk < 1.5 else 0.25
        p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=spd,
                    movement=(float(g[0]), float(g[1]), 0.0), facing=fv)
        time.sleep(0.02)
    for _ in range(int(1.5 / 0.02)):
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
        time.sleep(0.02)
    # The walk-up lands with 0.1-0.2 m of gait scatter, which shifts the
    # can's bearing; re-realize the SAMPLED facing offset (delta between
    # facing and bearing is what the rel sample encodes) from where the
    # robot actually ended up.
    s = p.obs()
    d2 = p.object_pos(obj_key, s)[:2] - p.pelvis_xy(s)
    bearing = float(np.arctan2(d2[1], d2[0]))
    delta = -float(np.arctan2(rel[1], rel[0]))
    p.turn_to(bearing + delta, timeout=8.0)
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)

    # -- the grasp, same path as collect_episode ---------------------------
    st = p.obs()
    if stream_reach.toppled(st, obj_key):
        return False, "[gym] object toppled during the settle"
    cam = st["objects"][obj_key].get("cam")
    if "cam" in st["objects"][obj_key] and (
            cam is None or abs(cam[0]) > 0.8 or abs(cam[1]) > 0.8):
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
        replan=replan, ghost=show_ghost, nudge=False, pin=reset.pin_arms)
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
