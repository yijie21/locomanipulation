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
from typing import NamedTuple

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_grasp  # noqa: E402
import grasping  # noqa: E402
import paths
import reach_map  # noqa: E402
import sim_vis
import stream_reach  # noqa: E402
import upper_ref  # noqa: E402
from collect_episode import (  # noqa: E402
    PALM_DRIFT_MAX,
    STAND_SETTLE,
    obj_top,
    palm_frame_offset,
)
from execution_record import ExecutionRecord, RoundResult
from primitives import (  # noqa: E402
    READY_ARM,
    READY_UPPER,
    STANDING_PELVIS_Z,
    UPPER_BODY_NAMES,
    LocomotionMode,
    Primitives,
    StreamMode,
    upper_body_from_mj,
    wrap,
)
from privileged_pub import EpisodeControlClient  # noqa: E402
from reach_config import DEFAULT_REACH, ReachConfig
from tabletop_grasp import TabletopGrasp, TabletopTarget

SCENE_META = paths.SCENE_META

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
FACING_MAX = np.deg2rad(30.0)

# Ego-frame visibility gate: (u_x, u_y) is the object's position in the ego
# camera frame, normalised so the frame EDGE is 1.0 (diag_ego_geom.frame_pos).
# A camera-geometry constant (fovy, mount pitch), not spawn distance --
# --view-gate-x/-y (defaults below) thread as view_gate_x/view_gate_y through
# run_round/_run_round to the spawn check and the arrival check.

# Per-round deploy restart (GYM_RESTART_DEPLOY=0 disables). The deploy keeps
# locomotion/heading state across sim resets; round 1 of a fresh deploy is
# always clean, rounds 2+ wander, fall on the walk-up, or land metres off.
# ~30 s a round buys back the dominant failure class.
RESTART_DEPLOY = os.environ.get("GYM_RESTART_DEPLOY", "1") == "1"
RESTART_SCRIPT = f"{paths.REPO}/scripts/restart_deploy.sh"


class TargetPlan(NamedTuple):
    """Everything about one candidate target that does not change per round."""

    key: str
    phrase: str  # how the instruction names it
    nom: np.ndarray  # nominal xy on the surface
    grip_z: float  # world z of the grip centre
    rel_cands: np.ndarray  # pelvis-frame (x, y) stands that put it in the core
    squat_hint: float | None
    ax: int  # 0 = x edge, 1 = y edge -- the edge it hugs
    sgn: float  # which side of that axis
    stand3: np.ndarray = np.zeros((0, 3))  # coverage-first (x, y, drop) cells, GYM_STAND_COVER


def resolve_pool(meta: dict, spec: str) -> list[str]:
    """--targets -> a list of object keys, checked against the scene."""
    if spec == "pool":
        want = list(meta.get("target_pool") or [meta["target_object"]])
    elif spec == "default":
        want = [meta["target_object"]]
    else:
        want = [t.strip() for t in spec.split(",") if t.strip()]
    missing = [t for t in want if t not in meta["objects"]]
    if missing:
        raise SystemExit(
            f"[gym] targets not in the scene: {missing}. "
            f"Rebuild it with build_sim_scene.py, or pick from "
            f"{sorted(meta['objects'])}"
        )
    return want


def plan_for_target(meta: dict, key: str, rm: reach_map.ReachMap, config=DEFAULT_REACH) -> TargetPlan:
    """Solve the stand plan for one target.

    Split out of main() when the target stopped being fixed. The edge the
    object hugs is derived here too: the sampler slides the target along
    ITS OWN nearest edge, so a pool spread around the table is also a pool
    of approach directions.
    """
    surf = meta["surfaces"][meta["objects"][key].get("surface", "A")]
    cxy = np.asarray(surf["xy"], dtype=float)
    half = np.asarray(surf.get("half_extent", [0.6, 0.6])[:2], dtype=float)
    nom = np.asarray(meta["objects"][key]["init_pos"][:2], dtype=float)
    grip_z = obj_top(meta, key) - grasping.GRIP_BELOW_RIM
    if config.ready_grasp_recovery:
        obj = meta["objects"][key]
        grip_z = obj["init_pos"][2] + obj["box"].get("center", [0.0, 0.0, 0.0])[2] - config.grip_below_center

    e0 = nom - cxy
    ax = 0 if abs(abs(e0[0]) - half[0]) < abs(abs(e0[1]) - half[1]) else 1
    sgn = float(np.sign(e0[ax]))

    plan_cs = rm.plan_core_stand(float(grip_z - STANDING_PELVIS_Z))
    if plan_cs is not None:
        rel_cands, want_drop, _ = plan_cs
        # plan_core_stand's drop is a DELIVERED pelvis drop; arm.squat takes
        # a command whose delivery runs ~1.5x (measured: cmd 5/9/13 cm ->
        # 4.4/12/19.5 cm) — invert that curve for the hint.
        squat_hint = float(np.interp(want_drop, [0.0, 0.044, 0.12, 0.195], [0.0, 0.05, 0.09, 0.13]))
    else:
        rel_cands = build_rel_candidates(rm, float(grip_z - STANDING_PELVIS_Z))
        squat_hint = None
    phrase = meta.get("object_phrase", {}).get(key, f"the {key}")
    stand3 = rm.stand_cells(float(grip_z - STANDING_PELVIS_Z))
    return TargetPlan(key, phrase, nom, grip_z, rel_cands, squat_hint, ax, sgn, stand3)


def _drop_to_hint(want_drop: float) -> float:
    # plan_core_stand's drop is a DELIVERED pelvis drop; arm.squat's command
    # delivers ~1.5x (cmd 5/9/13 cm -> 4.4/12/19.5 cm) -- invert for the hint
    return float(np.interp(want_drop, [0.0, 0.044, 0.12, 0.195], [0.0, 0.05, 0.09, 0.13]))


STAND_COVER = os.environ.get("GYM_STAND_COVER", "1") == "1"


def sample_stand(pl, rng):
    """Coverage-first stand: stratified in y over the whole core, plus a
    +-3 cm crouch jitter (the policy's own crouch scatters that much).
    Returns (rel_xy, squat_hint). Falls back to the classic aim cells."""
    if not STAND_COVER or len(pl.stand3) == 0:
        rel = pl.rel_cands[rng.integers(len(pl.rel_cands))]
        return rel, pl.squat_hint
    ys = pl.stand3[:, 1]
    bins = np.arange(np.floor(ys.min() / 0.05) * 0.05, ys.max() + 0.05, 0.05)
    live = [b for b in bins if ((ys >= b) & (ys < b + 0.05)).any()]
    b = live[rng.integers(len(live))]
    cells = pl.stand3[(ys >= b) & (ys < b + 0.05)]
    x, y, dz = cells[rng.integers(len(cells))]
    hint = float(np.clip(_drop_to_hint(dz) + rng.uniform(-0.03, 0.03), 0.0, 0.13))
    return np.array([x, y]), hint


def instruction(phrase: str) -> str:
    """The episode's task string. One place, so the dataset and any future
    eval harness cannot drift apart."""
    return f"pick up {phrase}"


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
    ap.add_argument(
        "--dump-sample",
        default=None,
        help="write this round's scene sample to a JSON file so a "
        "replay can reproduce the identical initial condition "
        "(diag_token_replay.py)",
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="drive the data exporter too, and keep only the episodes that pass every stage predicate",
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--spawn-off",
        type=float,
        nargs="+",
        default=[1.0, 3.0],
        metavar="M",
        help="spawn this far behind the stand and walk in. One "
        "value fixes it, two give a per-round range. Long "
        "enough that the approach is a real walk: the arm "
        "rises over the last RAISE_LEAD metres instead of "
        "being held out level the whole way, and the ego "
        "camera picks the object up at 1.77 m "
        "(diag_ego_geom).",
    )
    ap.add_argument(
        "--spawn-edge",
        type=float,
        nargs="+",
        default=None,
        metavar="M",
        help="spawn this far from the TABLE, overriding "
        "--spawn-off. One value fixes it, two give a range. "
        "The table is what a person measures from, and it is "
        "the quantity the clearance floor (EDGE_MIN) is in; "
        "--spawn-off is measured behind the STAND, which "
        "itself sits 0.17-0.32 m from the edge and moves with "
        "the target, so the same --spawn-off is a different "
        "table distance every round. Solved per round by "
        "bisection on the sampled approach line.",
    )
    ap.add_argument(
        "--face-object",
        action="store_true",
        help="wake up pointing at the OBJECT rather than along the "
        "approach line, and take --spawn-yaw-err as jitter "
        "around that. The two coincide only for a spawn dead "
        "on the line: with a lateral offset, and especially "
        "with a close --spawn-edge, facing the approach "
        "heading leaves the object well off centre.",
    )
    ap.add_argument(
        "--spawn-lateral",
        type=float,
        default=0.35,
        metavar="M",
        help="spawn up to this far to either side of the approach "
        "line. 0 puts every episode dead on it, which is a "
        "correction the model then never sees.",
    )
    ap.add_argument(
        "--spawn-yaw-err",
        type=float,
        default=90.0,
        metavar="DEG",
        help="spawn facing up to this far off the approach "
        "heading, and turn to it before walking. The turn "
        "happens AT THE SPAWN, metres from the table: near "
        "the object a heading correction is a 110-160 deg "
        "pirouette that walks the pelvis into the edge.",
    )
    ap.add_argument(
        "--edge-jitter",
        type=float,
        default=0.10,
        metavar="M",
        help="how far the target may slide ALONG its edge, either "
        "way, from its nominal spot. The default 0.10 puts "
        "it in a 20 cm segment of a 100 cm edge, which is "
        "enough variation for the grasp but lets the "
        "walk-up be learnt as 'go to the +y edge, reach a "
        "little left' rather than as locating the object. "
        "Widen it when the batch has few targets and the "
        "approach is what has to generalise. The rejection "
        "sampler validates the stand and both walk legs "
        "against the table, so a value too large costs "
        "retries, not bad samples.",
    )
    ap.add_argument(
        "--stand-retries",
        type=int,
        choices=(0, 1),
        default=1,
        help="reset retries after a stand-stage failure; use 0 for fixed-case acceptance",
    )
    ap.add_argument(
        "--until-ok",
        type=int,
        default=None,
        metavar="N",
        help="stop as soon as N rounds have SUCCEEDED instead of "
        "running all --rounds. --rounds is then the cap on "
        "how many attempts that may take, so a run that "
        "never reaches N still terminates.",
    )
    ap.add_argument(
        "--one-shot",
        action="store_true",
        help="discard any round whose first approach parks out "
        "of reach instead of walking in again (flow200: "
        "the repair re-walk is recorded hesitation, and "
        "the dataset doctrine is one continuous arrival "
        "or nothing).",
    )
    ap.add_argument(
        "--mid-band",
        type=float,
        default=0.30,
        metavar="M",
        help="distance before plan_r where the approach drops "
        "from COARSE to the MID deceleration speed "
        "(default 0.30, the calibrated brake; flow200 "
        "runs 0.20 for a shorter crawl-in at a small "
        "carry-risk cost)",
    )
    ap.add_argument(
        "--targets",
        default="pool",
        help="which objects to draw the round's target from: "
        "'pool' (every entry in the scene's target_pool), "
        "'default' (the single scene target, the old "
        "behaviour), or a comma-separated list of names",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="append one JSON line per round: the target, the "
        "instruction, and whether the episode was kept. "
        "convert_to_hf_format.py joins on this to give each "
        "episode its own task string, which the exporter "
        "cannot do -- its --task-prompt is per SESSION.",
    )
    # 0.8 / 0.9, the values both gates were hardcoded to before they became
    # parameters. Making them settable is right; moving the DEFAULT is a
    # different change: it decides which rounds enter the dataset, so the
    # 191-episode walk200 batch and the next one would be selected by
    # different rules. The horizontal bound is the stricter of the two on
    # purpose -- yaw scatter moves the object across the frame fast.
    ap.add_argument(
        "--view-gate-x",
        type=float,
        default=0.80,
        metavar="U",
        help="ego-frame visibility gate, horizontal "
        "(|u_x| must stay under this; frame edge is 1.0). "
        "Camera geometry, not spawn distance.",
    )
    ap.add_argument(
        "--view-gate-y",
        type=float,
        default=0.90,
        metavar="U",
        help="ego-frame visibility gate, vertical "
        "(|u_y| must stay under this; frame edge is 1.0). "
        "Camera geometry, not spawn distance.",
    )
    ap.add_argument("--reach-config", help="versioned JSON tabletop reach calibration")
    ap.add_argument("--cases", help="fixed JSON case bank; bypass random sampling")
    ap.add_argument("--write-cases", help="write exact sampled conditions as JSON (exclusive output)")
    ap.add_argument("--sample-only", action="store_true", help="generate cases without starting the robot")
    args = ap.parse_args()
    if args.sample_only and not args.write_cases:
        ap.error("--sample-only requires --write-cases")
    if args.cases and args.until_ok:
        ap.error("fixed cases cannot be combined with success-count stopping")
    from experiment_cases import load_cases, save_cases

    fixed_cases = load_cases(args.cases, SCENE_META) if args.cases else None
    if fixed_cases is not None and args.rounds > len(fixed_cases):
        ap.error("requested more rounds than fixed cases")
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
    if getattr(args, "one_shot", False):
        os.environ["GYM_ONE_SHOT"] = "1"  # stream_reach: discard instead of repair
    if args.spawn_edge is not None:
        edge_range = tuple(sorted((args.spawn_edge * 2)[:2]))
        if edge_range[0] < EDGE_MIN:
            raise SystemExit(f"--spawn-edge {edge_range[0]} is inside the pelvis-to-edge floor EDGE_MIN={EDGE_MIN}")
    if off_lo > off_hi:
        off_lo, off_hi = off_hi, off_lo
    # World-pose sidecar: one jsonl per attempt, next to the manifest. Only
    # when recording -- an eval round has its own trace. LOCOMANI_POSE_SIDECAR=0
    # opts out. Fail-safe: a dead sidecar prints and costs nothing.
    sidecar = None
    if args.record and args.manifest and os.environ.get("LOCOMANI_POSE_SIDECAR", "1") == "1":
        from privileged_pub import PoseSidecar

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
            f"cells ({len(pl.stand3)} coverage cells, y "
            f"{pl.stand3[:, 1].min() if len(pl.stand3) else float('nan'):.2f}..{pl.stand3[:, 1].max() if len(pl.stand3) else float('nan'):.2f}, "
            f"{'ON' if STAND_COVER else 'off'})  squat_hint "
            f"{'none' if pl.squat_hint is None else f'{pl.squat_hint * 100:.1f} cm'}"
            f"  edge {'+' if pl.sgn > 0 else '-'}{'xy'[pl.ax]}"
        )

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
        p.cmd.start_control()
        p.cmd.hold(2.5, mode=LocomotionMode.IDLE)

    wins = 0
    played = 0
    deck: list[str] = []
    for rnd in range(args.rounds):
        # --until-ok collects a fixed number of SUCCESSES, which is what a
        # dataset is counted in; --rounds stays the attempt cap so a run
        # whose success rate collapses still ends.
        if args.until_ok is not None and wins >= args.until_ok:
            break
        played += 1
        t0 = time.time()
        # -- sample --------------------------------------------------------
        # Draw the round's target first: everything after it -- the edge to
        # hug, the grip height, the stand candidates, the squat -- is a
        # property of WHICH object this round is about.
        #
        # A shuffled deck drawn without replacement, not iid uniform. What
        # this feeds is a dataset, and a dataset wants BALANCE: an iid draw
        # over four targets left one of them unseen for the first eight
        # rounds of a twelve-round batch, which at collection scale is a
        # class the model gets much less of for no reason anyone chose. The
        # deck guarantees every target appears once per cycle and still
        # randomises the order within it.
        if fixed_cases is not None:
            case = fixed_cases[rnd]
            obj_key = case["target"]
            pl = plans[obj_key]
            can_nom = pl.nom
            task = instruction(pl.phrase)
            can_xy, stand, rel = (np.asarray(case[k], dtype=float) for k in ("can_xy", "stand", "rel"))
            psi, spawn_off, spawn_lat, dpsi = (case[k] for k in ("psi", "spawn_off", "spawn_lat", "dpsi"))
            squat_hint, appearance = case["squat_hint"], case["appearance"]
        else:
            if not deck:
                deck = list(pool)
                rng.shuffle(deck)
            obj_key = deck.pop()
            pl = plans[obj_key]
            can_nom, rel_cands, squat_hint = pl.nom, pl.rel_cands, pl.squat_hint
            ax, sgn = pl.ax, pl.sgn
            task = instruction(pl.phrase)
            # One integer per episode decides the whole look of the scene, so the
            # manifest can record it and a replay can reproduce the exact frame.
            appearance = int(rng.integers(0, 2**31 - 1))
            # The target lives in a band hugging its nominal edge (radial 5-10 cm
            # inboard, +-args.edge_jitter along it): a pelvis 10 cm OUTSIDE the edge
            # then reaches it at rel depth 0.15-0.24 — comfortably inside the
            # map instead of at its rim, which is what lets the no-contact
            # stand-off exist at all.
            for try_i in range(500):
                can_xy = can_nom.copy()
                can_xy[ax] = cxy[ax] + sgn * (half[ax] - rng.uniform(*CAN_INBOARD))
                can_xy[1 - ax] = can_nom[1 - ax] + rng.uniform(-args.edge_jitter, args.edge_jitter)
                if reach_config.ready_grasp_baseline:
                    from ready_grasp_policy import sample_stand as sample_comfortable_stand

                    rel, squat_hint = sample_comfortable_stand(pl, rng, reach_config)
                else:
                    rel, squat_hint = sample_stand(pl, rng)
                # Face the table from outside, along the can's inward edge
                # normal, +-25 deg (a uniform world yaw almost never yields a
                # stand outside the table).
                normal = np.zeros(2)
                normal[ax] = -sgn
                psi = float(
                    np.arctan2(normal[1], normal[0])
                    + rng.uniform(*((-0.14, 0.14) if reach_config.ready_grasp_baseline else (-0.44, 0.44)))
                )
                c, s = np.cos(psi), np.sin(psi)
                stand = can_xy - np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
                # Prefer slack over the bare floor: a stand SAMPLED at edge
                # 0.17 exactly leaves the +-4 cm execution scatter nowhere to
                # land (the edge-hold retry loop — a yaw -90 draw spent six
                # re-approaches hovering at 0.11-0.15). First 300 tries demand
                # 3 cm of headroom; only then fall back to the bare band.
                edge_lo = EDGE_MIN + (0.03 if try_i < 300 else 0.0)
                if not (edge_lo <= edge_dist(stand) <= EDGE_MAX):
                    continue
                # Where the robot starts. Sampled INSIDE the rejection loop
                # because the clearance check below depends on all three: a
                # spawn 3 m out with 0.35 m of lateral offset crosses different
                # furniture than one 1 m out dead on the line.
                spawn_lat = float(rng.uniform(-args.spawn_lateral, args.spawn_lateral))
                fv2 = np.array([np.cos(psi), np.sin(psi)])
                perp = np.array([-fv2[1], fv2[0]])
                if edge_range is None:
                    spawn_off = float(rng.uniform(off_lo, off_hi))
                else:
                    # Bisect for the offset that lands the spawn the requested
                    # distance from the table. edge_dist grows monotonically as
                    # the robot backs off along -fv2, so 60 halvings of [0, 6]
                    # resolve it far below any distance that matters.
                    want = float(rng.uniform(*edge_range))
                    lo_off, hi_off = 0.0, 6.0
                    for _ in range(60):
                        mid = 0.5 * (lo_off + hi_off)
                        if edge_dist(stand - mid * fv2 + spawn_lat * perp) < want:
                            lo_off = mid
                        else:
                            hi_off = mid
                    spawn_off = 0.5 * (lo_off + hi_off)
                spawn_pt = stand - spawn_off * fv2 + spawn_lat * perp
                jitter = float(np.deg2rad(rng.uniform(-args.spawn_yaw_err, args.spawn_yaw_err)))
                if args.face_object:
                    # Face the object FROM THE SPAWN, then jitter. dpsi stays the
                    # quantity the episode turns out at the spawn, so everything
                    # downstream (the turn, its guards, the manifest) is unchanged.
                    to_obj = can_xy - spawn_pt
                    dpsi = float(wrap(np.arctan2(to_obj[1], to_obj[0]) - psi) + jitter)
                else:
                    dpsi = jitter
                # The WALK must clear the table too: the batch's "fell during
                # the walk-up" class was the robot brushing the table edge on
                # the way in (video: hip catches the rim, robot tips backwards
                # onto the tabletop). Check the ACTUAL two legs, not one
                # straight line: with a lateral offset the robot goes
                # spawn -> lead_tgt (a diagonal) -> stand (on the line), and
                # the diagonal is the segment that swings widest.
                lead_pt = stand - (0.30 + RAISE_LEAD) * fv2
                legs = [(spawn_pt, lead_pt), (lead_pt, stand)]
                n_chk = max(2, int(np.ceil(spawn_off / 0.25)) + 1)
                pts = [q for a, b in legs for q in np.linspace(a, b, n_chk, endpoint=False)]
                if all(edge_dist(q) >= EDGE_MIN for q in pts):
                    break
            else:
                print("EPISODE FAILED [gym] no feasible sample found")
                continue
        case_id = fixed_cases[rnd]["case_id"] if fixed_cases is not None else f"{args.seed}:{rnd}"
        sampled_cases.append(
            dict(
                case_id=case_id,
                target=obj_key,
                appearance=int(appearance),
                can_xy=can_xy.tolist(),
                stand=stand.tolist(),
                rel=rel.tolist(),
                psi=float(psi),
                spawn_off=float(spawn_off),
                spawn_lat=float(spawn_lat),
                dpsi=float(dpsi),
                squat_hint=None if squat_hint is None else float(squat_hint),
            )
        )
        if fixed_cases is not None and "object_yaw" in case:
            sampled_cases[-1]["object_yaw"] = case["object_yaw"]
        if args.sample_only:
            continue
        side = "right"
        goal = f"{args.until_ok} ok" if args.until_ok is not None else f"{args.rounds}"
        print(
            f"[gym] round {rnd + 1}/{goal} (ok {wins}): {obj_key} "
            f"{np.round(can_xy, 2)}"
            f" rel [{rel[0]:.2f} {rel[1]:+.2f}] yaw {np.degrees(psi):.0f}"
            f" side {side} | spawn {spawn_off:.2f}m lat {spawn_lat:+.2f}m "
            f'yaw_err {np.degrees(dpsi):+.0f}deg | "{task}"'
        )

        # -- reset + settle -----------------------------------------------
        # A stand-stage failure may reflect navigation or initialization.
        # Keep legacy retries available, but fixed-case acceptance can disable
        # them rather than hide a failed first attempt behind reset success.
        for attempt in range(1 + args.stand_retries):
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
                json.dump(
                    dict(
                        target=obj_key,
                        appearance=appearance,
                        d_can=list(map(float, can_xy - can_nom)),
                        d_robot=list(map(float, stand - robot_nom)),
                        psi=float(psi),
                        side=side,
                        rel=list(map(float, rel)),
                        spawn_off=float(spawn_off),
                        spawn_lat=float(spawn_lat),
                        dpsi=float(dpsi),
                        squat_hint=(None if squat_hint is None else float(squat_hint)),
                        stand=list(map(float, stand)),
                        robot_nom=list(map(float, robot_nom)),
                    ),
                    open(args.dump_sample, "w"),
                    indent=1,
                )
            if sidecar:
                sidecar.start(rnd, attempt)
            round_result = run_round(
                p,
                reset,
                meta,
                obj_key,
                surf_a,
                edge_dist,
                can_xy - can_nom,
                stand - robot_nom,
                psi,
                side,
                rel,
                rm,
                stand,
                squat_hint,
                spawn_off=spawn_off,
                spawn_lat=spawn_lat,
                dpsi=dpsi,
                appearance=appearance,
                record=args.record,
                view_gate_x=args.view_gate_x,
                view_gate_y=args.view_gate_y,
                one_shot=args.one_shot,
                mid_band=args.mid_band,
                reach_config=reach_config,
                object_yaw=(case.get("object_yaw") if fixed_cases is not None else None),
            )
            ok, why, produced = round_result
            if args.record and produced and ok:
                # 'c' finishes RECORDING -> NEED_TO_SAVE and the exporter's
                # own loop writes the episode. Failures need no branch here
                # any more: run_round discards them itself, before the retry
                # below can toggle the exporter out of phase.
                p.cmd.key("c")
                # facing MATTERS here: an IDLE hold without one commands the
                # default world +x, which pirouetted the robot away from the
                # approach heading right after the back-off -- the "grasp,
                # retreat, then spin round" seen in every stream video. The
                # episode itself was never contaminated (the 'c' above ends
                # it first; fast250 tails hold yaw to 7-16 deg), but the spin
                # wastes 2-4 s of settling and reads as part of the task.
                p.cmd.hold(
                    4.0, mode=LocomotionMode.IDLE, facing=(float(np.cos(psi)), float(np.sin(psi)), 0.0)
                )  # let the write finish
            if sidecar:
                sidecar.stop()
            if args.manifest:
                # One line per ATTEMPT, not per round: a stand retry that got
                # as far as recording leaves an episode in the session, so
                # the converter's join has to see it too. It filters on
                # `produced`; the rest is for reading per-target success.
                with open(args.manifest, "a") as f:
                    # The scene's actual dimensions, not the seed that
                    # produced them: they are what an analysis wants, they
                    # survive a change to the sampling ranges, and they cost
                    # nothing since the scene already declares them.
                    o = meta["objects"][obj_key]
                    # The INITIAL CONDITIONS, not just the outcome. Without
                    # these a finished dataset cannot be replayed: an episode
                    # says what happened but not what it started from, so a
                    # closed-loop policy cannot be put on the same scene a
                    # recorded episode used, and "same initial conditions" is
                    # the only way to compare two policies without paying for
                    # the variance of independent draws. Cheap to write, and
                    # impossible to recover afterwards -- the collection that
                    # produced the first 200-episode bottle set omitted them
                    # and the holdout episodes' scenes are simply gone.
                    # World-frame, which is what EpisodeControlClient.reset
                    # takes: object xy, the robot's spawn xy, and its yaw.
                    fv2 = np.array([np.cos(psi), np.sin(psi)])
                    spawn_xy = stand - spawn_off * fv2 + spawn_lat * np.array([-fv2[1], fv2[0]])
                    rec_line = dict(
                        case_id=case_id,
                        round=rnd,
                        attempt=attempt,
                        target=obj_key,
                        task=task,
                        ok=bool(ok),
                        why=why,
                        appearance=appearance,
                        init=dict(
                            obj_xy=[round(float(v), 5) for v in can_xy],
                            robot_xy=[round(float(v), 5) for v in spawn_xy],
                            stand_xy=[round(float(v), 5) for v in stand],
                            yaw=round(float(psi + dpsi), 5),
                            psi=round(float(psi), 5),
                            side=side,
                            spawn_off=round(float(spawn_off), 4),
                            spawn_lat=round(float(spawn_lat), 4),
                        ),
                        table_top=round(float(surf_a["top"]), 4),
                        obj_height=round(float(o["height"]), 4),
                        obj_diameter=round(float(o["box"]["half_extent"][0]) * 2, 4),
                        produced=bool(args.record and produced),
                        # The park distances this dataset was missing: without
                        # them the cliff analysis has to be re-mined from pose
                        # sidecars.
                        park_first=round_result.execution["metrics"].get("first_stop"),
                        park=round_result.execution["metrics"].get("park"),
                        transition=round_result.execution["transition"],
                        execution=round_result.execution,
                        seconds=round(time.time() - t0, 1),
                    )
                    f.write(json.dumps(rec_line) + chr(10))
            if ok or not why.startswith("[stand]") or attempt >= args.stand_retries:
                break
            p.log(f"[gym] stand-stage retry ({why})")
        dt = time.time() - t0
        if ok:
            wins += 1
            print(f"EPISODE OK in {dt:.0f}s")
        else:
            print(f"EPISODE FAILED {why} in {dt:.0f}s")
        reset.end_video()
    if args.write_cases:
        save_cases(args.write_cases, sampled_cases, SCENE_META)

    print(f"[gym] {wins}/{played} OK")


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


def sample_object_yaw(fixed=None):
    """Optional exact-case yaw; historical callers retain random orientation."""
    if fixed is not None:
        if type(fixed) not in (int, float) or not np.isfinite(fixed):
            raise ValueError("object_yaw must be finite")
        return float(fixed)
    return float(np.random.uniform(-np.pi, np.pi))


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
    # Owned by run_round; direct diagnostic callers may use a local record.
    execution = execution if execution is not None else ExecutionRecord()
    # Spawn 0.30 m behind the stand. Directly ON the stand the STAND-UP
    # fails (13/24 never rose): the nominal pose is a deep crouch whose
    # knees ride ~0.25 m forward, and the rise needs more clearance than
    # standing does. 0.30 m back gives the rise ~0.5 m of room and leaves
    # only a two-step walk to the stand.
    fv2 = np.array([np.cos(psi), np.sin(psi)])
    perp = np.array([-fv2[1], fv2[0]])
    d_robot = np.asarray(d_robot, dtype=float) - spawn_off * fv2 + spawn_lat * perp
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
    if VLA_POLICY is not None:
        dpsi = 0.0
    psi_spawn = float(psi + dpsi)
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
    object_yaw = sample_object_yaw(object_yaw)
    execution.metrics["object_yaw"] = object_yaw
    reset_objects = {obj_key: {"dxy": [float(d_can[0]), float(d_can[1])], "yaw": object_yaw}}
    reset_robot = {"dxy": [float(d_robot[0]), float(d_robot[1])], "yaw": psi_spawn}
    reset.reset(objects=reset_objects, robot=reset_robot, appearance=appearance)
    time.sleep(0.3)
    landed_xy = np.asarray(p.pelvis_xy(p.obs()))
    want_xy = np.asarray(spawn_xy) - spawn_off * fv2 + spawn_lat * perp
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
    fv = (float(np.cos(psi)), float(np.sin(psi)), 0.0)
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
    p.cmd.hold(1.5 if VLA_POLICY is not None else 4.0, mode=LocomotionMode.IDLE, facing=fv_spawn)
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
    if abs(wrap(dpsi)) > 0.10 and VLA_POLICY is None:
        turned_from = np.asarray(p.pelvis_xy(p.obs()))
        if not p.turn_to(psi, tol=0.16, timeout=20.0, mode=LocomotionMode.SLOW_WALK):
            return False, (
                f"[stand] could not turn {np.degrees(dpsi):+.0f} deg to the approach heading at the spawn ({p.aborted})"
            )
        s = p.obs()
        if s["fallen"]:
            return False, "[stand] fell while turning at the spawn"
        # The turn is allowed to shuffle, not to walk away: it happens
        # before the sampler's clearance guarantees mean anything, since
        # those were computed for the spawn POINT.
        walked = float(np.linalg.norm(np.asarray(p.pelvis_xy(s)) - turned_from))
        p.log(
            f"[gym] spawn turn {np.degrees(dpsi):+.0f} deg -> "
            f"yaw_err {np.degrees(wrap(p.pelvis_yaw(s) - psi)):+.0f} deg, "
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
        if VLA_POLICY is not None:
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
    if VLA_POLICY is None:
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
    include_init_video = os.environ.get("SIM_VIDEO_INCLUDE_INIT") == "1"
    if include_init_video:
        stamp = time.time()
        reset.start_video(f"gym_{int(stamp)}")
        execution.metrics["video_start_wall"] = stamp
    if reach_config.ready_grasp_baseline and VLA_POLICY is None:
        from ready_grasp_policy import prepare_ready

        # This drives the expert encoder's upper-body reference. Token-only
        # deployment initializes through its own VLA prelude; asking it to
        # execute this reference can time out before the first model request.
        if not prepare_ready(p):
            return False, "[stand] baseline ready posture not delivered"
    elif reach_config.ready_grasp_baseline and VLA_POLICY is not None and os.environ.get("TF_INIT_TOKEN_FILE"):
        from initial_pose import prepare_token_pose
        from ready_grasp_policy import ready_q

        if not prepare_token_pose(p, ready_q()[15:]):
            return False, "[stand] token-only ready posture not delivered"
    _o0 = p.obs()["objects"][obj_key]
    cam0 = _o0.get("cam")
    # `cam is None` means BEHIND the camera and must fail too -- the arrival
    # gate below always treated it that way, this one used to let it through.
    # Guarded on the key's presence, not its value, so a publisher built
    # without a camera still runs (the key is absent then, not None).
    if "cam" in _o0 and (cam0 is None or abs(cam0[0]) > view_gate_x or abs(cam0[1]) > view_gate_y):
        return False, (
            "[view] object not in ego view at spawn (cam="
            + ("behind camera)" if cam0 is None else f"[{cam0[0]:+.2f} {cam0[1]:+.2f}])")
        )
    if not include_init_video:
        stamp = time.time()
        reset.start_video(f"gym_{int(stamp)}")
        execution.metrics["video_start_wall"] = stamp
    _wmark("video")
    if record:
        # 'c' toggles the exporter IDLE -> RECORDING; a second 'c' saves and
        # 'x' discards (zmq_driver.SonicCommander.key). Recording starts HERE,
        # after the settle, so the first frame of the episode is the first
        # frame of the task rather than a robot standing still.
        p.cmd.key("c")
        if rec is not None:
            rec["on"] = True
    task_grasp = TabletopGrasp(
        p,
        TabletopTarget(
            key=obj_key,
            side=side,
            below_center=reach_config.grip_below_center if reach_config.ready_grasp_recovery else None,
            grip_z=obj_top(meta, obj_key) - grasping.GRIP_BELOW_RIM,
            surface_top=surf_a["top"],
            box=meta["objects"][obj_key]["box"],
            body=meta["objects"][obj_key]["body"],
            surface_bodies=tuple(s["body"] for s in meta["surfaces"].values()),
        ),
    )
    grip_za = float(task_grasp.grip(p.obs())[2])

    def rel_dbg(tag, s2):
        y2 = p.pelvis_yaw(s2)
        c2, s2n = np.cos(y2), np.sin(y2)
        dv2 = p.object_pos(obj_key, s2)[:2] - p.pelvis_xy(s2)
        rx = c2 * dv2[0] + s2n * dv2[1]
        ry = -s2n * dv2[0] + c2 * dv2[1]
        yerr = np.degrees((y2 - psi + np.pi) % (2 * np.pi) - np.pi)
        p.log(f"[gym] {tag}: rel [{rx:.2f} {ry:+.2f}] yaw_err {yerr:+.0f} deg")

    rel_dbg("pre-walk ", p.obs())

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
    if VLA_POLICY is not None:
        s_fork = p.obs()
        d_fork = float(np.linalg.norm(np.asarray(p.object_pos(obj_key, s_fork)[:2]) - p.pelvis_xy(s_fork)))
        p.log(f"[gym] fork->vla: pelvis {np.round(p.pelvis_xy(s_fork), 3)} obj-dist {d_fork:.3f} m")

        def replace():
            # Mid-prelude re-teleport: the gait limit cycle walks drift-prone
            # spawn configs ~0.15 m/s through every planner countermeasure
            # (hold/squat/walk_to -- all measured), so the position is
            # RESTORED physically instead. Identical args -> identical
            # scene; label "replace" so frame_rate_guard's marker rule can
            # exclude it. qvel is zeroed by the reset, which also kills the
            # cycle's momentum.
            reset.reset(objects=reset_objects, robot=reset_robot, appearance=appearance, label="replace")
            time.sleep(0.8)  # stance settle; deploys tolerate teleports

        # spawn_xy is the ACCEPTANCE REFERENCE for tstart_spawn_off. It
        # used to be landed_xy (the raw teleport landing), which folded the
        # planner-mode settle (~0.13 m median, walk200) into the metric --
        # a segment collection rounds share, so the 0.1 m acceptance
        # complained every round about a displacement training also saw.
        # s_fork is the post-settle pose at handover: from here the metric
        # measures only what the VLA prelude itself adds (hold + blend).
        execution.enter("vla")
        return VLA_POLICY(
            p=p,
            reset=reset,
            meta=meta,
            obj_key=obj_key,
            psi=psi,
            fv=fv,
            grip_za=grip_za,
            record=record,
            replace=replace,
            spawn_xy=[float(v) for v in p.pelvis_xy(s_fork)],
        )

    execution.enter("navigate")

    def rel_now(s2):
        y2 = p.pelvis_yaw(s2)
        c2, s2n = np.cos(y2), np.sin(y2)
        dv2 = p.object_pos(obj_key, s2)[:2] - p.pelvis_xy(s2)
        return [
            c2 * dv2[0] + s2n * dv2[1],
            -s2n * dv2[0] + c2 * dv2[1],
            float(task_grasp.grip(s2)[2]) - float(s2["pelvis"]["pos"][2]),
        ]

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
    lead_tgt = np.asarray(spawn_xy) - (0.30 + RAISE_LEAD) * np.array(fv[:2])
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
    along = float(np.dot(lead_tgt - np.asarray(p.pelvis_xy(p.obs())), np.array(fv[:2])))
    if along > 0.20:
        # face_yaw, not face_target=False: hold the LINE, not the current
        # heading. See Primitives.walk_to -- holding the heading locks in the
        # spawn turn's dead-zone residual, which a long walk then magnifies
        # into a lateral error the deadband approach cannot erase.
        if not p.walk_to(lead_tgt, stop_dist=0.10, timeout=10.0 + 8.0 * spawn_off, face_target=False, face_yaw=psi):
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
    if reach_config.ready_grasp_baseline:
        from ready_grasp_policy import ready_q

        q_free = upper_body_from_mj(ready_q())
    arm_ramp = upper_ref.UpperRamp(p.cmd, q_free)
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY

    coordinated = None
    if reach_config.ready_grasp_baseline:
        from ready_grasp_policy import ReadyGraspPolicy

        arrival_task = task_grasp
        controller = ReadyGraspPolicy
        if reach_config.ready_grasp_recovery:
            from recoverable_grasp_policy import RecoverableGraspPolicy

            controller = RecoverableGraspPolicy
        if reach_config.coupled_squat:
            from coupled_grasp_policy import CoupledGraspPolicy

            controller = CoupledGraspPolicy
        coordinated = controller(p, arrival_task, reach_config, arm_ramp, execution)

    def raise_arm(_s, dist):
        if coordinated is not None:
            coordinated.walk_tick(_s)
            return
        u = float(np.clip(1.0 - (dist - 0.06) / RAISE_LEAD, 0.0, 1.0))
        sm = 3 * u * u - 2 * u**3

        return arm_ramp.push(q_free + sm * (READY_UPPER - q_free))

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
    # Three speed bands, from the measured gait quantum (diag_shuffle.py):
    #   0.26  -> 0.155 m/s, 11 cm per step
    #   0.215 -> 0.040 m/s,  5 cm per step
    #   0.20  -> 0.013 m/s,  1-2 cm per step
    # The gate fires at x 0.37-0.42 (matching the offline margin profile),
    # so the braking has to be DONE by then. With a single 0.12 m shuffle
    # band the robot only slowed at x 0.43 and an 11 cm coarse stride was
    # still in flight when the gate fired: measured carry 0.08-0.24 m, and
    # every round whose carry passed ~0.15 m ended inside the edge floor.
    # COARSE at 0.30, not 0.26 (2026-08-25). At 0.26 the gait initiation
    # minces: the first seconds of every episode showed a step taken and taken
    # back before the walk committed -- recorded hesitation a VLA then learns
    # as "dither first". 0.30 is not a guess: diag_shuffle measured the
    # release policy at speed 0.30 delivering a committed 17 cm/step natural
    # stride. Only the far band changes; the MID/SHUFFLE deceleration bands
    # that give the arrival its braking accuracy are untouched.
    APPR_COARSE, APPR_MID, APPR_SHUFFLE = 0.30, 0.215, 0.20
    # The expert's own 1-2 cm/s shuffle-imitating crawl let the learned policy
    # set its own stop point independent of where the expert stopped; removing
    # it (formerly GYM_SHUFFLE_BAND=0) moved 07d 30% -> 48%, McNemar p=0.052
    # (docs/experiments.md:234). MID speed now holds all the way to the brake.
    MID_BAND = mid_band

    # Coverage-first (2026-09-04): steer the walk at a point offset laterally
    # from the planned stand, so the last 0.3 m has to correct sideways --
    # the correction the policy needs is then IN the data instead of the
    # expert always arriving aligned. The creep closes both axes.
    appr_lat = float(np.random.uniform(-0.15, 0.15)) if STAND_COVER and coordinated is None else 0.0
    aim_xy = np.asarray(spawn_xy) + appr_lat * np.array([-fv[1], fv[0]])
    advance = float(np.clip(reach_config.arrival_advance, 0.0, 0.10))
    VISIBLE = bool(getattr(reach_config, "visible_expert", False))
    if VISIBLE:
        # Visible expert: no latch, no advance. The walk stops the tick the
        # MEASURED object depth reaches visible_brake_depth and decelerates
        # over visible_decel_band above it -- both functions of the relative
        # pose the cameras see, nothing per-episode.
        advance = 0.0
    aim_xy = aim_xy + advance * np.asarray(fv[:2])
    if appr_lat:
        p.log(f"[gym] approach aim offset {appr_lat * 100:+.0f} cm lateral")

    def do_approach(seconds: float, ramp: bool) -> str:
        appr_end = time.time() + seconds
        why, stall_x, stall_t = "timeout", None, time.time()
        brake_x, brake_e = float("nan"), float("nan")
        from arrival_validation import ArrivalAdvance

        advance_gate = ArrivalAdvance(advance)
        while time.time() < appr_end:
            s = p.obs()
            if s["fallen"]:
                return "fell"
            r = rel_now(s)
            if ramp or coordinated is not None:
                raise_arm(s, float(np.linalg.norm(np.asarray(spawn_xy) - p.pelvis_xy(s))))
            if coordinated is not None and coordinated.error:
                return coordinated.error
            reachable = rm.margin(r, "right", cap=MARGIN_STOP + 0.01) >= MARGIN_STOP
            if VISIBLE:
                stop_here = r[0] <= reach_config.visible_brake_depth
            else:
                stop_here = advance_gate.stop(r[0], reachable) if advance else reachable
            if stop_here:
                why = "visible brake depth" if VISIBLE else "advanced reachable stop" if advance else "can reachable"
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
            to_stand = np.asarray(aim_xy) - p.pelvis_xy(s)
            n_ts = float(np.linalg.norm(to_stand))
            mv = to_stand / n_ts if n_ts > 1e-3 else np.array(fv[:2])
            fv_now = fv
            speed = APPR_MID if r[0] - plan_r <= MID_BAND else APPR_COARSE
            if VISIBLE and reach_config.visible_decel_band > 0:
                # Monotone in the measured depth: full speed above the band,
                # ramping to the gait's slow floor at the brake depth.
                u = float(
                    np.clip((r[0] - reach_config.visible_brake_depth) / reach_config.visible_decel_band, 0.0, 1.0)
                )
                speed = min(speed, APPR_SHUFFLE + (speed - APPR_SHUFFLE) * u)
            p.cmd.frame(
                mode=LocomotionMode.SLOW_WALK, facing=fv_now, movement=(float(mv[0]), float(mv[1]), 0.0), speed=speed
            )
            time.sleep(0.02)
        if ramp and coordinated is None:
            # Finish whatever is left of the rise: the stop can fire before
            # the slew reaches READY, and a snap there would be a step in the
            # reference. The origin is the RAMP's own commanded state, not the
            # measured pose -- re-anchoring on the measurement here (which is
            # what this used to do) put a step of the whole tracking error
            # back in, every time it ran. Costs nothing when the rise already
            # finished during the walk: the first push returns converged and
            # no frame is sent, so no standing is recorded.
            t0 = time.time()
            while time.time() - t0 < RAISE_FINISH_MAX_S:
                if arm_ramp.push(READY_UPPER) < 1e-3:
                    break
                p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
                time.sleep(0.02)
            arm_ramp.settle()
            if time.time() - t0 > 0.10:
                p.log(f"[gym] arm rise finished at the table: {time.time() - t0:.2f} s of standing")
        # No settle hold at all any more (was 1.0 s, then 0.2 s). The grasp
        # is a 50 Hz servo that re-aims on the measured object pose every
        # tick, so a settled base buys nothing -- and every recorded standing
        # frame here teaches a VLA that standing is what one does at the
        # table (the round-1/2 stall).
        x_end = rel_now(p.obs())[0]
        p.log(
            f"[gym] approach: exit={why} braked at x {brake_x:.2f} "
            f"edge {brake_e:.2f} -> stopped x {x_end:.2f} "
            f"edge {edge_dist(p.pelvis_xy(p.obs())):.2f} "
            f"(carry {brake_x - x_end:+.2f}, plan_r {plan_r:.2f})"
        )
        execution.metrics.setdefault("arrival_measurements", []).append(
            dict(
                requested_advance_m=advance,
                first_reachable_depth_m=advance_gate.first_reachable,
                brake_depth_m=brake_x,
                brake_edge_m=brake_e,
                end_depth_m=x_end,
                end_edge_m=edge_dist(p.pelvis_xy(p.obs())),
                reason=why,
            )
        )
        rel_dbg("post-approach", p.obs())

        return why

    approach_result = do_approach(45.0, ramp=True)
    if coordinated is not None and coordinated.error:
        return False, coordinated.error
    if approach_result == "fell":
        return False, "[stand] fell during the approach"
    _r0 = rel_now(p.obs())
    execution.metrics["first_stop"] = round(float(np.hypot(_r0[0], _r0[1])), 3)

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
    for _ in range(0 if one_shot or coordinated is not None else 2):
        s = p.obs()
        rel_a = rel_now(s)
        too_far = rel_a[0] > plan_r + 0.06 or not rm.graspable(rel_a, "right")
        if not too_far:
            break
        p.log(f"[gym] still far: rel [{rel_a[0]:.2f} {rel_a[1]:+.2f}] — walking in again")
        if do_approach(12.0, ramp=False) == "fell":
            return False, "[stand] fell walking in again"
    s = p.obs()
    rel_a = rel_now(s)
    execution.metrics["park"] = round(float(np.hypot(rel_a[0], rel_a[1])), 3)
    # A real safety floor, not the planning floor. EDGE_MIN (0.17) is where
    # the PLANNER aims; rounds that actually stopped at edge 0.09-0.17 went
    # 12/12 with zero furniture contacts in the whole batch, so failing them
    # here would only discard good episodes now that nothing backs up to
    # rescue them.
    if edge_dist(p.pelvis_xy(s)) < 0.06:
        return False, "[stand] pelvis ended against the table edge"
    if not rm.graspable(rel_a, "right"):
        return False, (f"[stand] arrival left the object out of reach (rel [{rel_a[0]:.2f} {rel_a[1]:+.2f}])")

    # -- the grasp, same path as collect_episode ---------------------------
    st = p.obs()
    if stream_reach.toppled(st, obj_key):
        return False, "[gym] object toppled during the settle"
    cam = st["objects"][obj_key].get("cam")
    # Vertical bound view_gate_y, horizontal view_gate_x: the 52-deg-pitched
    # camera puts a close can low in the frame (fy -0.82..-0.94 killed 4
    # otherwise-good rounds/batch). fy 0.9 is still inside the physical image
    # (edge 1.0), so on real hardware the object stays detectable -- the
    # horizontal bound keeps the stricter margin because yaw scatter moves it
    # fast.
    if "cam" in st["objects"][obj_key] and (cam is None or abs(cam[0]) > view_gate_x or abs(cam[1]) > view_gate_y):
        return False, f"[view] object outside ego view at stand (cam={cam})"

    execution.enter("grasp_plan")
    grip = task_grasp.grip(st)

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

    plan = None
    if not reach_config.ready_grasp_baseline:
        plan = task_grasp.initial(st)
        p.log(adaptive_grasp.describe(plan))
        if not plan.ok:
            return False, f"[grasp] {plan.why}"
        show_ghost(plan)

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
        obj_key,
        grip,
        plan,
        side=side,
        surface_top=surf_a["top"],
        replan=task_grasp.replan,
        ghost=show_ghost,
        nudge=False,
        execution=execution,
        config=reach_config,
        coordinated=coordinated,
        transport_guard_factory=lambda ik: task_grasp.transport_guard(ik, reach_config),
        pin=(reset.pin_arms if os.environ.get("GYM_ARM_PIN", "0") == "1" else None),
        squat_hint=squat_hint,
        plan_r=float(np.hypot(rel[0], rel[1])),
        stand_rel=None,
        envelope=None,
    )
    if not grasp_result.ok:
        return False, f"[grasp] {grasp_result.why}"

    execution.enter("verify_visibility")
    vis = grasp_result.visibility
    if vis is not None:
        p.log(f"  ego visibility through servo: {vis * 100:.0f}%")
        if vis < 0.7:
            return False, (f"[view] object in ego view only {vis * 100:.0f}% of the servo")

    execution.enter("carry_and_verify")
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
    if not p.walk_to(back_tgt, stop_dist=0.08, timeout=8.0, face_target=False):
        return False, "[hold] fell while backing away from the table"
    # Keep validation in raw data; conversion can take a contiguous prefix
    # using the same sequence key as the exporter, without wall-clock guesses.
    execution.metrics["training_tail"] = dict(end_seq=int(p.obs()["seq"]), reason="post_transport_hold_verification")
    execution.enter("verify_terminal_hold")
    p.cmd.hold(2.0, mode=LocomotionMode.IDLE, facing=fv)
    s = p.obs()
    drift = float(np.linalg.norm(palm_frame_offset(s, obj_key, side) - hold_ref))
    if drift >= PALM_DRIFT_MAX:
        return False, (f"[hold] object slipped {drift * 100:.1f} cm in the palm frame during the hold")
    if float(p.object_pos(obj_key, s)[2]) <= surf_a["top"] + 0.03:
        return False, "[hold] object is back on the table after the hold"
    return True, ""


if __name__ == "__main__":
    main()
