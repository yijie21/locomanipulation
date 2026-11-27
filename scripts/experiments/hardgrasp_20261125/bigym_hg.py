"""selfmodel_hardgrasp_20261125, BiGym part: direct execution (D: cuRobo's joint path to the goal, bigym_bench's J) and
self-model execution (S: the same route, each goal moved by -(g + p) with M2's p, cuRobo replanned, scaled 2/3 and
1/3 when it fails, else D's route), paired by episode on BiGym's native controller (GR00T-WBC legs and waist, the arms'
PD, Dex1). The registration: docs/handoff/notebook-20261120/selfmodel_hardgrasp_20261125.md.

Stages (one look per episode, cached, read by both groups):
    look    reset(seed); one 640x360 RGB-D frame of the head camera at the reset (bigym_bench.Look); SAM 3 with the
            task's prompt; the target instance: score >= SAM_MIN, >= MIN_PTS points (the robot's pixels dropped), the
            nearest to the pelvis in the horizontal plane; its points (F0, the pelvis at the look) -> <dir>/<name>_obj.npz,
            the frame -> <dir>/<name>_look.npz
    grasps  graspgen_dex1.py on the clouds (GraspGen-X, Dex1) -> <dir>/<name>_grasps.npz (right_end_effector poses, F0)
    run     per seed and group: reset (the look's state hash checked), the grasp candidates (rules below), the turn,
            the walk to the stance, the squat, the pick among the candidates by IK at the stance, the two reaches
            (pregrasp, then a straight line in to the grasp), close, lift, hold; scored by the simulator's truth after
            the episode. The arms stay at the reset's hold until reach 1 (no ready pose, see the design record).

Decisions use the look and the robot's own readings only (the pelvis by its floating-base readings, as bigym_bench;
known to be optimistic against a real robot). The simulator's object state is read only for the row's scoring columns.

Grasp candidates (the look only): GraspGen-X's grasps whose feed is within FEED_CONE of straight down and whose gripper
(boxes GRIP_BOXES in the site frame, open fingers) at the grasp and at 4 points back along the feed to the pregrasp holds
at most COLL_MAX scene points (the look's points less the robot's and the target's: the target's mask dilated OBJ_DIL px
and points within OBJ_CLEAR of its cloud); the grasp point re-centred along the closing axis on the cloud's points
between the pads (dropped: wider than MAX_WIDTH or under 5 points); GraspGen-X score >= GG_MIN (Z_MIN_Q, DEPTH: off);
the best TOP_K by GraspGen-X's score (RANK "score"). The stance from the cloud's median g0: turn TURN_DEG (left) in
place, then WALK_ROUNDS [sideways, ahead] rounds until g0 sits STAND_X ahead and STAND_Y to the left of the pelvis, the ahead walk capped so the pelvis stays FRONT_CLEAR short of the nearest scene point ahead in the
legs' band (the look's points); then the height command HEIGHT and the torso pitch command PITCH, settled SQUAT_S. At
the stance, the grasp: the first candidate (score order) whose pregrasp and grasp DLS IK solutions (either of the two
finger twins, 180 deg about the feed: the Dex1 is symmetric) are within IK_TOL and keep the arm off the body (1 mm), the
twin with the smaller joint motion.

Reaches (one command each, nothing measured of the hand corrects anything):
    1  the pregrasp (PRE_M back along the feed): D = cuRobo's plan (bigym_bench.plan_j, the look's voxels less the
       target's), retimed by bigym_bench.joint_refs; no plan: bigym_bench.line_refs (recorded). S: M2's p on D's
       command at the command's state, the goal moved by -(g1 + p) (g1 = the reference end's pinch centre less the
       goal), replanned; failing, the move scaled 2/3, 1/3; failing, D's route.
       D has no plan (line): S runs D's line unshifted (SF: episodes without J's route run J's line).
    2  hold SETTLE_S, then the line in: bigym_bench.line_refs from the reach's last target's hand pose to the grasp (D)
       or to the grasp moved by reach 1's applied shift (S; IK short of the moved point by more than IK_TOL or stuck:
       scaled 2/3, 1/3, else D's line). M2's own prediction at reach 2's command is recorded, not used (out of its
       band, see the design record). Hold SETTLE_S.
    then close (gripper 1) CLOSE_S, lift: a line LIFT_M up (world z) from the held target's hand pose over LIFT_S, hold
    HOLD_S. Every phase after a reach holds the arm at its last target, so M2's label window (0.5 s) is clean.

Score (simulator, after the episode): success = the target prop held (robot.is_gripper_holding_object, right) at the
end, never fell, and its centre RISE_MIN or more above its centre at the reset at every control step of the last
HOLD_OK_S. The target prop: the task's prop nearest the look's target centroid at the reset. Failure classes, in the
registration's order: perception (no target, or the look's target centroid farther than PERCEPT_BAD from the target
prop's centre: scored after the episode), reach (no candidate / no IK at the stance), fell, furniture (the right arm
or gripper touched a non-target scene body by >= FURN_MM during the reaches), missed (not held after the close),
dropped (held after the close), other.

    source /wuji-vefps-D/wuji-il/yijie/tools/bigym2/env.sh; export OMP_NUM_THREADS=1; cd <repo>
    python .../bigym_hg.py look --task dishwasher_unload_cups --seeds 732000:732010 --dir <out>/look
    python .../bigym_hg.py run  --task dishwasher_unload_cups --seeds 732000:732010 --groups D,S --dir <out>/look \
        --out <out>/rows/x.jsonl --plan-socket <sock>
    python .../bigym_hg.py analyze <rows.jsonl ...>

DESIGN RECORD (design seeds 732000-732009 only; frozen 2026-11-25 before any official seed)
  Tasks kept: dishwasher_unload_cups only. Dropped on the design seeds:
    take_cups                  the mugs sit in the wall cabinet above the head camera's field of view at the reset: 0
                               of their pixels in the frame on 3 of 3 seeds checked (segmentation), still 0 with the
                               torso pitch command at its -0.2 rad floor; SAM 3 ("mug") finds nothing on 10 of 10.
    dishwasher_unload_cutlery  the fork is out of the reset frame (0 pixels on 3 of 3 seeds; ~60 after a 90 deg turn,
                               mostly behind the robot's own right gripper); SAM 3 ("fork") finds nothing on 10 of 10.
    dishwasher_unload_plates   the plate stands in the middle of the bottom tray, 0.60-0.67 m ahead of the pelvis at
                               the nearest stance that keeps the legs FRONT_CLEAR off the door and tray: no GraspGen-X
                               grasp has an IK solution there on 7 of 10 seeds (squat 0.62, torso pitch 0.3 rad), and
                               on the other 3 SAM 3 ("plates") picked a non-plate region (no grasp passes the filters).
  Look: SAM 3 scores these renders 0.10-0.65 -> SAM_MIN 0.1 (it returns nothing under about 0.1).
  Arms: grasp_tol's ready pose (both arms up) put 20 of 29 joints (all 7 of the right arm) outside M2's band at the
  command and M2 predicted 1.1-2.2 m offsets -> no ready pose (the arms stay at the reset's hold): 0-4 of 29 out.
  Walk: turn 90 deg left in place (the mugs are to the robot's left at the reset), then the stance; the forward stop
  from the look's points in the legs' band (the floor excluded: LEG_BAND), FRONT_CLEAR 0.22 (0.30 kept the robot
  0.45 m short). STAND_X 0.33 (with the stance from the cloud's median: 0.37 no candidate with IK on 9/9 seeds,
  0.40 on 7/9, 0.45 on 9/9; 0.33 IK on 9/9), STAND_Y -0.15, standing (squat 0.62 / 0.55 at STAND_X 0.37 under an
  earlier candidate order: no IK on 9/9).
  Grasp order: GraspGen-X's score. Tried on D only: the near end along the heading, then the attitude to a 45 deg
  forward feed ("near": 1/9 lifted, its picks deep in the list, the cup pushed 56-197 mm before the close on 6 of 8);
  MAX_WIDTH 0.03 (0/9), DEPTH 0.015 / 0.02 / 0.035 (0/9, 0/9, 0/9), Z_MIN_Q 0.5 / 0.75 (0/9, 0/9), CLOSE_S 1.5
  (0/9), CLOSE_S 1.5 + LIFT_S 5 (0/9), "near" + CLOSE_S 1.5 (1/9). Kept the detector's own order: one episode is
  within the noise of 9 and "near" is a heuristic of ours on top of GraspGen-X.
  Design-seed outcome at the frozen rule: 0 of 9 lifted in either group (one seed has no look target). The gripper
  holds the mug after the close (is_gripper_holding_object) and loses it as the hand rises 9-14 cm: after the lift the
  mug touches only the tray and its tines, the gripper not at all. The main metric is at the floor on the design seeds.
  M2 at reach 2 (from the pregrasp): every right-arm joint outside its training band (M2's training reaches all start
  with the arm at rest: each right-arm joint's 0.5-99.5 % band is under 0.04 rad wide); its predictions there are
  0.1-2 m -> S's line in is D's line moved by reach 1's applied shift (recorded; M2's own reach-2 prediction is
  recorded, not used). When D has no plan (cuRobo IK_FAIL: the look's voxels; the same goal plans without them), S
  runs D's line unshifted (SF's rule: episodes without J's route run J's line).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bigym_20261124"))
import bigym_bench as B  # noqa: E402

REPO = HERE.parents[2]
OUT = REPO / "outputs/hardgrasp_20261125/bigym"
EVAL_LO, EVAL_HI = 620000, 630000
DESIGN = (732000, 732010)
TASKS = {
    "take_cups": dict(prompt="mug", props="cups"),
    "dishwasher_unload_cups": dict(prompt="mug", props="cups"),
    "dishwasher_unload_plates": dict(prompt="plates", props="plates"),  # "plate" finds nothing on 9 of 10 design seeds
    "dishwasher_unload_cutlery": dict(prompt="fork", props="cutlery"),
}

# ------------------------------------------------------------------------------------- the design (frozen at the end)
SAM_MIN, MIN_PTS = 0.1, 30  # SAM 3 scores these rendered mugs and plates 0.1-0.65 (design seeds); it returns nothing under ~0.1
FEED_CONE = {"dishwasher_unload_cups": 60.0, "dishwasher_unload_plates": 60.0, "dishwasher_unload_cutlery": 45.0,
             "take_cups": 90.0}  # fmt: skip
COLL_MAX, TOP_K = 3, 200
GG_MIN = 0.5  # GraspGen-X's score floor
DEPTH = 0.0  # m: the grasp point moved this far along the feed (deeper) after GraspGen-X's
Z_MIN_Q = None  # drop grasps whose point lies below this quantile of the object cloud's height (None: off)
RANK = "score"  # the candidates' order: "score" (GraspGen-X's) or "near" (near end, then attitude)
NEAR_BAND = 0.04  # m: grasp points this close to the object's near end along the heading go first
LIFT_ROT_W = 0.1  # the lift line's attitude weight (the hand's attitude is free-ish while it rises)
PREF_TILT = 45.0  # deg: candidates ranked by their feed's angle to straight down turned this far toward the heading
READY_ARMS = False  # the ready pose (grasp_tol's) before the walk: puts every arm joint outside M2's training band
PAD_X, PAD_Z, MAX_WIDTH = (-0.029, 0.037), 0.013, 0.080  # the Dex1 pads (grasp_tol.py), the widest object across
TURN_DEG = {"dishwasher_unload_cups": 90.0, "dishwasher_unload_plates": 90.0, "dishwasher_unload_cutlery": 90.0,
            "take_cups": 0.0}  # fmt: skip
STAND_X = {"dishwasher_unload_cups": 0.33, "dishwasher_unload_plates": 0.40, "dishwasher_unload_cutlery": 0.37,
           "take_cups": 0.37}  # fmt: skip
STAND_Y = {"dishwasher_unload_cups": -0.15, "dishwasher_unload_plates": -0.15, "dishwasher_unload_cutlery": -0.15,
           "take_cups": -0.15}  # fmt: skip
FRONT_CLEAR = 0.22  # m: the pelvis stays this far short of the nearest scene point ahead in the legs' band
LEG_BAND = (-0.62, -0.20)  # m, the band's height relative to the pelvis at the look (above the floor, below the hips)
LEG_HALF_W = 0.22  # m: the band's half width across the walk
HEIGHT = {"dishwasher_unload_cups": None, "dishwasher_unload_plates": 0.62, "dishwasher_unload_cutlery": 0.62,
          "take_cups": None}  # fmt: skip
PITCH = {"dishwasher_unload_cups": 0.0, "dishwasher_unload_plates": 0.3, "dishwasher_unload_cutlery": 0.3,
         "take_cups": 0.0}  # fmt: skip
SQUAT_S = 3.0
PRE_M = 0.08
IK_TOL, IK_TOL_DEG = 0.01, 10.0
WALK_ROUNDS = 2
TURN_K, TURN_MAX, TURN_TOL, TURN_SETTLE_S = 1.5, 0.5, math.radians(2.0), 2.0
YAW_K, YAW_MAX = 1.5, 0.3
READY_P = (0.26, -0.18, 0.28)
READY_TILT = 45.0
FAMILY = (-1.0, -1.2, 1.0, -0.6, -0.8, -0.3, 1.3)
LEFT_READY = (0.3, 0.2, 0.0, -1.0, 0.0, 0.0, 0.0)
READY_S, READY_HOLD_S = 2.0, 0.5
SETTLE_S, CLOSE_S, LIFT_M, LIFT_S, HOLD_S = 0.5, 0.6, 0.15, 1.5, 1.5
RISE_MIN, HOLD_OK_S = 0.10, 1.0
FURN_MM = 2.0
PERCEPT_BAD = 0.10  # m: a look target farther than this from every prop's centre is a perception failure (scoring)
SCALES = (1.0, 2.0 / 3.0, 1.0 / 3.0)
# the open Dex1 in the right_end_effector frame (x feed, y the fingers' axis): finger pads and the base, boxes
# (centre, half extents), the fingers' inner faces at +-0.0447 open (grasp_tol's 89.4 mm opening)
GRIP_BOXES = (((0.004, 0.054, 0.0), (0.034, 0.009, 0.014)), ((0.004, -0.054, 0.0), (0.034, 0.009, 0.014)),
              ((-0.075, 0.0, 0.0), (0.045, 0.065, 0.035)))  # fmt: skip
DESIGN_KEYS = ("FEED_CONE", "COLL_MAX", "TOP_K", "TURN_DEG", "STAND_X", "STAND_Y", "FRONT_CLEAR", "LEG_BAND", "LEG_HALF_W",
               "HEIGHT", "PITCH", "SQUAT_S", "PRE_M", "IK_TOL", "READY_P", "WALK_ROUNDS", "MAX_WIDTH",
               "GG_MIN", "PREF_TILT", "READY_ARMS", "NEAR_BAND", "LIFT_ROT_W", "RANK", "DEPTH", "Z_MIN_Q", "CLOSE_S", "LIFT_S", "LIFT_M")  # fmt: skip
DT = B.DT
LEFT = [n.replace("right_", "left_") for n in B.ARM]


def ready_R(tilt_deg: float) -> np.ndarray:
    t = math.radians(tilt_deg)
    x = np.array([math.sin(t), 0.0, -math.cos(t)])
    y = np.array([0.0, -1.0, 0.0])
    return np.column_stack([x, y, np.cross(x, y)])


def make_env(task: str):
    from bigym.loco import make
    from bigym.loco.eval import protocol_violations

    env = make(task, camera_keys=())  # our own renderer reads the head camera; no observation is read
    viol = [v for v in protocol_violations(env) if not v.startswith("camera_keys:")]
    if viol:
        raise SystemExit(f"not the official config: {viol}")
    return env


def state_hash(d) -> str:
    h = hashlib.md5()
    h.update(np.round(d.qpos, 9).tobytes())
    h.update(np.round(d.qvel, 6).tobytes())
    return h.hexdigest()[:12]


def check_seeds(seeds, allow_eval: bool):
    bad = [s for s in seeds if EVAL_LO <= s < EVAL_HI]
    if bad and not allow_eval:
        raise SystemExit(f"seeds {bad[:3]}... are in the eval block; --allow-eval-seeds (only after the freeze)")


def parse_seeds(s: str) -> list:
    if ":" in s:
        lo, hi = map(int, s.split(":"))
        return list(range(lo, hi))
    return [int(x) for x in s.split(",")]


# ------------------------------------------------------------------------------------------------------------ the look
class CachedLook:
    """bigym_bench.Look's attributes from the cached frame."""

    def __init__(self, z):
        self.rgb, self.depth, self.own = z["rgb"], z["depth"].astype(float), z["own"].astype(bool)
        self.cam_p, self.cam_R = z["cam_p"], z["cam_R"]
        self.valid = np.isfinite(self.depth) & (self.depth > 0.05) & (self.depth < 3.0)
        v, u = np.mgrid[0 : B.H, 0 : B.W]
        zz = self.depth
        pc = np.stack([(u - B.CX) / B.FX * zz, -(v - B.CY) / B.FX * zz, -zz], axis=-1)
        self.P = pc @ self.cam_R.T + self.cam_p

    def points(self, mask):
        return self.P[mask & self.valid & ~self.own]


def cmd_look(a) -> None:
    seeds = parse_seeds(a.seeds)
    check_seeds(seeds, a.allow_eval_seeds)
    env = make_env(a.task)
    kin = B.Kin(env)
    m, d = env.inner_env.model, env.inner_env.data
    keep = int(m.vis.quality.offsamples)
    m.vis.quality.offsamples = 0
    renderer = mujoco.Renderer(m, B.H, B.W)
    m.vis.quality.offsamples = keep
    os.makedirs(a.dir, exist_ok=True)
    prompt = TASKS[a.task]["prompt"]
    for seed in seeds:
        t0 = time.time()
        env.reset(seed=seed)
        kin.load(d)
        h = state_hash(d)
        look = B.Look(env, kin, renderer)
        inst = B.sam3(look.rgb, prompt)
        pp, Rp = d.xpos[kin.pelvis].copy(), d.xmat[kin.pelvis].reshape(3, 3).copy()
        cands = []
        for i, (s, mk) in enumerate(inst):
            X = look.points(mk)
            rec = dict(i=i, score=round(s, 3), px=int(mk.sum()), n_pts=len(X))
            if len(X):
                c = np.median(X, 0)
                rec.update(c=B.r3(c), dist=round(float(np.linalg.norm(c[:2])), 4))
            rec["ok"] = bool(s >= SAM_MIN and len(X) >= MIN_PTS)
            cands.append(rec)
        ok = [r for r in cands if r["ok"]]
        name = f"{a.task}_{seed}"
        info = dict(task=a.task, seed=seed, prompt=prompt, state_hash=h, cands=cands, pelvis=B.r3(pp),
                    pelvis_R=B.r3(Rp.ravel(), 6))  # fmt: skip
        mask = np.zeros((B.H, B.W), bool)
        if ok:
            top = min(ok, key=lambda r: r["dist"])
            mask = inst[top["i"]][1]
            X = look.points(mask)
            info.update(target=top["i"], n_pts=len(X), centroid=B.r3(np.median(X, 0)))
            np.savez_compressed(f"{a.dir}/{name}_obj.npz", pts=X.astype(np.float32))
        else:
            info["target"] = None
        np.savez_compressed(f"{a.dir}/{name}_look.npz", rgb=look.rgb, depth=look.depth.astype(np.float32), own=look.own,
                            mask=mask, cam_p=look.cam_p, cam_R=look.cam_R)  # fmt: skip
        info["wall_s"] = round(time.time() - t0, 2)
        with open(f"{a.dir}/{name}_look.json", "w") as f:
            json.dump(info, f)
        print(json.dumps({k: info.get(k) for k in ("task", "seed", "target", "n_pts", "centroid", "wall_s")}),
              [(r["score"], r["n_pts"], r.get("dist")) for r in cands], flush=True)  # fmt: skip
    renderer.close()
    env.close()


# ------------------------------------------------------------------------------------------------- grasp candidates
def grip_hits(Xs: np.ndarray, p: np.ndarray, R: np.ndarray) -> int:
    """Scene points inside the open gripper's boxes with right_end_effector at (p, R)."""
    L = (Xs - p) @ R
    n = 0
    for c, e in GRIP_BOXES:
        n += int(np.all(np.abs(L - np.array(c)) <= np.array(e), axis=1).sum())
    return n


def centre_close(obj: np.ndarray, p: np.ndarray, R: np.ndarray):
    """The grasp point moved along the closing axis to the middle of the object's points between the pads (the pads'
    footprint PAD_X x +-PAD_Z, within +-0.08 m across) -> (point, extent across) or (p, None) with under 5 points."""
    L = (obj - p) @ R
    sel = (L[:, 0] >= PAD_X[0]) & (L[:, 0] <= PAD_X[1]) & (np.abs(L[:, 2]) <= PAD_Z) & (np.abs(L[:, 1]) <= 0.08)
    if sel.sum() < 5:
        return p, None
    lo, hi = float(L[sel, 1].min()), float(L[sel, 1].max())
    return p + 0.5 * (lo + hi) * R[:, 1], hi - lo


def candidates(task: str, look: CachedLook, mask: np.ndarray, obj: np.ndarray, G: np.ndarray, score: np.ndarray):
    """-> list of (index, T (4,4) F0, info) in score order after the cone and collision filters, at most TOP_K; and
    the counts."""
    keep = look.valid & ~look.own & ~ndimage.binary_dilation(mask, iterations=B.OBJ_DIL)
    Xs = look.P[keep]
    from scipy.spatial import cKDTree

    if len(obj):
        Xs = Xs[cKDTree(obj).query(Xs)[0] >= B.OBJ_CLEAR]
    c = np.median(obj, 0)
    Xs = Xs[np.linalg.norm(Xs - c, axis=1) < 0.4]  # the gripper never reaches beyond
    down = np.array([0.0, 0.0, -1.0])
    out, n_cone, n_coll, n_wide, n_low, n_lowz = [], 0, 0, 0, 0, 0
    t = math.radians(TURN_DEG[task])
    heading = np.array([math.cos(t), math.sin(t), 0.0])  # F0: the robot's heading after the turn
    pref = math.cos(math.radians(PREF_TILT)) * down + math.sin(math.radians(PREF_TILT)) * heading
    for i in range(len(G)):
        R, p = G[i, :3, :3], G[i, :3, 3]
        if R[:, 0] @ down < math.cos(math.radians(FEED_CONE[task])):
            n_cone += 1
            continue
        hits = max(grip_hits(Xs, p - s * PRE_M * R[:, 0], R) for s in (0.0, 0.25, 0.5, 0.75, 1.0))
        if hits > COLL_MAX:
            n_coll += 1
            continue
        p2, width = centre_close(obj, p, R)
        if width is None or width > MAX_WIDTH:
            n_wide += 1
            continue
        T = G[i].copy()
        T[:3, 3] = p2 + DEPTH * R[:, 0]
        if Z_MIN_Q is not None and T[2, 3] < float(np.quantile(obj[:, 2], Z_MIN_Q)):
            n_lowz += 1
            continue
        if float(score[i]) < GG_MIN:
            n_low += 1
            continue
        ang = math.degrees(math.acos(float(np.clip(R[:, 0] @ pref, -1.0, 1.0))))
        out.append((i, T, dict(score=round(float(score[i]), 4), hits=hits, width_mm=round(1000 * width, 1), ang=round(ang, 1),
                               moved_mm=round(1000 * float(np.linalg.norm(p2 - p)), 1))))  # fmt: skip
    # the near end of the object along the heading (the mug's handle faces the robot): candidates whose grasp point
    # lies within NEAR_BAND of the cloud's nearest point along the heading first, then by the feed's angle to `pref`
    # plus the closing axis's angle to the horizontal lateral (grasp_tol's frozen handle grasp: feed 45 deg forward,
    # fingers closing across)
    lat = np.cross(np.array([0.0, 0.0, 1.0]), heading)
    near0 = float((obj @ heading).min())
    for c in out:
        R = c[1][:3, :3]
        c[2]["near_cm"] = round(100 * (float(c[1][:3, 3] @ heading) - near0), 1)
        c[2]["close_ang"] = round(math.degrees(math.acos(min(1.0, abs(float(R[:, 1] @ lat))))), 1)
    if RANK == "near":
        out.sort(key=lambda c: (c[2]["near_cm"] > 100 * NEAR_BAND, c[2]["ang"] + c[2]["close_ang"], -c[2]["score"]))
    else:  # "score": GraspGen-X's own order (the grasps file is best first)
        out.sort(key=lambda c: -c[2]["score"])
    out = out[:TOP_K]
    return out, dict(n=len(G), cone_out=n_cone, coll_out=n_coll, wide_out=n_wide, low_out=n_low, lowz_out=n_lowz, kept=len(out))


# ------------------------------------------------------------------------------------------------------- an episode
class Ep:
    def __init__(self, env, kin, task, m2, a):
        self.env, self.kin, self.task, self.m2_, self.a = env, kin, task, m2, a
        self.d = env.inner_env.data
        m, ns = env.inner_env.model, env.inner_env.robot.namespace
        self.m = m
        self.scene_root = {m.body_rootid[kin.pelvis]}

    # -- helpers ----------------------------------------------------------------------------------------------
    def yaw(self) -> float:
        R = self.run.pelvis()[1]
        return math.atan2(R[1, 0], R[0, 0])

    def turn(self, ang: float) -> float:
        run = self.run
        y_goal = self.yaw() + ang
        for _ in range(int(round((abs(ang) / 0.3 + 4.0) / DT))):
            e = (y_goal - self.yaw() + math.pi) % (2 * math.pi) - math.pi
            if abs(e) < TURN_TOL:
                break
            run.hold[3] = float(np.clip(TURN_K * e, -TURN_MAX, TURN_MAX))
            if run.step("turn"):
                break
        run.hold[3] = 0.0
        self.wait("turn_settle", TURN_SETTLE_S)
        return math.degrees(self.yaw() - (y_goal - ang))

    def walk(self, phase, dist, vx, cap_s, axis) -> float:
        run = self.run
        p0, R0 = run.pelvis()
        h = np.array([R0[0, axis], R0[1, axis], 0.0])
        h /= np.linalg.norm(h)
        sign = 1.0 if vx > 0 else -1.0
        for _ in range(int(round(cap_s / DT))):
            if float(sign * (run.pelvis()[0] - p0) @ h) >= dist - B.WALK_LEAD:
                break
            err = (self.yaw() - self.yaw_ref + math.pi) % (2 * math.pi) - math.pi
            run.hold[3] = float(np.clip(-YAW_K * err, -YAW_MAX, YAW_MAX))
            if run.step(phase, None, None, *((vx, 0.0) if axis == 0 else (0.0, vx))):
                break
        run.hold[3] = 0.0
        return float(sign * (run.pelvis()[0] - p0) @ h)

    def wait(self, phase, seconds, grip=None) -> None:
        for _ in range(int(round(seconds / DT))):
            if self.run.step(phase, None, grip):
                return

    def to_now(self, p0):
        """F0 (the pelvis at the look) -> the pelvis frame now (floating-base readings); (3,) or (n, 3)."""
        pN, RN = self.run.pelvis()
        return (np.asarray(p0, float) @ self.RF0.T + self.pF0 - pN) @ RN

    def R_now(self, R0):
        return self.run.pelvis()[1].T @ self.RF0 @ R0

    def touch_scene(self) -> dict:
        """The right arm's / gripper's contacts with scene bodies other than the target prop (deepest mm)."""
        m, d, kin, out = self.m, self.d, self.kin, {}
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in kin.right) == (b2 in kin.right):
                continue
            other = b2 if b1 in kin.right else b1
            if kin.robot_bodies[other] or m.body_rootid[other] in self.target_roots:
                continue
            k = m.body(other).name
            out[k] = max(out.get(k, 0.0), round(-1000 * float(c.dist), 1))
        return out

    def target_contacts(self) -> list:
        """Diagnostics (scored, after the close): the target prop's contacts: [its body, the other body, mm]."""
        m, d, out = self.m, self.d, {}
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            t1, t2 = m.body_rootid[b1] in self.target_roots, m.body_rootid[b2] in self.target_roots
            if t1 == t2:
                continue
            mine, other = (b1, b2) if t1 else (b2, b1)
            k = (m.body(mine).name, m.body(other).name)
            out[k] = max(out.get(k, 0.0), round(-1000 * float(c.dist), 1))
        return [[x, y, v] for (x, y), v in sorted(out.items())]

    def body_contacts(self) -> list:
        """The robot's contacts with the scene (not the floor), the right arm's excepted: [robot body, scene body, mm]."""
        m, d, kin, out = self.m, self.d, self.kin, {}
        for c in d.contact[: d.ncon]:
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = kin.robot_bodies[b1], kin.robot_bodies[b2]
            if r1 == r2:
                continue
            mine, other, og = (b1, b2, c.geom2) if r1 else (b2, b1, c.geom1)
            if m.geom_type[og] == mujoco.mjtGeom.mjGEOM_PLANE or mine in kin.right:
                continue
            k = (m.body(mine).name.split("/")[-1], m.body(other).name)
            out[k] = max(out.get(k, 0.0), round(-1000 * float(c.dist), 1))
        return [[x, y, v] for (x, y), v in sorted(out.items())]

    def path(self, phase, refs, grip=0.0) -> None:
        run = self.run
        for r in refs:
            if run.step(phase, r, grip):
                return
            t = self.touch_scene()
            for k, v in t.items():
                self.furn[k] = max(self.furn.get(k, 0.0), v)

    # -- the two commands -------------------------------------------------------------------------------------------
    def plan(self, q_meas, ctrl0, p, R, vox, T_vp):
        """cuRobo to (p, R) (pelvis frame) -> refs, info, or (None, info)."""
        try:
            Q, pinfo = B.plan_j(self.a.plan_socket, self.kin, q_meas, vox, p, R, T_vp, self.a.plan_timeout)
        except (OSError, EOFError) as e:
            Q, pinfo = None, dict(ok=False, status=f"no plan server: {type(e).__name__}")
        if Q is None:
            return None, pinfo
        refs, info = B.joint_refs(self.kin, Q, ctrl0)
        return refs, dict(pinfo, **info)

    def pred(self, i_cmd, refs, ctrl0, dur):
        out = self.m2_.predict(self.run.film, i_cmd, refs, ctrl0, dur, self.h_cmd, jnt=self.kin.jnt)
        import via_point as VP

        F = VP.features(dict(x=out["x"][None], ref=out["ref"][None].astype(np.float32), key=self.m2_.key,
                             hand2=self.m2_.hand2), np.arange(1))  # fmt: skip
        z = np.abs((F - self.m2_.mu_x) / self.m2_.sd_x).max(0)
        top = np.argsort(-z)[:4]
        out["z"] = dict(max=round(float(z.max()), 1), top=[[int(i), round(float(z[i]), 1)] for i in top])
        Rp = self.run.pelvis()[1]
        return out, Rp.T @ out["p_world"]  # p in the pelvis frame (offsets: rotation only)

    # -- the episode ------------------------------------------------------------------------------------------------
    def go(self, seed: int, group: str, look_dir: str) -> dict:
        from bigym.const import HandSide

        env, kin, task, a = self.env, self.kin, self.task, self.a
        inner, d, m = env.inner_env, self.d, self.m
        t_wall = time.time()
        name = f"{task}_{seed}"
        info = json.load(open(f"{look_dir}/{name}_look.json"))
        env.reset(seed=seed)
        inner._success_hold_steps = 10**9  # no episode end on the task's success (no placement here)
        budget = int(env._time_limit)
        env.bigym._time_limit = 10**9  # nor on the time limit (recorded: the official budget in steps)
        kin.load(d)
        row = dict(task=task, seed=seed, group=group, budget_steps=budget, bigym_commit="35ec409",
                   look_hash_ok=state_hash(d) == info["state_hash"])  # fmt: skip
        self.run = run = B.Run(env, kin, task)
        self.furn = {}
        self.pF0, self.RF0 = run.pelvis()
        props = getattr(inner, TASKS[task]["props"])
        props0 = [np.asarray(pr.get_position(), float).copy() for pr in props]
        # the target prop (scoring only): the one nearest the look's target centroid at the reset
        if info.get("target") is not None:
            c_w = self.RF0 @ np.array(info["centroid"]) + self.pF0
            ti = int(np.argmin([np.linalg.norm(p - c_w) for p in props0]))
        else:
            ti = 0
        target = props[ti]
        tops = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0 and not kin.robot_bodies[b]]
        tb = min(tops, key=lambda b: float(np.linalg.norm(d.xpos[b] - props0[ti])))
        self.target_roots = {int(m.body_rootid[tb])}
        self.left_q = np.array([m.jnt_qposadr[m.joint(inner.robot.namespace + n).id] for n in LEFT])
        self.left0 = d.qpos[self.left_q].copy()
        self.tb = tb
        row["truth"] = dict(target_prop=ti, prop0=B.r3(props0[ti]))
        if info.get("target") is not None:  # scoring only: the look's target centroid against the prop's true centre
            row["truth"]["percept_err_cm"] = round(100 * float(np.linalg.norm(c_w - props0[ti])), 2)
        self.h_cmd = -1.0

        def finish(fail, why):
            row.update(success=False, fail=fail, why=why, steps=run.k, wall_s=round(time.time() - t_wall, 2),
                       fell=bool(env.episode_fell()))  # fmt: skip
            return row

        if info.get("target") is None:
            return finish("perception", "no target instance")
        look = CachedLook(np.load(f"{look_dir}/{name}_look.npz"))
        obj = np.load(f"{look_dir}/{name}_obj.npz")["pts"].astype(float)
        gfile = f"{look_dir}/{name}_grasps.npz"
        if not os.path.exists(gfile):
            return finish("perception", "no grasps file")
        gz = np.load(gfile)
        G, gs = gz["T"], gz["score"]
        mask = np.load(f"{look_dir}/{name}_look.npz")["mask"].astype(bool)
        cand, cinfo = candidates(task, look, mask, obj, G, gs)
        row["cand"] = cinfo
        if not cand:
            return finish("reach", "no grasp candidate after the cone and collision filters")

        # the ready pose (both arms up off the furniture)
        lay = env.wholebody_action_layout()
        lslots = np.array([lay["base_dim"] + list(lay["limb_names"]).index(n) for n in LEFT])
        if READY_ARMS:
            q_ready, e_r, _ = kin.ik(np.array(READY_P), ready_R(READY_TILT), np.array(FAMILY), 1.0, iters=400, tol_p=1e-4)
            a0, l0, l1 = run.ref.copy(), run.hold[lslots].copy(), np.array(LEFT_READY)
            N = int(round(READY_S / DT))
            for k in range(N):
                u = B.minjerk((k + 1) / N)
                run.hold[lslots] = l0 + u * (l1 - l0)
                run.step("ready", a0 + u * (q_ready - a0), 0.0)
        self.wait("ready_hold", READY_HOLD_S)
        # the turn and the walk to the stance (from the first candidate's grasp point)
        g0 = np.median(obj, 0)  # the stance from the target's cloud (its median point)
        walk = dict(turn_err_deg=None, walks=[])
        if TURN_DEG[task]:
            walk["turn_err_deg"] = round(self.turn(math.radians(TURN_DEG[task])), 2)
        self.yaw_ref = self.yaw()
        # the forward cap: scene points of the look in the legs' band (F0)
        keep = look.valid & ~look.own & ~ndimage.binary_dilation(mask, iterations=B.OBJ_DIL)
        Xs = look.P[keep]
        Xs = Xs[(Xs[:, 2] > LEG_BAND[0]) & (Xs[:, 2] < LEG_BAND[1])]
        for rnd in range(WALK_ROUNDS):
            for axis, stand in ((1, STAND_Y[task]), (0, STAND_X[task])):
                g_now = self.to_now(g0)
                err = float(g_now[axis]) - stand
                if axis == 0:
                    Xn = self.to_now(Xs) if len(Xs) else np.zeros((0, 3))
                    ahead = Xn[(np.abs(Xn[:, 1]) < LEG_HALF_W) & (Xn[:, 0] > 0)] if len(Xn) else Xn
                    cap = float(ahead[:, 0].min()) - FRONT_CLEAR if len(ahead) else np.inf
                    walk["front_cap_cm"] = round(100 * cap, 1) if np.isfinite(cap) else None
                    err = min(err, cap)
                if abs(err) <= B.WALK_MIN:
                    continue
                moved = self.walk(f"walk{rnd}{'yx'[1 - axis]}", abs(err), math.copysign(B.WALK_VX, err),
                                  abs(err) / B.WALK_VX + 2.0, axis)  # fmt: skip
                self.wait(f"settle{rnd}{'yx'[1 - axis]}", B.WALK_SETTLE_S)
                walk["walks"].append(["yx"[1 - axis], round(100 * err, 1), round(100 * math.copysign(moved, err), 1)])
        if HEIGHT[task] is not None:
            run.hold[2] = HEIGHT[task]
            self.h_cmd = float(HEIGHT[task])
        run.hold[4] = PITCH[task]
        if HEIGHT[task] is not None or PITCH[task]:
            self.wait("squat", SQUAT_S)
        walk.update(g_pelvis=B.r3(self.to_now(g0)), yaw_deg=round(math.degrees(self.yaw()), 2),
                    pelvis_z=round(float(run.pelvis()[0][2]), 3), waist_deg=B.r3(np.degrees(d.qpos[kin.waist_q]), 1))  # fmt: skip
        row["walk"] = walk
        row["prefix_hash"] = state_hash(d)
        row["prefix_steps"] = run.k
        if env.episode_fell() or run.done:
            return finish("fell", "fell before the reach")

        # the grasp at the stance: the first candidate whose pregrasp and grasp IK hold
        kin.load(d)
        q_meas, ctrl0 = d.qpos[kin.arm_q].copy(), run.ref.copy()
        pick = None
        tried = []
        for ci, (gi, T0, cinf) in enumerate(cand):
            g, Rg = self.to_now(T0[:3, 3]), self.R_now(T0[:3, :3])
            best, diag = None, []
            for R_ in (Rg, Rg @ np.diag([1.0, -1.0, -1.0])):
                pre = g - PRE_M * R_[:, 0]
                qp, ep, er = kin.ik(pre, R_, q_meas, 1.0, iters=300, tol_p=1e-4)
                qg, eg, erg = kin.ik(g, R_, qp, 1.0, iters=300, tol_p=1e-4)
                pen = max(kin.self_contact(qp), kin.self_contact(qg))
                okk = max(ep, eg) < IK_TOL and math.degrees(max(er, erg)) < IK_TOL_DEG and pen <= 1.0
                mot = float(np.abs(qp - q_meas).sum())
                diag.append([round(1000 * ep, 1), round(1000 * eg, 1), round(math.degrees(max(er, erg)), 1), round(pen, 1)])
                if okk and (best is None or mot < best[0]):
                    best = (mot, R_, qp, pre)
            tried.append([gi, None if best is None else round(math.degrees(best[0]), 1), diag, cinf])
            if best is not None:
                pick = (ci, gi, g, best[1], best[2], best[3])
                break
        row["ik_tried"] = tried
        if pick is None:
            return finish("reach", "no candidate with IK at the stance")
        ci, gi, g, Rh, q_pre, pre = pick
        row["grasp"] = dict(cand=ci, gg_index=int(gi), score=cand[ci][2]["score"], g_pelvis=B.r3(g),
                            feed=B.r3(Rh[:, 0]), close=B.r3(Rh[:, 1]))  # fmt: skip
        g_world = run.pelvis()[1] @ g + run.pelvis()[0]  # the target in the world, by the pelvis readings now
        Rt = d.xmat[self.tb].reshape(3, 3)
        row["truth"]["grasp_in_prop_cm"] = B.r3(100 * Rt.T @ (g_world - d.xpos[self.tb]), 2)  # scoring only
        # J's voxels (the look less the robot and the target), into the virtual pelvis frame
        pN, RN = run.pelvis()
        RF, tF = RN.T @ self.RF0, RN.T @ (self.pF0 - pN)
        tw, Rw = kin.torso_pose()
        tw0, Rw0 = kin.torso_pose(np.zeros(3))
        T_vp = (Rw @ Rw0.T, tw - (Rw @ Rw0.T) @ tw0)
        vox, n_obj = B.voxels(look, mask, obj, T_vp, (RF, tF))
        gp, gR = kin.site_pose(q_pre)  # the pregrasp pose aimed at (FK of the IK solution)
        n29, n7, names = self.m2_.out_of_band(run.film, run.k, jnt=kin.jnt)
        row["range"] = dict(n_out29=n29, n_out_arm7=n7, names=names)

        # reach 1: the pregrasp
        refs, pinfo = self.plan(q_meas, ctrl0, gp, gR, vox, T_vp)
        r1 = dict(plan=pinfo)
        if refs is None:
            if self.a.diag:  # design only: the same goal without obstacles
                r_, i_ = self.plan(q_meas, ctrl0, gp, gR, np.zeros((0, 3), np.float32), T_vp)
                r1["diag_novox"] = i_.get("status") if r_ is None else "ok"
            refs, linfo = B.line_refs(kin, *kin.site_pose(q_meas), gp, gR, ctrl0, q_meas)
            r1.update(fallback_line=True, line=linfo)
            dur = linfo["dur_s"]
        else:
            dur = pinfo["dur_s"]
        i_cmd = run.k
        pred, p_pel = self.pred(i_cmd, refs, ctrl0, dur)
        g1 = kin.site_pose(refs[-1])[0] - gp
        r1.update(p_cm=B.r3(100 * p_pel, 2), g_cm=B.r3(100 * g1, 2), z=pred["z"])
        shift1_applied = np.zeros(3)
        if group == "S" and r1.get("fallback_line"):
            r1["scale"] = None  # D has no plan: S is D (SF's rule: episodes without J's route run J's line)
        elif group == "S":
            shift = -(g1 + p_pel)
            r1["shift_cm"] = B.r3(100 * shift, 2)
            used = None
            for sc in SCALES:
                rs, pi_ = self.plan(q_meas, ctrl0, gp + sc * shift, gR, vox, T_vp)
                if rs is not None:
                    used = (sc, rs, pi_)
                    break
            if used is None:
                r1["scale"] = 0.0
            else:
                r1["scale"] = round(used[0], 3)
                refs, dur = used[1], used[2]["dur_s"]
                shift1_applied = used[0] * shift
                pred, p_pel = self.pred(i_cmd, refs, ctrl0, dur)  # the prediction on the command sent (reported)
                r1["p_sent_cm"] = B.r3(100 * p_pel, 2)
        r1["dur_s"] = dur
        r1["n"] = len(refs)
        r1["pred_land_pelvis"] = B.r3(kin.site_pose(refs[-1])[0] + p_pel)
        self.path("reach", refs)
        self.wait("settle1", SETTLE_S)
        r1["label_head_cm"] = B.r3(100 * self.m2_.label(run.film, i_cmd, len(refs), dur, jnt=kin.jnt), 2)
        r1["pred_err_cm"] = round(float(np.linalg.norm(np.array(r1["label_head_cm"]) - 100 * pred["p_head"])), 2)
        r1["hand_err_cm"] = round(100 * float(np.linalg.norm(run.hand() - gp)), 2)
        row["reach1"] = r1

        # reach 2: the line in
        q_end = refs[-1]
        p_pre, R_pre = kin.site_pose(q_end)
        kin.load(d)
        ctrl0 = run.ref.copy()
        r_in, i_in = B.line_refs(kin, p_pre, R_pre, g, R_pre, q_end, q_end)
        i_cmd2 = run.k
        pred2, p2 = self.pred(i_cmd2, r_in, ctrl0, i_in["dur_s"])
        g2 = kin.site_pose(r_in[-1])[0] - g
        r2 = dict(p_cm=B.r3(100 * p2, 2), g_cm=B.r3(100 * g2, 2), line=i_in, z=pred2["z"])
        n29b, n7b, namesb = self.m2_.out_of_band(run.film, i_cmd2, jnt=kin.jnt)
        row["range2"] = dict(n_out29=n29b, n_out_arm7=n7b, names=namesb, body_contacts=self.body_contacts(),
                             left_dev_deg=B.r3(np.degrees(d.qpos[self.left_q] - self.left0), 1))  # fmt: skip
        if group == "S" and np.any(shift1_applied):
            # the line in moved by reach 1's applied shift: M2's training reaches all start with the right arm at rest
            # (its 0.5-99.5 % band of every right-arm joint at the command is under 0.04 rad wide), so its prediction
            # for a reach from the pregrasp is out of its domain (recorded: p_cm, z, range2), not used
            r2["shift_cm"] = B.r3(100 * shift1_applied, 2)
            r2["scale"] = 0.0
            for sc in SCALES:
                rr, ii = B.line_refs(kin, p_pre, R_pre, g + sc * shift1_applied, R_pre, q_end, q_end)
                if ii["ik_end_mm"] <= 1000 * IK_TOL and ii["stuck_at"] is None:
                    r_in, i_in, r2["scale"] = rr, ii, round(sc, 3)
                    break
        r2["dur_s"] = i_in["dur_s"]
        self.path("in", r_in)
        self.wait("settle2", SETTLE_S)
        r2["label_head_cm"] = B.r3(100 * self.m2_.label(run.film, i_cmd2, len(r_in), i_in["dur_s"], jnt=kin.jnt), 2)
        r2["pred_err_cm"] = round(float(np.linalg.norm(np.array(r2["label_head_cm"]) - 100 * pred2["p_head"])), 2)
        row["reach2"] = r2
        # the hand at the close (simulator, scored): to the grasp target (world, by the pelvis readings)
        hand_w = d.site_xpos[kin.site].copy()
        e = hand_w - g_world
        row["hand_target_cm"] = round(100 * float(np.linalg.norm(e)), 2)
        row["hand_target_vec_grasp_cm"] = B.r3(100 * (run.pelvis()[1] @ Rh).T @ e, 2)  # feed, close, third
        o_pre = np.asarray(target.get_position(), float)
        row["obj_moved_before_close_mm"] = round(1000 * float(np.linalg.norm(o_pre - props0[ti])), 1)
        # close, lift, hold
        q_hold = run.ref.copy()
        hand_z0 = float(d.site_xpos[kin.site][2])
        for _ in range(int(round(CLOSE_S / DT))):
            run.step("close", q_hold, 1.0)
        held_close = bool(inner.robot.is_gripper_holding_object(target, HandSide.RIGHT))
        kin.load(d)
        p_h, R_h = kin.site_pose(q_hold)
        up = run.pelvis()[1].T @ np.array([0.0, 0.0, LIFT_M])
        r_up, i_up = B.line_refs(kin, p_h, R_h, p_h + up, R_h, q_hold, q_hold, rot_w=LIFT_ROT_W)
        for r in r_up:
            if run.step("lift", r, 1.0):
                break
        row["target_contacts_after_lift"] = self.target_contacts()
        rises = []
        for _ in range(int(round(HOLD_S / DT))):
            if run.step("hold", None, 1.0):
                break
            rises.append(float(np.asarray(target.get_position(), float)[2] - props0[ti][2]))
        n_ok = int(round(HOLD_OK_S / DT))
        row["hand_rise_cm"] = round(100 * (float(d.site_xpos[kin.site][2]) - hand_z0), 2)
        row["obj_rise_trace_cm"] = [round(100 * x, 1) for x in rises[::15]]
        held_end = bool(inner.robot.is_gripper_holding_object(target, HandSide.RIGHT))
        fell = bool(env.episode_fell())
        lifted = len(rises) >= n_ok and min(rises[-n_ok:]) >= RISE_MIN
        others = [j for j, pr in enumerate(props) if j != ti and float(np.asarray(pr.get_position())[2] - props0[j][2]) >= RISE_MIN]
        row.update(success=bool(lifted and held_end and not fell), held_after_close=held_close, held_end=held_end,
                   rise_end_cm=round(100 * rises[-1], 2) if rises else None,
                   rise_min_last_cm=round(100 * min(rises[-n_ok:]), 2) if len(rises) >= n_ok else None,
                   fell=fell, furniture=self.furn, other_props_lifted=others, steps=run.k,
                   lift_ik_end_mm=i_up["ik_end_mm"], within_budget=bool(run.k <= budget))  # fmt: skip
        if row["success"]:
            row["fail"] = None
        elif row["truth"].get("percept_err_cm", 0.0) > 100 * PERCEPT_BAD:
            row["fail"] = "perception"  # the look's target is not the prop (scored after the episode)
        elif fell:
            row["fail"] = "fell"
        elif any(v >= FURN_MM for v in self.furn.values()):
            row["fail"] = "furniture"
        elif not held_close:
            row["fail"] = "missed"
        elif held_close:
            row["fail"] = "dropped"
        else:
            row["fail"] = "other"
        row["wall_s"] = round(time.time() - t_wall, 2)
        return row


def cmd_run(a) -> None:
    for kv in a.set or []:  # design trials only
        k, v = kv.split("=", 1)
        if k not in DESIGN_KEYS:
            raise SystemExit(f"--set {k}: not a design key")
        cur, val = globals()[k], json.loads(v)
        if isinstance(cur, dict) and not isinstance(val, dict):
            val = {**cur, a.task: val}
        globals()[k] = val
    seeds = parse_seeds(a.seeds)
    check_seeds(seeds, a.allow_eval_seeds)
    if a.set and any(EVAL_LO <= s < EVAL_HI for s in seeds):
        raise SystemExit("--set is for the design seeds only")
    import m2 as M2

    env = make_env(a.task)
    kin = B.Kin(env)
    model = M2.M2(a.m2_weights, a.m2_band)
    ep = Ep(env, kin, a.task, None, a)
    ep.m2_ = model
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["seed"], r["group"]))
    for s in seeds:
        for grp in a.groups.split(","):
            if (s, grp) in done:
                continue
            try:
                row = ep.go(s, grp, a.dir)
            except Exception as e:
                import traceback

                row = dict(task=a.task, seed=s, group=grp, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-2500:])
            if a.set:
                row["set"] = a.set
            with open(a.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row.get(k) for k in ("task", "seed", "group", "success", "fail", "why", "hand_target_cm",
                                                      "rise_end_cm", "held_after_close", "error", "wall_s")}), flush=True)  # fmt: skip
    env.close()


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n)


def sign_test(d) -> float:
    d = [x for x in d if x != 0]
    n, k = len(d), sum(x < 0 for x in d)
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2.0**n)


FAIL_ORDER = ("perception", "reach", "fell", "furniture", "missed", "dropped", "other")


def cmd_analyze(a) -> None:
    rows = []
    for pat in a.rows:
        for fn in sorted(glob.glob(pat)):
            rows += [json.loads(x) for x in open(fn)]
    errs = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    print(f"rows {len(rows)}, error rows {len(errs)}")
    for r in errs[:5]:
        print("  error:", r["task"], r["seed"], r["group"], r["error"])
    allpairs = []
    for task in sorted({r["task"] for r in rows}):
        R = {(r["seed"], r["group"]): r for r in rows if r["task"] == task}
        seeds = sorted({s for s, _ in R})
        pairs = [(R[(s, "D")], R[(s, "S")]) for s in seeds if (s, "D") in R and (s, "S") in R]
        allpairs += pairs
        print(f"\n=== {task} === {len(pairs)} pairs (seeds {seeds[0]}-{seeds[-1]})")
        print(f"  look hash ok: {sum(d['look_hash_ok'] and s['look_hash_ok'] for d, s in pairs)}/{len(pairs)};"
              f" prefix (reset..stance) identical D vs S: {sum(d.get('prefix_hash') == s.get('prefix_hash') for d, s in pairs)}"
              f"/{len(pairs)}; within the official budget D {sum(bool(d.get('within_budget')) for d, _ in pairs)}"
              f" S {sum(bool(s.get('within_budget')) for _, s in pairs)}")  # fmt: skip
        report_pairs(pairs)
    if len({r["task"] for r in rows}) > 1:
        print("\n=== all tasks pooled ===")
        report_pairs(allpairs)


def report_pairs(pairs) -> None:
    kd, ks = sum(d["success"] for d, _ in pairs), sum(s["success"] for _, s in pairs)
    b = sum(d["success"] and not s["success"] for d, s in pairs)
    c = sum(s["success"] and not d["success"] for d, s in pairs)
    n = len(pairs)
    print(f"  lift success: D {kd}/{n} ({100 * kd / max(n, 1):.1f} %), S {ks}/{n} ({100 * ks / max(n, 1):.1f} %),"
          f" S - D {100 * (ks - kd) / max(n, 1):+.1f} pp; discordant D-only {b}, S-only {c}; McNemar exact p = {mcnemar_exact(b, c):.4f}")  # fmt: skip
    hp = [(d["hand_target_cm"], s["hand_target_cm"]) for d, s in pairs
          if d.get("hand_target_cm") is not None and s.get("hand_target_cm") is not None]  # fmt: skip
    if hp:
        dd = [y - x for x, y in hp]
        print(f"  hand to the grasp target at the close (cm), {len(hp)} pairs with both closes: D median"
              f" {np.median([x for x, _ in hp]):.2f}, S median {np.median([y for _, y in hp]):.2f}; paired S - D median"
              f" {np.median(dd):+.2f} (S nearer on {sum(x < 0 for x in dd)}, farther on {sum(x > 0 for x in dd)});"
              f" sign test p = {sign_test(dd):.4f}")  # fmt: skip
    for g, k in (("D", 0), ("S", 1)):
        fc = {}
        for p in pairs:
            f = p[k].get("fail") or "success"
            fc[f] = fc.get(f, 0) + 1
        print(f"  failure classes {g}: " + ", ".join(f"{x} {fc[x]}" for x in ("success",) + FAIL_ORDER if x in fc))
        whys = {}
        for p in pairs:
            if p[k].get("why"):
                whys[p[k]["why"]] = whys.get(p[k]["why"], 0) + 1
        if whys:
            print(f"    reasons: {whys}")
    rg = [d["range"] for d, _ in pairs if d.get("range")]
    if rg:
        o29 = np.array([x["n_out29"] for x in rg])
        o7 = np.array([x["n_out_arm7"] for x in rg])
        print(f"  start state vs M2's training band (29 joints at reach 1's command, D episodes, {len(rg)}): out-of-band"
              f" joints median {np.median(o29):.0f}, max {o29.max()}, episodes with any {int((o29 > 0).sum())}; right arm"
              f" (7) median {np.median(o7):.0f}, episodes with any {int((o7 > 0).sum())}")  # fmt: skip
        names = {}
        for x in rg:
            for nm in x["names"]:
                names[nm] = names.get(nm, 0) + 1
        print(f"    joints out (episodes): {dict(sorted(names.items(), key=lambda t: -t[1]))}")
        rg2 = [d["range2"] for d, _ in pairs if d.get("range2")]
        if rg2:
            print(f"  reach 2's command (from the pregrasp): out-of-band joints median"
                  f" {np.median([x['n_out29'] for x in rg2]):.0f}, right arm median {np.median([x['n_out_arm7'] for x in rg2]):.0f}")  # fmt: skip
    for g, k in (("D", 0), ("S", 1)):
        e1 = [p[k]["reach1"]["pred_err_cm"] for p in pairs if p[k].get("reach1")]
        e2 = [p[k]["reach2"]["pred_err_cm"] for p in pairs if p[k].get("reach2")]
        lab = [np.linalg.norm(p[k]["reach1"]["label_head_cm"]) for p in pairs if p[k].get("reach1")]
        if e1:
            print(f"  M2 prediction error, {g} episodes (|predicted - actual settled palm offset|, cm): reach 1 median"
                  f" {np.median(e1):.2f} p90 {np.percentile(e1, 90):.2f} (n {len(e1)}; the actual offset's own size"
                  f" median {np.median(lab):.2f}); reach 2 median {np.median(e2):.2f} p90 {np.percentile(e2, 90):.2f}")  # fmt: skip
    sc1 = [p[1]["reach1"].get("scale") for p in pairs if p[1].get("reach1")]
    sc2 = [p[1]["reach2"].get("scale") for p in pairs if p[1].get("reach2")]
    if sc1:
        cnt = lambda v: {str(x): v.count(x) for x in sorted(set(v), key=str)}  # noqa: E731
        print(f"  S's scales: reach 1 {cnt(sc1)} (None: D had no plan, S = D), reach 2 {cnt(sc2)}")
        pk = [(p[0].get("reach1", {}).get("fallback_line", False), p[1].get("reach1", {}).get("fallback_line", False))
              for p in pairs if p[0].get("reach1") and p[1].get("reach1")]  # fmt: skip
        print(f"  D's plan failed (line): in D episodes {sum(x for x, _ in pk)}, in S episodes {sum(y for _, y in pk)}"
              f" of {len(pk)} (cuRobo is not run-to-run deterministic)")  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("look")
    lk.add_argument("--task", required=True, choices=list(TASKS))
    lk.add_argument("--seeds", required=True)
    lk.add_argument("--dir", required=True)
    lk.add_argument("--allow-eval-seeds", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--task", required=True, choices=list(TASKS))
    r.add_argument("--seeds", required=True)
    r.add_argument("--groups", default="D,S")
    r.add_argument("--dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--plan-socket", default=str(OUT / "plan1.sock"))
    r.add_argument("--plan-timeout", type=float, default=10.0)
    r.add_argument("--m2-weights", default=str(OUT / "m2/m2_weights.npz"))
    r.add_argument("--m2-band", default=str(OUT / "m2/band.npz"))
    r.add_argument("--set", action="append")
    r.add_argument("--diag", action="store_true", help="design only: on a failed plan, plan again without obstacles")
    r.add_argument("--allow-eval-seeds", action="store_true")
    z = sub.add_parser("analyze")
    z.add_argument("rows", nargs="+")
    a = ap.parse_args()
    if a.cmd == "look":
        cmd_look(a)
    elif a.cmd == "run":
        cmd_run(a)
    else:
        cmd_analyze(a)


if __name__ == "__main__":
    main()
