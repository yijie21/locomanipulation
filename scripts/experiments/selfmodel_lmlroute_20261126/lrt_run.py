"""selfmodel_lmlroute_20261126: lmlcmp3's frozen lc3_run.py (md5 eca94321..., = lmlcmp3's code/) with two groups more,
marked "lmlroute:"; nothing else changed (alone, selfmodel, guard, reachable, entryinv run exactly lmlcmp3's code; the
new groups' episodes that take no route are compared with lmlcmp3's A / R records one by one).

(l) `route` (C, id 5) and `route_reachable` (CR, id 6). At the first reach command (after the walk, its correction walk
    and the settle wait; the same step on which A sends its first command), before anything else of that command:
    1. A's straight reference at this command is computed by our IK (plan_rec's placement, the planner's reach): to the
       via point when the plan has one, then from that reference's end to the pre-grasp (straight_chain);
    2. the request (the measured start, that chain, the look's furniture / object points mapped through odometry as
       command_b + plan_rec place a target) goes to lrt_plan.py (cuRobo, SIMPLE's venv, one GPU) through a file queue
       under $LRT_ROUTES (default $LC_OUT/routes); the answer is cached per (controller, task, episode): C and CR (one
       process runs an episode's C, then its CR) and every rerun read the same file; its md5 and the request's md5 are
       recorded (lmlroute.route);
    3. lrt_plan's gate: A's reference arm (cuRobo's spheres) on the way (hand > 10 cm from the pre-grasp) comes within
       5 cm of the look's voxels -> plan a route; no route (gate clear, a failed plan, no inner via point, a route IK
       failure here) -> the episode goes on exactly as A (C) / R (CR);
    4. a route: the route command takes the first slot (k = -1, reach_no 0, direct, no model): one WholeBodyIK through
       the route's inner via points (hand point and wrist attitude; Planner.reach's settings, 3 cm steps, 0.15 m/s per
       leg, at least 0.6 s in all), placed as plan_rec places a reach (the measured pelvis, the reference shift); it
       ends at the last via point. Then the pre-grasp (k = 0: C the goal, CR R's shrink-back fixed point), the entry
       (k = 1, likewise), the close and the lift, scheduled and gated as after a via point.
    The contact window (n_contacts, first_contact) also runs from the route command on (A's window starts at the first
    command's scheduled step; a routed episode moves that step to the pre-grasp). Every furniture contact row in the
    window is recorded with the hand point (lmlroute.contacts: truth, scoring only). The route segment's records carry
    n_via / via_p / via_r6 / via_gen (heading frame at the command, d3_collect.augment_b's form) and via_route 2.
(m) Shards by episode (an episode's groups in one process, in --methods order).
(n) The looks (revision 1, the user's decision of 2026-11-26): LRT_LOOKS set -> every episode's look from there (the
    round's looks with the simulator's mask, hg_look HG_MASK=gt); the episode record's look block then also states
    mask_source, gt_mask_px, gt_mask_pts and the look file. A and R run here too (no reuse of lmlcmp3's records).
(o) The formal run's queue (revision 1 addendum, the coordinator before the formal run): `work` claims episodes on vePFS
    by mkdir ($LC_OUT/claims/<backend>/<task>/ep+0030/, with claim.json: host, pid, time; done.json when every group's
    record is written), one episode = all its groups (A R C CR) in one process on one machine, so that 4090 and H20
    workers can share the queue and every pair is run on one machine; a worker goes through --tasks in order, builds a
    task's driver only when it got an episode of it, and ends when it gets none. `release`: this machine's claims whose
    process is gone without done.json lose the episode's partial records and the claim (the episode runs again, whole).
    Every record states the machine (host). `run` (by shard) is unchanged apart from the host field.
Output: $LC_OUT (default loco_outputs/selfmodel_lmlroute_20261126). lmlcmp3's docstring follows.

selfmodel_lmlcmp3_20261126: lmlcmp2's frozen lc2_run.py (7f3d1e1c) with three changes, marked "lmlcmp3:"; nothing
else changed (alone, selfmodel and reachable run exactly lmlcmp2's code: on lmlcmp2's design episodes they are compared
with lmlcmp2's records one by one).

(h) `entryinv` (E): the pre-grasp command is the goal itself (as alone), the entry's comes from selfmodel's fixed point
    (6 steps, the same choice rule; no shrink, no guard). The via point stays direct, as in every group.
(i) Every reach (via point, pre-grasp, entry) records the end of its reference: the hand point (HAND_OFF ahead of the
    right wrist_yaw_link) at the plan's last IK waypoint -- FK of rec's ref_pel / ref_q on the planner's model, in an own
    MjData (the planner's is never touched) -- as `ref_end_w` (the plan's measured world, the frame of target_w: its
    distance to target_w is ik_err) and `ref_end_h` (the command's heading frame, the frame of cmd_h and act_h).
(j) The new seeds' looks: episodes 30..119 read their cached look from LOOK3 (lmlcmp3's own looks/, made by hardgrasp's
    hg_run.py look phase with HG_OUT = lmlcmp3's hg/); every other episode reads hardgrasp's looks/ as before.
(k) The run's code check (added before the formal run, revision 1 addendum; the episodes are untouched): with
    LC_CODE_ROOT set (lc3_formal.sh: the round's private copy, loco_outputs/selfmodel_lmlcmp3_20261126/code), every run
    process prints the env.py it imported and stops unless env.py and every loaded module that comes from the shared
    repo tree or the copy lie in the copy -- once the driver and the models are up, and again after every episode
    (imports made inside an episode). Without LC_CODE_ROOT (dev / smoke) nothing is checked.
Output: $LC_OUT (default loco_outputs/selfmodel_lmlcmp3_20261126).

lmlcmp2's docstring follows.

selfmodel_lmlcmp2_20261126: lmlcmp's frozen lc_run.py (9810df23) with two more groups, marked "lmlcmp2:"; nothing
else changed (alone and selfmodel run exactly lmlcmp's code: their episodes are compared with lmlcmp's one by one).

(f) `guard`: selfmodel's fixed point (6 steps, the same candidate chosen), then two guards on the chosen command t:
    (1) |t - goal| > CAP (10 cm) -> t moved back along t - goal to |t - goal| = 10 cm (planned and predicted again);
    (2) D's predicted landing for the command after (1) still > FALLBACK (3 cm) from the goal -> the direct command (the
    goal itself, candidate 0). Recorded per reach in `guard`.
(g) `reachable`: selfmodel's fixed point, but every candidate after the first (the first is the goal) is checked
    before it is evaluated: its plan's IK error (the last waypoint's hand error, as ik_err_cm) > SHRINK_IK (2 cm) -> the
    candidate moved back along t - goal to the largest fraction a of the correction with IK error <= 2 cm (bisection on
    a in [0, 1], SHRINK_STEPS halvings; a = 0, the goal, when no fraction passes); the next step starts from the
    shrunk candidate. The choice (least |D(t) - goal|^2 + LAMBDA |spread|^2) is selfmodel's. Recorded in `shrink`.
Output: $LC_OUT (default loco_outputs/selfmodel_lmlcmp2_20261126). HEFT's gait copies: lmlcmp's heft_gait/.

lmlcmp's docstring follows.

selfmodel_lmlcmp_20261126: LocoManiLab's 7 real tasks (hardgrasp / lmlreal), direct execution (`alone`) against
execution through the self model (`selfmodel`), ScaleBFM, HEFT and SONIC, the robot settled before every command.
2026-11-26.

What runs is lmlreal's lr_run.episode (frozen, b783eaef; its drivers, plan_rec, delay alignment and d3 records are
imported unchanged) copied here with the registration's changes, marked "lmlcmp:":

(a) Settle before every command (the robot's own rule, both groups). A command waits from the time lmlreal's schedule
    gives it until, on each of the last 0.5 s of steps, both hand points (13 cm ahead of wrist_yaw_link) in the pelvis
    frame -- FK of the joint encoders on robot_fk with the pelvis at the origin -- move < 2 cm/s and every one of the 12
    leg joints < 0.05 rad/s (encoder position differences); speeds are displacements over the last 0.1 s (data3's
    windowing), evaluated every step; at most 3 s, then the command goes anyway, recorded as not settled. The commands:
    the first reach after the look / walk (its correction-walk decision is made after the same wait), the next reach
    (pre-grasp after a via point, entry after the pre-grasp), the close and the lift. The walk that follows the look is
    not gated (it starts at the cached look's fixed time, as data3's walks start at a fixed time). The schedule after a
    wait is lmlreal's, moved by the wait; the episode's time limit is unchanged. data3's truth rule (simulator pelvis
    and both hand points < 2 cm/s for 0.5 s, d3_collect.Rig.quiet) is only measured: at every command row, and on every
    step of every wait (both rules' agreement).
(b) The landing (each reach's end label): data3's rule -- the first 10 Hz row (from the command) at least 0.5 s after
    the reference end + ref_delay_f where the truth rule holds, searched up to 3 s after that opening (then that row,
    settled_end 0), or up to the next command / the close if it comes first (the last 10 Hz row before it,
    settled_end 0). The label only records; it never times anything.
(c) HEFT walks on gait_heft.HEFT (src/locomani/backends/heft/gait_heft.py, not in this repo copy: byte-identical copies
    of gait.py and gait_heft.py in heft_gait/, imported as locomani.backends.heft.*). gait_heft holds the start's upper
    body; the approach tasks start leaned back for the look, so the upper body (waist and arms) is blended to DEFAULT_Q
    over the first BLEND_S exactly as hg_walk.walk_frames does (legs and pelvis are gait_heft's). ScaleBFM and SONIC walk
    on hg_walk.walk_frames, unchanged.
(d) Every reach (via point, pre-grasp, entry) records the predictions of three end models of its controller for the
    command it executes: T3 (train3: data3, frac 1.0, seed 0; chain input clipped to 1), M (train3b: data3 + data3b,
    seed 0), D (train3b: M's data + the mid-point samples, seed 0). Features: d3_collect.segments on the live rows up to
    the command row (rows after it are placeholders; they only feed the labels) + data.py's derive, exactly what the
    recording of this reach gives (lc_check compares).
(e) `selfmodel`: the pre-grasp and entry commands come from D -- hardgrasp's fixed point (lml_tasks ITERS 6 steps:
    t <- clip_t(t + goal - D(t)), the candidate with the least |D(t) - goal|^2 + LAMBDA |D's spread|^2), each candidate
    planned from the measured state as the drive will. The via point and the lift are direct in both groups. `alone`:
    the goal itself (lml_tasks `alone`).

Everything else is lmlreal's: hardgrasp's 7 formal tasks, our jitter, the episode ids, the cached look (hardgrasp's
looks/, ScaleBFM's, for every controller), the stance and walk rules, leg odometry, the lost / fall rules, scoring
(simulator truth only after the episode starts closing, hardgrasp's lift judgement), the d3 records (+ the fields
below). Simulator truth decides nothing: not the settle waits, not the stance, not a command (the plan is anchored on
the measured pelvis as in hardgrasp, shared by both groups).

Records (per backend): episodes/<task>/<method>/ep*.json; segs/<task>/<method>/ep*.npz (+ arm/) in data3's layout with,
per segment, lmlreal's added scalars and: method_id (0 alone, 1 selfmodel), prop_settled (the own rule held when the
command went), prop_v_hand_cms / prop_v_leg_rads (the own rule's largest speeds over the 0.5 s before the command),
truth_first_s (the wait's first step the truth rule held, s after the wait opened; -1 none), label_kind (0 settled, 1 3 s
timeout, 2 cut by the next command), win_rows (50 Hz rows from the command to the label: furn_rows / win_rows is the
contact share), d_prev_x/y, d_this_x/y (the reference shifts, for C[k0]), pre_wait_s = the wait (s).

    cd $G; PYTHONPATH=$G:$W/src python lc_run.py run --backend scalebfm --task T --eps -1:30 --methods alone,selfmodel \
        --shard K --shards N
    python lc_run.py merge --backend scalebfm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import types
from collections import deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for _k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):  # train3 / train3b's flags (lml_tasks.reachwm sets the same)
    os.environ.setdefault(_k, "1")
    assert os.environ[_k] == "1", f"{_k} must be 1"
for _k in ("REACHWM_X0_REF", "REACHWM_FDVEL"):
    assert os.environ.get(_k, "0") == "0", f"{_k} must be off"
sys.path[:0] = [str(EXP / "selfmodel_lmlreal_20261125")]
import lr_run as LR  # noqa: E402  (lmlreal's frozen runner: drivers, plan_rec, constants)

D3, HG, L, LT = LR.D3, LR.HG, LR.L, LR.LT

ROOT = Path("/wuji-vefps-D/wuji-il/yijie/loco_outputs")
OUT = Path(os.environ.get("LC_OUT", str(ROOT / "selfmodel_lmlroute_20261126")))  # lmlroute
LOOK3 = ROOT / "selfmodel_lmlcmp3_20261126" / "hg" / "looks"  # lmlcmp3 (j): the new seeds' looks (fixed, whatever LC_OUT)
NEW_EPS = range(30, 120)  # lmlcmp3 (j)
LMLREAL = ROOT / "selfmodel_lmlreal_20261125"
TASKS, BACKENDS = LR.TASKS, LR.BACKENDS
METHODS = ("alone", "selfmodel", "guard", "reachable", "entryinv", "route", "route_reachable")  # lmlcmp2: alone / selfmodel keep lmlcmp's ids (0, 1); lmlcmp3: + E (4); lmlroute: + C (5), CR (6)
INVERT = ("selfmodel", "guard", "reachable", "route_reachable")  # lmlcmp2: the groups whose pre-grasp / entry go through D (lmlroute: + CR)
SHRINK = ("reachable", "route_reachable")  # lmlroute: R's shrink-back (lmlcmp2 (g)), also CR's
ROUTE = ("route", "route_reachable")  # lmlroute (l): the groups that may take a route
ROUTES = Path(os.environ.get("LRT_ROUTES", str(OUT / "routes")))  # lmlroute (l): req/ (the queue), routes/<be>/<task>/ (the answers)
ROUTE_WAIT_S, ROUTE_REQUEUE_S = 3600.0, 300.0  # lmlroute: give up (episode error) / re-queue a request no planner holds
ENTRY_ONLY = ("entryinv",)  # lmlcmp3 (h): the entry only (k = 1) goes through D
CAP, FALLBACK = 0.10, 0.03  # lmlcmp2 (f): m
SHRINK_IK, SHRINK_STEPS = 0.02, 6  # lmlcmp2 (g): m, bisection halvings
LMLCMP_DIR = EXP / "selfmodel_lmlcmp_20261126"  # lmlcmp2: heft_gait/ copies live there
MODEL_NAMES = ("T3", "M", "D")
CHAIN_COL = 205  # data.py x0: S[k0], S[k0-1], S[k0-2], jaw0, rot0, walked, chain, ...
T3_CHAIN_MAX = 1

CODE_ROOT = os.environ.get("LC_CODE_ROOT")  # lmlcmp3 (k): the private copy the formal run must import from
SHARED_TREE = Path("/wuji-vefps-D/wuji-il/yijie/ws/code/loco")  # lmlcmp3 (k): the repo tree other rounds sync into


def code_check(tag: str) -> None:
    """lmlcmp3 (k): env.py and every loaded module from the shared tree or the copy must lie in CODE_ROOT, else stop."""
    if not CODE_ROOT:
        return
    root, shared = Path(CODE_ROOT).resolve(), SHARED_TREE.resolve()
    inside = lambda p: p == root or root in p.parents  # noqa: E731
    env = sys.modules.get("env")
    env_f = Path(env.__file__).resolve() if env is not None and getattr(env, "__file__", None) else None
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if shared in p.parents and not inside(p):  # from the shared tree (directly or through a link in the copy)
            bad.append(f"{name} {p}")
    print(f"lc code check {tag}: env.py {env_f}; modules outside {root}: {len(bad)}", flush=True)
    if env_f is None or not inside(env_f) or bad:
        raise SystemExit(f"lc code check {tag}: FAIL -- env.py {env_f}, outside the copy: {bad[:8]}")


def model_roots(backend: str) -> dict:
    return {"T3": ROOT / "selfmodel_data3_20261125" / backend / "models_s0",
            "M": ROOT / "selfmodel_train3b_20261125" / backend / "models_M_s0",
            "D": ROOT / "selfmodel_train3b_20261125" / backend / "models_D_s0"}  # fmt: skip


FPS, TICKS, DT_S = L.FPS, D3.TICKS, L.DT
SETTLE_F, SETTLE_MAX_F = D3.SETTLE_F, D3.SETTLE_MAX_F  # 0.5 s, 3 s
HAND_V, LEG_V = 0.02, 0.05  # m/s, rad/s (the registration's own rule)
EXTRA_HOLD_S = 6.5  # every reference hold lengthened by two waits + 0.5 s: the motion never runs out while waiting
RESET0 = 9_200_000
EXTRA = LR.EXTRA + ("method_id", "prop_settled", "prop_v_hand_cms", "prop_v_leg_rads", "truth_first_s", "label_kind",
                    "win_rows", "d_prev_x", "d_prev_y", "d_this_x", "d_this_y")  # fmt: skip
SCALARS_LC = D3.SCALARS + EXTRA


def reset_id(task: str, method: str, ep: int) -> int:
    return RESET0 + 100_000 * METHODS.index(method) + 10_000 * TASKS.index(task) + (ep + 1)


# -- (c) HEFT's gait --------------------------------------------------------------------------------------------------
def heft_gait():
    """locomani.backends.heft.{gait, gait_heft} from the byte-identical copies in heft_gait/."""
    name = "locomani.backends.heft"
    if name not in sys.modules:
        import locomani.backends as B

        pkg = types.ModuleType(name)
        pkg.__path__ = [str(LMLCMP_DIR / "heft_gait")]  # lmlcmp2: lmlcmp's copies
        pkg.__package__ = name
        sys.modules[name] = pkg
        B.heft = pkg
    from locomani.backends.heft import gait as G
    from locomani.backends.heft import gait_heft as GH

    return G, GH


def walk_ctrl(dr, backend: str, x: dict, ref_frame: np.ndarray, goal_xy, goal_yaw: float, pose, info: dict | None = None):
    import hg_walk as HW

    if backend != "heft":
        return HW.walk_frames(dr, x, ref_frame, goal_xy, goal_yaw, pose=pose, info=info)
    G, GH = heft_gait()
    if not hasattr(dr, "_heft_legs"):
        dr._heft_legs = G.LegModel(dr.pl.ik)
    wf = GH.walk_frames(dr._heft_legs, ref_frame, pose, goal_xy, goal_yaw, GH.HEFT, info)
    q0 = np.asarray(ref_frame[7:], float)
    t = (np.arange(len(wf)) + 1) / FPS
    bl = np.array([L.minjerk(min(1.0, v / HW.BLEND_S)) for v in t])
    wf[:, 7 + 12 :] = (q0[12:][None] + bl[:, None] * (L.DEFAULT_Q[12:] - q0[12:])[None]).astype(np.float32)
    if info is not None:
        info["gait"] = "gait_heft.HEFT"
    return wf


# -- (a) the robot's own settle rule ----------------------------------------------------------------------------------
class PropSettle:
    """Both hand points in the pelvis frame (robot_fk FK of the encoders, pelvis at the origin; own MjData) and the 12 leg
    joints (encoders); speeds over the last 0.1 s; settled when every one of the last SETTLE_F steps is below both limits."""

    def __init__(self, dr):
        pl = dr.pl
        self.mj, self.m = pl.mj, pl.ik.model
        self.d = self.mj.MjData(self.m)
        self.base, self.qadr, self.wrist = int(pl.ik.base_qadr), np.asarray(pl.ik.qadr), pl.wrist
        self.to_mj = dr.to_mj
        self.hist: deque = deque(maxlen=TICKS + 1)
        self.ok: deque = deque(maxlen=SETTLE_F)
        self.vv: deque = deque(maxlen=SETTLE_F)

    def clear(self) -> None:
        self.hist.clear()
        self.ok.clear()
        self.vv.clear()

    def hands(self, q_mj: np.ndarray) -> np.ndarray:
        d = self.d
        d.qpos[:] = 0
        d.qpos[self.base + 3] = 1.0
        d.qpos[self.qadr] = q_mj
        self.mj.mj_kinematics(self.m, d)
        return np.stack([d.xpos[b] + d.xmat[b].reshape(3, 3) @ L.HAND_OFF for b in (self.wrist["right"], self.wrist["left"])])

    def update(self, x: dict) -> None:
        q_mj = np.asarray(x["q"], float)[self.to_mj]
        h, legs = self.hands(q_mj), q_mj[:12].copy()
        self.hist.append((h, legs))
        if len(self.hist) > TICKS:
            h0, l0 = self.hist[0]
            vh = float(np.linalg.norm(h - h0, axis=1).max() / (TICKS * DT_S))
            vl = float(np.abs(legs - l0).max() / (TICKS * DT_S))
        else:
            vh = vl = math.inf
        self.vv.append((vh, vl))
        self.ok.append(vh < HAND_V and vl < LEG_V)

    def quiet(self) -> bool:
        return len(self.ok) == SETTLE_F and all(self.ok)

    def vmax(self) -> tuple[float, float]:
        v = np.array(self.vv) if self.vv else np.full((1, 2), np.inf)
        return float(v[:, 0].max()), float(v[:, 1].max())


# -- (d), (e) the three end models ------------------------------------------------------------------------------------
class Models:
    """T3, M, D of one controller (train.load_model; the normalisation is part of each model's state, so the dataset
    handed to the rebuild is only its template: lmlreal's records of the controller -- lc_check verifies)."""

    def __init__(self, backend: str):
        import torch

        DT, M = LT.reachwm()
        import train as TR

        torch.set_num_threads(1)
        self.torch, self.DT, self.M = torch, DT, M
        base = DT.load(str(LMLREAL / backend / "data" / "all.npz"))
        self.m = {n: TR.load_model("E3_ensemble", 1.0, base, "cpu", root=r) for n, r in model_roots(backend).items()}

    def features(self, dr, rig, rows: dict, f: int, frames: np.ndarray, c: dict, chain: int, walked: bool,
                 d_prev: np.ndarray, wrec: dict, delay: int):  # fmt: skip
        """(plan, z): command c planned at row f (plan_rec) and its one-segment dataset (d3_collect.segments + derive)."""
        blend, last, rec = dr.plan(rows[f], frames[f], c)  # LinAlgError / ValueError here: the plan failed (hg_run's)
        try:
            return (blend, last, rec), self._z(dr, rig, rows, f, frames, blend, rec, chain, walked, d_prev, wrec, delay)
        except Exception as e:  # anything after the plan is a fault of this code, never a "plan failed"
            raise RuntimeError(f"features: {type(e).__name__}: {e}") from e

    def _z(self, dr, rig, rows, f, frames, blend, rec, chain, walked, d_prev, wrec, delay) -> dict:
        rc = reach_rc(rec, f, f + len(blend), chain, d_prev, slot=0)
        rc.update(f_label=f + TICKS, settled_end=False, settle_s=-1.0)
        cand = np.concatenate([frames[: f + 1], blend])
        fake = {i: rows[i] for i in range(max(min(rows), f - 30), f + 1)}
        fake.update({i: rows[f] for i in range(f + 1, f + TICKS + 1)})
        info3 = dict(fell=None, lost=None, walk=wrec, walked=walked, delay_f=delay)
        segs = D3.segments(fake, info3, cand, [rc], 0, dr.pl, rig)
        assert len(segs) == 1, "features: the placeholder segment was not built"
        z, _ = D3.pack(segs)
        z["walk"] = np.zeros((1, 3), np.float32)
        z["target_b"] = z["target"]
        self.DT.derive(z)
        return z

    def predict(self, z: dict, names=MODEL_NAMES) -> dict:
        torch = self.torch
        GD = self.M.tensors(z, "cpu")
        assert float(GD["x0"][0, CHAIN_COL]) == float(z["chain"][0]), "x0's chain column moved"
        out = {}
        for n in names:
            G = GD
            if n == "T3":
                G = dict(GD)
                G["x0"] = GD["x0"].clone()
                G["x0"][:, CHAIN_COL] = G["x0"][:, CHAIN_COL].clamp(max=T3_CHAIN_MAX)
            with torch.no_grad():
                o = self.m[n].predict(self.M.sub(G, torch.tensor([0]), False))
            out[n] = (o["pos"][0].numpy().astype(np.float64), o["pos_std"][0].numpy().astype(np.float64))
        return out


def reach_rc(rec: dict, f: int, f_end: int, chain: int, d_prev: np.ndarray, slot: int) -> dict:
    """lr_run's reach in d3_collect's record form (one hand, roll 0)."""
    return dict(mode="right", pelvis_z=rec["pelvis_z"], reach_s=rec["reach_s"], ref_q=np.asarray(rec["ref_q"]),
                ref_pel=np.asarray(rec["ref_pel"]), delta=np.asarray(rec["delta"]), RT=rec["RT"], RQ=rec["RQ"], RP=rec["RP"],
                hands={"right": dict(target_w=np.asarray(rec["target_w"]), rot_w=np.asarray(rec["rot_w"]), dyaw=rec["dyaw"],
                                     pitch=rec["pitch"], roll=0.0, core=False, cmd_target=np.asarray(rec["cmd_target"]),
                                     ik_err=rec["ik_err"])},
                slot=slot, chain=chain, f_cmd=f, f_end=f_end, pre_wait_s=-1.0, d_prev=np.asarray(d_prev, float).copy(),
                f_label=None, settled_end=False, settle_s=-1.0)  # fmt: skip


# -- lmlcmp2: the two new groups' rules ---------------------------------------------------------------------------------
def shrink(dr, x: dict, ref: np.ndarray, c: dict, gh: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, dict]:
    """(g) candidate t's plan IK error > SHRINK_IK -> back along t - gh to the largest fraction a (bisection, SHRINK_STEPS
    halvings) whose IK error <= SHRINK_IK; a = 0 (the goal) when none passes. A plan that fails counts as not passing."""

    def ik(tt: np.ndarray) -> float:
        try:
            return float(dr.plan(x, ref, dict(c, target=tt.copy()))[2]["ik_err"])
        except (np.linalg.LinAlgError, ValueError):
            return float("inf")

    e1 = ik(t)
    sh = dict(ik0_cm=100 * e1 if np.isfinite(e1) else None, corr0_cm=100 * float(np.linalg.norm(t - gh)), alpha=1.0, shrunk=False, n_plans=1)
    if e1 <= SHRINK_IK:
        return t, sh
    lo, hi, n = 0.0, 1.0, 1
    for _ in range(SHRINK_STEPS):
        mid = 0.5 * (lo + hi)
        n += 1
        if ik(gh + mid * (t - gh)) <= SHRINK_IK:
            lo = mid
        else:
            hi = mid
    t2 = gh + lo * (t - gh)
    sh.update(alpha=lo, shrunk=True, n_plans=n, corr_cm=100 * float(np.linalg.norm(t2 - gh)))
    return t2, sh


def guard(best: dict, evs: list[dict], gh: np.ndarray, c: dict, feat, MD) -> tuple[dict, dict]:
    """(f) on selfmodel's chosen candidate: (1) cap |t - gh| at CAP along t - gh (planned and predicted again); (2) D's
    predicted miss of the command after (1) > FALLBACK -> the direct command (candidate 0, t = gh)."""
    corr = np.asarray(best["t"]) - gh
    n = float(np.linalg.norm(corr))
    gd = dict(s_corr_cm=100 * n, s_pred_miss_cm=100 * float(np.linalg.norm(best["pos"] - gh)), capped=False, fallback=False,
              already_direct=bool(n == 0.0))  # fmt: skip
    cand = best
    if n > CAP:
        t2 = gh + corr * (CAP / n)
        gd.update(capped=True, cap_t=t2.tolist())
        try:
            pl_, zz = feat(dict(c, target=t2.copy()))
            pos, std = MD.predict(zz, ("D",))["D"]
            cand = dict(t=t2, pos=pos, std=std, l=float(((pos - gh) ** 2).sum() + LT.LAMBDA * (std**2).sum()), plan=pl_, z=zz)
            gd["cap_pred_miss_cm"] = 100 * float(np.linalg.norm(pos - gh))
        except (np.linalg.LinAlgError, ValueError) as e:
            gd["cap_plan_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            cand = None
    miss = float(np.linalg.norm(cand["pos"] - gh)) if cand is not None else float("inf")
    gd["pred_miss_cm"] = 100 * miss if np.isfinite(miss) else None
    if miss > FALLBACK:
        gd["would_fall_back"] = True
        if cand is None or float(np.linalg.norm(np.asarray(cand["t"]) - gh)) > 0.0:
            gd["fallback"] = True
        cand = evs[0]  # the goal itself; planned (a plan failure ends the loop, so a planned best means a planned evs[0])
    gd["final_corr_cm"] = 100 * float(np.linalg.norm(np.asarray(cand["t"]) - gh))
    return cand, gd


# -- lmlcmp3: the reference's end and the new seeds' looks -------------------------------------------------------------
_REF_D: dict = {}


def ref_end(dr, rec: dict, x: dict) -> dict:
    """(i) The hand point at the reference's last IK waypoint: Planner.fk_world's FK (right wrist_yaw_link + HAND_OFF) of
    rec's ref_pel (pelvis pos + quat, measured world) and ref_q (MuJoCo order) in an own MjData of the planner's model;
    also in the command's heading frame (plan_rec's frame of the target: the measured root and its yaw at the command)."""
    pl = dr.pl
    d = _REF_D.get(id(pl))
    if d is None:
        d = _REF_D[id(pl)] = pl.mj.MjData(pl.ik.model)
    rp = np.asarray(rec["ref_pel"], float)
    b0 = int(pl.ik.base_qadr)
    d.qpos[:] = 0
    d.qpos[b0 : b0 + 3] = rp[:3]
    d.qpos[b0 + 3 : b0 + 7] = rp[3:7]
    d.qpos[pl.ik.qadr] = np.asarray(rec["ref_q"], float)
    pl.mj.mj_kinematics(pl.ik.model, d)
    b = pl.wrist["right"]
    hw = d.xpos[b] + d.xmat[b].reshape(3, 3) @ L.HAND_OFF
    p = x["root"][:3].astype(float)
    Rh = L.rot_z(L.yaw_of(L._qmat(x["root"][3:].astype(float))))
    hh = np.r_[(Rh.T @ np.r_[hw[:2] - p[:2], 0.0])[:2], hw[2]]
    return dict(ref_end_w=hw.tolist(), ref_end_h=hh.tolist())


LRT_LOOKS = os.environ.get("LRT_LOOKS")  # lmlroute (n): the round's own looks (simulator masks), every episode


def look_path(task: str, ep: int) -> Path:
    """(j) Episodes 30..119: lmlcmp3's looks; the rest: hardgrasp's (HG.look_path). lmlroute (n): LRT_LOOKS set -> that
    directory for every episode."""
    if LRT_LOOKS:
        return Path(LRT_LOOKS) / task / f"ep{ep:+05d}"
    return LOOK3 / task / f"ep{ep:+05d}" if ep in NEW_EPS else HG.look_path(task, ep)


# -- lmlroute (l): the route ----------------------------------------------------------------------------------------------
REQ_KEYS = ("start_P", "start_Q", "start_q", "start_hand", "chain_RP", "chain_RQ", "chain_hand", "furn", "obj")  # lrt_plan's


def req_md5(z: dict) -> str:
    """lrt_plan.req_md5: md5 over the request's arrays (fixed key order, float64, shape included)."""
    import hashlib

    h = hashlib.md5()
    for k in REQ_KEYS:
        a = np.ascontiguousarray(np.asarray(z[k], np.float64))
        h.update(k.encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def straight_chain(dr, x: dict, plan: dict, pose) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A's straight reference at this command, our IK (the planner's reach on plan_rec's placement: the measured pelvis,
    the targets from command_b on the odometry pose): to the via point when the plan has one, then from that reference's
    end to the pre-grasp. Returns RP (pelvis pos + quat wxyz), RQ (MuJoCo order), the right hand point per waypoint."""
    pl = dr.pl
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    Rh = L.rot_z(yaw)
    cur = (p, quat, x["q"][dr.to_mj].astype(float))
    poses = [cur]
    for goal in ([plan["via"]] if plan["via"] is not None else []) + [plan["pre"]]:
        c, _ = HG.command_b(pose, goal, plan["ad"], plan["pz"])
        target_w = np.r_[p[:2] + (Rh @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
        rot_w = L.rot_z(yaw + c["dyaw"]) @ L.rot_y(c["pitch"])
        ps, _, _, _ = pl.reach(cur, c["side"], target_w, rot_w, c["pelvis_z"])
        poses += ps[1:]
        cur = ps[-1]
    RP = np.array([np.r_[a, b] for a, b, _ in poses], float)
    RQ = np.array([j for _, _, j in poses], float)
    hand = np.array([pl.fk_world(a, b, j)["right"][0] for a, b, j in poses], float)
    return RP, RQ, hand


def route_request(dr, x: dict, plan: dict, pose, lk: dict) -> dict:
    """The planner's request: the measured start, A's chain, the look's points in the measured world (the look frame ->
    the odometry belief at this command -> the measured pelvis, as command_b + plan_rec place a target)."""
    RP, RQ, hand = straight_chain(dr, x, plan, pose)
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yc = L.yaw_of(L._qmat(quat))
    po, yo = np.asarray(pose[0], float), float(pose[1])
    Mh = (L.rot_z(yc) @ L.rot_z(yo).T)[:2, :2]

    def to_meas(X):
        X = np.asarray(X, float).reshape(-1, 3)
        return np.c_[p[:2] + (X[:, :2] - po[:2]) @ Mh.T, X[:, 2]]

    q_mj = x["q"][dr.to_mj].astype(float)
    return dict(start_P=p, start_Q=quat, start_q=q_mj, start_hand=dr.pl.fk_world(p, quat, q_mj)["right"][0], chain_RP=RP,
                chain_RQ=RQ, chain_hand=hand, furn=to_meas(lk["furn_pts"]), obj=to_meas(lk["obj_pts"]))  # fmt: skip


def route_get(backend: str, task: str, ep: int, z: dict) -> tuple[dict, dict, dict]:
    """The cached answer for (backend, task, ep), else queue the request and wait for lrt_plan. Returns (meta, arrays,
    record). A second request for the same episode (CR after C, a rerun) reads the first answer."""
    import hashlib

    name = f"{backend}__{task}__ep{ep:+05d}"
    out = ROUTES / "routes" / backend / task / f"ep{ep:+05d}.npz"
    md5 = req_md5(z)
    t0 = time.time()
    cached = out.exists()
    if not cached:
        req = ROUTES / "req"
        req.mkdir(parents=True, exist_ok=True)

        def queue():
            tmp = req / f".{name}.{os.getpid()}.tmp"  # not *.npz: the planner takes only finished requests
            with open(tmp, "wb") as fh:
                np.savez(fh, **{k: np.asarray(z[k], np.float64) for k in REQ_KEYS})
            os.replace(tmp, req / f"{name}.npz")

        queue()
        t_q, t_say = time.time(), time.time()
        while not out.exists():
            time.sleep(0.1)
            now = time.time()
            if now - t0 > ROUTE_WAIT_S:
                raise RuntimeError(f"route: no answer for {name} after {ROUTE_WAIT_S:.0f} s")
            if now - t_q > ROUTE_REQUEUE_S and not (req / f"{name}.npz").exists() and not out.exists():
                print(f"lrt route {name}: no answer {now - t0:.0f} s after queueing and no planner holds it in req/: queued again", flush=True)
                queue()
                t_q = now
            if now - t_say > 60:
                print(f"lrt route {name}: waiting for the planner {now - t0:.0f} s", flush=True)
                t_say = now
    raw = out.read_bytes()
    zz = np.load(out, allow_pickle=False)
    meta = json.loads(str(zz["meta"]))
    arr = {k: zz[k] for k in zz.files if k != "meta"}
    rec = dict(meta, cached=cached, wall_wait_s=round(time.time() - t0, 2), file_md5=hashlib.md5(raw).hexdigest(), req_md5_here=md5,
               req_match=meta.get("req_md5") == md5)  # fmt: skip
    return meta, arr, rec


def route_reach(pl, pose, pts, rots, pz) -> tuple[list, list, list]:
    """One WholeBodyIK through the via points (Planner.reach's settings; dev wbik_route.py): 3 cm steps, the wrist turned
    along each leg from one via attitude to the next, the left hand held (weight 0.3), REACH_SPEED per leg."""
    from scipy.spatial.transform import Rotation

    from locomani.primitives.whole_body_ik import WholeBodyIK

    pos, quat, q = pose
    solver = WholeBodyIK(pl.ik, dict(joints=q, pelvis=dict(pos=pos, quat=quat)), L.IK_CFG, tool_offset=tuple(L.HAND_OFF),
                         knee_max=1.5, pelvis_forward_max=0.20, pelvis_height_target=pz, jacobian="analytic")  # fmt: skip
    if pz is not None:
        solver.hi[2] = max(solver.hi[2], pz + 0.01)
    solver.seed = np.clip(solver.initial, solver.lo, solver.hi)
    cur = pl.fk_world(pos, quat, q)
    a, Ra = cur["right"]
    poses, durs, errs = [pose], [], []
    for b, Rb in zip(pts, rots):
        b, Rb = np.asarray(b, float), np.asarray(Rb, float)
        length = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(length / L.REACH_STEP_M)))
        dur = length / L.REACH_SPEED
        rv = Rotation.from_matrix(Ra.T @ Rb).as_rotvec()
        for k in range(1, n + 1):
            u = k / n
            tp = [cur["left"][0], a + u * (b - a)]
            tr = [cur["left"][1], Ra @ Rotation.from_rotvec(u * rv).as_matrix()]
            solver.hand_weights = np.array([0.3, 1.0])
            sol = solver.solve(np.array(tp), np.array(tr), hand_rotation_weight=L.ROT_W)
            qw = Rotation.from_matrix(np.asarray(sol.rotation)).as_quat()[[3, 0, 1, 2]]
            poses.append((np.asarray(sol.position, float), qw, np.asarray(sol.joints, float).copy()))
            durs.append(dur / n)
            errs.append(float(sol.hand_errors[1]))
        a, Ra = b, Rb
    tot = float(sum(durs))
    if tot < L.REACH_MIN_S:  # at least REACH_MIN_S in all (Planner.reach's floor)
        durs = [d * L.REACH_MIN_S / max(tot, 1e-9) for d in durs]
    return poses, durs, errs


def plan_route(dr, x: dict, ref_w: np.ndarray, vias: np.ndarray, rots: np.ndarray, pz):
    """plan_rec for the route command: the IK from the measured state through the via points, the poses shifted by the
    reference offset, blend_frames; rec in plan_rec's form (target_w / rot_w = the last via point; dyaw / pitch read off
    its attitude as rot_z(yaw + dyaw) rot_y(pitch) would give its x axis)."""
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    q_mj = x["q"][dr.to_mj].astype(float)
    poses, durs, errs = route_reach(dr.pl, (p, quat, q_mj), vias, rots, pz)
    delta = np.r_[ref_w[:2] - p[:2], 0.0]
    ref_pose = (ref_w[:3].astype(float), ref_w[3:7].astype(float), ref_w[7:].astype(float))
    shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
    last = poses[-1]
    tw, Rw = np.asarray(vias[-1], float), np.asarray(rots[-1], float)
    Rh = L.rot_z(yaw)
    gh = np.r_[(Rh.T @ np.r_[tw[:2] - p[:2], 0.0])[:2], tw[2]]
    dyaw = wrap(math.atan2(Rw[1, 0], Rw[0, 0]) - yaw)
    pitch = math.asin(float(np.clip(-Rw[2, 0], -1.0, 1.0)))
    rec = dict(side="right", target_w=tw.tolist(), rot_w=Rw.tolist(), dyaw=dyaw, pitch=pitch, pelvis_z=pz, cmd_target=gh.tolist(),
               ik_err=errs[-1], reach_s=float(sum(durs)), ref_q=last[2].tolist(), ref_pel=np.r_[last[0], last[1]].tolist(),
               delta=delta.tolist(), RT=np.r_[0.0, np.cumsum(durs)], RQ=np.array([pp[2] for pp in poses]),
               RP=np.array([np.r_[pp[0], pp[1]] for pp in poses]), ik_err_max=float(max(errs)))  # fmt: skip
    c = dict(side="right", target=gh, dyaw=dyaw, pitch=pitch, pelvis_z=pz)
    return (L.blend_frames(shifted, durs), shifted[-1], rec), c, gh


# -- one episode (lr_run.episode + lmlcmp) ----------------------------------------------------------------------------
wrap = LR.wrap


def episode(dr, rig, ps: PropSettle, ep: int, MD: Models, backend: str, method: str) -> tuple[dict, list[dict]]:
    import hg_look as HL

    t0 = time.time()
    spec = HG.TASKS[dr.task]
    jit = dr.reset_hg(ep)
    rig.clear()
    ps.clear()  # lmlcmp
    frames = dr.start_frames()
    f_look = int(round(L.START_HOLD_S * L.FPS))
    if spec["kind"] == "approach":  # the look lean (same for both groups)
        n_l, n_a = int(round(HG.LEAN_S * L.FPS)), int(round(HG.LOOK_AFTER_S * L.FPS))
        f0 = 5
        frames = np.concatenate([frames, np.repeat(frames[-1:], n_l + n_a + 10, axis=0)])
        for j in range(f0, len(frames)):
            frames[j, 7 + HG.WAIST_PITCH] = L.DEFAULT_Q[HG.WAIST_PITCH] + HG.LOOK_LEAN * L.minjerk(min(1.0, (j - f0) / n_l))
        f_look = max(f_look, f0 + n_l + n_a)
    rows: dict[int, dict] = {}
    R = dict(task=dr.task, ep=ep, method=method, backend=backend, jitter=jit, obj_name=dr.obj_name, scene=dr.scene)
    f = 0
    phase, fell, why = "stance", None, ""
    odo, plan, lk = None, None, None
    sched: dict[str, int] = {}
    reaches, contact_hits = [], []
    lift_run, lift_best, rise_max, official_at = 0, 0, 0.0, None
    anchor = None
    walk_info: dict = {}
    n_walks = 0
    first = None
    rlo, rhi = HG.train_range()
    delay = shift = None
    t_max = 60 * L.FPS
    d3r: list[dict] = []
    cur_delta = np.zeros(3)
    active = None
    vq: deque = deque(maxlen=D3.SETTLE_F)
    lost = None
    walking = False
    wrec = dict(asked=None, real=None, dev_cm=0.0)
    x_walk0 = None
    # lmlcmp: the waits (key -> opening frame, per-step stats), their log, the frames fired this step without a second wait
    waits: dict[str, dict] = {}
    wlog: list[dict] = []
    agree = dict(both=0, neither=0, own_only=0, truth_only=0)
    # lmlroute (l): the route decision (made once, at the first reach command), its arrays, whether a route command went;
    # the furniture contact rows of the window with the hand point (truth, scoring only)
    route, route_arr, route_on = None, None, False
    furn_log: list = []

    def close_window(fe: int, kind: int = 2) -> None:
        """The pending reach's window ends at row fe: no label yet -> the last 10 Hz row inside it (settled_end 0)."""
        nonlocal active
        if active is not None and active["f_label"] is None:
            g = active["f_cmd"] + ((fe - active["f_cmd"]) // D3.TICKS) * D3.TICKS
            hw = rows[g]["wr"]["right"]
            active.update(f_label=g, settled_end=False, settle_s=(g - active["f_end"]) / L.FPS, label_kind=kind,
                          live_err_cm=100 * float(np.linalg.norm(hw[0] + hw[1] @ L.HAND_OFF - active["hands"]["right"]["target_w"])))  # fmt: skip
        active = None

    def gate(key: str) -> bool:
        """lmlcmp (a): may the command `key` go at this step? Opens its wait at the first call (lmlreal's time for it);
        True when the own rule holds or 3 s have passed, else the command moves to the next step."""
        w = waits.setdefault(key, dict(open=f, tf=None, n=dict(both=0, neither=0, own_only=0, truth_only=0)))
        qo, qt = ps.quiet(), rig.quiet()
        cat = "both" if qo and qt else "neither" if not (qo or qt) else "own_only" if qo else "truth_only"
        w["n"][cat] += 1
        agree[cat] += 1
        if qt and w["tf"] is None:
            w["tf"] = f
        if qo or f - w["open"] >= SETTLE_MAX_F:
            vh, vl = ps.vmax()
            tv = 100 * float(max(rig.v)) if len(rig.v) else float("inf")  # data3's truth speed over the last 0.5 s (measurement)
            wlog.append(dict(key=key, open=w["open"], fire=f, wait_s=(f - w["open"]) / L.FPS, prop_settled=bool(qo), truth_quiet=bool(qt), truth_v_max_cms=tv,
                             truth_first_s=-1.0 if w["tf"] is None else (w["tf"] - w["open"]) / L.FPS, steps=w["n"],
                             prop_v_hand_cms=100 * vh, prop_v_leg_rads=vl, phase=phase))  # fmt: skip
            del waits[key]
            return True
        sched[key] = f + 1
        return False

    while True:
        dr.push(frames[min(f, len(frames) - 1)])
        if dr.buf is None:
            dr.buf = dr._find_buffer()
        dr.step()
        if delay is None:
            delay = D3.ref_delay_f(dr)
            shift = delay - LR.BASE_DELAY_F
            if shift > 0:
                frames = np.concatenate([frames, np.repeat(frames[-1:], shift + 10, axis=0)])
                f_look += shift
                t_max += 6 * shift
        if f == 1 and getattr(getattr(dr, "comp", None), "_settling", False):
            raise RuntimeError("the bridge is still settling: our reference is not being tracked")
        x = rows[f] = dr.sample()
        rig.record(x)
        ps.update(x)  # lmlcmp
        vq.append(np.linalg.norm(rig.pts[-1] - rig.pts[0], axis=1) / (D3.TICKS * L.DT) if len(rig.pts) > D3.TICKS else np.full(3, np.inf))
        if odo is not None:
            odo.update(dr, x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, f"fell during {phase}"
            break
        if anchor is not None:
            fr = frames[min(max(f - shift, anchor[2]), len(frames) - 1)]
            dev = (x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])
            if walking:
                wrec["dev_cm"] = max(wrec["dev_cm"], 100 * float(np.linalg.norm(dev)))
            if float(np.linalg.norm(dev)) > L.LOST_M:
                fell, why = f, f"lost during {phase}"
                lost = f
                break
        # the pending reach's end label: data3's rule on its 10 Hz grid; lmlcmp (b): at most 3 s after the opening
        if active is not None and active["f_label"] is None and (f - active["f_cmd"]) % D3.TICKS == 0:
            if f >= active["f_end"] + delay + D3.SETTLE_F and rig.quiet():
                hw = x["wr"]["right"]
                active.update(f_label=f, settled_end=True, settle_s=(f - active["f_end"]) / L.FPS, label_kind=0,
                              live_err_cm=100 * float(np.linalg.norm(hw[0] + hw[1] @ L.HAND_OFF - active["hands"]["right"]["target_w"])))  # fmt: skip
            elif f >= active["f_end"] + delay + SETTLE_MAX_F:
                close_window(f, kind=1)
        # contacts during the reach (after its command row, up to its label row)
        if active is not None and f > active["f_cmd"] and (active["f_label"] is None or active["f_label"] == f):
            active["furn_rows"] += int(dr.arm_scene_contact() is not None)
            active["obj_rows"] += int(dr.obj_contacts()[0])
            active["win_rows"] += 1
        # scoring (truth)
        z = float(dr.d.xpos[dr.obj_bid][2]) - dr.obj_z0
        if "close" in sched and f >= sched["close"] and phase in ("close", "lift"):
            hand_c, world_c = dr.obj_contacts()
            rise_max = max(rise_max, z)
            lift_run = lift_run + 1 if (z >= HG.LIFT_MIN and not world_c) else 0
            lift_best = max(lift_best, lift_run)
            try:
                if official_at is None and bool(dr.env._check_success()):
                    official_at = f
            except Exception:
                pass
        # lr_run's window: from the first reach's time up to the close (here: up to the step the close goes; that step's
        # own contact is dropped when it goes, as lr_run's f < sched["close"])
        if first and ((first in sched and sched[first] <= f) or route_on) and phase not in ("close", "lift"):  # lmlroute: or after a route command
            h = dr.arm_scene_contact()
            if h is not None:
                contact_hits.append((f, h))
                if method in ROUTE:  # lmlroute (l): the hand point of every contact row (truth, scoring only)
                    furn_log.append([f, h, *map(float, dr.hand_point(x))])
        # -- the look ---------------------------------------------------------------------------------------------
        if f == f_look:
            lp = look_path(dr.task, ep)  # lmlcmp3 (j)
            lk = HL.load(lp)
            pel = np.asarray(lk["pelvis"], float)
            R["look_state_dev_mm"] = 1000 * float(np.abs(pel[:3] - x["root"][:3]).max())
            R["look"] = dict(status=lk["meta"].get("status"), mask_score=float(lk["mask_score"]), n_obj=len(lk["obj_pts"]),
                             n_grasps=len(lk["grasp_T"]), prompt=lk["meta"].get("prompt_used"))  # fmt: skip
            if "mask_source" in lk["meta"]:  # lmlroute (n): the mask's source and size (absent: SAM 3, as before)
                R["look"].update(mask_source=lk["meta"]["mask_source"], gt_mask_px=lk["meta"].get("gt_mask_px"),
                                 gt_mask_pts=lk["meta"].get("gt_mask_pts"), look_file=str(lp))  # fmt: skip
            tp = dr.obj_pos()
            R["look"]["obj_center_err_cm"] = 100 * float(np.linalg.norm(np.median(lk["obj_pts"], 0) - tp)) if len(lk["obj_pts"]) else None
            p0, yaw0 = x["root"][:2].copy(), L.yaw_of(L._qmat(x["root"][3:]))
            import hg_walk as HW

            odo = HW.LegOdo(dr, x, (p0, yaw0))
            plan = HG.plan_from_look(lk, spec["kind"], x["root"][:3], yaw0)
            R["plan"] = HG._js({k: v for k, v in plan.items() if k not in ("stance",)})
            R["plan"]["stance"] = HG._js(plan.get("stance"))
            if plan["fail"]:
                why = plan["fail"]
                break
            st = plan["stance"]
            d_move = float(np.linalg.norm(st["xy"] - p0))
            d_turn = abs(wrap(st["yaw"] - yaw0))
            R["move"] = dict(dist=d_move, turn_deg=math.degrees(d_turn))
            if d_move >= HG.MOVE_MIN or d_turn >= HG.TURN_MIN:
                wf = walk_ctrl(dr, backend, x, frames[f], st["xy"], st["yaw"], pose=odo.pose(), info=walk_info)
                frames = np.concatenate([frames[: f + 1], wf, np.repeat(wf[-1:], max(shift, 0) + 10 + int(EXTRA_HOLD_S * L.FPS), 0)])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                first = "cmd0" if plan["via"] is not None else "cmd1"
                sched[first] = f + len(wf) - 25 + shift
                phase = "walk"
                n_walks = 1
                dp = L.rot_z(yaw0).T @ np.r_[st["xy"] - p0, 0.0]
                wrec["asked"] = [float(dp[0]), float(dp[1]), wrap(float(st["yaw"]) - yaw0)]
                cur_delta = np.r_[frames[f, :2] - x["root"][:2], 0.0]
                x_walk0, walking = x, True
            else:
                frames = np.concatenate([frames[: f + 1], L.hold((frames[f][:3], frames[f][3:7], frames[f][7:]), 2.0 + shift / L.FPS + EXTRA_HOLD_S)])
                first = "cmd0" if plan["via"] is not None else "cmd1"
                sched[first] = f + int(round(L.PRE_S * L.FPS))
        # -- reaches ----------------------------------------------------------------------------------------------
        fired = set()  # lmlcmp: a command whose wait ended on this step (no second wait)
        if first and sched.get(first) == f and phase == "walk" and n_walks < 1 + HG.WALK_FIX and gate(first):
            pose = odo.pose()
            st = plan["stance"]
            d_left = float(np.linalg.norm(st["xy"] - pose[0]))
            R.setdefault("walk_left_cm", []).append(100 * d_left)
            if d_left >= HG.FIX_MIN and n_walks < HG.WALK_FIX + 1:
                wlog[-1]["then"] = "correction walk"
                wf = walk_ctrl(dr, backend, x, frames[f], st["xy"], st["yaw"], pose=pose)
                frames = np.concatenate([frames[: f + 1], wf, np.repeat(wf[-1:], max(shift, 0) + 10 + int(EXTRA_HOLD_S * L.FPS), 0)])
                anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                sched[first] = f + len(wf) - 25 + shift
                n_walks += 1
                cur_delta = np.r_[frames[f, :2] - x["root"][:2], 0.0]
            else:
                fired.add(first)
        for k, key in ((-1, "cmd0"), (0, "cmd1"), (1, "cmd2")):
            if sched.get(key) != f or (phase == "walk" and n_walks < 1 + HG.WALK_FIX and key == first and key not in fired):
                continue
            if key not in fired and not gate(key):
                continue
            wl = wlog[-1]
            route_now = None  # lmlroute (l): the decision at the first reach command (C and CR alike; nothing has moved yet)
            if method in ROUTE and key == first and route is None:
                tic_r = time.time()
                zreq = route_request(dr, x, plan, odo.pose(), lk)
                meta_r, route_arr, route = route_get(backend, dr.task, ep, zreq)
                route.update(f=f, key=key, think_wall_s=round(time.time() - tic_r, 2), routed_here=False)
                if meta_r.get("routed"):
                    try:
                        route_now = plan_route(dr, x, frames[f], route_arr["via_p"][1:-1], route_arr["via_R"][1:-1], plan["pz"])
                    except (np.linalg.LinAlgError, ValueError) as e:  # our IK cannot follow it: A's / R's commands (fallback)
                        route.update(fallback="route IK failed", plan_error=f"{type(e).__name__}: {str(e)[:160]}")
                    else:
                        route["routed_here"] = True
                        k = -1  # the route command takes the first slot (reach_no 0); the pre-grasp follows as after a via point
            wl["reach_no"] = LR.REACH_NO[k]
            phase = f"reach{k + 1}"
            close_window(f)  # the previous reach's window ends at this command
            walking = False
            if x_walk0 is not None and wrec["real"] is None:
                pw0, yw0 = x_walk0["root"][:2].astype(float), L.yaw_of(L._qmat(x_walk0["root"][3:]))
                dp = L.rot_z(yw0).T @ np.r_[x["root"][:2] - pw0, 0.0]
                wrec["real"] = [float(dp[0]), float(dp[1]), wrap(L.yaw_of(L._qmat(x["root"][3:])) - yw0)]
            chain = len(d3r)
            pose = odo.pose()
            if route_now is not None:  # lmlroute (l): the route's last via point, back in the look frame (command_b's inverse)
                (blend, last, rec), c, gh = route_now
                goal = np.r_[pose[0] + (L.rot_z(pose[1]) @ np.r_[gh[:2], 0.0])[:2], gh[2]]
            else:
                goal = plan["via"] if k == -1 else plan["pre"] if k == 0 else plan["goal"]
                c, gh = HG.command_b(pose, goal, plan["ad"], plan["pz"])
            info = dict(k=k, reach_no=LR.REACH_NO[k], chain=chain, goal_w=goal.tolist(), goal_h=gh.tolist(),
                        dyaw_deg=math.degrees(c["dyaw"]), pitch_deg=math.degrees(c["pitch"]), pz=c["pelvis_z"],
                        pose_odo=[pose[0].tolist(), pose[1]], pose_true=[x["root"][:2].tolist(), L.yaw_of(L._qmat(x["root"][3:]))],
                        wait_s=wl["wait_s"], prop_settled=wl["prop_settled"], truth_settled=bool(rig.quiet()))  # fmt: skip
            qx = x["q"]
            out_j = int(((qx < rlo) | (qx > rhi)).sum())
            info.update(q_out=out_j, q_in_range=out_j == 0)
            tic = time.time()
            feat = lambda cc: MD.features(dr, rig, rows, f, frames, cc, chain, n_walks > 0, cur_delta, wrec, delay)  # noqa: E731
            best = None
            if (method in INVERT and k >= 0) or (method in ENTRY_ONLY and k == 1):  # (e): D's fixed point; the via point stays direct (lmlcmp2: + guard, reachable; lmlcmp3: + entryinv's entry)
                t, evs = gh.copy(), []
                shr = [] if method in SHRINK else None  # lmlroute: R's and CR's shrink-back
                for it in range(LT.ITERS + 1):
                    if shr is not None and it > 0:  # lmlcmp2 (g): the candidate shrunk back to IK-reachable first
                        t, sh = shrink(dr, rows[f], frames[f], c, gh, t)
                        shr.append(dict(sh, it=it))
                    cc = dict(c, target=t.copy())
                    try:
                        pl_, zz = feat(cc)
                    except (np.linalg.LinAlgError, ValueError) as e:  # this candidate cannot be planned
                        evs.append(dict(t=t.copy(), l=np.inf, err=f"{type(e).__name__}: {str(e)[:120]}"))
                        break
                    pos, std = MD.predict(zz, ("D",))["D"]
                    evs.append(dict(t=t.copy(), pos=pos, std=std, l=float(((pos - gh) ** 2).sum() + LT.LAMBDA * (std**2).sum()), plan=pl_, z=zz))
                    t = LT.clip_t(t + (gh - pos), "right")
                bi = min(range(len(evs)), key=lambda i: evs[i]["l"])  # hg_run's min(evs, key=l): the first least
                best = evs[bi]
                info["inv"] = [dict(t=e["t"].tolist(), pos=e["pos"].tolist() if "pos" in e else None, std=e["std"].tolist() if "std" in e else None,
                                    l=e["l"], err=e.get("err")) for e in evs]  # fmt: skip
                if shr is not None:
                    info["shrink"] = shr  # lmlcmp2 (g)
                if "plan" in best:
                    if method == "guard":  # lmlcmp2 (f): the two guards on the chosen candidate
                        best, info["guard"] = guard(best, evs, gh, c, feat, MD)
                    c["target"] = best["t"]
                    info.update(inv_best=bi, pred_miss0_cm=100 * float(np.linalg.norm(evs[0]["pos"] - gh)),
                                pred_miss_cm=100 * float(np.linalg.norm(best["pos"] - gh)),
                                corr_cm=100 * float(np.linalg.norm(best["t"] - gh)))  # fmt: skip
                    (blend, last, rec), zz = best["plan"], best["z"]
                else:
                    info["model_fallback"] = True
                    best = None
            if best is None and route_now is None:  # lmlroute: the route command is planned above
                try:
                    (blend, last, rec), zz = feat(c)
                except (np.linalg.LinAlgError, ValueError) as e:
                    why = f"reach: plan failed {type(e).__name__}"
                    info["plan_error"] = str(e)[:200]
                    reaches.append(info)
                    fell = None
                    break
            if route_now is None:  # lmlroute: no end model for the route command (none is trained on via points)
                pr = MD.predict(zz)  # (d): T3, M, D on the executed command
                info["pred_h"] = {n: v[0].tolist() for n, v in pr.items()}
                info["pred_std"] = {n: v[1].tolist() for n, v in pr.items()}
                info["feat_x0"], info["feat_u"] = zz["x0"][0].tolist(), zz["u"][0].tolist()
            else:
                route_on = True
                info.update(route=True, n_via=len(route_arr["via_p"]) - 2, ik_err_max_cm=100 * float(rec["ik_err_max"]))
            info["think_s"] = time.time() - tic
            rec.update(chain=chain, f_cmd=f, f_end=f + len(blend))
            post = int(round(L.POST_S * L.FPS))
            frames = np.concatenate([frames[: f + 1], blend, L.hold(last, L.POST_S + L.PRE_S + L.MARGIN_S + 0.5 + shift / L.FPS + EXTRA_HOLD_S)])
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
            info.update(cmd_h=np.asarray(c["target"]).tolist(), f_cmd=f, f_end=rec["f_end"], f_win=rec["f_end"] + post + shift,
                        ik_err_cm=100 * float(rec["ik_err"]), p_cmd=x["root"][:3].tolist(), yaw_cmd=L.yaw_of(L._qmat(x["root"][3:])))  # fmt: skip
            info.update(ref_end(dr, rec, x))  # lmlcmp3 (i)
            reaches.append(info)
            vv = np.array(vq)
            vh, vl = ps.vmax()
            rc = reach_rc(rec, f, rec["f_end"], chain, cur_delta, slot=LR.REACH_NO[k])
            rc.update(pre_wait_s=wl["wait_s"], live_err_cm=-1.0, pre_settled=bool(rig.quiet()),
                      pre_v_pel_cms=100 * float(vv[:, 0].max()), pre_v_hand_cms=100 * float(vv[:, 1].max()),
                      pre_v_other_cms=100 * float(vv[:, 2].max()), pre_v_max_cms=100 * float(vv.max()), furn_rows=0, obj_rows=0,
                      win_rows=0, prop_settled=wl["prop_settled"], prop_v_hand_cms=100 * vh, prop_v_leg_rads=vl,
                      truth_first_s=wl["truth_first_s"], label_kind=-1)  # fmt: skip
            cur_delta = rc["delta"].copy()
            d3r.append(rc)
            active = rc
            if k == -1:
                sched["cmd1"] = rec["f_end"] + post + int(round(L.PRE_S * L.FPS)) + shift
            elif k == 0:
                sched["cmd2"] = rec["f_end"] + post + int(round(L.PRE_S * L.FPS)) + shift
            else:
                sched["close"] = rec["f_end"] + post + shift
            break  # one command per step
        if why.startswith("reach: plan failed"):
            break
        if sched.get("close") == f and phase == "reach2" and gate("close"):
            wlog[-1]["reach_no"] = -1
            close_window(f)  # the entry's window ends at the close
            contact_hits = [h for h in contact_hits if h[0] < f]
            phase = "close"
            hp = dr.hand_point(x)
            R["close_dist_cm"] = 100 * float(np.linalg.norm(hp - plan["goal"]))
            R["close_obj_cm"] = 100 * float(np.linalg.norm(hp - dr.obj_pos()))
            sched["lift"] = f + int(round((HG.CLOSE_S + HG.SETTLE_S) * L.FPS))
        if phase in ("close", "lift") and dr.grip < HG.GRIP:
            dr.grip = min(HG.GRIP, (f - sched["close"] + 1) / (0.3 * L.FPS))
        if sched.get("lift") == f and phase == "close" and gate("lift"):
            wlog[-1]["reach_no"] = -2
            phase = "lift"
            pose = odo.pose()
            goal = plan["goal"] + np.array([0.0, 0.0, HG.LIFT])
            c, gh = HG.command_b(pose, goal, plan["ad"], None)
            try:
                blend, last, rec = dr.plan(x, frames[f], c)
            except (np.linalg.LinAlgError, ValueError):
                why = "lift: plan failed"
                break
            frames = np.concatenate([frames[: f + 1], blend, L.hold(last, HG.HOLD_S + L.MARGIN_S + 0.5 + shift / L.FPS)])
            anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
            sched["end"] = f + len(blend) + int(round(HG.HOLD_S * L.FPS)) + shift
            R["lift_cmd"] = dict(goal_h=gh.tolist(), ik_err_cm=100 * float(rec["ik_err"]))
        if sched.get("end") is not None and f >= sched["end"]:
            break
        if f >= len(frames) - 1:
            why = why or "motion ended"
            break
        if f > t_max:
            why = "timeout"
            break
        f += 1
    # -- post-hoc --------------------------------------------------------------------------------------------------
    for w, rc in zip([w for w in reaches if "f_win" in w], d3r):
        fl = rc["f_label"] if rc["f_label"] is not None else min(w["f_win"], max(rows))
        hand = dr.hand_point(rows[fl]) if fl in rows else None
        if hand is None:
            continue
        p0c, Rh = np.asarray(w["p_cmd"]), L.rot_z(w["yaw_cmd"])
        act_h = np.r_[(Rh.T @ np.r_[hand[:2] - p0c[:2], 0.0])[:2], hand[2]]
        w["err_cm"] = 100 * float(np.linalg.norm(hand - np.asarray(w["goal_w"])))
        w["act_h"], w["f_label"], w["settled_end"] = act_h.tolist(), fl, bool(rc["settled_end"])
        w["pred_err_cm"] = {n: 100 * float(np.linalg.norm(np.asarray(v) - act_h)) for n, v in w.get("pred_h", {}).items()}
    if walk_info:
        R["walk"] = HG._js({k: v for k, v in walk_info.items() if k in ("walk_s", "n_steps", "ik_err", "knots", "ik_err_cm", "kinds", "gait")})
    if first and first in sched and odo is not None and 0 < sched[first] <= max(rows):
        xr = rows[min(sched[first], max(rows))]
        st = plan["stance"]
        R["arrive"] = dict(pos_err_cm=100 * float(np.linalg.norm(xr["root"][:2] - st["xy"])),
                           yaw_err_deg=math.degrees(abs(wrap(L.yaw_of(L._qmat(xr["root"][3:])) - st["yaw"]))))  # fmt: skip
    own_ok = official_at is not None
    lift10 = lift_best >= HG.LIFT_HOLD
    success = own_ok if spec["own"] else lift10
    R.update(success=bool(success), official=own_ok, official_step=official_at, lift10=bool(lift10), rise_max_cm=100 * rise_max,
             fell=fell, why=why, steps=len(rows), n_walks=n_walks, reaches=reaches, n_contacts=len(contact_hits),
             first_contact=contact_hits[0] if contact_hits else None, wall_s=round(time.time() - t0, 1))  # fmt: skip
    R["reason"] = HG.classify(R, phase)
    R["waits"], R["agree_steps"] = wlog, agree  # lmlcmp
    info3 = dict(fell=fell, lost=lost, walk=wrec, walked=n_walks > 0, delay_f=delay)
    segs = D3.segments(rows, info3, frames, d3r, reset_id(dr.task, method, ep), dr.pl, rig)
    for s in segs:
        rc = d3r[s["chain"]]
        s.update(reach_no=rc["slot"], task_id=TASKS.index(dr.task), ep=ep, backend_id=BACKENDS.index(backend), method_id=METHODS.index(method),
                 d_prev_x=float(rc["d_prev"][0]), d_prev_y=float(rc["d_prev"][1]), d_this_x=float(rc["delta"][0]), d_this_y=float(rc["delta"][1]),
                 **{k: rc[k] for k in ("pre_settled", "pre_v_pel_cms", "pre_v_hand_cms", "pre_v_other_cms", "pre_v_max_cms", "live_err_cm",
                                       "furn_rows", "obj_rows", "win_rows", "prop_settled", "prop_v_hand_cms", "prop_v_leg_rads",
                                       "truth_first_s", "label_kind")})  # fmt: skip
        if route_on and s["chain"] == 0:  # lmlroute (l): the route command's via points (d3_collect.augment_b's form), via_route 2
            p0s, Rhs = s["start"][:3].astype(float), L.rot_z(float(s["start"][3]))
            vps, vrs = route_arr["via_p"][1:-1][:16], route_arr["via_R"][1:-1][:16]
            vp, vr, vg = np.zeros((16, 3), np.float32), np.zeros((16, 6), np.float32), np.zeros(16, np.float32)
            for i, (pv, Rv) in enumerate(zip(vps, vrs)):
                vp[i] = np.r_[(Rhs.T @ np.r_[pv[:2] - p0s[:2], 0.0])[:2], pv[2]]
                Rl = Rhs.T @ np.asarray(Rv, float)
                vr[i], vg[i] = np.r_[Rl[:, 0], Rl[:, 1]], 1.0
            s.update(n_via=len(vps), via_p=vp, via_r6=vr, via_gen=vg, via_route=2)
    if method in ROUTE:  # lmlroute (l)
        keep = {h[0] for h in contact_hits}
        R["lmlroute"] = dict(route=route, route_on=route_on, contacts=[c for c in furn_log if c[0] in keep])
    R.update(delay_f=delay, shift_f=shift, walk_rec=wrec, n_segments=len(segs),
             d3=[dict(reach_no=rc["slot"], chain=rc["chain"], f_cmd=rc["f_cmd"], f_end=rc["f_end"], f_label=rc["f_label"],
                      settled_end=rc["settled_end"], settle_s=rc["settle_s"], label_kind=rc["label_kind"], pre_settled=rc["pre_settled"],
                      prop_settled=rc["prop_settled"], pre_wait_s=rc["pre_wait_s"], pre_v_max_cms=rc["pre_v_max_cms"],
                      prop_v_hand_cms=rc["prop_v_hand_cms"], prop_v_leg_rads=rc["prop_v_leg_rads"], live_err_cm=rc["live_err_cm"],
                      furn_rows=rc["furn_rows"], obj_rows=rc["obj_rows"], win_rows=rc["win_rows"]) for rc in d3r])  # fmt: skip
    for s in segs:
        R["d3"][s["chain"]].update(jaw_err_cm=float(s["jaw_err_cm"]), seg_fell=bool(s["fell"]))
    return R, segs


# -- runners ----------------------------------------------------------------------------------------------------------
def ep_path(backend: str, task: str, method: str, ep: int) -> Path:
    return OUT / backend / "episodes" / task / method / f"ep{ep:+05d}.json"


HOST = os.uname().nodename  # lmlroute (o): every record states the machine


def run_one(dr, rig, ps, MD, backend: str, task: str, m: str, e: int):
    """One group of one episode, recorded (lmlroute (o): the loop body of run, unchanged but the host field). Returns
    (dr, rig, ps): rebuilt after an error, as run did."""
    try:
        r, segs = episode(dr, rig, ps, e, MD, backend, m)
    except Exception as err:  # recorded, not retried
        r, segs = dict(task=task, ep=e, method=m, backend=backend, error=f"{type(err).__name__}: {err}",
                       tb=traceback.format_exc()[-2000:], success=False, reason="other"), []  # fmt: skip
        print(f"lc {backend} {task} {m} ep {e}: ERROR {err}", flush=True)
        dr = LR.make_driver(backend, task)
        rig, ps = D3.Rig(dr), PropSettle(dr)
    r["host"] = HOST
    D3.save_chunk(segs, OUT / backend / "segs" / task / m, f"ep{e:+05d}", rig, SCALARS_LC)
    p = ep_path(backend, task, m, e)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{HOST}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(r, default=LT._json))
    os.replace(tmp, p)
    print(f"lc {backend} {task} {m} ep {e}: success {r.get('success')} reason {r.get('reason')} why {r.get('why')} close "
          f"{r.get('close_dist_cm', float('nan')):.1f} cm waits {[(w['key'], round(w['wait_s'], 2), w['prop_settled'], w['truth_quiet']) for w in r.get('waits', [])]} "
          f"labels {[(d['reach_no'], d['label_kind'], round(d.get('jaw_err_cm', -1), 1)) for d in r.get('d3', [])]} "
          f"inv {[(w['reach_no'], round(w.get('corr_cm', 0), 1), round(w.get('pred_miss_cm', -1), 2)) for w in r.get('reaches', []) if 'inv' in w]} "
          f"{r.get('wall_s')} s", flush=True)  # fmt: skip
    code_check(f"after {m} ep {e}")  # lmlcmp3 (k)
    return dr, rig, ps


def claim_dir(backend: str, task: str, e: int) -> Path:
    return OUT / "claims" / backend / task / f"ep{e:+05d}"


def claim(backend: str, task: str, eps: list[int], methods: list[str]) -> int | None:
    """lmlroute (o): the first episode of eps whose claim directory this process creates (mkdir: one winner across the
    machines on vePFS) and whose groups are not all recorded yet; None when there is none."""
    for e in eps:
        d = claim_dir(backend, task, e)
        if d.exists() or all(ep_path(backend, task, m, e).exists() for m in methods):
            continue
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(d)
        except FileExistsError:
            continue
        (d / "claim.json").write_text(json.dumps(dict(host=HOST, pid=os.getpid(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))))
        return e
    return None


def work(backend: str, tasks: list[str], eps: list[int], methods: list[str]) -> None:
    """lmlroute (o): claim episodes (all groups of one) until none is left, task by task."""
    n = 0
    MD = None
    for task in tasks:
        dr = None
        while (e := claim(backend, task, eps, methods)) is not None:
            if dr is None:
                dr = LR.make_driver(backend, task)
                rig, ps = D3.Rig(dr), PropSettle(dr)
                if MD is None:
                    MD = Models(backend)
                print(f"lc work {backend} {task} on {HOST} pid {os.getpid()}: pelvis offset c {np.round(dr.c, 4).tolist()}, root check "
                      f"{rig.check_root():.2e} m; models { {n_: str(r) for n_, r in model_roots(backend).items()} }", flush=True)  # fmt: skip
                code_check("start")  # lmlcmp3 (k)
            t0 = time.time()
            for m in methods:
                if not ep_path(backend, task, m, e).exists():
                    dr, rig, ps = run_one(dr, rig, ps, MD, backend, task, m, e)
            (claim_dir(backend, task, e) / "done.json").write_text(json.dumps(dict(host=HOST, pid=os.getpid(), s=round(time.time() - t0, 1))))
            n += 1
    print(f"lc work {backend} on {HOST} pid {os.getpid()}: {n} episodes, nothing left to claim", flush=True)


def release(backend: str, methods: list[str]) -> None:
    """lmlroute (o): claims of this machine whose process is gone without done.json -> the episode's records of every
    group (json, segs, arm) and the claim removed, so that the whole episode runs again in one process (pairs on one
    machine). Other machines' claims are left alone."""
    n = 0
    for d in sorted((OUT / "claims" / backend).glob("*/ep*")):
        if (d / "done.json").exists():
            continue
        try:
            c = json.loads((d / "claim.json").read_text())
        except (OSError, ValueError):
            c = {}
        if c.get("host") not in (HOST, None):
            continue
        pid = c.get("pid")
        if pid is not None:
            try:
                os.kill(int(pid), 0)
                continue  # still running
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
        task, e = d.parent.name, int(d.name[2:])
        gone = []
        for m in methods:
            for f in (ep_path(backend, task, m, e), OUT / backend / "segs" / task / m / f"ep{e:+05d}.npz", OUT / backend / "segs" / task / m / "arm" / f"ep{e:+05d}.npz"):
                if f.exists():
                    f.unlink()
                    gone.append(f.name)
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
        n += 1
        print(f"lc release {backend} {task} ep {e}: claim of pid {pid} on {HOST} without done.json; removed {gone}", flush=True)
    print(f"lc release {backend} on {HOST}: {n} claims released", flush=True)


def run(backend: str, task: str, eps: list[int], methods: list[str], shard: int, shards: int) -> None:
    jobs = [(m, e) for e in eps[shard::shards] for m in methods]  # lmlroute (m): by episode (C, then CR, in one process)
    todo = [(m, e) for m, e in jobs if not ep_path(backend, task, m, e).exists()]
    print(f"lc {backend} {task} shard {shard}/{shards}: {len(jobs) - len(todo)}/{len(jobs)} done", flush=True)
    if not todo:
        return
    dr = LR.make_driver(backend, task)
    rig, ps = D3.Rig(dr), PropSettle(dr)
    MD = Models(backend)
    print(f"lc {backend} {task}: pelvis offset c {np.round(dr.c, 4).tolist()}, root check {rig.check_root():.2e} m; models "
          f"{ {n: str(r) for n, r in model_roots(backend).items()} }", flush=True)  # fmt: skip
    code_check("start")  # lmlcmp3 (k)
    for m, e in todo:
        dr, rig, ps = run_one(dr, rig, ps, MD, backend, task, m, e)


def merge(backend: str) -> None:
    """segs/<task>/<method>/ep*.npz (+ arm/) -> data/all.npz, data/arm.npz: TASKS order, then METHODS, then ep."""
    base = OUT / backend
    names = []
    for t in TASKS:
        for m in METHODS:
            names += sorted((base / "segs" / t / m).glob("ep*.npz"), key=lambda p: int(p.stem[2:]))
    d = base / "data"
    d.mkdir(parents=True, exist_ok=True)
    for src, dst in ((None, d / "all.npz"), ("arm", d / "arm.npz")):
        zs = [dict(np.load(p if src is None else p.parent / "arm" / p.name)) for p in names]
        keep = ("radius", "link")
        allz = {k: (zs[0][k] if k in keep else np.concatenate([z[k] for z in zs])) for k in zs[0]}
        tmp = dst.with_name(f".{dst.stem}.tmp.npz")
        np.savez_compressed(tmp, **allz)
        os.replace(tmp, dst)
    z = np.load(d / "all.npz")
    print(f"{backend}: {len(names)} episode files, {len(z['n_steps'])} segments, {int(z['n_steps'].sum())} rows, fell/lost "
          f"{int(z['fell'].sum())}, settled at the label {int(z['settled_end'].sum())}; by method "
          f"{dict(zip(*[v.tolist() for v in np.unique(z['method_id'], return_counts=True)]))}; reach_no "
          f"{dict(zip(*[v.tolist() for v in np.unique(z['reach_no'], return_counts=True)]))}")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "merge", "work", "release"])  # lmlroute (o): work = the claim queue
    ap.add_argument("--backend", choices=BACKENDS, default="scalebfm")
    ap.add_argument("--task", default=TASKS[0])
    ap.add_argument("--tasks", default=",".join(TASKS), help="work: the tasks in the order the worker goes through them")
    ap.add_argument("--eps", default="-1:30")
    ap.add_argument("--ep-list", default=None, help="comma-separated episode ids (instead of --eps)")
    ap.add_argument("--methods", default="alone,reachable,route,route_reachable")  # lmlroute: A, R, C, CR (lrt_formal.sh passes them)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    a = ap.parse_args()
    if a.phase == "merge":
        merge(a.backend)
    elif a.phase == "release":
        release(a.backend, a.methods.split(","))
    else:
        if a.ep_list:
            eps = [int(v) for v in a.ep_list.split(",")]
        else:
            lo, hi = (int(v) for v in a.eps.split(":"))
            eps = list(range(lo, hi))
        if a.phase == "work":
            work(a.backend, a.tasks.split(","), eps, a.methods.split(","))
        else:
            run(a.backend, a.task, eps, a.methods.split(","), a.shard, a.shards)
