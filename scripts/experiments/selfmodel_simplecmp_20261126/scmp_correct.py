"""selfmodel_simplecmp_20261126: simple_bench.py's --correct (2026-11-26). Off by default; off, simple_bench never imports
this file and nothing of an episode changes.

With --correct <model dir> (model D: selfmodel_simpledata_20261126's E3_ensemble, frac 1.0, seed 0; REACHWM_GOAL_REF /
APPROACH / QFIX / FDVEL = 1, X0_REF off; or, 2026-11-26 selfmodel_simpleheft_20261126, HEFT with train3b's D, FDVEL = 0)
every episode carries sd_real.py's recorder (EpRec with the task label: the
episode's own rhythm, nothing waits) and the straight whole-body-IK reaches named by --correct-stages (default the
pregrasp and the entry) go through lmlcmp3's inversion on D before they are sent:

  --correct-rule R     lmlcmp2 (g) / lmlcmp3 "reachable": t <- clip_t(t + goal - D(t)), 6 steps (7 candidates, the first
                       the goal itself); every candidate after the first is shrunk first: its plan's IK error > 2 cm ->
                       back along t - goal to the largest fraction a (6 halvings on [0, 1]) with IK error <= 2 cm, a = 0
                       (the goal itself) when none passes; the next step starts from the shrunk candidate; the choice is
                       the first least |D(t) - goal|^2 + LAMBDA |D's spread|^2. A candidate that cannot be planned ends
                       the loop; no planned candidate at all -> the goal itself (model_fallback).
  --correct-rule S     the same fixed point without the shrink (lmlcmp's `selfmodel`, lmlcmp3's S).
  --correct-rule dryR  R's inversion is run and recorded, the goal itself is sent (A's command, J's entry): planning the
                       candidates must change nothing (the rows equal A's / J's), and D's prediction of the command that
                       was sent is recorded.
In SIMPLE's terms (lmlcmp's are LocoManiLab's):
  - a command is the palm pose simple_bench sends (Motor.reach's point, REACH_HAND_OFF 0.0415); D's command target is the
    13 cm point of that pose (lml_selfmodel.HAND_OFF; sd_real's convention): target = palm + R (HAND_OFF - OFF). The
    attitude is the goal's and never changes; a candidate moves the palm and the 13 cm point together.
  - the candidates t are 13 cm-point targets in the heading frame of the command row (data.py's frame); clip_t is
    lml_tasks.clip_t (the data's WIDE box), copied here with ITERS 6 and LAMBDA 1.0.
  - --correct-point palm (default): D's landing is turned into the palm, palm = pos + R(rot6) (OFF - HAND_OFF), and the
    goal is the palm goal (simple_bench's goal_w, what err_cm measures); p13: D's 13 cm point against the goal's 13 cm
    point (lmlcmp's own quantity). The spread term is D's 13 cm-point member std either way.
  - IK error (R's shrink): the 13 cm point of the plan's last pose (the IK model's FK, sd_real's ik_err, the training
    data's) to the candidate's 13 cm target; the palm's error at that pose is recorded beside it.
  - a candidate is planned exactly as the reach will be (Motor.reach with simple_bench's arguments: the entry's speed,
    start and ik_from_ref), stopped inside Motor._blend before the motor is touched (plan_invert.plan_of's way).
  - D's features for a candidate: d3_collect.segments on the recorder's rows up to the command row (the rows after it
    are placeholders: they only feed labels) + data.py's load defaults and derive, the same path as the recording of the
    reach (and sd_real / sx_eval offline); the executed reach's online prediction is compared with the offline one.
Joint-path reaches (--avoid --joint, J's pregrasp) are never changed here: the recorder takes the path's reference
(Motor.reach_joints' end pose and duration, computed as it computes them) and D's prediction of it is recorded with the
reference end as the command (its 13 cm point, IK error 0); scmp_route.py turns that prediction into the route arm's
shifted goal offline, as S-full did with its model.
Decision inputs: the recorder's rows -- joint encoders (FDVEL: velocities from their differences), the IMU's gravity
and yaw change, the hand from the joints; the simulator's pelvis is only the frame's origin (the goal relative to it is the
robot's belief, place.Belief.to_sim) and its height (what leg kinematics give on the flat floor) -- the inputs evaluation
3 of selfmodel_simpledata validated. Simulator truth enters only the records (landed, the labels).

Records: each changed or recorded stage gets s["correct"] (the candidates, the shrink, the choice, D's prediction of the
sent command as 13 cm point and palm, both in the command row's heading frame and world, the IK errors, plan_same, the
landing at the stage's end); per episode --correct-out/data/<task>.npz (+ arm/) in data3's layout with the task label (D3
segments of the sent commands, sd_real's) and log/<task>.jsonl (sd_real's log plus the stage records); the files are
named <task>__<episodes> (one pair a process).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for _k in ("REACHWM_GOAL_REF", "REACHWM_APPROACH", "REACHWM_QFIX"):
    assert os.environ.get(_k) == "1", f"--correct: {_k}=1 (model D's switches) must be set before simple_bench starts"
assert os.environ.get("REACHWM_X0_REF", "0") == "0", "--correct: REACHWM_X0_REF must be off"
# REACHWM_FDVEL as the model was trained (2026-11-26, selfmodel_simpleheft_20261126): 1 for selfmodel_simpledata's D
# (SIMPLE, ScaleBFM: every run before), 0 for train3b's D (LocoManiLab; HEFT here). ModelD checks the model against it.
# selfmodel_simplesonicdata_20261126's D_S (SIMPLE, SONIC, in place, 5 cm/s; simpledata's recipe): 1.
MODEL_FDVEL = {"selfmodel_simpledata_20261126": "1", "selfmodel_train3b_20261125": "0", "selfmodel_simplesonicdata_20261126": "1"}
assert os.environ.get("REACHWM_FDVEL") in ("0", "1"), "--correct: REACHWM_FDVEL (0 or 1, the model's) must be set"
# the controllers --correct runs with (2026-11-26, selfmodel_simpleheft_20261126: HEFT added; its Motor is ScaleBFM's
# Motor subclass, so the recorder, the candidates' planning and the hand point are the same code; its reference delay in
# this body is 0 frames, as ScaleBFM's, sc_collect.DELAY_F). 2026-11-26, selfmodel_simplesonicdata_20261126: SONIC added
# (backends.sonic_onnx.SonicMotor keeps Motor's reach / _blend / reference logic and only changes how a reference reaches
# the controller, so the recorder, the candidates' planning and the hand point are again the same code; sd_real's record
# keeps its delay 0, a scalar no feature reads).
CONTROLLERS = ("scalebfm", "heft", "sonic")
sys.path[:0] = [str(EXP / "selfmodel_simpledata_20261126"), str(EXP / "reachwm_20261030")]
import sd_real as SDR  # noqa: E402  (its EpRec, sc_collect, sx_collect, d3_collect, lml_selfmodel)

SC, SX, D3, L = SDR.SC, SDR.SX, SDR.D3, SDR.L
TICKS = L.TICKS
ITERS, LAMBDA = 6, 1.0  # lml_tasks.ITERS, lml_tasks.LAMBDA
SHRINK_IK, SHRINK_STEPS = 0.02, 6  # lmlcmp2 (g)
RULES = ("R", "S", "dryR")
ID0 = 95_000_000  # the records' reset ids: + 1_000_000 * arm + 1000 * task + 100 * level + index
TASKS = SDR.TASKS


def clip_t(t: np.ndarray, side: str) -> np.ndarray:
    """lml_tasks.clip_t: the target kept inside the data's WIDE box (mirrored for the left hand)."""
    lo, hi = L.WIDE
    sd = 1.0 if side == "right" else -1.0
    y = np.clip(t[1] * sd, lo[1], hi[1]) * sd
    return np.r_[np.clip(t[0], lo[0], hi[0]), y, np.clip(t[2], lo[2], hi[2])]


def gram(r6: np.ndarray) -> np.ndarray:
    a, b = np.asarray(r6[:3], float), np.asarray(r6[3:6], float)
    c0 = a / np.linalg.norm(a)
    b = b - (c0 @ b) * c0
    c1 = b / np.linalg.norm(b)
    return np.c_[c0, c1, np.cross(c0, c1)]


class _PlanOnly(Exception):
    pass


def r3(v) -> list:
    return np.round(np.asarray(v, float), 5).tolist()


# -- the recorder ---------------------------------------------------------------------------------------------------
class CorrRec(SDR.EpRec):
    """sd_real's EpRec (task label) plus Motor.reach_joints: a joint-path reach is recorded with its reference end."""

    def __init__(self, env, a):
        super().__init__(env, "task", a)
        m = self.mot
        rj0 = m.reach_joints
        self.pending = None  # (goal palm, goal R) of the joint reach simple_bench is about to send

        def reach_joints(side, path, joint_idx, **kw):
            n = self.n_reach
            self.n_reach += 1
            if n >= 2 or side != self.side:
                return rj0(side, path, joint_idx, **kw)
            f_cmd = self.last()
            x = self.rows[f_cmd]
            open_f = self.walk_end_f if self.walk_end_f is not None else f_cmd
            cap, end_p, end_R = joint_plan(m, side, path, joint_idx, kw.get("speed"))
            out = rj0(side, path, joint_idx, **kw)
            rc = self.rec("pre" if n == 0 else "entry", n, x, f_cmd, open_f, end_p, end_R, {"speed": kw.get("speed")}, cap, 0.0)
            self.reaches.append(rc)
            return out

        m.reach_joints = reach_joints


def joint_plan(m, side: str, path, joint_idx, speed):
    """Motor.reach_joints' reference without moving: (cap with the start and end poses and the duration, the end's palm
    point, the end's wrist rotation). The same arithmetic as reach_joints (the end is joints_at(1.0): the path's last
    row, the start gap faded out)."""
    MO = sys.modules[type(m).__module__]
    cfg = m.cfg
    base = m.ref(m.t)
    P, idx = np.asarray(path, float), list(joint_idx)

    def jaw_of(q):
        j = base.joints.copy()
        j[idx] = q
        p, quat = m.kp(base.pos, base.quat, j)[MO.WRIST[side]]
        Rw = MO.R.mat(quat)
        return p + Rw @ MO.R.OFF, Rw

    jaw = np.array([jaw_of(q)[0] for q in P])
    length = float(np.sum(np.linalg.norm(np.diff(jaw, axis=0), axis=1)))
    total = max(cfg.reach_min_s, length / (speed or cfg.reach_speed))
    j_end = base.joints.copy()
    j_end[idx] = P[-1]
    end = MO.Pose(np.asarray(base.pos, float).copy(), np.asarray(base.quat, float).copy(), j_end, np.zeros(29))
    p_end, R_end = jaw_of(P[-1])
    return dict(poses=[base, end], durs=[total]), p_end, R_end


# -- model D ----------------------------------------------------------------------------------------------------------
class ModelD:
    """D rebuilt with train.load_model (its normalisation is in its state; the template only gives the shapes)."""

    def __init__(self, model_dir: Path, template: Path):
        import torch

        import data as DT
        import train as TR

        torch.set_num_threads(1)
        self.torch, self.DT, self.M = torch, DT, TR.M
        model_dir = Path(model_dir)
        assert model_dir.name == "E3_ensemble_f1", model_dir
        fd = [v for k, v in MODEL_FDVEL.items() if k in str(model_dir)]
        assert len(fd) == 1 and fd[0] == os.environ["REACHWM_FDVEL"], f"--correct: {model_dir} needs REACHWM_FDVEL={fd}"
        base = DT.load(str(template))
        self.m = TR.load_model("E3_ensemble", 1.0, base, "cpu", root=model_dir.parent)
        self.dir = str(model_dir)

    def z_of(self, segs: list[dict]) -> dict:
        z, _ = D3.pack(segs, SDR.SCALARS)  # sc_collect.save_chunk's arrays, then data.load's defaults and derive
        z["d_this"] = np.stack([s["d_this"] for s in segs])
        z["d_prev"] = np.stack([s["d_prev"] for s in segs])
        n = len(z["n_steps"])
        z.setdefault("walk", np.zeros((n, 3), np.float32))  # data.load's defaults (REACHWM_APPROACH)
        z.setdefault("target_b", z["target"])
        self.DT.derive(z)
        return z

    def predict(self, z: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        GD = self.M.tensors(z, "cpu")
        with torch.no_grad():
            o = self.m.predict(self.M.sub(GD, torch.tensor([0]), self.m.step))  # models.predict_batched's call
        return (o["pos"][0].numpy().astype(np.float64), o["pos_std"][0].numpy().astype(np.float64),
                o["rot6"][0].numpy().astype(np.float64))  # fmt: skip


# -- the switch ------------------------------------------------------------------------------------------------------
class Corrector:
    def __init__(self, a):
        assert a.correct_rule in RULES, a.correct_rule
        self.a = a
        self.rule, self.point = a.correct_rule, a.correct_point
        self.stages = tuple(s for s in a.correct_stages.split(",") if s)
        self.D = ModelD(Path(a.correct), Path(a.correct_template))
        self.out = Path(a.correct_out)
        for sub in ("data", "log"):
            (self.out / sub).mkdir(parents=True, exist_ok=True)
        self.rec: CorrRec | None = None
        self.segs_all: list[dict] = []
        self.stage_recs: list[dict] = []
        print(f"--correct: model {self.D.dir}, rule {self.rule}, point {self.point}, stages {self.stages}, out {self.out}", flush=True)

    # frames and the heading frame of the command row
    def _frame(self, f: int):
        x = self.rec.rows[f]
        p = x["root"][:3].astype(float)
        yaw = L.yaw_of(L._qmat(x["root"][3:].astype(float)))
        return x, p, yaw, L.rot_z(yaw)

    @staticmethod
    def to_h(v_w, p, Rh) -> np.ndarray:
        v = np.asarray(v_w, float)
        return np.r_[(Rh.T @ np.r_[v[:2] - p[:2], 0.0])[:2], v[2]]

    @staticmethod
    def from_h(t_h, p, Rh) -> np.ndarray:
        t = np.asarray(t_h, float)
        return np.r_[p[:2] + (Rh @ np.r_[t[:2], 0.0])[:2], t[2]]

    def features(self, f_cmd: int, rc: dict) -> dict:
        """The candidate's one-segment dataset (Models._z of lmlcmp on SIMPLE's recorder)."""
        rec = self.rec
        rows = rec.rows
        rc = dict(rc, f_label=f_cmd + TICKS, settled_end=False, settle_s=-1.0)
        fake = {i: rows[i] for i in range(f_cmd - 30, f_cmd + 1) if i in rows}
        fake.update({i: rows[f_cmd] for i in range(f_cmd + 1, f_cmd + TICKS + 1)})
        ks = sorted(k for k in rec.frames if k <= f_cmd)
        F = np.array([rec.frames[k] if k in rec.frames else rec.frames[max(j for j in ks if j < k)] for k in range(ks[-1] + 1)])
        F = np.concatenate([F, np.repeat(F[-1:], TICKS + 2, 0)])
        walked = rec.walk0 is not None
        winfo = dict(asked=None, real=None, dev_cm=0.0, plan=None)
        if walked:
            winfo["asked"] = [float(rec.walk_asked[0]), float(rec.walk_asked[1]), 0.0]
            f1 = rec.reaches[0]["f_cmd"] if rec.reaches else f_cmd
            x0, x1 = rows[rec.walk0], rows[f1]
            p0, y0 = x0["root"][:2].astype(float), L.yaw_of(L._qmat(x0["root"][3:]))
            dp = L.rot_z(y0).T @ np.r_[x1["root"][:2] - p0, 0.0]
            winfo["real"] = [float(dp[0]), float(dp[1]), D3.wrap(L.yaw_of(L._qmat(x1["root"][3:])) - y0)]
        info = dict(fell=None, lost=None, why="", walked=walked, walk=winfo, delay_f=0)
        segs = D3.segments(fake, info, F, [rc], 0, rec.h.nfk, rec.rig)
        assert len(segs) == 1, "features: the placeholder segment was not built"
        rl = [None] * rc["chain"] + [rc]  # add_fields / augment_b find the reach by chain
        SX.add_fields(segs, rl)
        D3.augment_b(segs, rl)
        for s in segs:
            s["pert_rms_deg"], s["walk_task"] = 0.0, int(walked)
        return self.D.z_of(segs)

    def plan(self, side, palm_w, rot_w, kw: dict):
        """Motor.reach's plan for this command from the current state, stopped inside _blend (nothing moves)."""
        m = self.rec.mot
        got: dict = {}

        def stop(poses, durations):
            got["poses"], got["durs"] = list(poses), list(durations)
            raise _PlanOnly

        had = "_blend" in m.__dict__
        saved = m.__dict__.get("_blend")
        m._blend = stop
        t0 = time.time()
        try:
            type(m).reach(m, side, np.asarray(palm_w, float), np.asarray(rot_w, float), **kw)
        except _PlanOnly:
            pass
        finally:
            if had:
                m._blend = saved
            else:
                del m._blend
        assert "poses" in got, "plan: Motor.reach returned before its _blend"
        return got, time.time() - t0

    def evaluate(self, n: int, f_cmd: int, open_f: int, palm_w, rot_w, kw: dict) -> dict:
        """One candidate: its plan, reach record, features and D's prediction (13 cm point and palm, heading frame)."""
        rec = self.rec
        x, p, yaw, Rh = self._frame(f_cmd)
        kwr = dict(kw, servo=False)
        cap, plan_s = self.plan(rec.side, palm_w, rot_w, kwr)
        rc = rec.rec("pre" if n == 0 else "entry", n, x, f_cmd, open_f, palm_w, rot_w, kwr, cap, plan_s)
        z = self.features(f_cmd, rc)
        pos, std, r6 = self.D.predict(z)
        Rp = gram(r6)
        palm = pos + Rp @ (rec.h.off - L.HAND_OFF)
        hk = rec.h.nfk.fk_world(*_root_of(rec.h, cap["poses"][-1]))
        palm_last = hk[rec.side][0] + hk[rec.side][1] @ (rec.h.off - L.HAND_OFF)
        return dict(rc=rc, z=z, pos=pos, std=std, rot6=r6, palm=palm, ik13=float(rc["hands"][rec.side]["ik_err"]),
                    ik_palm=float(np.linalg.norm(palm_last - np.asarray(palm_w, float))), plan_s=plan_s)  # fmt: skip

    # -- the straight reaches ----------------------------------------------------------------------------------------
    def command(self, stage: str, goal_w, rot_w, pz: float, kw: dict):
        """-> (the palm command to send, the stage record). kw: simple_bench's planner options of this reach."""
        rec = self.rec
        side = rec.side
        n = rec.n_reach
        assert stage == ("pregrasp" if n == 0 else "entry"), (stage, n)
        f_cmd = rec.last()
        open_f = (rec.walk_end_f if rec.walk_end_f is not None else f_cmd) if n == 0 else rec.reaches[-1]["f_end"]
        x, p, yaw, Rh = self._frame(f_cmd)
        goal_w, rot_w = np.asarray(goal_w, float), np.asarray(rot_w, float)
        lever = rot_w @ (L.HAND_OFF - rec.h.off)  # palm -> 13 cm point of the commanded pose
        gh13, ghp = self.to_h(goal_w + lever, p, Rh), self.to_h(goal_w, p, Rh)
        g_pt = ghp if self.point == "palm" else gh13
        kwp = dict(kw, pelvis_z=None if pz < 0 else float(pz))
        self.goal_w = goal_w
        tic = time.time()
        t, evs, shr = gh13.copy(), [], []
        shrink_on = self.rule in ("R", "dryR")
        for it in range(ITERS + 1):
            if shrink_on and it > 0:
                t, sh = self.shrink(n, f_cmd, open_f, gh13, t, lever, p, Rh, rot_w, kwp)
                shr.append(dict(sh, it=it))
            palm_c = goal_w.copy() if np.array_equal(t, gh13) else self.from_h(t, p, Rh) - lever
            try:
                ev = self.evaluate(n, f_cmd, open_f, palm_c, rot_w, kwp)
            except (np.linalg.LinAlgError, ValueError) as e:  # this candidate cannot be planned
                evs.append(dict(t=t.copy(), l=math.inf, err=f"{type(e).__name__}: {str(e)[:120]}"))
                break
            pt = ev["palm"] if self.point == "palm" else ev["pos"]
            ev.update(t=t.copy(), palm_cmd=palm_c, l=float(((pt - g_pt) ** 2).sum() + LAMBDA * (ev["std"] ** 2).sum()))
            evs.append(ev)
            t = clip_t(t + (g_pt - pt), side)
        bi = min(range(len(evs)), key=lambda i: evs[i]["l"])  # the first least
        best = evs[bi]
        fallback = "rc" not in best
        sent_i = 0 if (self.rule == "dryR" or fallback) else bi
        if "rc" not in evs[sent_i]:  # the goal itself could not be planned: the reach is sent as it is, unrecorded here
            sent = None
        else:
            sent = evs[sent_i]
        cmd = goal_w.copy() if (sent_i == 0 or sent is None) else evs[sent_i]["palm_cmd"].copy()
        srec = dict(rule=self.rule, point=self.point, stage=stage, chain=n, f_cmd=f_cmd, yaw=yaw, p_cmd=r3(p),
                    goal_w=r3(goal_w), goal_h13=r3(gh13), goal_hp=r3(ghp), best=bi, sent=sent_i, model_fallback=fallback,
                    think_s=round(time.time() - tic, 3),
                    cands=[dict(t=r3(e["t"]), l=e["l"], pos=r3(e["pos"]) if "pos" in e else None,
                                palm=r3(e["palm"]) if "palm" in e else None, std=r3(e["std"]) if "std" in e else None,
                                ik13_cm=round(100 * e["ik13"], 4) if "ik13" in e else None,
                                ik_palm_cm=round(100 * e["ik_palm"], 4) if "ik_palm" in e else None, err=e.get("err"))
                           for e in evs],
                    shrink=shr if shrink_on else None, cmd_w=r3(cmd), corr_cm=round(100 * float(np.linalg.norm(cmd - goal_w)), 4),
                    choice_corr_cm=round(100 * float(np.linalg.norm(best["t"] - gh13)), 4) if "t" in best else None)  # fmt: skip
        if sent is not None:
            srec.update(self.pred_record(sent, p, Rh, ghp, gh13))
            srec["_rc"] = sent["rc"]  # compared with the executed plan in landed(); popped there
        return cmd, srec

    def shrink(self, n, f_cmd, open_f, gh13, t, lever, p, Rh, rot_w, kwp):
        """lmlcmp2 (g): IK error > SHRINK_IK -> the largest fraction a of t - gh13 with IK error <= SHRINK_IK."""

        def ik(tt: np.ndarray) -> float:
            palm_c = self.goal_w.copy() if np.array_equal(tt, gh13) else self.from_h(tt, p, Rh) - lever
            try:
                return self.evaluate_ik(n, f_cmd, open_f, palm_c, rot_w, kwp)
            except (np.linalg.LinAlgError, ValueError):
                return math.inf

        e1 = ik(t)
        sh = dict(ik0_cm=100 * e1 if np.isfinite(e1) else None, corr0_cm=100 * float(np.linalg.norm(t - gh13)), alpha=1.0,
                  shrunk=False, n_plans=1)  # fmt: skip
        if e1 <= SHRINK_IK:
            return t, sh
        lo, hi, k = 0.0, 1.0, 1
        for _ in range(SHRINK_STEPS):
            mid = 0.5 * (lo + hi)
            k += 1
            if ik(gh13 + mid * (t - gh13)) <= SHRINK_IK:
                lo = mid
            else:
                hi = mid
        t2 = gh13 + lo * (t - gh13)
        sh.update(alpha=lo, shrunk=True, n_plans=k, corr_cm=100 * float(np.linalg.norm(t2 - gh13)))
        return t2, sh

    def evaluate_ik(self, n, f_cmd, open_f, palm_w, rot_w, kw) -> float:
        rec = self.rec
        x = rec.rows[f_cmd]
        kwr = dict(kw, servo=False)
        cap, plan_s = self.plan(rec.side, palm_w, rot_w, kwr)
        rc = rec.rec("pre" if n == 0 else "entry", n, x, f_cmd, open_f, palm_w, rot_w, kwr, cap, plan_s)
        return float(rc["hands"][rec.side]["ik_err"])

    def pred_record(self, ev: dict, p, Rh, ghp, gh13) -> dict:
        pw = self.from_h(ev["palm"], p, Rh)
        p13w = self.from_h(ev["pos"], p, Rh)
        return dict(pred_h13=r3(ev["pos"]), pred_hp=r3(ev["palm"]), pred_std=r3(ev["std"]), pred_rot6=r3(ev["rot6"]),
                    pred_w13=r3(p13w), pred_wp=r3(pw), pred_miss_palm_cm=round(100 * float(np.linalg.norm(ev["palm"] - ghp)), 4),
                    pred_miss13_cm=round(100 * float(np.linalg.norm(ev["pos"] - gh13)), 4),
                    ik13_cm=round(100 * ev["ik13"], 4), ik_palm_cm=round(100 * ev["ik_palm"], 4),
                    in_table=in_table(ev["rc"], self.rec.side), x0=r3(ev["z"]["x0"][0]), u=r3(ev["z"]["u"][0]))  # fmt: skip

    # -- the joint-path reach (J's pregrasp): recorded only --------------------------------------------------------
    def joint(self, stage: str, goal_w, rot_w, path, joint_idx, speed) -> dict:
        rec = self.rec
        n = rec.n_reach
        f_cmd = rec.last()
        open_f = rec.walk_end_f if rec.walk_end_f is not None else f_cmd
        x, p, yaw, Rh = self._frame(f_cmd)
        cap, end_p, end_R = joint_plan(rec.mot, rec.side, path, joint_idx, speed)
        rc = rec.rec("pre" if n == 0 else "entry", n, x, f_cmd, open_f, end_p, end_R, {"speed": speed}, cap, 0.0)
        z = self.features(f_cmd, rc)
        pos, std, r6 = self.D.predict(z)
        palm = pos + gram(r6) @ (rec.h.off - L.HAND_OFF)
        goal_w, rot_w = np.asarray(goal_w, float), np.asarray(rot_w, float)
        ghp = self.to_h(goal_w, p, Rh)
        gh13 = self.to_h(goal_w + rot_w @ (L.HAND_OFF - rec.h.off), p, Rh)
        ev = dict(pos=pos, std=std, rot6=r6, palm=palm, ik13=float(rc["hands"][rec.side]["ik_err"]), ik_palm=0.0, rc=rc, z=z)
        srec = dict(rule="joint", point=self.point, stage=stage, chain=n, f_cmd=f_cmd, yaw=yaw, p_cmd=r3(p), goal_w=r3(goal_w),
                    goal_h13=r3(gh13), goal_hp=r3(ghp), ref_end_w=r3(end_p), ref_end_h=r3(self.to_h(end_p, p, Rh)),
                    ref_gap_cm=round(100 * float(np.linalg.norm(end_p - goal_w)), 4), dur_s=round(float(cap["durs"][0]), 4))  # fmt: skip
        srec.update(self.pred_record(ev, p, Rh, ghp, gh13))
        srec["_rc"] = rc
        return srec

    # -- after the reach ----------------------------------------------------------------------------------------
    def landed(self, srec: dict, palm_w, R_w) -> None:
        """The stage's landing (simple_bench's measure after POST_S; record only) in the command row's heading frame, and
        whether the executed plan is the evaluated one."""
        rc0 = srec.pop("_rc", None)
        p, yaw = np.asarray(srec["p_cmd"], float), float(srec["yaw"])
        Rh = L.rot_z(yaw)
        palm_w = np.asarray(palm_w, float)
        p13 = palm_w + np.asarray(R_w, float) @ (L.HAND_OFF - self.rec.h.off)
        lp, l13 = self.to_h(palm_w, p, Rh), self.to_h(p13, p, Rh)
        srec.update(land_hp=r3(lp), land_h13=r3(l13))
        if "pred_hp" in srec:
            srec.update(pred_err_palm_cm=round(100 * float(np.linalg.norm(lp - np.asarray(srec["pred_hp"]))), 4),
                        pred_err13_cm=round(100 * float(np.linalg.norm(l13 - np.asarray(srec["pred_h13"]))), 4))  # fmt: skip
        ex = self.rec.reaches[-1] if self.rec.reaches else None
        if rc0 is not None and ex is not None and ex["f_cmd"] == srec["f_cmd"]:
            srec["plan_same"] = bool(np.array_equal(ex["ref_q"], rc0["ref_q"]) and np.array_equal(ex["ref_pel"], rc0["ref_pel"])
                                     and ex["reach_s"] == rc0["reach_s"])  # fmt: skip
        else:
            srec["plan_same"] = None
        self.stage_recs.append({k: v for k, v in srec.items() if k not in ("cands", "x0", "u", "shrink")})


def _root_of(h, pose):
    """A Motor Pose as NativeFK.fk_world's arguments (the free joint's position, quat, joints)."""
    return np.asarray(pose.pos, float), np.asarray(pose.quat, float), np.asarray(pose.joints, float)


def in_table(rc: dict, side: str) -> bool | None:
    """The command target inside the reach table's box for its hand and pelvis height (sd_range's 'table' check)."""
    try:
        table = _table()
        hc = rc["hands"][side]
        return bool(D3.in_table(table["rows"][side][D3.pz_key(rc["pelvis_z"])], np.asarray(hc["cmd_target"], float)))
    except Exception as e:  # noqa: BLE001  (a record field only)
        return f"{type(e).__name__}: {e}"[:80]  # type: ignore[return-value]


_TAB: dict = {}


def _table() -> dict:
    if "t" not in _TAB:
        _TAB["t"] = json.loads(Path(D3.TABLE).read_text())
    return _TAB["t"]


# -- simple_bench's hooks ------------------------------------------------------------------------------------------
def install(SB, a) -> Corrector:
    """simple_bench (the running module) gets: a CorrRec on every episode's Env, the walk's legs recorded (sd_real's),
    and its episode() wrapped to save the episode's record. Returns the Corrector (a.corr)."""
    assert a.method == "direct", "--correct goes with --method direct"
    assert a.root == "odo" and a.controller in CONTROLLERS, f"--correct: {CONTROLLERS} with leg odometry (the models')"
    assert a.correct_out and a.correct_template, "--correct needs --correct-out and --correct-template"
    C = Corrector(a)
    E, GR = SB.E, SB.GR
    env_init0 = E.Env.__init__

    def env_init(self, *args, **kw):
        env_init0(self, *args, **kw)
        C.rec = CorrRec(self, a)
        C.stage_recs = []

    E.Env.__init__ = env_init
    open_walk0 = GR.open_walk

    def open_walk(m, forward, left, turn):
        r = C.rec
        r.walk_leg(float(forward), float(left))
        assert turn == 0.0, "simple_bench walks without a turn"
        out = open_walk0(m, forward, left, turn)
        r.walk_end_f = r.last()
        return out

    GR.open_walk = open_walk
    episode0 = SB.episode
    ti = {t: i for i, t in enumerate(TASKS)}

    def episode(ep, method, sb_a, models, rep=None):
        row = episode0(ep, method, sb_a, models, rep)
        rec = C.rec
        rid = ID0 + 1_000_000 * int(a.correct_arm_id) + 1000 * ti[ep["task"]] + 100 * int(ep["level"]) + int(ep["index"])
        segs, log = rec.finish(rid)
        log.update(task=ep["task"], level=ep["level"], index=ep["index"], success_45=row.get("success_45"),
                   success_official=row.get("success_official"), why=row.get("why"), palm_cm=row.get("palm_cm"),
                   rule=C.rule, point=C.point, model=C.D.dir, stages_correct=C.stage_recs,
                   stages=[{k: s.get(k) for k in ("stage", "err_cm", "err_deg", "ik_cm")} for s in row.get("stages", [])])  # fmt: skip
        C.segs_all.extend(segs)
        name = f"{ep['task']}__{a.episodes.replace(',', '_')[:40]}"  # one file pair a process (several may share --correct-out)
        with open(C.out / "log" / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(log, default=D3._json) + "\n")
        SC.save_chunk(C.segs_all, C.out / "data", name, rec.rig, SDR.SCALARS)
        row["correct_reset"] = rid
        return row

    SB.episode = episode
    return C
