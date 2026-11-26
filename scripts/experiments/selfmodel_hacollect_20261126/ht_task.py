"""selfmodel_hatask_20261126: HumanoidArena (HA) task episodes, A (direct) against R (lmlcmp3's `reachable` through the
controller's HA self model), P&PBox and DoubleDesk, ScaleBFM / HEFT / SONIC.

The episode up to the pre-grasps is hr_real's (selfmodel_hadata_20261126 revision 2a and its addendum, imported and run
as it is: hr_real.Episode's settle / walk / reach loop, hr_real.Look's looks and target rules, hr_real.FLOWS's look,
approach and perceived pre-grasps, the via rule, the stand clearance per controller, HEFT on gait_heft). Then
ha_bench's fixed post-reach sequence (its TASK RULES), with our walk and reach (d3_collect.plan_reach's whole-body IK,
hg_walk / gait_heft):
  ppbox       squeeze (both hands 2 cm into the perceived faces; 10 cm in from the pre-grasps), close both, hold 25, lift
              10 cm, look for "blue metal shelf" turning to the start heading +45 / +90 / +135 deg, the board (ha_bench's
              find_shelf: the highest level below the box's bottom), approach (stand rule, board centre >= 0.34 m ahead),
              the box's centre over the board centre with its bottom 3 cm over the board, open both, hold 50
  doubledesk  down 8 cm to the grasp point, close the right hand, hold 30, lift 15 cm, look for "bin" / "basket" (start
              heading +45 / +90 / +135 deg), approach (rim centre >= 0.40 m ahead, 0.20 m right), the grasp point 15 cm
              over the rim centre, open, hold 75
Every command waits for data3's settle rule first (as hr_real: up to 6 s, then it goes anyway). After the hands close,
every walk holds the upper body (joints 12+: waist and arms) at the reference row the walk starts from. The lift is the
executed squeeze / descend command raised (heading frame, rotation kept); the place / over-basket command is the lift
command moved by (where the perceived target is now - where the lift put the perceived grasp / box centre), rotation
kept, so R's corrections carry on in R and A's commands are ha_bench's.

The commands through the model in R: the pre-grasps (reach_no 1; P&PBox both hands, one at a time), the squeeze
(two-hand, reachable_two) and the descend (reach_no 2). Direct in both arms: the via points (reach_no 0), the lift (3),
the place / over-basket (4). R = ht_inv.reachable_one (lc3_run's block verbatim) / reachable_two, planned and predicted
at the command row as the drive will execute it (the chosen candidate's plan is executed; a candidate whose plan fails
ends the iteration; no candidate planned -> the goal). The model's features: lc3_run's Models._z on this episode's rows
and plan (d3_collect.segments of a placeholder label, pack, walk 0, target_b = target, data.derive), with the reach
fields hr_real's Episode.reach gives the real record.

Pairing: per seed A, then R, in the same process. R's looks before its first pre-grasp command replay A's colour images
(hr_real's --replay-looks, from A's looks of the same name); later looks are live. Both arms record a digest of the rows
and reference frames up to the first pre-grasp command row (and that command's goal): the consistency check.

The episode ends at the first of: HA's reward reaches 1 (raw_total >= 1), HA's fall detector (confirmed), the step
budget (--budget-x times the official one: P&PBox 1450, DoubleDesk 2000 steps); when the sequence ends first (done, a
target not seen, a plan that fails) the robot holds --hold steps more (ha_bench's default 100). d3's fall (0.45 m) and
lost (0.5 m) are recorded only. Simulator truth only scores: the reward, the fall detector, the object's pose every 10
steps and the grasp-and-lift (the object >= LIFT_DZ over its height after the reset for >= LIFT_N consecutive steps).

Records (per arm under <out>/<A|R>/: data/<name>.npz (+ arm/) as hr_real's, log/<name>.json, looks/): per command the
goal, the executed command, R's iterations (each candidate, its shrink, prediction, spread, loss), the IK errors; R's
queries (every plan's IK error and every prediction, keyed by the candidate) for ht_equiv.py; the executed candidate's
model input (x0, u) for the feature check.

    SX_DEPTH=1 SX_HA_TASK=ppbox HC_SCRIPT=ht_task.py REACHWM_FDVEL=1 bash hc_ha.sh <gpu> --controller scalebfm --out DIR \
        --seeds 380-389 --repeats 0 --arms A,R --model-root <models_X_s0> --model-template <npz>
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hr_real as HR  # noqa: E402  (hc_ha, sx_ha, ha_sim, ha_bench, perceive)
import ht_inv as INV  # noqa: E402

HC, SXH, ha_sim, B = HR.HC, HR.SXH, HR.ha_sim, HR.B
End = HR.End

TASK_ID = HR.TASK_ID
ARM_ID = {"A": 0, "R": 1}
RID0 = 96_000_000  # record ids: + 1e6 arm + 1e4 task + 10 seed + repeat (hr_real 95M, data3 91-94M)
BUDGET = {"ppbox": 1450, "doubledesk": 2000}  # ha_sim.TASKS' official step budgets
OBJ = {"ppbox": "box", "doubledesk": "object_l"}  # ha_bench's diag_step keys
LIFT_DZ, LIFT_N = 0.05, 50  # grasp-and-lift: m over the reset height, consecutive steps (1 s)
SQUEEZE_IN, LIFT = 0.02, {"ppbox": 0.10, "doubledesk": 0.15}
CLOSE_HOLD = {"ppbox": 25, "doubledesk": 30}
OPEN_HOLD = {"ppbox": 50, "doubledesk": 75}
PLACE_DZ = 0.03  # the box's bottom over the board
OVER_DZ = 0.15  # the grasp point over the basket rim
SEARCH_ABS = (45, 90, 135)  # deg from the episode's start heading
MODEL_NAMES = ("D",)
CHAIN_COL = 205  # data.py x0's chain column (lc3_run)


# -- the reach loop: hr_real's Episode with the task's end rules ----------------------------------------------------------
class TaskEp(HR.Episode):
    def __init__(self, h, rig, SX, D3, L, HW, task: str, budget: int, hands_fn):
        super().__init__(h, rig, SX, D3, L, HW)
        self.task, self.budget, self.hands_fn = task, int(budget), hands_fn
        self.success_at = self.ha_fell = None
        self.d3_fell = self.d3_lost = None
        self.carry = False
        self.phase, self.phases = "stand", [(0, "stand")]
        self.obj, self.obj_z0, self.lift_run, self.lifted_at, self.max_lift = [], None, 0, None, -np.inf
        self.hands_state, self.hand_events = (False, False), []
        self.set_hands(False, False)

    def set_phase(self, p: str) -> None:
        self.phase = p
        self.phases.append((self.f + 1, p))

    def set_hands(self, left: bool, right: bool) -> None:
        self.hands_fn(bool(left), bool(right))
        self.hands_state = (bool(left), bool(right))
        self.hand_events.append((self.f + 1, bool(left), bool(right)))

    def score(self, f: int) -> None:
        """Simulator truth, scoring only (nothing below is read by a command)."""
        h = self.h
        env = h.sim.env
        if isinstance(h, HR.HRSonic):
            raw = float(h.sim.reward)  # ha_sim.Sim.step computed it (HAHost.step calls it)
        else:
            raw = float(ha_sim.sev._extract_reward_info(env)["raw_total"])
        if raw >= 1.0 and self.success_at is None:
            self.success_at = f + 1
        if self.ha_fell is None and h.ha_fell_at is not None:
            self.ha_fell = int(h.ha_fell_at)
        p = env.scene[OBJ[self.task]].data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        if self.obj_z0 is None:
            self.obj_z0 = float(p[2])
        dz = float(p[2]) - self.obj_z0
        self.max_lift = max(self.max_lift, dz)
        self.lift_run = self.lift_run + 1 if dz >= LIFT_DZ else 0
        if self.lift_run >= LIFT_N and self.lifted_at is None:
            self.lifted_at = f + 1
        if f % 10 == 0:
            self.obj.append([f + 1] + p.round(4).tolist() + [round(raw, 4)])

    def tick(self) -> dict:
        """hr_real's tick with the task's end rules: success, HA's fall, the budget end the episode; d3's fall and lost
        are recorded only; the motion never 'ends' (the last frame is held)."""
        D3, L, h = self.D3, self.L, self.h
        self.f += 1
        f = self.f
        if f >= len(self.frames):  # the last frame held: appended, so every row keeps its own frame (frames[f] = row f's)
            self.frames = np.concatenate([self.frames, np.repeat(self.frames[-1:], 500, 0)])
        h.push(self.frames[f])
        h.step()
        if self.delay is None:
            self.delay = D3.ref_delay_f(h)
        x = self.rows[f] = h.sample()
        self.rig.record(x)
        self.x = x
        if self.odo is not None:
            self.odo.update(h, x)
        self.note_contacts(f)
        h.sim.k = f + 1
        self.score(f)
        if float(x["root"][2]) < L.FALL_Z and self.d3_fell is None:
            self.d3_fell = f
        if self.anchor is not None and not self.walking and self.d3_lost is None:
            fr = self.frames[min(max(f - self.delay, self.anchor[2]), len(self.frames) - 1)]
            if float(np.linalg.norm((x["root"][:2] - self.anchor[0]) - (fr[:2] - self.anchor[1]))) > L.LOST_M:
                self.d3_lost = f
        if self.success_at is not None:
            self.why = "success"
            raise End("success")
        if self.ha_fell is not None:
            self.why = "fell"
            raise End("fell")
        if f + 1 >= self.budget:
            self.why = "budget"
            raise End("budget")
        return x

    def walk_to(self, xy, yaw, tag: str) -> dict:
        """hr_real's walk; carrying: the upper body (joints 12+) held at this row's reference."""
        if not self.carry:
            return super().walk_to(xy, yaw, tag)
        hold = np.asarray(self.frames[self.f][7 + 12 :]).copy()
        HW0, hwf0 = self.HW, HR.heft_walk_frames

        class Held:
            def __getattr__(self, k):
                return getattr(HW0, k)

            @staticmethod
            def walk_frames(*args, **kw):
                wf = HW0.walk_frames(*args, **kw)
                wf[:, 7 + 12 :] = hold
                return wf

        def heft_held(*args, **kw):
            wf = hwf0(*args, **kw)
            wf[:, 7 + 12 :] = hold
            return wf

        self.HW, HR.heft_walk_frames = Held(), heft_held
        try:
            w = super().walk_to(xy, yaw, tag)
        finally:
            self.HW, HR.heft_walk_frames = HW0, hwf0
        w["carry"] = True
        return w

    def reach(self, cmd: dict, extra: dict, plan=None) -> dict:
        """hr_real's Episode.reach with the plan given (R's chosen candidate, planned on this row) or made here."""
        D3, L = self.D3, self.L
        f_ready, pre_ok = self.ready
        x, f = self.x, self.f
        assert f == f_ready, (f, f_ready)
        if plan is None:
            try:
                plan = D3.plan_reach(self.h, x, self.frames[f], cmd)
            except (np.linalg.LinAlgError, ValueError) as e:
                self.why = f"ik {type(e).__name__}: {e}"
                raise End("ik")
        blend, last, rec = plan
        self.plan_s += rec["plan_s"]
        rec.update(slot=len(self.reaches), chain=len(self.reaches), f_cmd=f, f_end=f + len(blend), pre_wait_s=self.wait_s,
                   d_prev=self.cur_delta.copy(), f_label=None, settled_end=False, settle_s=-1.0, pre_settled=bool(pre_ok), **extra)  # fmt: skip
        self.frames = np.concatenate([self.frames[: f + 1], blend, L.hold(last, D3.HOLD_S)])
        self.anchor = (x["root"][:2].copy(), self.frames[f, :2].copy(), f)
        self.cur_delta = rec["delta"]
        self.reaches.append(rec)
        fl, ok = self.settle(rec["f_end"] + self.delay, grid0=f)
        rec.update(f_label=fl, settled_end=ok, settle_s=(fl - rec["f_end"]) / D3.FPS)
        self.wait_s = 0.0
        return rec

    def ready_now(self) -> None:
        """data3's wait before a command that follows a hold (close / open): from this row."""
        self.settle(self.f)

    def hold_n(self, n: int) -> None:
        for _ in range(int(n)):
            self.tick()


# -- the model (R) ------------------------------------------------------------------------------------------------------
class Model:
    """One E3_ensemble (train.load_model: the normalisation is part of the state; the template only sizes the rebuild),
    on the CPU with one thread, as lc3_run's Models. Every prediction is recorded (the equivalence check's queries)."""

    def __init__(self, root: Path, template: Path):
        import torch

        exp = Path(os.environ["HT_REACHWM_REPO"]) / "scripts" / "experiments"
        sys.path[:0] = [str(exp / "reachwm_20261030"), str(exp / "selfwm_20261029")]
        for k in ("data", "models", "train", "env", "fk", "wm"):
            assert k not in sys.modules or str(exp) in str(getattr(sys.modules[k], "__file__", "")), f"{k} already imported from elsewhere"
        import data as DT
        import models as M
        import train as TR

        for mod in (DT, M, TR):
            assert str(exp) in mod.__file__, mod.__file__
        torch.set_num_threads(1)
        self.torch, self.DT, self.M = torch, DT, M
        self.files = {k: dict(file=sys.modules[k].__file__, md5=hashlib.md5(Path(sys.modules[k].__file__).read_bytes()).hexdigest()[:8])
                      for k in ("data", "models", "train", "env", "fk", "wm") if k in sys.modules}  # fmt: skip
        self.flags = dict(FDVEL=bool(DT.FDVEL), GOAL_REF=bool(DT.GOAL_REF), APPROACH=bool(DT.APPROACH), QFIX=bool(DT.QFIX), X0_REF=bool(DT.X0_REF))
        base = DT.load(str(template))
        self.m = TR.load_model("E3_ensemble", 1.0, base, "cpu", root=Path(root))
        mp = Path(root) / "E3_ensemble_f1" / "model.pt"
        self.info = dict(root=str(root), model_md5=hashlib.md5(mp.read_bytes()).hexdigest()[:8], template=str(template), flags=self.flags, files=self.files)
        self.queries = []  # (key t, pos, std) per prediction, for ht_equiv
        self.key = {}  # id(z) -> the candidate's target (feat sets it)

    def _row(self, GD, i: int):
        torch = self.torch
        with torch.no_grad():
            o = self.m.predict(self.M.sub(GD, torch.tensor([i]), False))
        return o["pos"][0].numpy().astype(np.float64), o["pos_std"][0].numpy().astype(np.float64)

    def predict(self, z: dict, names=MODEL_NAMES) -> dict:
        GD = self.M.tensors(z, "cpu")
        assert float(GD["x0"][0, CHAIN_COL]) == float(z["chain"][0]), "x0's chain column moved"
        pos, std = self._row(GD, 0)
        self.queries.append(dict(t=self.key.get(id(z)), pos=pos.tolist(), std=std.tolist()))
        return {"D": (pos, std)}

    def predict2(self, z: dict) -> dict:
        """Each row of a two-hand plan's segments alone (a one-hand prediction per hand)."""
        GD = self.M.tensors(z, "cpu")
        out = {}
        for i in range(len(z["n_steps"])):
            assert float(GD["x0"][i, CHAIN_COL]) == float(z["chain"][i]), "x0's chain column moved"
            side = "right" if int(z["side"][i]) == 1 else "left"
            out[side] = self._row(GD, i)
        k = self.key.get(id(z))
        self.queries.append(dict(t=k, pos={s: v[0].tolist() for s, v in out.items()}, std={s: v[1].tolist() for s, v in out.items()}))
        return out


def walk_info(ep: TaskEp) -> dict:
    """hr_real's info()['walk'] as the episode's record will have it: the last walk before the first reach's command
    (before this row while no reach is made)."""
    f0 = ep.reaches[0]["f_cmd"] if ep.reaches else ep.f + 1
    before = [w for w in ep.walks if w["f"] < f0]
    w = before[-1] if before else None
    return dict(asked=w["asked"] if w else None, real=w.get("real") if w else None, dev_cm=w["dev_cm"] if w else 0.0,
                plan=w["plan"] if w else None)  # fmt: skip


def features(ep: TaskEp, plan, extra: dict, DT) -> dict:
    """lc3_run's Models._z on this episode: the reach fields as TaskEp.reach will set them, a placeholder label one
    10 Hz row after the command, d3_collect.segments + pack, walk 0, target_b = target, data.derive."""
    D3 = ep.D3
    blend, last, rec = plan
    f = ep.f
    rc = dict(rec)
    rc.update(slot=len(ep.reaches), chain=len(ep.reaches), f_cmd=f, f_end=f + len(blend), pre_wait_s=ep.wait_s,
              d_prev=ep.cur_delta.copy(), f_label=None, settled_end=False, settle_s=-1.0, pre_settled=bool(ep.ready[1]), **extra)  # fmt: skip
    rc.update(f_label=f + D3.TICKS, settled_end=False, settle_s=-1.0)
    cand = np.concatenate([ep.frames[: f + 1], blend])
    fake = {i: ep.rows[i] for i in range(max(min(ep.rows), f - 30), f + 1)}
    fake.update({i: ep.rows[f] for i in range(f + 1, f + D3.TICKS + 1)})
    info3 = dict(fell=None, lost=None, walk=walk_info(ep), walked=bool(ep.walks), delay_f=ep.delay)
    segs = D3.segments(fake, info3, cand, [rc], 0, ep.h.pl, ep.rig)
    assert len(segs) == len(rec["hands"]), "features: the placeholder segments were not built"
    z, _ = D3.pack(segs)
    z["walk"] = np.zeros((len(segs), 3), np.float32)
    z["target_b"] = z["target"]
    DT.derive(z)
    return z


class OneDr:
    """lc3_run's driver interface for one hand: plan(x, ref, c) -> (blend, last, rec) with rec['ik_err'], c = {'target'}
    placed into this command's hand (every IK error recorded, keyed by the candidate)."""

    def __init__(self, ep: TaskEp, cmd: dict, side: str):
        self.ep, self.cmd, self.side, self.ik = ep, cmd, side, []

    def cmd_of(self, c: dict) -> dict:
        hc = dict(self.cmd["hands"][self.side], target=np.asarray(c["target"], float).copy())
        return dict(self.cmd, hands={self.side: hc})

    def plan(self, x, ref, c):
        try:
            blend, last, rec = self.ep.D3.plan_reach(self.ep.h, x, ref, self.cmd_of(c))
        except (np.linalg.LinAlgError, ValueError) as e:
            self.ik.append(dict(t=np.asarray(c["target"], float).tolist(), ik=None, err=type(e).__name__))
            raise
        e = float(rec["hands"][self.side]["ik_err"])
        self.ik.append(dict(t=np.asarray(c["target"], float).tolist(), ik=e))
        return blend, last, dict(rec, ik_err=e)


class TwoDr:
    """The same for a two-hand command: c2 = {side: {'target'}}; rec['ik_err'] = {side: error}."""

    def __init__(self, ep: TaskEp, cmd: dict):
        self.ep, self.cmd, self.ik = ep, cmd, []

    def cmd_of(self, c2: dict) -> dict:
        return dict(self.cmd, hands={s: dict(self.cmd["hands"][s], target=np.asarray(c2[s]["target"], float).copy()) for s in self.cmd["hands"]})

    def plan(self, x, ref, c2):
        key = {s: np.asarray(c2[s]["target"], float).tolist() for s in c2}
        try:
            blend, last, rec = self.ep.D3.plan_reach(self.ep.h, x, ref, self.cmd_of(c2))
        except (np.linalg.LinAlgError, ValueError) as e:
            self.ik.append(dict(t=key, ik=None, err=type(e).__name__))
            raise
        e = {s: float(rec["hands"][s]["ik_err"]) for s in rec["hands"]}
        self.ik.append(dict(t=key, ik=e))
        return blend, last, dict(rec, ik_err=e)


def strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "ik_err"}


def run_r(ep: TaskEp, MD: Model, cmd: dict, extra: dict) -> tuple[tuple | None, dict]:
    """R on one command (one hand: reachable_one; two hands: reachable_two). Returns (the plan to execute or None for
    the goal, the record)."""
    sides = list(cmd["hands"])
    rows, f, frames = ep.rows, ep.f, ep.frames
    q0 = len(MD.queries)
    t0 = time.time()
    if len(sides) == 1:
        side = sides[0]
        gh = np.asarray(cmd["hands"][side]["target"], float).copy()
        dr = OneDr(ep, cmd, side)
        c = dict(target=gh.copy())

        def feat(cc):
            pl_ = dr.plan(rows[f], frames[f], cc)  # LinAlgError / ValueError here: the plan failed (lc3_run's)
            try:
                z = features(ep, (pl_[0], pl_[1], strip(pl_[2])), extra, MD.DT)
            except Exception as e:  # anything after the plan is a fault of this code, never a "plan failed" (lc3_run's)
                raise RuntimeError(f"features: {type(e).__name__}: {e}") from e
            MD.key[id(z)] = np.asarray(cc["target"], float).tolist()
            return (pl_[0], pl_[1], strip(pl_[2])), z

        best, bi, evs, shr = INV.reachable_one(dr, rows, f, frames, c, gh, feat, MD, side)
        inv = [dict(t=e["t"].tolist(), pos=e["pos"].tolist() if "pos" in e else None, std=e["std"].tolist() if "std" in e else None,
                    l=e["l"], err=e.get("err")) for e in evs]  # fmt: skip
        goal = dict(side=side, gh=gh.tolist())
    else:
        gh2 = {s: np.asarray(cmd["hands"][s]["target"], float).copy() for s in sides}
        dr = TwoDr(ep, cmd)
        c2 = {s: dict(target=gh2[s].copy()) for s in sides}

        def feat2(cc):
            pl_ = dr.plan(rows[f], frames[f], cc)
            try:
                z = features(ep, (pl_[0], pl_[1], strip(pl_[2])), extra, MD.DT)
            except Exception as e:
                raise RuntimeError(f"features: {type(e).__name__}: {e}") from e
            MD.key[id(z)] = {s: np.asarray(cc[s]["target"], float).tolist() for s in cc}
            return (pl_[0], pl_[1], strip(pl_[2])), z

        best, bi, evs, shr = INV.reachable_two(dr, rows, f, frames, c2, gh2, feat2, MD)
        inv = [dict(t={s: v.tolist() for s, v in e["t"].items()}, pos={s: v.tolist() for s, v in e["pos"].items()} if "pos" in e else None,
                    std={s: v.tolist() for s, v in e["std"].items()} if "std" in e else None, l=e["l"], err=e.get("err")) for e in evs]  # fmt: skip
        goal = dict(sides=sides, gh={s: v.tolist() for s, v in gh2.items()})
    r = dict(goal, n_iter=len(evs), inv=inv, inv_best=bi, shrink=shr, ik_queries=dr.ik, pred_queries=MD.queries[q0:],
             think_s=round(time.time() - t0, 3))  # fmt: skip
    if "plan" not in best:
        r["model_fallback"] = True
        return None, r
    if len(sides) == 1:
        r["t_exec"] = best["t"].tolist()
        r["pred_miss_cm"] = 100 * float(np.linalg.norm(best["pos"] - gh))
        r["corr_cm"] = 100 * float(np.linalg.norm(best["t"] - gh))
    else:
        r["t_exec"] = {s: v.tolist() for s, v in best["t"].items()}
        r["pred_miss_cm"] = {s: 100 * float(np.linalg.norm(best["pos"][s] - gh2[s])) for s in sides}
        r["corr_cm"] = {s: 100 * float(np.linalg.norm(best["t"][s] - gh2[s])) for s in sides}
    r["feat_x0"], r["feat_u"] = best["z"]["x0"].tolist(), best["z"]["u"].tolist()
    return best["plan"], r


# -- commands ------------------------------------------------------------------------------------------------------------
def two(cmds: dict) -> dict:
    """d3's two-hand command from two one-hand commands (right first, as d3_collect.sample_command orders them)."""
    return dict(mode="both", pelvis_z=None, hands={s: cmds[s]["hands"][s] for s in ("right", "left") if s in cmds})


def shifted(cmd_exec: dict, dp: dict) -> dict:
    """A command with each hand's heading-frame target moved by dp[side] (rotation kept)."""
    hands = {s: dict(hc, target=np.asarray(hc["target"], float) + np.asarray(dp[s], float)) for s, hc in cmd_exec["hands"].items()}
    return dict(mode=cmd_exec["mode"], pelvis_z=cmd_exec["pelvis_z"], hands=hands)


def exec_cmd(rec: dict) -> dict:
    """The command a reach executed (its record: heading-frame targets, rotation)."""
    hands = {s: dict(target=np.asarray(hc["cmd_target"], float).copy(), core=bool(hc["core"]), dyaw=float(hc["dyaw"]),
                     pitch=float(hc["pitch"]), roll=float(hc["roll"])) for s, hc in rec["hands"].items()}  # fmt: skip
    return dict(mode=rec["mode"], pelvis_z=rec["pelvis_z"], hands=hands)


def head(ep: TaskEp, pw) -> np.ndarray:
    """A world point (belief) in the current pelvis heading frame (z as is)."""
    p, yaw = ep.pose()
    L = ep.L
    v = L.rot_z(yaw).T @ np.r_[np.asarray(pw[:2], float) - p, 0.0]
    return np.r_[v[:2], float(pw[2])]


class Arm:
    """One arm's command path: A executes the command; R runs it through the model first (when `through`)."""

    def __init__(self, arm: str, MD: Model | None, log: dict):
        self.arm, self.MD, self.log = arm, MD, log

    def go(self, ep: TaskEp, cmd: dict, extra: dict, through: bool, label: str) -> dict:
        info = dict(label=label, arm=self.arm, through=bool(through), f_cmd=ep.f, mode=cmd["mode"],
                    goal={s: np.asarray(hc["target"], float).tolist() for s, hc in cmd["hands"].items()},
                    rot={s: [hc["dyaw"], hc["pitch"], hc["roll"]] for s, hc in cmd["hands"].items()},
                    pre_settled=bool(ep.ready[1]))  # fmt: skip
        self.log.setdefault("commands", []).append(info)
        plan = None
        if through and self.arm == "R":
            plan, info["r"] = run_r(ep, self.MD, cmd, extra)
        rec = ep.reach(cmd, extra, plan=plan)
        info.update(chain=rec["chain"], f_label=rec["f_label"], settled_end=rec["settled_end"],
                    exec={s: np.asarray(hc["cmd_target"], float).tolist() for s, hc in rec["hands"].items()},
                    ik_err_cm={s: round(100 * float(hc["ik_err"]), 3) for s, hc in rec["hands"].items()})  # fmt: skip
        return rec


def digest(ep: TaskEp) -> dict:
    """The consistency check's digest: every row's joints, joint speeds and root, and the reference frames, up to now."""
    f = ep.f
    hr, hf = hashlib.md5(), hashlib.md5()
    for i in range(f + 1):
        x = ep.rows[i]
        for k in ("q", "dq", "root"):
            hr.update(np.ascontiguousarray(x[k], np.float64).tobytes())
    hf.update(np.ascontiguousarray(ep.frames[: f + 1], np.float64).tobytes())
    return dict(f=f, rows=hr.hexdigest(), frames=hf.hexdigest())


def search_abs(ep: TaskEp, lk, find, prompts, y0: float, steps_deg, tag: str):
    """ha_bench.turn_until_seen(from_start=True): look; not found: turn (in place) to the start heading + each step."""
    lkk, ms = HR.look(ep, lk, prompts, f"{tag}0")
    t = find(lkk, ms, None)
    for i, dy in enumerate(steps_deg):
        if t is not None:
            break
        ep.walk_to(ep.pose()[0], y0 + math.radians(dy), f"search_{tag}{i}")
        lkk, ms = HR.look(ep, lk, prompts, f"{tag}{i + 1}")
        t = find(lkk, ms, None)
    return t


def find_basket_of(lk):
    """ha_bench.run_doubledesk's find_basket (self -> the Look)."""

    def find_basket(lk_, ms_, prev):
        m, pts = lk.pick(lk_, ms_, near=None if prev is None else prev["p"], min_px=20)
        if m is None:
            return None
        top = float(np.percentile(pts[:, 2], 95))
        rim = pts[pts[:, 2] > top - 0.03]
        ctr = HR.P.min_area_rect(rim[:, :2])[0] if len(rim) >= 10 else np.median(pts[:, :2], axis=0)
        p = np.array([ctr[0], ctr[1], top])
        zs, _ = HR.P.support_level(lk_.world, lk_.ok, m)
        bt = dict(p=p)
        bt.update(lk.stand_on_support("basket_table", lk_, p, zs, 0.40, -0.20))
        return bt

    return find_basket


# -- one episode ---------------------------------------------------------------------------------------------------------
def pregrasps(a, ep: TaskEp, lk, plan, arm: Arm, log: dict, on_first) -> None:
    """hr_real.run's reach loop: per hand its via point (direct) and its pre-grasp (through the model in R)."""
    L, D3 = ep.L, ep.D3
    first = True
    for hand_no, (side, pg, Rw, off, g, support) in enumerate(plan):
        goals = [(1, np.asarray(pg, float))]
        # -- hr_real.run's via rule (verbatim) --
        if a.via and a.via_rule == "lift" and support is not None:
            x = ep.xb()
            pw, Rw0 = x["wr"][side]
            hp0 = np.asarray(pw, float) + np.asarray(Rw0, float) @ L.HAND_OFF
            # the via is the hand point's; command() takes the grasp point: shift by the same offset
            hp_pg = np.asarray(pg, float) + np.asarray(Rw, float) @ (L.HAND_OFF - D3.PALM_OFF[side] - np.asarray(off, float))
            via_h = np.r_[hp0[:2], max(float(support) + HR.LIFT_CLEAR, float(hp_pg[2]) + HR.VIA_UP)]
            goals.insert(0, (0, via_h - (hp_pg - np.asarray(pg, float))))
        elif a.via and a.via_rule != "lift" and (support is not None if a.via_rule == "support" else float(g[2]) >= HR.VIA_Z):
            x = ep.xb()
            yaw = L.yaw_of(L._qmat(x["root"][3:]))
            hd = np.array([math.cos(yaw), math.sin(yaw)])
            xh = float((np.asarray(pg[:2], float) - x["root"][:2]) @ hd)
            back = min(HR.VIA_BACK, max(0.0, xh - HR.VIA_XMIN))
            goals.insert(0, (0, np.r_[np.asarray(pg[:2], float) - back * hd, float(pg[2]) + HR.VIA_UP]))
        # -- end of the via rule --
        for no, goal in goals:
            cmd, cinfo = HR.command(L, D3, ep.xb(), side, goal, Rw, off)
            ok = all(cinfo["in_range"].values())
            if no == 1 and first:
                on_first(cmd)
                first = False
            ep.set_phase(f"{'via' if no == 0 else 'pregrasp'}_{side}")
            extra = dict(reach_no=no, hand_no=hand_no, cmd_info=cinfo, cmd_in_range=ok, support=support)
            arm.go(ep, cmd, extra, through=no == 1, label=f"{'via' if no == 0 else 'pregrasp'}_{side}")


def post_ppbox(ep: TaskEp, lk, arm: Arm, log: dict, y0: float) -> str:
    L, D3 = ep.L, ep.D3
    t = lk.target
    faces, a_, Rw, half_a = lk.box_faces(t)
    sq = {"right": faces["right"] + SQUEEZE_IN * a_, "left": faces["left"] - SQUEEZE_IN * a_}
    cmds, cinfo = {}, {}
    for s in ("left", "right"):
        cmds[s], cinfo[s] = HR.command(L, D3, ep.xb(), s, sq[s], Rw, B.G_BOX[s])
    box_h = head(ep, t["p"])  # the perceived box centre, heading frame at the squeeze
    log["perceived"]["squeeze"] = {s: np.round(v, 4).tolist() for s, v in sq.items()}
    ep.set_phase("squeeze")
    rec = arm.go(ep, two(cmds), dict(reach_no=2, hand_no=-1, cmd_info=cinfo, cmd_in_range=all(all(c["in_range"].values()) for c in cinfo.values()),
                                     support=t["support"]), through=True, label="squeeze")  # fmt: skip
    ep.set_phase("close")
    ep.set_hands(True, True)
    ep.carry = True
    ep.hold_n(CLOSE_HOLD["ppbox"])
    ep.ready_now()
    dz = np.array([0.0, 0.0, LIFT["ppbox"]])
    lift = shifted(exec_cmd(rec), {s: dz for s in rec["hands"]})
    ep.set_phase("lift")
    rec_l = arm.go(ep, lift, dict(reach_no=3, hand_no=-1, cmd_info=None, cmd_in_range=True, support=t["support"]), through=False, label="lift")
    box_lift_h = box_h + dz
    # ha_bench: the grasp points' height after the lift - h / 2 (its reference; here the perceived squeeze points, the
    # same in A and R)
    box_bottom_w = 0.5 * (float(sq["left"][2]) + float(sq["right"][2])) + LIFT["ppbox"] - 0.5 * float(t["h"])

    def find_s(lk_, ms_, prev):
        return lk.find_shelf(lk_, ms_, prev, box_bottom_w)

    sprom = ["blue metal shelf"]
    ep.set_phase("search_shelf")
    s = search_abs(ep, lk, find_s, sprom, y0, SEARCH_ABS, "shelf")
    if s is None:
        return "shelf not seen"
    lk.target = s
    log["shelf_first"] = dict(p=np.round(s["p"], 3).tolist(), levels=np.round(s["levels"], 3).tolist(), stand_x=round(s["x"], 3), edge=s["edge"])
    ep.set_phase("carry")
    cl = log.setdefault("carry", {})
    HR.approach(ep, lk, find_s, sprom, cl)
    s = lk.target
    tgt_h = head(ep, np.asarray(s["p"], float) + np.array([0.0, 0.0, 0.5 * float(t["h"]) + PLACE_DZ]))
    shift = tgt_h - box_lift_h
    log["perceived"]["board"] = np.round(s["p"], 4).tolist()
    log["perceived"]["place_shift_h"] = np.round(shift, 4).tolist()
    ep.set_phase("place")
    arm.go(ep, shifted(exec_cmd(rec_l), {sd: shift for sd in rec_l["hands"]}),
           dict(reach_no=4, hand_no=-1, cmd_info=None, cmd_in_range=True, support=float(s["p"][2])), through=False, label="place")  # fmt: skip
    ep.set_phase("release")
    ep.set_hands(False, False)
    ep.hold_n(OPEN_HOLD["ppbox"])
    return "sequence done"


def post_doubledesk(ep: TaskEp, lk, arm: Arm, log: dict, y0: float, plan) -> str:
    L, D3 = ep.L, ep.D3
    side, pg, Rw, off, g, support = plan[0]
    cmd, cinfo = HR.command(L, D3, ep.xb(), side, g, Rw, off)
    g_h = head(ep, g)
    ep.set_phase("descend")
    rec = arm.go(ep, cmd, dict(reach_no=2, hand_no=0, cmd_info=cinfo, cmd_in_range=all(cinfo["in_range"].values()), support=support),
                 through=True, label="descend")  # fmt: skip
    ep.set_phase("close")
    ep.set_hands(False, True)
    ep.carry = True
    ep.hold_n(CLOSE_HOLD["doubledesk"])
    ep.ready_now()
    dz = np.array([0.0, 0.0, LIFT["doubledesk"]])
    ep.set_phase("lift")
    rec_l = arm.go(ep, shifted(exec_cmd(rec), {side: dz}), dict(reach_no=3, hand_no=0, cmd_info=None, cmd_in_range=True, support=support),
                   through=False, label="lift")  # fmt: skip
    g_lift_h = g_h + dz
    bprom = ["bin", "basket"]
    find_b = find_basket_of(lk)
    ep.set_phase("search_basket")
    b = search_abs(ep, lk, find_b, bprom, y0, SEARCH_ABS, "basket")
    if b is None:
        return "basket not seen"
    lk.target = b
    log["basket_first"] = dict(p=np.round(b["p"], 3).tolist(), stand_x=round(b["x"], 3), edge=b["edge"])
    ep.set_phase("carry")
    cl = log.setdefault("carry", {})
    HR.approach(ep, lk, find_b, bprom, cl)
    b = lk.target
    over_h = head(ep, np.asarray(b["p"], float) + np.array([0.0, 0.0, OVER_DZ]))
    shift = over_h - g_lift_h
    log["perceived"]["basket"] = np.round(b["p"], 4).tolist()
    log["perceived"]["over_shift_h"] = np.round(shift, 4).tolist()
    ep.set_phase("over_basket")
    arm.go(ep, shifted(exec_cmd(rec_l), {side: shift}), dict(reach_no=4, hand_no=0, cmd_info=None, cmd_in_range=True, support=None),
           through=False, label="over_basket")  # fmt: skip
    ep.set_phase("release")
    ep.set_hands(False, False)
    ep.hold_n(OPEN_HOLD["doubledesk"])
    return "sequence done"


def episode(a, ctx: dict, arm_name: str, sd: int, rp: int) -> dict:
    h, rig, lk, SX, D3, L, HW = (ctx[k] for k in ("h", "rig", "lk", "SX", "D3", "L", "HW"))
    task = a.ha_task
    name = f"{task}_s{sd}_r{rp}"
    out = a.out / arm_name
    es = ha_sim.episode_seed(a.task, sd, rp)
    rid = RID0 + 1_000_000 * ARM_ID[arm_name] + 10_000 * TASK_ID[task] + 10 * sd + rp
    h.ep_seed = es
    h.dev = dict(xy=0.0, z=0.0, rot_deg=0.0)
    log = dict(task=task, seed=sd, repeat=rp, arm=arm_name, episode_seed=es, reset=rid, controller=a.controller,
               budget=ctx["budget"], model=ctx["MD"].info if ctx["MD"] is not None else None)  # fmt: skip
    ep = TaskEp(h, rig, SX, D3, L, HW, task, ctx["budget"], ctx["hands_fn"])
    lk.new_episode(name, ep)
    lk.look_dir = str(out / "looks")
    a.replay_looks = (a.out / "A" / "looks") if arm_name == "R" else None
    ep.heft_gait = bool(a.heft_gait) and a.controller == "heft"
    arm = Arm(arm_name, ctx["MD"], log)

    def on_first(cmd):  # the first pre-grasp command: the consistency digest; R's looks are live from here
        d = digest(ep)
        d["goal"] = {s: np.asarray(hc["target"], float).tolist() for s, hc in cmd["hands"].items()}
        log["prefix"] = d
        a.replay_looks = None

    end = seq = None
    try:
        ep.stand_still()
        y0 = ep.pose()[1]
        log["y_start"] = y0
        log["clear"] = B.CLEAR
        ep.set_phase("look")
        plan = HR.FLOWS[task](ep, lk, log)
        if isinstance(plan, str):
            seq = plan
        else:
            pregrasps(a, ep, lk, plan, arm, log, on_first)
            seq = post_ppbox(ep, lk, arm, log, y0) if task == "ppbox" else post_doubledesk(ep, lk, arm, log, y0, plan)
    except End as e:
        if str(e) in ("success", "fell", "budget"):
            end = str(e)
        else:
            seq = f"abandoned: {e} ({ep.why})"
    except Exception as e:  # a harness error: record it, keep Kit for the next episode
        traceback.print_exc()
        end = f"error {type(e).__name__}: {e}"
    a.replay_looks = None
    if end is None:  # the sequence ended first: hold
        ep.set_phase("hold_end")
        try:
            ep.hold_n(a.hold)
            end = "held"
        except End as e:
            end = str(e)
        except Exception as e:
            traceback.print_exc()
            end = f"error {type(e).__name__}: {e}"
    info = ep.info()
    info.update(fell=ep.d3_fell if ep.d3_fell is not None else ep.d3_lost, lost=ep.d3_lost)
    segs = []
    if ep.reaches:
        segs = D3.segments(ep.rows, info, ep.frames, ep.reaches, rid, h.pl, rig)
        SX.add_fields(segs, ep.reaches)
        HC.settle_fields(segs, ep.rows, ep.reaches, L, D3)
        info["touch"] = SXH.add_touch(h, segs, ep.reaches)
        log["contacts"] = HR.reach_contacts(ep, log)
        for sg in segs:
            rc = ep.reaches[sg["chain"]]
            sg.update(task_id=TASK_ID[task], seed=sd, repeat=rp, reach_no=rc["reach_no"], hand_no=rc["hand_no"],
                      pre_settled=rc["pre_settled"], cmd_in_range=rc["cmd_in_range"], arm=ARM_ID[arm_name])  # fmt: skip
        HC.save_chunk(SX, D3, segs, out / "data", name, rig, ctx["scalars"])
    steps = ep.f + 1
    log.update(end=end, sequence=seq, success=ep.success_at is not None, success_step=ep.success_at,
               success_official=ep.success_at is not None and ep.success_at <= BUDGET[task], fell_step=ep.ha_fell,
               d3_fell=ep.d3_fell, d3_lost=ep.d3_lost, lifted_step=ep.lifted_at, max_lift_cm=round(100 * float(ep.max_lift), 2),
               steps=steps, phase_at_end=ep.phase, phases=ep.phases, hands=ep.hand_events, obj=ep.obj, segments=len(segs),
               looks=lk.looks, info=info, wall_s=round(time.time() - ep.t0, 1))  # fmt: skip
    (out / "log" / f"{name}.json").write_text(json.dumps(log, default=D3._json, indent=1))
    print(f"ht_task {a.controller} {task} {arm_name} {name}: end={end!r} seq={seq!r} success_step={ep.success_at} lifted={ep.lifted_at} "
          f"fell={ep.ha_fell} phase={ep.phase} steps={steps} reaches={len(ep.reaches)} segs={len(segs)} wall={time.time() - ep.t0:.0f} s", flush=True)  # fmt: skip
    return log


# -- run -----------------------------------------------------------------------------------------------------------------
def parser():
    p = HR.parser()
    p.add_argument("--arms", default="A,R")
    p.add_argument("--hold", type=int, default=100, help="steps held after the sequence ends (ha_bench's --hold)")
    p.add_argument("--budget-x", type=float, default=3.0, help="the step budget: this times the official one")
    p.add_argument("--model-root", type=Path, default=None, help="R's model: the models_<arm>_s0 directory (E3_ensemble_f1/)")
    p.add_argument("--model-template", type=Path, default=None, help="a data file of the model's layout (sizes the rebuild)")
    return p


def run(a) -> int:
    _orig = ha_sim.sev._configure_renderer_gpu

    def _cfg(args):
        _orig(args)
        SXH.kit_single_gpu(args)

    ha_sim.sev._configure_renderer_gpu = _cfg
    a = ha_sim.finish_args(a, a.ha_task)
    assert a.ha_task in BUDGET, a.ha_task
    assert int(a.max_steps) == BUDGET[a.ha_task], (a.max_steps, BUDGET[a.ha_task])
    a.no_walker = True
    a.replay_looks = None
    if a.clear is not None:
        B.CLEAR = float(a.clear)
    elif a.controller in HR.CLEAR_BY:
        B.CLEAR = HR.CLEAR_BY[a.controller]
    arms = [s for s in a.arms.split(",") if s]
    assert set(arms) <= set(ARM_ID) and arms == sorted(arms), arms  # A before R
    seeds, repeats = HR.rng_list(a.seeds), HR.rng_list(a.repeats)
    for sd in seeds:
        ha_sim.check_seed(sd)
    for arm in ("A", "R"):
        for k in ("data", "data/arm", "log", "looks"):
            (a.out / arm / k).mkdir(parents=True, exist_ok=True)
    t_up = time.time()
    sim = ha_sim.Sim(a)
    SXH.setup_paths()
    sys.path.insert(0, str(HC.W_REPO / "src"))
    import sx_collect as SX

    import hg_walk as HW

    D3, L = SX.D3, SX.L
    D3.SETTLE_MAX_F = int(round(HC.SETTLE_MAX_S * D3.FPS))
    D3.HOLD_S = HC.HOLD_S
    D3.SETTLE_V = HC.SETTLE_V[a.controller]
    INV.set_L(L)
    hands_now = np.zeros(2)
    if a.controller == "sonic":
        if a.no_camera:
            sim.provider._should_refresh_lerobot_visuals_next_step = lambda: False
        h = HR.HRSonic(sim, L, D3, SXH.SONIC_DELAY_F, "native", 0.0, None)
        _step = sim.step

        def step_with_hands():  # HAHost.step zeroes sim.hands before calling this: the hands as commanded
            sim.hands = hands_now.copy()
            _step()

        sim.step = step_with_hands

        def hands_fn(left, right):
            hands_now[:] = [float(left), float(right)]  # ha_bench: sim.hands[0] left, [1] right
    else:
        h = HR.HRG1(sim, L, D3, a.controller, "lml", None, 0.0, None)

        def hands_fn(left, right):
            h.prov.set_hands(left, right)

    rig = SX.SRig(h)
    lk = HR.Look(sim, a, a.out / "A" / "looks")
    MD = None
    if "R" in arms:
        assert a.model_root is not None and a.model_template is not None, "R needs --model-root and --model-template"
        MD = Model(a.model_root, a.model_template)
    h.sample()
    tag = f"ht_task {a.controller} {a.ha_task}"
    print(f"{tag}: up in {time.time() - t_up:.0f} s; dt {sim.dt:.3f}; palm vs wrist + PALM_OFF max dev {HR.palm_check(sim):.1e}; "
          f"delay_f {h.ref_delay_f}; CLEAR {B.CLEAR}; WIDE {[v.tolist() for v in L.WIDE]}; model {None if MD is None else MD.info}", flush=True)  # fmt: skip
    ctx = dict(h=h, rig=rig, lk=lk, SX=SX, D3=D3, L=L, HW=HW, MD=MD, hands_fn=hands_fn, budget=int(round(a.budget_x * BUDGET[a.ha_task])),
               scalars=D3.SCALARS + HC.EXTRA + ("task_id", "seed", "repeat", "reach_no", "hand_no", "pre_settled", "cmd_in_range", "arm"))  # fmt: skip
    for sd in seeds:
        for rp in repeats:
            name = f"{a.ha_task}_s{sd}_r{rp}"
            for arm in arms:
                if (a.out / arm / "log" / f"{name}.json").exists():
                    continue
                if arm == "R" and not (a.out / "A" / "log" / f"{name}.json").exists():
                    print(f"{tag} {name}: R needs A's looks; A missing", flush=True)
                    continue
                lg = episode(a, ctx, arm, sd, rp)
                if arm == "R":
                    la = json.loads((a.out / "A" / "log" / f"{name}.json").read_text())
                    pa, pr = la.get("prefix"), lg.get("prefix")
                    same = pa is not None and pr is not None and pa == pr
                    print(f"{tag} {name}: prefix A == R: {same} (A {pa and pa['f']}, R {pr and pr['f']})", flush=True)
    print(f"{tag}: DONE", flush=True)
    return 0


if __name__ == "__main__":
    args = parser().parse_args()
    try:
        rc = run(args)
    except Exception:
        traceback.print_exc()
        rc = 1
    sys.stdout.flush()
    os._exit(rc)
