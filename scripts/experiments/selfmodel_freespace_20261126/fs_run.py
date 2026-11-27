"""selfmodel_freespace_20261126 E2a: one in-place right-hand reach in LocoManiLab's EmptyEnv, direct (A) against the
self model's corrections (S, R), ScaleBFM / HEFT / SONIC, model D. 2026-11-26.

What runs, from lmlcmp3's frozen lc3_run.py (imported, not edited): the drivers' host pieces and plan_rec (lmlreal), the
robot's own settle rule (PropSettle, lmlcmp (a)), the landing rule (lmlcmp (b) = data3's), the models (Models: T3, M,
D; features through d3_collect.segments + data.py's derive), the inversion (6 fixed-point steps, 7 candidates, the
first least |D(t) - goal|^2 + LAMBDA |spread|^2 -- copied line for line from lc3_run.episode), shrink (R), ref_end,
reach_rc and the code check (LC_CODE_ROOT). Only the task layer is new: no perception, no walk, one reach.

Host: lml_host's EmptyEnv (data3's): HostDriver (ScaleBFM), lmlreal's HeftBase (HEFT), d3_sonic.SonicHost (SONIC),
with plan = lr_run.plan_rec. Before every reset the env's generator is set to fs_grid.reset_rng(cid, ep): robosuite's
robot init noise (EmptyEnv's "default", 0.02 rad Gaussian per joint, data3's start perturbation) is then a function of
the cell and the episode alone, so every group of a (cell, ep) starts from the same state.

One episode (cell, ep, group):
  1. reset; data3's reference (start_frames + HOLD_S hold).
  2. stand wait: opens at data3's FIRST_OPEN_F (0.2 s), goes at the first step >= 0.5 s after the opening (data3's
     earliest first command, 0.7 s) where the own rule holds (PropSettle: both hand points in the pelvis frame from the
     encoders < 2 cm/s and the 12 leg joints < 0.05 rad/s, every step of the last 0.5 s), at most 3 s after the opening.
  3. squat start only: the squat (fs_grid.squat_plan from the measured state: both hands lowered with the pelvis to
     0.60 m, d3_collect.reach_both), spliced as a reach is (protocol 2: planned from the measured state, shifted onto the
     reference root); then the same own-rule wait, opening at the squat reference's end + ref_delay_f (data3's next
     reach), at most 3 s. The reach's model inputs are those of a reach that follows another (data3's chain 1):
     chain = 1, d_prev = the squat's reference shift.
  4. the reach: target = fs_grid.goal(cell, ep) in the pelvis heading frame at the command row (measured root),
     dyaw = the cell's direction, pitch = the cell's (grid.json), roll 0, pelvis height none (stand) or 0.60 m.
     A: the goal itself. S (`selfmodel`): lc3's fixed point. R (`reachable`): lc3's fixed point with shrink.
  5. the landing: data3's rule from the command (10 Hz grid, >= 0.5 s after the reference end + ref_delay_f where the
     simulator pelvis and both hand points are < 2 cm/s for 0.5 s; else the grid row at 3 s, settled_end 0). The
     episode ends there. Score: the right hand point (13 cm ahead of wrist_yaw_link) to the goal (world, from the
     command row's measured pelvis), cm -- simulator truth, scoring only.
  Ends early: a fall (pelvis < 0.45 m) or a lost reach (pelvis displacement off the reference's by 0.5 m since the
  last splice, the reference taken ref_delay_f behind: data3's rule), an IK exception.
Simulator truth decides nothing: the waits are the own rule's; the landing row only records.

Records ($FS_OUT/<backend>/): runs/<chunk>.jsonl (one episode per line) and segs/<chunk>.npz (+ arm/) in data3's
layout with SCALARS_FS; chunks of FLUSH episodes, named by the process; a job is done when its line exists (global over
all chunks, so the shard count may change between launches).

    cd $G; PYTHONPATH=$G:$W/src $PY fs_run.py run --backend scalebfm --eps 0:16 --shard K --shards N [--cells a:b]
    $PY fs_run.py replay --backend scalebfm --n 20 --out DIR      # the entry's same-command replay of data3
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path[:0] = [str(EXP / "selfmodel_lmlcmp3_20261126"), str(HERE)]
import lc3_run as LC  # noqa: E402  (lmlcmp3's frozen runner)
import fs_grid as FG  # noqa: E402

LR, D3, HG, L, LT = LC.LR, LC.D3, LC.HG, LC.L, LC.LT
ROOT = LC.ROOT
OUT = Path(os.environ.get("FS_OUT", str(ROOT / "selfmodel_freespace_20261126")))
GRID = Path(os.environ.get("FS_GRID", str(OUT / "grid" / "grid.json")))
BACKENDS = LR.BACKENDS
METHODS = LC.METHODS  # method_id = lc3's index: alone 0, selfmodel 1, reachable 3
GROUPS = {"scalebfm": ("alone", "selfmodel", "reachable"), "heft": ("alone", "reachable"), "sonic": ("alone", "reachable")}
INVERT = ("selfmodel", "reachable")
FPS, TICKS = L.FPS, L.TICKS
SETTLE_F, SETTLE_MAX_F = D3.SETTLE_F, D3.SETTLE_MAX_F
RESET0 = 200_000_000  # + 10 000 000 method_id + 10 000 cid + ep: 200,000,000..232,159,999 (hadata_sonic has 93M / 94M)
FLUSH = 10
SCALARS_FS = D3.SCALARS + ("cell", "ep", "method_id", "posture", "backend_id", "pre_settled", "prop_settled", "label_kind",
                           "live_err_cm", "d_prev_x", "d_prev_y", "d_this_x", "d_this_y")  # fmt: skip


def reset_id(cid: int, ep: int, method: str) -> int:
    assert 0 <= ep < 10_000 and 0 <= cid < 216
    return RESET0 + 10_000_000 * METHODS.index(method) + 10_000 * cid + ep


# -- host ------------------------------------------------------------------------------------------------------------
def host_bases(backend: str) -> tuple:
    if backend == "sonic":
        import d3_sonic

        return (d3_sonic.SonicHost,)
    if backend == "heft":
        return (LR.HeftBase,)
    return (LR.LH.HostDriver,)


def make_driver(backend: str, seed: int = 0, plan=None):
    cls = type(f"FS_{backend}", host_bases(backend), {"plan": plan or LR.plan_rec})
    dr = cls(env_name="EmptyEnv", seed=seed)
    if not hasattr(dr, "plan_s"):
        dr.plan_s = 0.0  # plan_rec adds its planning time to it
    return dr


def plan_roll(self, x: dict, ref_w: np.ndarray, c: dict):
    """lr_run.plan_rec with data3's wrist roll (R = Rz(yaw + dyaw) Ry(pitch) Rx(roll), d3_collect.plan_reach's): only for
    the replay of data3's commands. With roll 0 it is plan_rec bit for bit (fs_check compares)."""
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    yaw = L.yaw_of(L._qmat(quat))
    Rh = L.rot_z(yaw)
    target_w = np.r_[p[:2] + (Rh @ np.r_[c["target"][:2], 0.0])[:2], c["target"][2]]
    rot_w = L.rot_z(yaw + c["dyaw"]) @ L.rot_y(c["pitch"]) @ D3.rot_x(c.get("roll", 0.0))
    q_mj = x["q"][self.to_mj].astype(float)
    t0 = time.time()
    poses, durs, ik_err, dur = self.pl.reach((p, quat, q_mj), c["side"], target_w, rot_w, c["pelvis_z"])
    self.plan_s = getattr(self, "plan_s", 0.0) + time.time() - t0
    delta = np.r_[ref_w[:2] - p[:2], 0.0]
    ref_pose = (ref_w[:3].astype(float), ref_w[3:7].astype(float), ref_w[7:].astype(float))
    shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
    last = poses[-1]
    rec = dict(side=c["side"], target_w=target_w.tolist(), rot_w=rot_w.tolist(), dyaw=c["dyaw"], pitch=c["pitch"],
               pelvis_z=c["pelvis_z"], cmd_target=c["target"].tolist(), ik_err=ik_err, reach_s=dur,
               ref_q=last[2].tolist(), ref_pel=np.r_[last[0], last[1]].tolist(), delta=delta.tolist(),
               RT=np.r_[0.0, np.cumsum(durs)], RQ=np.array([pp[2] for pp in poses]),
               RP=np.array([np.r_[pp[0], pp[1]] for pp in poses]))  # fmt: skip
    return L.blend_frames(shifted, durs), shifted[-1], rec


def plan_squat(dr, x: dict, ref_w: np.ndarray):
    """Step 3: fs_grid.squat_plan from the measured state, spliced as plan_rec splices a reach."""
    p, quat = x["root"][:3].astype(float), x["root"][3:].astype(float)
    q_mj = x["q"][dr.to_mj].astype(float)
    poses, durs, errs, dur = FG.squat_plan(dr.pl, (p, quat, q_mj), FG.PZ_SQ)
    delta = np.r_[ref_w[:2] - p[:2], 0.0]
    ref_pose = (ref_w[:3].astype(float), ref_w[3:7].astype(float), ref_w[7:].astype(float))
    shifted = [ref_pose] + [(pp + delta, qq, jj) for pp, qq, jj in poses[1:]]
    rec = dict(delta=delta, hand_err_cm={k: 100 * v for k, v in errs.items()}, squat_s=dur, n_way=len(poses),
               end_pel=np.r_[poses[-1][0], poses[-1][1]].tolist())  # fmt: skip
    return L.blend_frames(shifted, durs), shifted[-1], rec


def hand_w(x: dict) -> np.ndarray:
    pw, Rw = x["wr"]["right"]
    return np.asarray(pw, float) + np.asarray(Rw, float) @ L.HAND_OFF


# -- one episode -----------------------------------------------------------------------------------------------------
def episode(dr, rig, ps, MD, backend: str, method: str, c: dict, ep: int, *, pitch_deg: float, env_rng="cell",
            cmd: dict | None = None, force_f: int | None = None) -> tuple[dict, list[dict]]:  # fmt: skip
    """One (cell, ep, group). env_rng: "cell" -> fs_grid.reset_rng(cid, ep) before the reset; None -> the env's own
    generator as it stands (replay). cmd / force_f: the replay's command (target, dyaw, pitch, roll) and command row."""
    t0 = time.time()
    if env_rng == "cell":
        dr.env.rng = FG.reset_rng(c["cid"], ep)
    dr.reset()
    rig.clear()
    ps.clear()
    frames = dr.start_frames()
    frames = np.concatenate([frames, np.repeat(frames[-1:], int(D3.HOLD_S * FPS), 0)])
    posture = c["posture"]
    gh = FG.goal(c, ep) if cmd is None else np.asarray(cmd["target"], float)
    R = dict(backend=backend, method=method, cid=c["cid"], loc=c["loc"], ep=ep, posture=posture, goal_h=gh.tolist(),
             pitch_deg=pitch_deg if cmd is None else math.degrees(cmd["pitch"]))  # fmt: skip
    rows: dict[int, dict] = {}
    f, fell, lost, why = 0, None, None, ""
    stage, open_f = "stand", D3.FIRST_OPEN_F
    anchor, cur_delta, active, chain = None, np.zeros(3), None, 0
    delay = None
    wrec = dict(asked=None, real=None, dev_cm=0.0)
    waits: list[dict] = []
    info: dict = {}
    squat = None
    while True:
        dr.push(frames[min(f, len(frames) - 1)])
        if getattr(dr, "buf", 0) is None:
            dr.buf = dr._find_buffer()
        dr.step()
        if delay is None:
            delay = D3.ref_delay_f(dr)
        if f == 1 and getattr(getattr(dr, "comp", None), "_settling", False):
            raise RuntimeError("the bridge is still settling: our reference is not being tracked")
        x = rows[f] = dr.sample()
        rig.record(x)
        ps.update(x)
        if float(x["root"][2]) < L.FALL_Z:
            fell, why = f, f"fell during {stage}"
            break
        if anchor is not None:  # data3's lost rule: the reference's displacement taken ref_delay_f behind
            fr = frames[min(max(f - delay, anchor[2]), len(frames) - 1)]
            dev = float(np.linalg.norm((x["root"][:2] - anchor[0]) - (fr[:2] - anchor[1])))
            if dev > L.LOST_M:
                fell, lost, why = f, f, f"lost during {stage}"
                break
        if stage == "label":
            if (f - active["f_cmd"]) % TICKS == 0:
                kind = None
                if f >= active["f_end"] + delay + SETTLE_F and rig.quiet():
                    kind = 0
                elif f >= active["f_end"] + delay + SETTLE_MAX_F:
                    kind = 1
                if kind is not None:
                    active.update(f_label=f, settled_end=kind == 0, settle_s=(f - active["f_end"]) / FPS, label_kind=kind,
                                  live_err_cm=100 * float(np.linalg.norm(hand_w(x) - np.asarray(active["hands"]["right"]["target_w"]))))  # fmt: skip
                    break
        else:  # a wait: stand (before the squat or the reach) or squat (before the reach)
            if force_f is not None:
                go = "forced" if f == force_f else None
            elif f >= open_f + SETTLE_F and ps.quiet():
                go = "settled"
            elif f >= open_f + SETTLE_MAX_F:
                go = "timeout"
            else:
                go = None
            if go:
                vh, vl = ps.vmax()
                waits.append(dict(stage=stage, open=open_f, fire=f, wait_s=(f - open_f) / FPS, how=go, prop_settled=go == "settled" or bool(ps.quiet()),
                                  truth_quiet=bool(rig.quiet()), prop_v_hand_cms=100 * vh, prop_v_leg_rads=vl))  # fmt: skip
                if stage == "stand" and posture == "squat":
                    blend, last, squat = plan_squat(dr, x, frames[f])
                    frames = np.concatenate([frames[: f + 1], blend, L.hold(last, D3.HOLD_S)])
                    anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                    cur_delta = np.asarray(squat["delta"], float).copy()
                    squat.update(f_cmd=f, f_end=f + len(blend))
                    chain, stage, open_f = 1, "squat", f + len(blend) + delay
                else:
                    # -- the reach (lc3_run.episode's inversion, line for line, on this one command) --------------------
                    wl = waits[-1]
                    c_ = dict(side="right", target=gh.copy(), dyaw=math.radians(c["th"]) if cmd is None else cmd["dyaw"],
                              pitch=math.radians(pitch_deg) if cmd is None else cmd["pitch"], pelvis_z=c["pz"] if cmd is None else cmd["pelvis_z"])  # fmt: skip
                    if cmd is not None and "roll" in cmd:
                        c_["roll"] = cmd["roll"]
                    info = dict(chain=chain, goal_h=gh.tolist(), dyaw_deg=math.degrees(c_["dyaw"]), pitch_deg=math.degrees(c_["pitch"]),
                                pz=c_["pelvis_z"], wait_s=wl["wait_s"], prop_settled=wl["prop_settled"], truth_settled=bool(rig.quiet()))  # fmt: skip
                    tic = time.time()
                    walked = False
                    feat = lambda cc: MD.features(dr, rig, rows, f, frames, cc, chain, walked, cur_delta, wrec, delay)  # noqa: E731
                    best = None
                    if method in INVERT:
                        t, evs = gh.copy(), []
                        shr = [] if method == "reachable" else None
                        for it in range(LT.ITERS + 1):
                            if shr is not None and it > 0:
                                t, sh = LC.shrink(dr, rows[f], frames[f], c_, gh, t)
                                shr.append(dict(sh, it=it))
                            cc = dict(c_, target=t.copy())
                            try:
                                pl_, zz = feat(cc)
                            except (np.linalg.LinAlgError, ValueError) as e:
                                evs.append(dict(t=t.copy(), l=np.inf, err=f"{type(e).__name__}: {str(e)[:120]}"))
                                break
                            pos, std = MD.predict(zz, ("D",))["D"]
                            evs.append(dict(t=t.copy(), pos=pos, std=std, l=float(((pos - gh) ** 2).sum() + LT.LAMBDA * (std**2).sum()), plan=pl_, z=zz))
                            t = LT.clip_t(t + (gh - pos), "right")
                        bi = min(range(len(evs)), key=lambda i: evs[i]["l"])
                        best = evs[bi]
                        info["inv"] = [dict(t=e["t"].tolist(), pos=e["pos"].tolist() if "pos" in e else None, std=e["std"].tolist() if "std" in e else None,
                                            l=e["l"], err=e.get("err")) for e in evs]  # fmt: skip
                        if shr is not None:
                            info["shrink"] = shr
                        if "plan" in best:
                            c_["target"] = best["t"]
                            info.update(inv_best=bi, pred_miss0_cm=100 * float(np.linalg.norm(evs[0]["pos"] - gh)),
                                        pred_miss_cm=100 * float(np.linalg.norm(best["pos"] - gh)),
                                        corr_cm=100 * float(np.linalg.norm(best["t"] - gh)))  # fmt: skip
                            (blend, last, rec), zz = best["plan"], best["z"]
                        else:
                            info["model_fallback"] = True
                            best = None
                    if best is None:
                        try:
                            if MD is None:
                                blend, last, rec = dr.plan(rows[f], frames[f], c_)
                                zz = None
                            else:
                                (blend, last, rec), zz = feat(c_)
                        except (np.linalg.LinAlgError, ValueError) as e:
                            why = f"reach: plan failed {type(e).__name__}"
                            info["plan_error"] = str(e)[:200]
                            break
                    if MD is not None:
                        pr = MD.predict(zz)
                        info["pred_h"] = {n: v[0].tolist() for n, v in pr.items()}
                        info["pred_std"] = {n: v[1].tolist() for n, v in pr.items()}
                        info["feat_x0"], info["feat_u"] = zz["x0"][0].tolist(), zz["u"][0].tolist()
                    info["think_s"] = time.time() - tic
                    rec.update(chain=chain, f_cmd=f, f_end=f + len(blend))
                    frames = np.concatenate([frames[: f + 1], blend, L.hold(last, D3.HOLD_S)])
                    anchor = (x["root"][:2].copy(), frames[f, :2].copy(), f)
                    info.update(cmd_h=np.asarray(c_["target"]).tolist(), f_cmd=f, f_end=rec["f_end"], ik_err_cm=100 * float(rec["ik_err"]),
                                p_cmd=x["root"][:3].tolist(), yaw_cmd=L.yaw_of(L._qmat(x["root"][3:])),
                                goal_w=np.r_[x["root"][:2] + (L.rot_z(L.yaw_of(L._qmat(x["root"][3:]))) @ np.r_[gh[:2], 0.0])[:2], gh[2]].tolist())  # fmt: skip
                    info.update(LC.ref_end(dr, rec, x))
                    rc = LC.reach_rc(rec, f, rec["f_end"], chain, cur_delta, slot=chain)
                    if "roll" in c_:
                        rc["hands"]["right"]["roll"] = float(c_["roll"])
                    rc.update(pre_wait_s=wl["wait_s"], live_err_cm=-1.0, pre_settled=bool(rig.quiet()), prop_settled=wl["prop_settled"], label_kind=-1)
                    cur_delta = np.asarray(rc["delta"], float).copy()
                    active, stage = rc, "label"
        if f >= len(frames) - 1:
            why = why or "motion ended"
            break
        f += 1
    # -- post-hoc (scoring: simulator truth) ------------------------------------------------------------------------------
    if active is not None and active.get("f_label") is not None:
        x = rows[active["f_label"]]
        hand = hand_w(x)
        p0c, Rh = np.asarray(info["p_cmd"]), L.rot_z(info["yaw_cmd"])
        act_h = np.r_[(Rh.T @ np.r_[hand[:2] - p0c[:2], 0.0])[:2], hand[2]]
        info.update(err_cm=100 * float(np.linalg.norm(hand - np.asarray(info["goal_w"]))), act_h=act_h.tolist(), hand_w=hand.tolist(),
                    f_label=active["f_label"], settled_end=bool(active["settled_end"]), label_kind=active["label_kind"],
                    settle_s=active["settle_s"])  # fmt: skip
        info["pred_err_cm"] = {n: 100 * float(np.linalg.norm(np.asarray(v) - act_h)) for n, v in info.get("pred_h", {}).items()}
    R.update(fell=fell, lost=lost, why=why, steps=len(rows), delay_f=delay, waits=waits, reach=info, squat=squat,
             err_cm=info.get("err_cm"), wall_s=round(time.time() - t0, 2))  # fmt: skip
    segs = []
    if active is not None:
        info3 = dict(fell=fell, lost=lost, walk=wrec, walked=False, delay_f=delay)
        segs = D3.segments(rows, info3, frames, [active], reset_id(c["cid"], ep, method) if cmd is None else 0, dr.pl, rig)
        for s in segs:
            s.update(cell=c["cid"], ep=ep, method_id=METHODS.index(method), posture=c["ip"], backend_id=BACKENDS.index(backend),
                     pre_settled=active["pre_settled"], prop_settled=active["prop_settled"], label_kind=active.get("label_kind", -1),
                     live_err_cm=active.get("live_err_cm", -1.0), d_prev_x=float(active["d_prev"][0]), d_prev_y=float(active["d_prev"][1]),
                     d_this_x=float(active["delta"][0]), d_this_y=float(active["delta"][1]))  # fmt: skip
    R["n_segments"] = len(segs)
    return R, segs


# -- runner ------------------------------------------------------------------------------------------------------------
def load_grid() -> dict:
    g = json.loads(GRID.read_text())
    return {r["cid"]: r for r in g["cells"]}


def done_set(backend: str) -> set:
    out = set()
    for p in glob.glob(str(OUT / backend / "runs" / "*.jsonl")):
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            out.add((r["cid"], r["ep"], r["method"]))
    return out


def job_list(backend: str, cids: list[int], eps: list[int], methods: tuple) -> list[tuple]:
    """ep-major (every cell gets its first episodes first), then cell, then group: a (cell, ep)'s groups are adjacent."""
    return [(cid, ep, m) for ep in eps for cid in cids for m in methods]


def run(backend: str, cids: list[int], eps: list[int], methods: tuple, shard: int, shards: int) -> None:
    grid = load_grid()
    jobs = job_list(backend, cids, eps, methods)[shard::shards]
    done = done_set(backend)
    todo = [j for j in jobs if j not in done]
    tag = f"fs {backend} shard {shard}/{shards}"
    print(f"{tag}: {len(jobs) - len(todo)}/{len(jobs)} done; grid {GRID}", flush=True)
    if not todo:
        return
    dr = make_driver(backend)
    rig, ps = D3.Rig(dr), LC.PropSettle(dr)
    MD = LC.Models(backend)
    print(f"{tag}: pelvis offset c {np.round(dr.c, 4).tolist()}, root check {rig.check_root():.2e} m; models "
          f"{ {n: str(r) for n, r in LC.model_roots(backend).items()} }", flush=True)  # fmt: skip
    LC.code_check("start")
    od, rd = OUT / backend / "segs", OUT / backend / "runs"
    rd.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    buf_r, buf_s, k = [], [], 0

    def flush() -> None:
        nonlocal buf_r, buf_s, k
        if not buf_r:
            return
        name = f"n{shards:02d}s{shard:03d}_{stamp}_{os.getpid()}_{k:04d}"
        D3.save_chunk(buf_s, od, name, rig, SCALARS_FS)
        tmp = rd / f".{name}.tmp"
        tmp.write_text("".join(json.dumps(r, default=LT._json) + "\n" for r in buf_r))
        os.replace(tmp, rd / f"{name}.jsonl")
        buf_r, buf_s, k = [], [], k + 1

    for cid, ep, m in todo:
        c = FG.cell(cid)
        g = grid[cid]
        try:
            r, segs = episode(dr, rig, ps, MD, backend, m, c, ep, pitch_deg=g["pitch_deg"])
        except Exception as err:  # recorded, not retried
            r, segs = dict(backend=backend, method=m, cid=cid, ep=ep, posture=c["posture"], error=f"{type(err).__name__}: {err}",
                           tb=traceback.format_exc()[-2000:], err_cm=None), []  # fmt: skip
            print(f"{tag} cell {cid} ep {ep} {m}: ERROR {err}", flush=True)
            dr = make_driver(backend)
            rig, ps = D3.Rig(dr), LC.PropSettle(dr)
        buf_r.append(r)
        buf_s += segs
        rr = r.get("reach", {})
        print(f"{tag} cell {cid} {c['posture']} ep {ep} {m}: err {r.get('err_cm') if r.get('err_cm') is None else round(r['err_cm'], 2)} cm, "
              f"why {r.get('why')!r}, waits {[(w['stage'], round(w['wait_s'], 2), w['how']) for w in r.get('waits', [])]}, label "
              f"{rr.get('label_kind')}, corr {round(rr.get('corr_cm', 0.0), 1)} cm, ik {round(rr.get('ik_err_cm', -1), 2)} cm, {r.get('wall_s')} s", flush=True)  # fmt: skip
        if len(buf_r) >= FLUSH:
            flush()
        LC.code_check(f"after {cid} {ep} {m}")
    flush()


# -- the replay of data3 (the entry's same-command check) -------------------------------------------------------------
def replay(backend: str, n: int, out: Path, j_max: int) -> None:
    """data3's standing, in-place, right-hand first reaches of the first j_max resets of chunk 0 in each shard (the
    shard's host built with data3's seed SEED + shard; the resets before it replayed, so the env's generator is where
    data3's was; data3's shard count per controller from its log files), commanded through episode() twice: at data3's own command row (forced) and by the own rule. Compared
    with data3's segment: the start row, the landing row and the landing hand point."""
    import d3_collect as D3c

    z = np.load(ROOT / "selfmodel_data3_20261125" / backend / "data" / "all.npz")
    n_steps = z["n_steps"].astype(int)
    so = np.r_[0, np.cumsum(n_steps)[:-1]]
    reset, chain, side, pz, walked, slot = z["reset"], z["chain"], z["side"], z["pelvis_z"], z["walked"], z["slot"]
    shards = len({Path(q).name[:4] for q in glob.glob(str(ROOT / "selfmodel_data3_20261125" / backend / "data" / "w*_log.json"))})  # 50 / 30 / 38
    table = json.loads(D3c.TABLE.read_text())
    cand = []
    for i in np.where((chain == 0) & (slot == 0) & (walked == 0) & (side > 0) & (pz < 0))[0]:
        idx = int(reset[i]) - D3c.ID0
        k, j = idx % shards, idx // shards
        if j < j_max:
            cand.append((k, j, int(i), idx))
    cand.sort()  # shard by shard (one host each, advanced reset by reset)
    picks = []
    for k, j, i, idx in cand:
        plan = D3c.reset_plan(np.random.default_rng(D3c.SEED + idx), table, True)
        if plan["walk"] is None and plan["cmds"][0]["mode"] == "right" and plan["cmds"][0]["pelvis_z"] is None:
            picks.append((k, j, i, idx, plan))
        if len(picks) >= n:
            break
    S, W = z["S"], z["W"]
    out.mkdir(parents=True, exist_ok=True)
    res = []
    cache = {}
    for k, j, i, idx, plan in picks:
        if k not in cache:
            cache.clear()
            dr = make_driver(backend, seed=D3c.SEED + k, plan=plan_roll)
            cache[k] = (dr, 0)
        dr, done_j = cache[k]
        assert done_j <= j
        for _ in range(j - done_j):  # the shard's resets before this one
            dr.reset()
        g0 = copy.deepcopy(dr.env.rng)
        h = plan["cmds"][0]["hands"]["right"]
        cmd = dict(target=np.asarray(h["target"], float), dyaw=h["dyaw"], pitch=h["pitch"], roll=h["roll"], pelvis_z=None)
        rig, ps = D3.Rig(dr), LC.PropSettle(dr)
        f_cmd = D3c.FIRST_OPEN_F + int(round(float(z["pre_wait_s"][i]) * FPS))
        c = dict(FG.cell(0), posture="stand", pz=None)
        out_i = dict(reset=int(reset[i]), shard=k, j=j, f_cmd_data3=f_cmd)
        for variant in ("forced", "own"):
            dr.env.rng = copy.deepcopy(g0)
            r, segs = episode(dr, rig, ps, None, backend, "alone", c, 0, pitch_deg=0.0, env_rng=None, cmd=cmd,
                              force_f=f_cmd if variant == "forced" else None)  # fmt: skip
            sg = segs[0] if segs else None
            d3_land = W[so[i] + n_steps[i] - 1, 4:7].astype(np.float32)
            v = dict(why=r["why"], f_cmd=r["reach"].get("f_cmd"), f_label=r["reach"].get("f_label"), err_cm=r.get("err_cm"),
                     d3_err_cm=float(z["jaw_err_cm"][i]))  # fmt: skip
            if sg is not None:
                land = sg["W"][-1, 4:7].astype(np.float32)
                v.update(n_rows=len(sg["S"]), d3_rows=int(n_steps[i]), start_same=bool(np.array_equal(sg["S"][0], S[so[i]])),
                         start_maxdiff=float(np.abs(sg["S"][0] - S[so[i]]).max()), land_same=bool(np.array_equal(land, d3_land)),
                         land_diff_cm=100 * float(np.linalg.norm(land.astype(float) - d3_land.astype(float))),
                         rows_same=bool(len(sg["S"]) == n_steps[i] and np.array_equal(sg["S"], S[so[i] : so[i] + n_steps[i]])))  # fmt: skip
            out_i[variant] = v
        cache[k] = (dr, j + 1)
        res.append(out_i)
        print(json.dumps(out_i), flush=True)
    (out / f"replay_{backend}.json").write_text(json.dumps(dict(shards=shards, picks=res), indent=1))
    for variant in ("forced", "own"):
        vs = [r[variant] for r in res if "land_diff_cm" in r[variant]]
        d = np.array([v["land_diff_cm"] for v in vs])
        print(f"{backend} {variant}: {len(vs)}/{len(res)} labelled; start identical {sum(v['start_same'] for v in vs)}, rows identical "
              f"{sum(v['rows_same'] for v in vs)}, landing identical {sum(v['land_same'] for v in vs)}; landing difference median "
              f"{np.median(d) if len(d) else float('nan'):.4f} cm, max {d.max() if len(d) else float('nan'):.4f} cm; command row = data3's "
              f"{sum(r[variant]['f_cmd'] == r['f_cmd_data3'] for r in res)}")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "replay"])
    ap.add_argument("--backend", choices=BACKENDS, default="scalebfm")
    ap.add_argument("--eps", default="0:16")
    ap.add_argument("--cells", default="0:216", help="a:b or comma-separated cell ids")
    ap.add_argument("--methods", default=None, help="default: the controller's groups (GROUPS)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--jmax", type=int, default=12)
    ap.add_argument("--out", type=Path, default=OUT / "replay")
    a = ap.parse_args()
    if a.phase == "replay":
        replay(a.backend, a.n, a.out, a.jmax)
    else:
        lo, hi = (int(v) for v in a.eps.split(":"))
        cids = list(range(*map(int, a.cells.split(":")))) if ":" in a.cells else [int(v) for v in a.cells.split(",")]
        methods = tuple(a.methods.split(",")) if a.methods else GROUPS[a.backend]
        run(a.backend, cids, list(range(lo, hi)), methods, a.shard, a.shards)
