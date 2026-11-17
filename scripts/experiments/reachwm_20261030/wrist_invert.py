"""Wrist attitude in the inversion with the wide-wrist model (2026-11-02, after pose_invert.py).

pose_invert.py found the ceiling in the training data: the controller follows about half of a commanded wrist turn, so
pre-compensating a wanted rotation needs commands about twice as far off it, roll included, and the model trained on
the dyaw/pitch family (no roll) splits there. gen_reach.py REACH_WRIST=1 adds reaches whose wrist command is turned a
further 0-90 deg about a random axis (data_wrist); models_goalref_wrist is the plan-aware ensemble retrained on
data_ref + data_side + data_wrist.

Same 345 core goals, final states and stored commands as pose_invert.py, the new model, three inversions:
    pos    position only (pose_invert pos)
    pose   pose_invert pose: the wanted command and the 15 deg dyaw/pitch grid over the old range, NB best refined
    pose3  pose's candidates, plus a joint fixed point from two seeds (the wanted command and pose's best):
           t <- t + (goal - predicted), R_cmd <- R_want R_pred^T R_cmd (the predicted wrist miss turned back into the
           command, in full 3D), the command's turn off the wanted rotation capped at gen_reach.WRIST_MAX_DEG; best
           iterate by pose_invert's loss
    pose3g pose3 with a trust gate (2026-11-02, after the first full run): only iterates whose members agree on the wrist
           (spread <= SPREAD_MAX_DEG) are eligible. Ungated, 13 of the 77 goals still > 30 deg off had a predicted miss
           < 15 deg at the 90 deg cap with spread ~12 deg (the edge of the trained range), and 10 of the 14 servo misses
           > 2 cm had spread > 6 deg. Run alone (WRIST_METHODS=pose3g, files h_w*_pose3g.*).
    w20, k3, cap12, cap15  the no-servo case (2026-11-02, user asked to weigh position more): pose3g's candidates and
           gate, other selection rules (RULES). Open loop, pose3g's picks missed > 2 cm on 45 goals where the model
           predicted 4: the fixed point already zeroes the predicted miss, what remains is the model's own position
           error on far-turned commands, which its position std ranks (executed picks with std < 1 / 1-1.5 / >= 1.5 cm
           missed > 2 cm on 2 / 24 / 63 %). w20 weighs the whole position term double; k3 the std three times; cap12 and
           cap15 drop candidates with std above 1.2 / 1.5 cm. Run with pose3g (WRIST_METHODS=pose3g,w20,k3,cap12,cap15);
           a candidate several rules pick is executed once.
Executed open loop and with the position-only late servo, as before. The report sets the new model's rows beside
pose_invert's old-model ones (same goals, states and executions).

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    REACHWM_GOAL_REF=1 $PY scripts/experiments/reachwm_20261030/wrist_invert.py x   # pickle models_goalref_wrist
    $PY scripts/experiments/reachwm_20261030/wrist_invert.py h --procs 6
    WRIST_METHODS=pose3g $PY scripts/experiments/reachwm_20261030/wrist_invert.py h --procs 6
    WRIST_METHODS=pose3g,w20,k3,cap12,cap15 $PY scripts/experiments/reachwm_20261030/wrist_invert.py h --procs 6
    $PY scripts/experiments/reachwm_20261030/wrist_invert.py d
    # smoke test with the old model, which must reproduce pose_invert's pos and pose rows bit for bit
    WRIST_WM=old WRIST_OUT=/tmp/x $PY scripts/experiments/reachwm_20261030/wrist_invert.py h --procs 1 --limit 3
Writes outputs/reachwm_20261030/invert/wrist_invert/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose_invert as PV  # noqa: E402  (imports servo_gate and recheck, which sets RESTANCE_DATA=side)

RS, RC, E, IE, PI, SG = PV.RS, PV.RC, PV.E, PV.IE, PV.PI, PV.SG
OUT = Path(os.environ.get("WRIST_OUT", IE.OUT / "wrist_invert"))
WM = OUT / "wm_ref.pt"
# another model (2026-11-02, far-turn data): WRIST_MODEL_ROOT + WRIST_DATA (its training datasets) for phase x, WRIST_OUT
# for its results, WRIST_PREV = an earlier results dir the report sets beside it as "prev"
MODEL_ROOT = os.environ.get("WRIST_MODEL_ROOT", "models_goalref_wrist")
DATA_KEYS = os.environ.get("WRIST_DATA", "ref,side,wrist").split(",")
PREV = os.environ.get("WRIST_PREV", "")
ITERS3 = 6
SPREAD_MAX_DEG = 6.0
# selection rules on pose3g's gated candidates: (degrees that weigh like 1 cm, weight on the position std, std cap cm)
RULES = {"w20": (20.0, 1.0, None), "k3": (10.0, 3.0, None), "cap12": (10.0, 1.0, 1.2), "cap15": (10.0, 1.0, 1.5)}
METHODS = tuple(os.environ.get("WRIST_METHODS", "pos,pose,pose3").split(","))
SUF = "" if METHODS == ("pos", "pose", "pose3") else "_" + "_".join(METHODS)
SERVOS = tuple(os.environ.get("WRIST_SERVOS", ",".join(PV.SERVOS)).split(","))  # WRIST_SERVOS=open: eyes closed only
# WRIST_STUDENT=<student .pt> (2026-11-04, controller plus self model): every execution runs the relabelled controller
# from the command to the end of the hold, as gen_reach.py REACH_STUDENT recorded the model's data; planning and the
# replays before it stay frozen
STUDENT = os.environ.get("WRIST_STUDENT", "")
angle = PV.angle


def rotvec_of(R: np.ndarray) -> np.ndarray:
    """Rotation vector of a rotation matrix (inverse of gen_reach.rotvec, angles below pi)."""
    a = angle(np.eye(3), R)
    if a < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(a))
    return a * w


def cmd_h(st: dict, cmd) -> np.ndarray:
    """The wrist command as a rotation in the heading frame: None the wanted one, (dyaw, pitch) the old family, or a
    3x3 heading-frame rotation."""
    import gen_reach as G

    if cmd is None:
        return st["Rh"].T @ st["rot_w"]
    if isinstance(cmd, tuple):
        return st["Rh"].T @ (E.rot_z(st["yaw"] + cmd[0]) @ G.rot_y(cmd[1]))
    return cmd


def evaluate(model, m, st: dict, t: np.ndarray, pz: float, cmd) -> dict:
    """pose_invert.evaluate (same bits for None and (dyaw, pitch) commands), plus a 3x3 heading-frame command, and the
    predicted rotation R_pred (heading frame) in the result."""
    import gen_reach as G
    import torch

    sd, p, Rh, goal, start, x0b = (st[k] for k in ("sd", "p", "Rh", "goal", "start", "x0b"))
    if cmd is None:
        rot_w, rot6 = st["rot_w"], st["rot6"]
    else:
        rot_w = E.rot_z(st["yaw"] + cmd[0]) @ G.rot_y(cmd[1]) if isinstance(cmd, tuple) else Rh @ cmd
        rot_h = Rh.T @ rot_w
        rot6 = np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32)
    tw = np.r_[p[:2] + (Rh @ np.r_[t[:2], 0.0])[:2], t[2]]
    try:
        pr = PI.plan_record(PI.plan_of(m, st["c2"]["side"], tw, rot_w, pz))
    except np.linalg.LinAlgError:
        # 2026-11-03 (onpolicy x5 eval): the IK's least squares failed to converge on one candidate; the IK raises before
        # reach touches the motor (plan_of restores _blend), so the candidate is just infeasible: infinite losses, never
        # picked, and the fixed point stays where it is (pos = goal)
        return dict(t=t.copy(), cmd=cmd, pos=np.asarray(goal, float), Rp=np.eye(3), pred_err=np.inf, pred_std=np.inf,
                    pred_att=np.inf, att_spread=np.inf, dev=np.inf, rec=None, x0=None, l_pos=np.inf, l_att=np.inf, ik_failed=True)  # fmt: skip
    x0 = np.r_[x0b, PI.plan_features(pr, x0b[:29], start)]
    u = np.r_[sd, t, rot6, float(pz < 0), 0.0 if pz < 0 else pz]
    F = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None])  # noqa: E731
    pos, std, Rp, spread = PV.predict_pose(
        model, dict(x0=F(x0), u=F(u), target=F(t), rot6=F(rot6), side=torch.tensor([sd]))
    )
    R_want = Rh.T @ st["rot_w"]
    return dict(t=t.copy(), cmd=cmd, pos=pos, Rp=Rp, pred_err=100 * np.linalg.norm(pos - goal), pred_std=100 * np.linalg.norm(std),
                pred_att=float(np.degrees(angle(Rp, R_want))), att_spread=float(np.degrees(spread)),
                dev=float(np.degrees(angle(Rh.T @ rot_w, R_want))), rec=pr, x0=x0,
                l_pos=float(((pos - goal) ** 2).sum() + IE.LAMBDA * (std**2).sum()),
                l_att=float(PV.MU * (angle(Rp, R_want) ** 2 + IE.LAMBDA * spread**2)))  # fmt: skip


def clip_t(st: dict, t: np.ndarray) -> np.ndarray:
    lo, hi = PI.WIDE
    sd = st["sd"]
    y = np.clip(t[1] * sd, lo[1], hi[1]) * sd
    return np.r_[np.clip(t[0], lo[0], hi[0]), y, np.clip(t[2], lo[2], hi[2])].astype(np.float32).astype(np.float64)


def fixed_point(model, m, st: dict, t0: np.ndarray, pz: float, cmd) -> list[dict]:
    """pose_invert.fixed_point: the position fixed point with the wrist command held."""
    t = t0.astype(np.float32).astype(np.float64)
    evs = []
    for k in range(PV.ITERS_POSE + 1):
        ev = evaluate(model, m, st, t, pz, cmd)
        ev["k"] = k
        evs.append(ev)
        if k < PV.ITERS_POSE:
            t = clip_t(st, t + (st["goal"] - ev["pos"]))
    return evs


def fixed_point3(model, m, st: dict, ev0: dict, pz: float) -> list[dict]:
    """Joint fixed point from an evaluated seed: target by the predicted position miss, wrist command by the predicted
    rotation miss (R_cmd <- R_want R_pred^T R_cmd, heading frame), the command's turn off the wanted rotation capped at
    WRIST_MAX_DEG."""
    import gen_reach as G

    R_want = st["Rh"].T @ st["rot_w"]
    cap = np.radians(G.WRIST_MAX_DEG)
    ev, evs = ev0, []
    for k in range(1, ITERS3 + 1):
        t = clip_t(st, ev["t"] + (st["goal"] - ev["pos"]))
        R = R_want @ ev["Rp"].T @ cmd_h(st, ev["cmd"])
        v = rotvec_of(R_want.T @ R)  # the command's turn off the wanted rotation, jaw frame
        if np.linalg.norm(v) > cap:
            R = R_want @ G.rotvec(v * cap / np.linalg.norm(v))
        ev = evaluate(model, m, st, t, pz, R)
        ev["k"] = k
        evs.append(ev)
    return evs


def refine(model, m, st: dict, t0: np.ndarray, pz: float) -> dict[str, tuple[dict, list]]:
    """pos and pose exactly as pose_invert.refine; pose3 = pose's iterates plus the joint fixed points from the wanted
    command (k = 0 of pos) and from pose's best."""
    Lp = lambda ev: ev["l_pos"]  # noqa: E731
    La = lambda ev: ev["l_pos"] + ev["l_att"]  # noqa: E731
    out = {}
    pos = fixed_point(model, m, st, t0, pz, None)
    for ev in pos:
        ev["stage"] = "pos"
    out["pos"] = (min(pos, key=Lp), pos, Lp)
    dy0, p0 = st["c2"]["dyaw"], st["c2"]["pitch"]
    cmds = {(0.0, 0.0): None}
    for a in PV.CMD_DYAWS:
        for b in PV.CMD_PITCHES:
            key = (round(float(a) - dy0, 9), round(float(b) - p0, 9))
            if key != (0.0, 0.0):
                cmds.setdefault(key, (float(a), float(b)))
    grid = []
    for key, c in cmds.items():
        ev = evaluate(model, m, st, t0.astype(np.float32).astype(np.float64), pz, c)
        ev.update(stage="grid", k=0)
        grid.append(ev)
    pose = list(grid)
    for g in sorted(grid, key=La)[: PV.NB]:
        fp = fixed_point(model, m, st, t0, pz, g["cmd"])
        for ev in fp:
            ev["stage"] = "fp"
        pose += fp
    out["pose"] = (min(pose, key=La), pose, La)
    p3 = list(pose)
    for name, seed in (("want", pos[0]), ("best", out["pose"][0])):
        f3 = fixed_point3(model, m, st, seed, pz)
        for ev in f3:
            ev["stage"] = f"fp3_{name}"
        p3 += f3
    out["pose3"] = (min(p3, key=La), p3, La)
    gated = [ev for ev in p3 if ev["att_spread"] <= SPREAD_MAX_DEG] or [pos[0]]
    out["pose3g"] = (min(gated, key=La), p3, La)
    for nm, (dpc, ks, cap) in RULES.items():
        mu = (0.01 / np.radians(dpc)) ** 2

        def Lr(ev, mu=mu, ks=ks):
            return float(
                (ev["pred_err"] / 100) ** 2
                + (ks * ev["pred_std"] / 100) ** 2
                + mu * (np.radians(ev["pred_att"]) ** 2 + np.radians(ev["att_spread"]) ** 2)
            )

        ok = [ev for ev in gated if cap is None or ev["pred_std"] <= cap] or [pos[0]]
        out[nm] = (min(ok, key=Lr), p3, Lr)
    res = {}
    for meth, (best, evs, L) in out.items():
        best = dict(best, loss=L(best))
        res[meth] = (
            best,
            [
                [ev["k"], L(ev), ev["pred_err"], ev["pred_std"], ev["pred_att"], ev["att_spread"], ev["dev"]]
                for ev in evs
            ],
        )
    return res


def worker(wid: int, gids: list[int]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF and DT.X0_DIM == 206 + PI.REF_DIM
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    gate = SG.Gate(env.m)
    wm = RS.WM_REF if os.environ.get("WRIST_WM") == "old" else WM
    model = torch.load(wm, map_location="cpu", weights_only=False).eval()
    print(f"p{wid} model {wm}" + (f", controller {STUDENT}" if STUDENT else ""), flush=True)
    G.STUDENT, reach = STUDENT, env.m.reach

    def student_reach(*a, **kw):
        G.use_weights(env.m, "student")
        return reach(*a, **kw)

    rows = {r["gid"]: r for r in SG.plans()["rows"]}
    res = []
    for gid in gids:
        p = rows[gid]
        tic = time.time()
        if p["kind"] == "none":

            def replay():
                c, walked = IE.setup(env, gid)
                return None if env.m.fell_at is not None else (c, walked)
        else:

            def replay():
                r = (
                    RS.step(env, rec, gid, p["d1"])
                    if p["kind"] == "one"
                    else RC.two_steps(env, rec, gid, p["d1"], p["d2"])
                )
                return None if r is None else (r if p["kind"] == "one" else r[0])

        st = PV.state(env, rec, p, replay)
        if st is None:
            continue
        walked = st.get("walked", True)
        msg = [f"p{wid} goal {gid} {p['kind']}"]
        plans_ = refine(model, env.m, st, p["t"], p["pz"])  # all from the planning state, before any execution
        R_want = st["Rh"].T @ st["rot_w"]
        done = {}  # (target, command, servo) -> its record: a candidate several rules pick is executed once
        for meth in METHODS:
            best, trace = plans_[meth]
            cmd = best["cmd"]
            rv = np.degrees(rotvec_of(R_want.T @ cmd_h(st, cmd)))
            key = best["t"].tobytes() + (b"want" if cmd is None else np.asarray(cmd, float).tobytes())
            for v in SERVOS:
                if (key, v) in done:
                    res.append(dict(done[(key, v)], method=meth, trace=np.array(trace).tolist(), reused=True))
                    continue
                if replay() is None:
                    break
                gate.mode, gate.radius = ("orig" if v == "open" else v), None
                gate.reset()
                cfg = env.m.cfg
                env.m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)
                servo = v != "open"
                goal = st["goal"].astype(np.float32)
                c_cmd, rot_h = st["c2"], None
                if isinstance(cmd, tuple):
                    c_cmd = dict(st["c2"], dyaw=cmd[0], pitch=cmd[1])
                elif cmd is not None:
                    rot_h = cmd
                if STUDENT:
                    env.m.reach = student_reach
                seg = IE.execute(env, rec, c_cmd, best["t"].astype(np.float32), p["pz"], servo, walked,
                                 servo_goal_h=goal if servo else None, rot_h=rot_h)  # fmt: skip
                if STUDENT:
                    env.m.reach = reach
                    G.use_weights(env.m, "frozen")
                env.m.cfg = cfg
                tracked = PI.plan_record(rec.plan)
                same_plan = all(np.array_equal(tracked[k], best["rec"][k]) for k in tracked)
                s = PI.summarize(seg, tracked, goal[None])
                R_w = E.rot_z(float(seg["start"][3]) + st["c2"]["dyaw"]) @ G.rot_y(st["c2"]["pitch"])
                att = float(np.degrees(angle(R_w, env.m.jaw_centre(st["c2"]["side"])[1])))
                res.append(dict(gid=gid, kind=p["kind"], method=meth, servo=v, err=float(s["err"][0]), att_deg=att,
                                fell=bool(s["fell"][0]), same_plan=same_plan, x0_diff=float(np.abs(s["x0"][0] - best["x0"]).max()),
                                k=best["k"], stage=best["stage"], cmd_rv_deg=rv.tolist(), pred_err=float(best["pred_err"]),
                                pred_std=float(best["pred_std"]), pred_att=float(best["pred_att"]),
                                att_spread=float(best["att_spread"]), dev=float(best["dev"]), trace=np.array(trace).tolist()))  # fmt: skip
                done[(key, v)] = res[-1]
            a = [x for x in res if x["gid"] == gid and x["method"] == meth]
            msg.append(f"{meth} {best['stage']} k {best['k']} pred att {best['pred_att']:.1f} spread {best['att_spread']:.1f} dev {best['dev']:.0f} | " +
                       " ".join(f"{x['servo']} {x['err']:.2f} cm {x['att_deg']:.1f} deg" for x in a))  # fmt: skip
        print(" | ".join(msg) + f" | {time.time() - tic:.0f} s", flush=True)
    json.dump(res, open(OUT / f"h_w{wid}{SUF}.json", "w"))


def launch(procs: int, gpus: str, limit: int) -> None:
    gids = [r["gid"] for r in SG.plans()["rows"]][: limit or None]
    print(f"{len(gids)} core goals x {METHODS} x {SERVOS}", flush=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(OUT / f"h_w{w}{SUF}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "worker", "--wid", str(w), "--gids", ",".join(map(str, gids[w::procs]))]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [p.wait() for p in ps], flush=True)


def export() -> None:
    """Phase x (REACHWM_GOAL_REF=1): models_goalref_wrist, pickled for the workers."""
    import data as DT
    import torch
    import train as TR

    assert DT.GOAL_REF
    spec = ",".join(str(DT.OUT / f"data_{k}" / "all.npz") for k in DATA_KEYS)
    D = DT.load(spec)
    ref = TR.load_model("E3_ensemble", 1.0, D, "cpu", DT.OUT / MODEL_ROOT)
    ref.qfix = DT.QFIX  # which form of the joint feature it learned (data.check_qfix)
    WM.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ref, WM)
    print(f"wm_ref.pt from {MODEL_ROOT} ({len(D['n_steps'])} segments) -> {WM}", flush=True)


def report() -> None:
    import gen_reach as G
    from scipy.stats import spearmanr

    load = lambda d: [x for p in sorted(Path(d).glob("h_w*.json")) for x in json.load(open(p))]  # noqa: E731
    by = {}
    redo = {}  # a method run twice (pose3g alone, then again with the no-servo rules): must give the same execution
    for x in load(OUT):
        d = by.setdefault(("new", x["method"], x["servo"]), {})
        if x["gid"] in d:
            redo[x["method"]] = redo.get(x["method"], 0) + int(
                d[x["gid"]]["err"] == x["err"] and d[x["gid"]]["att_deg"] == x["att_deg"]
            )
        d[x["gid"]] = x
    for x in load(PV.OUT):
        by.setdefault(("old", x["method"], x["servo"]), {})[x["gid"]] = x
    if PREV:
        for x in load(PREV):
            by.setdefault(("prev", x["method"], x["servo"]), {})[x["gid"]] = x
    gids = sorted(set.intersection(*(set(v) for v in by.values())))
    E_ = lambda w, m, v: np.array([99.0 if by[(w, m, v)][g]["fell"] else by[(w, m, v)][g]["err"] for g in gids])  # noqa: E731
    A_ = lambda w, m, v: np.array([by[(w, m, v)][g]["att_deg"] for g in gids])  # noqa: E731
    X = [x for (w, _, _), d in by.items() if w == "new" for x in d.values()]
    L = [f"{len(gids)} core goals; falls {sum(x['fell'] for x in X)}; plan matches the executed one "
         f"{np.mean([x['same_plan'] for x in X]):.3f}, max x0 diff {max(x['x0_diff'] for x in X):.1e}"]  # fmt: skip
    L += [f"{m} executed twice: identical on {n} of the {2 * len(gids)} repeats" for m, n in redo.items()]
    L += [
        f"models: old = pose_invert's (no wide-wrist data), new = {OUT.name}"
        + (f", prev = {Path(PREV).name}" if PREV else "")
    ]
    if os.environ.get("WRIST_WM") == "old":  # smoke test: the old model must reproduce pose_invert bit for bit
        for m in ("pos", "pose"):
            for v in SERVOS:
                same = sum(
                    by[("new", m, v)][g]["err"] == by[("old", m, v)][g]["err"]
                    and by[("new", m, v)][g]["att_deg"] == by[("old", m, v)][g]["att_deg"]
                    for g in gids
                )
                L.append(f"old model {m} {v}: reproduces pose_invert on {same}/{len(gids)}")
    order = ("pos", "pose", "pose3", "pose3g", *RULES)
    rows = [(w, m) for w in ("old", "prev", "new") for m in order if (w, m, "late_pos") in by]
    L += ["", "| model | inversion | servo | position median cm | <1 cm | >2 cm (n) | p90 | attitude median deg | <10 deg | "
          "<20 deg | >30 deg (n) | p90 deg |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    out = dict(n=len(gids), rows=[])
    for w, m in rows:
        for v in SERVOS:
            e, a = E_(w, m, v), A_(w, m, v)
            r = dict(model=w, method=m, servo=v, median=float(np.median(e)), lt1=float(np.mean(e < 1)), gt2=int(np.sum(e > 2)),
                     p90=float(np.percentile(e, 90)), att_median=float(np.median(a)), att_lt10=float(np.mean(a < 10)),
                     att_lt20=float(np.mean(a < 20)), att_gt30=int(np.sum(a > 30)), att_p90=float(np.percentile(a, 90)))  # fmt: skip
            out["rows"].append(r)
            L.append(f"| {w} | {m} | {v} | {r['median']:.2f} | {r['lt1']:.2f} | {r['gt2']} | {r['p90']:.2f} | {r['att_median']:.1f} | "
                     f"{r['att_lt10']:.2f} | {r['att_lt20']:.2f} | {r['att_gt30']} | {r['att_p90']:.1f} |")  # fmt: skip
    # paired against the old position-only inversion, by how far the wrist ended off there (servo on)
    a0, e0 = A_("old", "pos", "late_pos"), E_("old", "pos", "late_pos")
    var = [(w, m) for w, m in rows if (w, m) != ("old", "pos")]
    L += ["", "| attitude miss with old pos, late_pos | n | old pos | " + " | ".join(f"{w} {m}" for w, m in var) + " |",
          "|---|---|---|" + "---|" * len(var)]  # fmt: skip
    for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 180), (0, 180)):
        k = (a0 >= lo) & (a0 < hi)
        if k.any():
            L.append(f"| {lo}-{hi} deg | {int(k.sum())} | {np.median(a0[k]):.1f} | " +
                     " | ".join(f"{np.median(A_(w, m, 'late_pos')[k]):.1f}" for w, m in var) + " |")  # fmt: skip
    # where the wrist-compensating picks came from, how far they turned, and how much of it is roll (jaw x axis)
    L.append("")
    for w, m in rows:
        if w == "old" or m in ("pos", "pose"):
            continue
        b3 = [by[(w, m, "late_pos")][g] for g in gids]
        dev3 = np.array([x["dev"] for x in b3])
        st = {}
        for x in b3:
            st[x["stage"]] = st.get(x["stage"], 0) + 1
        rv = np.array([x["cmd_rv_deg"] for x in b3])
        L.append(f"{w} {m} best from: {st}; command turned off the wanted rotation median {np.median(dev3):.1f} deg, p90 "
                 f"{np.percentile(dev3, 90):.1f}, at the {G.WRIST_MAX_DEG:.0f} deg cap on {int(np.sum(dev3 > G.WRIST_MAX_DEG - 0.5))}; "
                 f"turn about the jaw x / y / z axes, median |deg| {np.round(np.median(np.abs(rv), 0), 1).tolist()}")  # fmt: skip
    # the no-servo view: open loop first, the servo case beside it
    nr = [(w, m) for w, m in rows if w != "old" and m in ("pos", "pose3g", *RULES)]
    if nr:
        L += ["", "| model | selection | open: position median cm | <1 cm | >2 cm (n) | p90 | wrist median deg | <10 deg | "
              ">30 deg (n) | servo: >2 cm (n) | servo wrist median deg | position std of the pick median cm | "
              "open >2 cm on picks turned 30-60 / >= 60 deg |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
        for w, m in nr:
            e, a = E_(w, m, "open"), A_(w, m, "open")
            es, as_ = E_(w, m, "late_pos"), A_(w, m, "late_pos")
            sd = np.median([by[(w, m, "open")][g]["pred_std"] for g in gids])
            dv = np.array([by[(w, m, "open")][g]["dev"] for g in gids])
            mid, far = (dv >= 30) & (dv < 60), dv >= 60
            L.append(f"| {w} | {m} | {np.median(e):.2f} | {np.mean(e < 1):.2f} | {int(np.sum(e > 2))} | {np.percentile(e, 90):.2f} | "
                     f"{np.median(a):.1f} | {np.mean(a < 10):.2f} | {int(np.sum(a > 30))} | {int(np.sum(es > 2))} | {np.median(as_):.1f} | "
                     f"{sd:.2f} | {int(np.sum(e[mid] > 2))}/{int(mid.sum())} / {int(np.sum(e[far] > 2))}/{int(far.sum())} |")  # fmt: skip
    L.append("")
    for w, m in rows:
        pa = np.array([by[(w, m, "open")][g]["pred_att"] for g in gids])
        aa = A_(w, m, "open")
        spr = np.array([by[(w, m, "open")][g]["att_spread"] for g in gids])
        pe = np.array([by[(w, m, "open")][g]["pred_err"] for g in gids])
        L.append(f"{w} {m} open loop: predicted attitude miss median {np.median(pa):.1f} deg vs measured {np.median(aa):.1f}; "
                 f"|predicted - measured| median {np.median(np.abs(pa - aa)):.1f} deg, p90 {np.percentile(np.abs(pa - aa), 90):.1f}; "
                 f"spread / |predicted - measured| Spearman {spearmanr(spr, np.abs(pa - aa)).correlation:.2f}; position predicted > 2 cm "
                 f"{int(np.sum(pe > 2))} vs measured {int(np.sum(E_(w, m, 'open') > 2))}")  # fmt: skip
    json.dump(out, open(OUT / "results.json", "w"), indent=1)
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["x", "h", "d", "worker"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0, help="first N goals only (smoke test)")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--gids", default="")
    a = ap.parse_args(argv)
    if a.what == "x":
        export()
    elif a.what == "h":
        OUT.mkdir(parents=True, exist_ok=True)
        launch(a.procs, a.gpus, a.limit)
    elif a.what == "d":
        report()
    else:
        worker(a.wid, [int(x) for x in a.gids.split(",") if x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
