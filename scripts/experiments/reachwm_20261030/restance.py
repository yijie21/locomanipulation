"""Step, then reach (2026-11-01, after plan-aware inversion): the 16 % of core goals plan_invert.py's ff+servo still misses by
more than 2 cm are the ones its ensemble is unsure about (std 2.06 vs 0.75 cm), and they sit at the workspace's edges
(close to the body, across it, low, far). When the plan-aware std at the starting stance is high, move the feet first
(Motor.sidestep, heading kept), then invert and reach from where the robot actually stands.

Where to step, chosen before moving, two ways:
    wm    the plain E3 ensemble scores every displacement d of the grid STEP_X x STEP_Y as if the robot stood d away
          with the same body (a translation: the goal becomes goal - d in the heading frame), each over the 6 pelvis
          candidates with invert_eval's loss |miss|^2 + LAMBDA * |std|^2; the lowest score + STEP_COST * |d| wins, and
          the robot only steps if that beats standing still by MARGIN_CM
    geo   no model: the smallest step that puts the goal inside COMFORT, the box where the plan-aware std is low (the
          interquartile box of the core goals with std <= 1.2 cm), snapped to the same grid
After the step: plan_invert's fixed-point plan-aware inversion from the new state (start aims = phase s's aims for the
chosen d, re-expressed against the new stance), then the chosen command open loop (step_aim) and as feedforward with the
servo on the wanted position (step_ff_servo). The wanted position and wrist rotation are kept in the world.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/reachwm_20261030/restance.py s           # stance scores and choices
    $PY scripts/experiments/reachwm_20261030/restance.py h --procs 6                        # step, plan, execute
    $PY scripts/experiments/reachwm_20261030/restance.py d                                  # tables
Writes outputs/reachwm_20261030/invert/restance/.

RESTANCE_DATA=side (2026-11-01, short steps in the training data): both world models retrained on data_ref + data_side
(gen_reach.py REACH_SIDESTEP=1, reaches that start right after a short heading-kept step). Phase s scores with
models_side/, phase x exports models_goalref_side/ to OUT/wm_ref.pt for phase h, OUT is invert/restance_side/, and
phase d also pairs every goal with the run above. The gate and the no-step baseline stay plan_invert's.
    RESTANCE_DATA=side REACHWM_GOAL_REF=1 $PY .../restance.py x
    RESTANCE_DATA=side CUDA_VISIBLE_DEVICES=1 $PY .../restance.py s
    RESTANCE_DATA=side $PY .../restance.py h --procs 6
    RESTANCE_DATA=side $PY .../restance.py d
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
import env as E  # noqa: E402
import invert_eval as IE  # noqa: E402
import plan_invert as PI  # noqa: E402

SIDE = os.environ.get("RESTANCE_DATA", "") == "side"
OUT = IE.OUT / ("restance_side" if SIDE else "restance")
DATA_SPEC = ",".join(str(IE.OUT.parent / d / "all.npz") for d in ("data_ref", "data_side")) if SIDE else None
PLAIN_ROOT = IE.OUT.parent / "models_side" if SIDE else None  # None: train.py's MODELS
WM_REF = OUT / "wm_ref.pt" if SIDE else PI.PLAN / "wm_ref.pt"
STEP_X = np.round(np.arange(-0.25, 0.151, 0.05), 2)  # forward, m (negative: step back)
STEP_Y = np.round(np.arange(-0.15, 0.151, 0.05), 2)  # lateral, mirrored to the right hand (negative: toward the hand)
STEP_COST = 1.0  # cm of predicted score per m of step
MARGIN_CM = 0.2
COMFORT = (np.array([0.35, -0.28]), np.array([0.57, -0.02]))  # forward, mirrored lateral
GATE_RUN = 0.83  # plan-aware std (cm) above which the sim runs a goal (the report sweeps the gate from here up)
METHODS = ("wm", "geo")


def grid() -> np.ndarray:
    return np.array([(x, y) for x in STEP_X for y in STEP_Y])


def invert_plain(model, x0, rot6, sd, g, pz, dev, CH: int = 50000):
    """invert_eval's loss |miss|^2 + LAMBDA * |std|^2 minimised over the command target by Adam on the plain ensemble,
    one row per (start features x0, wrist rot6, side, goal g in the heading frame, pelvis candidate pz); aims, losses."""
    import torch

    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
    lo, hi = T(PI.WIDE[0]), T(PI.WIDE[1])
    aims, losses = np.zeros((len(g), 3), np.float32), np.zeros(len(g))
    for a in range(0, len(g), CH):
        sl = slice(a, a + CH)
        x0_, rot6_, sd_ = T(x0[sl]), T(rot6[sl]), T(sd[sl])
        g_ = T(g[sl])
        stand, pzv, s = T((pz[sl] < 0).astype(np.float32)), T(np.where(pz[sl] < 0, 0.0, pz[sl])), sd_[:, None]
        t = g_.clone()
        with torch.no_grad():
            y = t[:, 1:2] * s
            t[:, 0].clamp_(lo[0], hi[0])
            t[:, 2].clamp_(lo[2], hi[2])
            t[:, 1:2] = y.clamp(lo[1], hi[1]) * s
        t.requires_grad_(True)
        opt = torch.optim.Adam([t], lr=IE.LR)

        def run():
            u = torch.cat([sd_[:, None], t, rot6_, stand[:, None], pzv[:, None]], -1)
            o = model.predict(dict(x0=x0_, u=u, target=t, rot6=rot6_, side=sd_))
            return ((o["pos"] - g_) ** 2).sum(-1) + IE.LAMBDA * (o["pos_std"] ** 2).sum(-1)

        for _ in range(IE.ITERS):
            loss = run()
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                y = t[:, 1:2] * s
                t[:, 0].clamp_(lo[0], hi[0])
                t[:, 2].clamp_(lo[2], hi[2])
                t[:, 1:2] = y.clamp(lo[1], hi[1]) * s
        with torch.no_grad():
            losses[sl] = run().cpu().numpy()
        aims[sl] = t.detach().cpu().numpy()
        if len(g) > CH:
            print(f"rows {min(a + CH, len(g))}/{len(g)}", flush=True)
    return aims, losses


def choose(bad: np.ndarray, Gd: np.ndarray, k_still: int) -> np.ndarray:
    """Per row of bad (N, K) cm: the displacement with the lowest score + STEP_COST * |d|, if it beats standing still
    by MARGIN_CM, else standing still."""
    N = len(bad)
    pen = bad + STEP_COST * np.linalg.norm(Gd, axis=1)[None]
    k = pen.argmin(1)
    return np.where(bad[np.arange(N), k_still] - pen[np.arange(N), k] >= MARGIN_CM, k, k_still)


def geo_k(goal: np.ndarray, side: np.ndarray, Gd: np.ndarray) -> np.ndarray:
    """The smallest grid step that puts each goal (heading frame) inside COMFORT."""
    g_m = np.c_[goal[:, 0], goal[:, 1] * side]
    need = g_m - np.clip(g_m, *COMFORT)  # the stance moves toward the goal by what sticks out of the box
    need = np.c_[np.clip(np.round(need[:, 0] / 0.05) * 0.05, STEP_X[0], STEP_X[-1]),
                 np.clip(np.round(need[:, 1] / 0.05) * 0.05, STEP_Y[0], STEP_Y[-1])]  # fmt: skip
    return np.array([int(np.argmin(np.abs(Gd - v).sum(1))) for v in need])


def score() -> None:
    """Phase s (GPU): plain-E3 inversion for every goal x displacement x pelvis candidate; the two choices."""
    import data as DT
    import torch
    import train as TR

    assert not DT.GOAL_REF, "run phase s without REACHWM_GOAL_REF"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    I = np.load(PI.PLAN / "init.npz")
    D = DT.load(DATA_SPEC)
    model = TR.load_model("E3_ensemble", 1.0, D, dev, PLAIN_ROOT or TR.MODELS)
    del D
    for p in model.parameters():
        p.requires_grad_(False)
    Gd = grid()
    N, K, P = len(I["gid"]), len(Gd), len(I["pz"])
    side = I["side"].astype(np.float64)
    d_h = np.zeros((N, K, 3))
    d_h[:, :, :2] = Gd[None]
    d_h[:, :, 1] *= side[:, None]
    goal_s = I["goal"][:, None, :] - d_h  # (N, K, 3) the goal seen from the displaced stance
    # rows (goal, displacement, pelvis candidate)
    ig = np.repeat(np.arange(N), K * P)
    ik = np.tile(np.repeat(np.arange(K), P), N)
    ic = np.tile(np.arange(P), N * K)
    aims, losses = invert_plain(model, I["x0"][ig], I["rot6"][ig], side[ig], goal_s[ig, ik], I["pz"][ic, ig], dev)
    L = losses.reshape(N, K, P)
    A = aims.reshape(N, K, P, 3)
    bad = 100 * np.sqrt(L.min(-1))  # (N, K) cm
    k_still = int(np.where((Gd[:, 0] == 0) & (Gd[:, 1] == 0))[0][0])
    k_wm = choose(bad, Gd, k_still)
    k_geo = geo_k(I["goal"], side, Gd)
    out = dict(gid=I["gid"], grid=Gd, bad=bad, k_still=k_still)
    for nm, k in (("wm", k_wm), ("geo", k_geo)):
        out[f"{nm}_k"] = k
        out[f"{nm}_d"] = d_h[np.arange(N), k]  # heading frame at the starting stance, m
        out[f"{nm}_goal_s"] = goal_s[np.arange(N), k]
        out[f"{nm}_t"] = A[np.arange(N), k].transpose(1, 0, 2)  # (P, N, 3) start aims, seen from the displaced stance
        out[f"{nm}_bad"] = bad[np.arange(N), k]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "choice.npz", **out)
    s0 = bad[:, k_still]
    print(f"{N} goals x {K} displacements x {P} pelvis; predicted score standing still median {np.median(s0):.2f} cm")
    for nm in METHODS:
        k = out[f"{nm}_k"]
        st = k != k_still
        print(f"{nm}: steps on {st.mean():.2f}; predicted score where it steps {np.median(s0[st]):.2f} -> "
              f"{np.median(out[f'{nm}_bad'][st]):.2f} cm; |d| median {100 * np.median(np.linalg.norm(Gd[k[st]], axis=1)):.0f} cm")  # fmt: skip


def export() -> None:
    """Phase x (RESTANCE_DATA=side, REACHWM_GOAL_REF=1): the retrained plan-aware ensemble, pickled for the workers."""
    import data as DT
    import torch
    import train as TR

    assert SIDE and DT.GOAL_REF
    R = DT.load(DATA_SPEC)
    ref = TR.load_model("E3_ensemble", 1.0, R, "cpu", IE.OUT.parent / "models_goalref_side")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(ref, WM_REF)
    print(f"wm_ref.pt from models_goalref_side ({len(R['n_steps'])} segments) -> {WM_REF}", flush=True)


def x0_now(rec, side: float, walked: bool, start: np.ndarray) -> np.ndarray:
    """The 206 start features data.derive builds for a command issued now (the recorder's samples so far)."""
    import data as DT

    seg = dict(S=np.array(rec.S), W=np.array(rec.W), J=np.array(rec.J), C=np.array(rec.C), k0=len(rec.S) - 1,
               side=side, target=np.zeros(3, np.float32), dyaw=0.0, pitch=0.0, rot6=np.zeros(6, np.float32),
               pelvis_z=-1.0, start=start, walked=walked, chain=0, fell=False, ok=True, jaw_err_cm=-1.0,
               att_err_deg=-1.0, ik_err_cm=-1.0, reach_s=0.0, reset=0)  # fmt: skip
    D = IE.pack([seg])
    D.update(ref_q=np.zeros((1, 29), np.float32), ref_pel=np.r_[0, 0, 0, 1, 0, 0, 0][None].astype(np.float32),
             ref_s=np.zeros(1, np.float32))  # fmt: skip
    DT.derive(D)
    return D["x0"][0, :206]


def step(env, rec, gid: int, d_h: np.ndarray):
    """Goal `gid`'s reset (and walk), the PRE_S hold the starting stance is judged from, then the sidestep by d_h
    (heading frame). Returns the goal command, walked, the wanted world position and the starting yaw; None once fallen."""
    import gen_reach as G

    c, walked = IE.setup(env, gid)
    m = env.m
    if m.fell_at is not None:
        return None
    rec.start(c["side"])
    m.hold(G.PRE_S)
    rec.on = False
    p, Rp = env.pelvis()
    yaw = E.yaw_of(Rp)
    g = c["target"].astype(np.float32).astype(np.float64)
    goal_w = np.r_[p[:2] + (E.rot_z(yaw) @ np.r_[g[:2], 0.0])[:2], g[2]]
    out = m.sidestep(left=float(d_h[1]), forward=float(d_h[0]))
    if m.fell_at is not None:
        return None
    return c, walked, goal_w, yaw, p, out


def post_step(env, rec, c: dict, goal_w: np.ndarray, yaw0: float, p0: np.ndarray) -> dict:
    """The PRE_S hold after a step and what a reach from here is judged by: the goal and wrist rotation re-expressed
    in the new heading frame (both kept in the world), the 206 start features, and how far the feet moved from p0."""
    import gen_reach as G

    m = env.m
    sd = 1.0 if c["side"] == "right" else -1.0
    rec.start(c["side"])
    m.hold(G.PRE_S)
    rec.on = False
    p, Rp = env.pelvis()
    yaw = E.yaw_of(Rp)
    Rh = E.rot_z(yaw)
    moved = (E.rot_z(yaw0).T @ np.r_[p[:2] - p0[:2], 0.0])[:2]
    goal = Rh.T @ np.r_[goal_w[:2] - p[:2], 0.0]
    goal = np.r_[goal[:2], goal_w[2]].astype(np.float32).astype(np.float64)
    c2 = dict(c, dyaw=float(c["dyaw"] + E.wrap(yaw0 - yaw)))  # the wanted wrist rotation stays the world one
    rot_w = E.rot_z(yaw + c2["dyaw"]) @ G.rot_y(c["pitch"])
    rot_h = Rh.T @ rot_w
    rot6 = np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32)
    start = np.r_[p, yaw].astype(np.float32)
    return dict(sd=sd, p=p, yaw=yaw, Rh=Rh, moved=moved, goal=goal, c2=c2, rot_w=rot_w, rot6=rot6, start=start,
                x0b=x0_now(rec, sd, True, start))  # fmt: skip


def invert_here(model, m, c: dict, st: dict, t0: np.ndarray, goal_s: np.ndarray, pz_c: np.ndarray):
    """plan_invert's fixed-point plan-aware inversion from the state post_step judged, for every pelvis candidate;
    the start aims t0 (P, 3) were found for goal_s and are re-expressed against st's goal. Best row and the trace."""
    import torch

    lo, hi = PI.WIDE
    sd, p, Rh, goal, rot_w, rot6, start, x0b = (
        st[k] for k in ("sd", "p", "Rh", "goal", "rot_w", "rot6", "start", "x0b")
    )
    best, trace = None, []
    for ci in range(len(t0)):
        pz = float(pz_c[ci])
        t = (goal + (t0[ci] - goal_s)).astype(np.float32).astype(np.float64)
        for k in range(PI.ITERS + 1):
            tw = np.r_[p[:2] + (Rh @ np.r_[t[:2], 0.0])[:2], t[2]]
            pr = PI.plan_record(PI.plan_of(m, c["side"], tw, rot_w, pz))
            x0 = np.r_[x0b, PI.plan_features(pr, x0b[:29], start)]
            u = np.r_[sd, t, rot6, float(pz < 0), 0.0 if pz < 0 else pz]
            F = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None])  # noqa: E731
            with torch.no_grad():
                o = model.predict(dict(x0=F(x0), u=F(u), target=F(t), rot6=F(rot6), side=torch.tensor([sd])))
            pos, std = o["pos"][0].numpy().astype(np.float64), o["pos_std"][0].numpy().astype(np.float64)
            loss = float(((pos - goal) ** 2).sum() + IE.LAMBDA * (std**2).sum())
            trace.append([ci, k, loss, 100 * np.linalg.norm(pos - goal), 100 * np.linalg.norm(std)])
            if best is None or loss < best["loss"]:
                best = dict(loss=loss, ci=ci, k=k, t=t.copy(), pz=pz, pred=pos, std=std, x0=x0, rec=pr)
            if k < PI.ITERS:
                t = t + (goal - pos)
                y = np.clip(t[1] * sd, lo[1], hi[1]) * sd
                t = (
                    np.r_[np.clip(t[0], lo[0], hi[0]), y, np.clip(t[2], lo[2], hi[2])]
                    .astype(np.float32)
                    .astype(np.float64)
                )
    return best, trace


def worker(wid: int, jobs: list[tuple[int, str]]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF and DT.X0_DIM == 206 + PI.REF_DIM
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    C = np.load(OUT / "choice.npz")
    model = torch.load(WM_REF, map_location="cpu", weights_only=False).eval()
    PZ = np.load(PI.PLAN / "init.npz")["pz"]
    rows = {}
    for gid, meth in jobs:
        j = int(np.where(C["gid"] == gid)[0][0])
        d_h = C[f"{meth}_d"][j]
        tic = time.time()
        r = step(env, rec, gid, d_h)
        if r is None:
            rows.setdefault(f"{meth}_fellstep", []).append(gid)
            continue
        c, walked0, goal_w, yaw0, p0, sout = r
        st = post_step(env, rec, c, goal_w, yaw0, p0)
        sd, goal, moved, c2 = st["sd"], st["goal"], st["moved"], st["c2"]
        best, trace = invert_here(model, env.m, c, st, C[f"{meth}_t"][:, j], C[f"{meth}_goal_s"][j], PZ[:, j])
        plan_s = time.time() - tic
        for v, servo in (("aim", False), ("ff_servo", True)):
            if step(env, rec, gid, d_h) is None:
                break
            seg = IE.execute(env, rec, c2, best["t"].astype(np.float32), best["pz"], servo, True,
                             servo_goal_h=goal.astype(np.float32) if servo else None)  # fmt: skip
            tracked = PI.plan_record(rec.plan)
            same_plan = all(np.array_equal(tracked[k], best["rec"][k]) for k in tracked)
            s = PI.summarize(seg, tracked, goal.astype(np.float32)[None])
            rows.setdefault(f"{meth}_{v}", []).append(dict(
                gid=gid, walked=walked0, d=d_h[:2], moved=moved, dyaw_step=float(np.degrees(E.wrap(st["yaw"] - yaw0))),
                stop_err_cm=float((sout.info or {}).get("stop_err_cm", -1.0)), goal_t=goal.astype(np.float32), side=sd,
                t=best["t"], pz=best["pz"], ci=best["ci"], k=best["k"], pred_err=100 * np.linalg.norm(best["pred"] - goal),
                pred_std=100 * np.linalg.norm(best["std"]), same_plan=same_plan,
                x0_diff=float(np.abs(s["x0"][0] - best["x0"]).max()), plan_s=plan_s, err=s["err"][0], fell=s["fell"][0],
                settled=s["settled"][0]))  # fmt: skip
        rows.setdefault(f"{meth}_trace", []).append(dict(gid=gid, trace=np.array(trace)))
        a, f = rows[f"{meth}_aim"][-1], rows[f"{meth}_ff_servo"][-1]
        print(f"h{wid} goal {gid} {meth} d {100 * d_h[0]:+.0f},{100 * d_h[1]:+.0f} moved {100 * moved[0]:+.1f},"
              f"{100 * moved[1]:+.1f} cm, {time.time() - tic:.0f} s | std {a['pred_std']:.2f} | aim {a['err']:.2f} "
              f"ff_servo {f['err']:.2f} same_plan {a['same_plan']} x0_diff {a['x0_diff']:.1e}", flush=True)  # fmt: skip
    flat = {}
    for v, rs in rows.items():
        if v.endswith("_fellstep"):
            flat[v] = np.array(rs)
        elif v.endswith("_trace"):
            flat[f"{v}_gid"] = np.array([r["gid"] for r in rs])
            flat[v] = np.array([r["trace"] for r in rs])
        else:
            flat.update({f"{v}_{k}": np.array([r[k] for r in rs]) for k in rs[0]})
    np.savez(OUT / f"h_w{wid}.npz", **flat)


def baseline() -> dict:
    """plan_invert's per-goal results (no step) by gid."""
    zs = [dict(np.load(p)) for p in sorted(PI.PLAN.glob("h_w*.npz"))]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0] if not k.startswith("trace")}


def launch(procs: int, gpus: str, limit: int) -> None:
    H = baseline()
    C = np.load(OUT / "choice.npz")
    gate = {int(g) for g, s in zip(H["ref_aim_gid"], H["ref_aim_pred_std"]) if s > GATE_RUN}
    j_of = {int(g): i for i, g in enumerate(C["gid"])}
    jobs = []
    for g in sorted(gate):
        ds = {}
        for meth in METHODS:
            k = int(C[f"{meth}_k"][j_of[g]])
            if k != int(C["k_still"]) and k not in ds.values():
                ds[meth] = k
        jobs += [(g, meth) for meth in ds]
    jobs = jobs[: limit or None]
    print(f"{len(gate)} gated goals, {len(jobs)} step runs", flush=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[w % len(cards)], OMP_NUM_THREADS="1", REACHWM_GOAL_REF="1")
        log = open(OUT / f"h_w{w}.log", "w")
        spec = ",".join(f"{g}:{meth}" for g, meth in jobs[w::procs])
        cmd = [str(E.SIM_PY), __file__, "worker", "--wid", str(w), "--jobs", spec]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [p.wait() for p in ps], flush=True)


def report() -> None:
    """Per goal: the no-step result (plan_invert), or the step result when the gate fires and the method steps."""
    from scipy.stats import spearmanr

    H, C = baseline(), np.load(OUT / "choice.npz")
    zs = [dict(np.load(p)) for p in sorted(OUT.glob("h_w*.npz"))]
    R = {}
    for z in zs:
        for k, v in z.items():
            R[k] = np.concatenate([R[k], v]) if k in R else v
    run = {}  # (gid, k) -> {aim, ff_servo, std, moved, d}
    for meth in METHODS:
        for i, g in enumerate(R.get(f"{meth}_aim_gid", [])):
            j = int(np.where(C["gid"] == g)[0][0])
            r = lambda v, f: R[f"{meth}_{v}_{f}"][i]  # noqa: E731
            run[(int(g), int(C[f"{meth}_k"][j]))] = dict(
                aim=r("aim", "err"), ff_servo=r("ff_servo", "err"), std=r("aim", "pred_std"), moved=r("aim", "moved"),
                d=r("aim", "d"), fell=bool(r("aim", "fell") or r("ff_servo", "fell")),
                same=bool(r("aim", "same_plan") and r("ff_servo", "same_plan")),
                x0d=float(max(r("aim", "x0_diff"), r("ff_servo", "x0_diff"))), walked=bool(r("aim", "walked")))  # fmt: skip
    fellstep = {(int(g), int(C[f"{meth}_k"][int(np.where(C["gid"] == g)[0][0])]))
                for meth in METHODS for g in R.get(f"{meth}_fellstep", [])}  # fmt: skip
    j_of = {int(g): i for i, g in enumerate(C["gid"])}
    g_all = H["ref_aim_gid"].astype(int)
    goal, side = H["ref_aim_goal_t"], H["ref_aim_side"]
    tt = np.c_[goal[:, 0], goal[:, 1] * side, goal[:, 2]]
    core = np.all((tt >= PI.CORE[0]) & (tt <= PI.CORE[1]), axis=1)
    ok = (H["ref_aim_fell"] == 0) & (H["ref_ff_servo_fell"] == 0)
    m = core & ok
    std0, walked = H["ref_aim_pred_std"], H["ref_aim_walked"].astype(bool)
    base = {"aim": H["ref_aim_err"], "ff_servo": H["ref_ff_servo_err"]}
    missing = set()  # gated step runs without a result (only in a partial run)
    BIG = 99.0  # a fall during the step or the reach counts as a miss of BIG cm

    def policy(meth: str, tau: float):
        """err per goal (aim, ff_servo), stepped mask, std after the step (nan when it did not step)."""
        out = {v: base[v].copy() for v in base}
        stepped, s1 = np.zeros(len(g_all), bool), np.full(len(g_all), np.nan)
        for i, g in enumerate(g_all):
            k = int(C[f"{meth}_k"][j_of[g]])
            if std0[i] <= tau or k == int(C["k_still"]):
                continue
            if (g, k) not in run and (g, k) not in fellstep:
                missing.add((g, k))
                continue
            stepped[i] = True
            if (g, k) in fellstep:
                out["aim"][i] = out["ff_servo"][i] = BIG
                continue
            r = run[(g, k)]
            out["aim"][i], out["ff_servo"][i], s1[i] = r["aim"], r["ff_servo"], r["std"]
            if r["fell"]:
                out["aim"][i] = out["ff_servo"][i] = BIG
        return out, stepped, s1

    stat = lambda e: dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), lt1=float(np.mean(e < 1)),  # noqa: E731
                          gt2=float(np.mean(e > 2)))  # fmt: skip
    lines = [f"core goals {int(m.sum())}; step runs {len(run)}, falls during the step {len(fellstep)}, "
             f"falls during the reach {sum(r['fell'] for r in run.values())}; plan matches the executed one "
             f"{np.mean([r['same'] for r in run.values()]):.3f}, max x0 diff {max(r['x0d'] for r in run.values()):.1e}", "",
             "| policy | gate std > cm | stepped | open loop median | <1 cm | >2 cm | p90 | ff+servo median | <1 cm | >2 cm | p90 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    res = dict(rows=[])
    a, f = stat(base["aim"][m]), stat(base["ff_servo"][m])
    lines.append(f"| no step | - | 0.00 | {a['median']:.2f} | {a['lt1']:.2f} | {a['gt2']:.2f} | {a['p90']:.2f} | "
                 f"{f['median']:.2f} | {f['lt1']:.2f} | {f['gt2']:.2f} | {f['p90']:.2f} |")  # fmt: skip
    for meth in METHODS:
        for tau in (GATE_RUN, 1.0, 1.2, 1.5, 2.0):
            out, st, _ = policy(meth, tau)
            a, f = stat(out["aim"][m]), stat(out["ff_servo"][m])
            res["rows"].append(dict(method=meth, tau=tau, stepped=float(st[m].mean()), aim=a, ff_servo=f))
            lines.append(f"| {meth} | {tau:.2f} | {st[m].mean():.2f} | {a['median']:.2f} | {a['lt1']:.2f} | {a['gt2']:.2f} | "
                         f"{a['p90']:.2f} | {f['median']:.2f} | {f['lt1']:.2f} | {f['gt2']:.2f} | {f['p90']:.2f} |")  # fmt: skip
    # the goals that step, paired against not stepping, by start kind
    lines += ["", "| method | goals that step (gate 0.83) | start | n | open loop no step / step median | better on | "
              "ff+servo no step / step median | better on | std before / after median |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for meth in METHODS:
        out, st, s1 = policy(meth, GATE_RUN)
        for nm, msk in (("all", m & st), ("reset", m & st & ~walked), ("walk", m & st & walked)):
            ea, fa = out["aim"][msk], out["ff_servo"][msk]
            lines.append(f"| {meth} | | {nm} | {int(msk.sum())} | {np.median(base['aim'][msk]):.2f} / {np.median(ea):.2f} | "
                         f"{np.mean(ea < base['aim'][msk]):.2f} | {np.median(base['ff_servo'][msk]):.2f} / {np.median(fa):.2f} | "
                         f"{np.mean(fa < base['ff_servo'][msk]):.2f} | {np.median(std0[msk]):.2f} / {np.nanmedian(s1[msk]):.2f} |")  # fmt: skip
    # the tail: core goals ff+servo misses by > 2 cm without a step
    tail = m & (base["ff_servo"] > 2)
    lines += ["", f"tail (core, no-step ff+servo > 2 cm): {int(tail.sum())} goals"]
    for meth in METHODS:
        out, st, s1 = policy(meth, GATE_RUN)
        lines.append(f"  {meth}: steps on {int((tail & st).sum())}; ff+servo now < 2 cm on {int((tail & (out['ff_servo'] < 2)).sum())}, "
                     f"< 1 cm on {int((tail & (out['ff_servo'] < 1)).sum())}; median {np.median(base['ff_servo'][tail]):.2f} -> "
                     f"{np.median(out['ff_servo'][tail]):.2f}; open loop median {np.median(base['aim'][tail]):.2f} -> "
                     f"{np.median(out['aim'][tail]):.2f}")  # fmt: skip
    # best of the three stances per goal (no step, wm, geo): how much a better stance choice could still give
    o_w, _, s_w = policy("wm", GATE_RUN)
    o_g, _, s_g = policy("geo", GATE_RUN)
    for v in ("aim", "ff_servo"):
        best = np.minimum(np.minimum(base[v], o_w[v]), o_g[v])
        byst = np.where(np.nan_to_num(s_w, nan=std0) <= np.nan_to_num(s_g, nan=std0), o_w[v], o_g[v])
        byst = np.where(np.minimum(np.nan_to_num(s_w, nan=9), np.nan_to_num(s_g, nan=9)) < std0, byst, base[v])
        b, y = stat(best[m]), stat(byst[m])
        lines.append(f"{v}: best of the three stances median {b['median']:.2f}, <1 cm {b['lt1']:.2f}, >2 cm {b['gt2']:.2f}; "
                     f"the stance with the lowest std after stepping (hindsight on std, not on error) median {y['median']:.2f}, "
                     f"<1 cm {y['lt1']:.2f}, >2 cm {y['gt2']:.2f}")  # fmt: skip
    # does the std after stepping still rank the error
    rs = [r for r in run.values() if not r["fell"]]
    rho = spearmanr([r["std"] for r in rs], [r["aim"] for r in rs]).correlation
    gap = [100 * np.linalg.norm(r["moved"] - r["d"]) for r in rs]
    lines.append(f"after the step: std/error Spearman {rho:.2f}; step landed off the asked displacement by median "
                 f"{np.median(gap):.1f} cm (p90 {np.percentile(gap, 90):.1f})")  # fmt: skip
    lines.append(f"step runs missing from the results: {len(missing)}")
    pol = dict(gid=g_all, core=m, base_aim=base["aim"], base_ff=base["ff_servo"], std0=std0)
    for meth in METHODS:
        out, st, s1 = policy(meth, GATE_RUN)
        pol.update({f"{meth}_aim": out["aim"], f"{meth}_ff": out["ff_servo"], f"{meth}_stepped": st, f"{meth}_std": s1,
                    f"{meth}_k": np.array([int(C[f"{meth}_k"][j_of[g]]) for g in g_all])})  # fmt: skip
    np.savez(OUT / "policy.npz", **pol)
    if SIDE:
        lines += [""] + pair(pol, dict(np.load(IE.OUT / "restance" / "policy.npz")))
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def pair(new: dict, old: dict) -> list[str]:
    """Goal by goal, this run (world models retrained with short steps) against the first restance run."""
    from scipy.stats import spearmanr

    assert np.array_equal(new["gid"], old["gid"])
    m = new["core"]
    L = ["| method (gate 0.83) | data | stepped | open loop median | >2 cm | p90 | ff+servo median | <1 cm | >2 cm | p90 | "
         "regressions | std/error Spearman after the step |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for meth in METHODS:
        for nm, P in (("old", old), ("with short steps", new)):
            a, f, st = P[f"{meth}_aim"][m], P[f"{meth}_ff"][m], P[f"{meth}_stepped"][m]
            reg = int(np.sum(st & (P["base_ff"][m] < 2) & (f > 2)))
            ok = st & (a < 90)
            rho = spearmanr(P[f"{meth}_std"][m][ok], a[ok]).correlation
            L.append(f"| {meth} | {nm} | {st.mean():.2f} | {np.median(a):.2f} | {np.mean(a > 2):.2f} | {np.percentile(a, 90):.2f} | "
                     f"{np.median(f):.2f} | {np.mean(f < 1):.2f} | {np.mean(f > 2):.2f} | {np.percentile(f, 90):.2f} | {reg} | {rho:.2f} |")  # fmt: skip
    L += ["", "| method | goals stepped in both runs | same stance | ff+servo old / new median | new better on | old >2 cm now <2 | "
          "old <2 cm now >2 | open loop old / new median | new better on |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for meth in METHODS:
        b = m & old[f"{meth}_stepped"] & new[f"{meth}_stepped"]
        fo, fn, ao, an = old[f"{meth}_ff"][b], new[f"{meth}_ff"][b], old[f"{meth}_aim"][b], new[f"{meth}_aim"][b]
        L.append(f"| {meth} | {int(b.sum())} | {np.mean(old[f'{meth}_k'][b] == new[f'{meth}_k'][b]):.2f} | {np.median(fo):.2f} / "
                 f"{np.median(fn):.2f} | {np.mean(fn < fo):.2f} | {int(np.sum((fo > 2) & (fn < 2)))} | {int(np.sum((fo < 2) & (fn > 2)))} | "
                 f"{np.median(ao):.2f} / {np.median(an):.2f} | {np.mean(an < ao):.2f} |")  # fmt: skip
    # the first run's regressions: stepped, fine without the step, > 2 cm with it
    r = m & old["wm_stepped"] & (old["base_ff"] < 2) & (old["wm_ff"] > 2)
    L.append("")
    L.append(f"first run's wm regressions ({int(r.sum())}): ff+servo old " + ", ".join(f"{v:.2f}" for v in old["wm_ff"][r]) +
             "; now " + ", ".join(f"{v:.2f}" for v in new["wm_ff"][r]) + "; now stepped " +
             ", ".join(str(bool(v)) for v in new["wm_stepped"][r]))  # fmt: skip
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["x", "s", "h", "d", "worker"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0, help="first N step runs only (smoke test)")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--jobs", default="")
    a = ap.parse_args(argv)
    if a.what == "x":
        export()
    elif a.what == "s":
        score()
    elif a.what == "h":
        launch(a.procs, a.gpus, a.limit)
    elif a.what == "d":
        report()
    else:
        worker(a.wid, [(int(x.split(":")[0]), x.split(":")[1]) for x in a.jobs.split(",") if x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
