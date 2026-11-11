"""Action selection (2026-11-09, step 2 of the whole-body paper plan): given only a hand target, the self model picks the
stopping point, the pelvis height, the hand and the wrist pitch, then walks and reaches. Nothing after the sighting reads
sim truth when run with BFM_ROOT=odo.

  direct  ScaleBFM alone: fixed's action, the goal commanded as it is, no model anywhere
  fixed   the rule of approach_eval: stop X_NOM short of the goal, pelvis_for, near hand, level wrist
  rule    fixed, the wrist pitch from a height table read off the data (level above 0.80 m, down 75 deg under 0.30 m)
  select  STANDOFFS x PELVIS x PITCHES x both hands scored by the whole-chain model (one correction step each), TOP_K of them
          run through the full fixed point, the lowest predicted miss (with the ensemble's spread, as l_pos) taken

  select2 the same actions, scored only by the whole-chain model's predicted miss of the plain command (with the spread, as
          l_pos): the action that needs the least correction. On map1 the correction the inversion ends up making orders the
          outcome (fixed: 0-3 cm 93 % under 2 cm, 10-15 cm 23 %) while the predicted miss after inversion is about 0 for every action.

Results of 320 fresh cases (map2, SELECT_SEED=14000000, BFM_ROOT=odo, 0 falls), under 2 cm: direct 7 %, fixed 76 %, rule 73 %, select2 70 %;
under 0.35 m select2 56 of 80 vs fixed 40, from 0.35 m up 168 of 240 vs 202. On map1, select tied fixed (+0.00 cm, 95 % -0.26..+0.28).

All of them end the same way: the walk, the walk model on the body after the walk (post_wkp), the reach model's inversion.
Cases: forward goal distance x height cells (X_BANDS x Z_BANDS), N_PER_CELL each, y within +-0.30 m, walked or fresh start.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    I=$PWD/outputs/reachwm_20261030
    export BFM_ROOT=odo OMP_NUM_THREADS=1 APPROACH_WM=$I/approach/wm_approach_odo.pt APPROACH_WM_POST=$I/invert/wrist_odo/wm_ref.pt
    export WALK_DATA=approach_odo,approach0_odo WALK_SUFFIX=_odo
    $PY scripts/experiments/reachwm_20261030/select_eval.py h --procs 6 --n 320 --tag _map1
    $PY scripts/experiments/reachwm_20261030/select_eval.py d --tag _map1
"""

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("REACHWM_APPROACH", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import approach_eval as AE  # noqa: E402
import numpy as np  # noqa: E402

E, IE, PI, PV, WI = AE.E, AE.IE, AE.PI, AE.PV, AE.WI
OUT = AE.OUT / "select"
SEED = int(
    os.environ.get("SELECT_SEED", 13000000)
)  # map1 13000000; map2 14000000 (fresh cases, the score select2 was found on map1)
X_BANDS = ((0.35, 0.55), (0.55, 0.80), (0.80, 1.05), (1.05, 1.30))
Z_BANDS = (
    (0.10, 0.20),
    (0.20, 0.35),
    (0.35, 0.50),
    (0.50, 0.65),
    (0.65, 0.80),
    (0.80, 0.95),
    (0.95, 1.10),
    (1.10, 1.25),
)
STANDOFFS = (0.30, 0.45, 0.60)
PELVIS = (-1.0, 0.66, 0.54)  # -1: stand
PITCHES = (0.0, 30.0, 60.0, 85.0)  # deg, down
TOP_K = 4
METHODS = ("direct", "fixed", "rule", "select", "select2")


def case(i: int) -> dict:
    rng = np.random.default_rng(SEED + i)
    ix, iz = divmod(i % (len(X_BANDS) * len(Z_BANDS)), len(Z_BANDS))
    g = np.array([rng.uniform(*X_BANDS[ix]), rng.uniform(-0.30, 0.30), rng.uniform(*Z_BANDS[iz])])
    walked = bool(rng.random() < 0.5)
    ang, d, yaw = rng.uniform(-math.pi, math.pi), rng.uniform(0.3, 1.2), rng.uniform(-math.pi, math.pi)
    return dict(i=i, ix=ix, iz=iz, goal=g, walked=walked, walk0=[d * math.cos(ang), d * math.sin(ang), yaw])


def rule_pitch(z: float) -> float:
    return 0.0 if z >= 0.80 else 30.0 if z >= 0.50 else 45.0 if z >= 0.30 else 75.0


def walk_for(g: np.ndarray, standoff: float) -> np.ndarray:
    lo, hi = AE.G().APPROACH_BOX
    return np.r_[np.clip(g[0] - standoff, lo[0], hi[0]), 0.0, 0.0]


def x0_side(rec, m, side: str, sd: float, walked: bool, start: np.ndarray) -> np.ndarray:
    """x0_here for either hand: the recorder's last jaw rotation is the hand it was started with."""
    keep = rec.J
    if side != rec.side:
        Rj = m.jaw_centre(side)[1]
        rec.J = [*keep[:-1], np.r_[Rj[:, 0], Rj[:, 1]].astype(np.float32)]
    try:
        return AE.x0_here(rec, sd, walked, start)
    finally:
        rec.J = keep


def score_one(model, m, st: dict, a: dict, tb: np.ndarray) -> dict | None:
    """The whole-chain model on one action (AE.evaluate, with the hand and the wrist pitch free)."""
    import data as DT
    import torch

    G = AE.G()
    rb, sd, x0b = G.rot_y(math.radians(a["pitch"])), a["sd"], st["x0"][a["side"]]
    p0, R0, yaw0, start, walk, pz = st["p"], st["Rh"], st["yaw"], st["start"], a["walk"], a["pz"]
    tw = np.r_[p0[:2] + (R0 @ np.r_[tb[:2], 0.0])[:2], tb[2]]
    try:
        pr = G.plan_only(m, a["side"], tw, E.rot_z(yaw0) @ rb, None if pz < 0 else float(pz))
    except np.linalg.LinAlgError:
        return None
    x0 = np.r_[x0b, PI.plan_features(pr, x0b[:29], start), DT.walk_features(walk)[0]]
    t0 = np.r_[walk[:2] + tb[:2], tb[2]]
    r6 = AE.rot6_of(rb)
    u = np.r_[sd, t0, r6, float(pz < 0), 0.0 if pz < 0 else pz]
    F = lambda v: torch.as_tensor(np.asarray(v, np.float32)[None])  # noqa: E731
    pos, std, _, _ = PV.predict_pose(model, dict(x0=F(x0), u=F(u), target=F(t0), rot6=F(r6), side=torch.tensor([sd])))
    g = st["goal"]
    return dict(tb=np.asarray(tb, float).copy(), pos=pos, pred_err=100 * float(np.linalg.norm(pos - g)), pred_std=100 * float(np.linalg.norm(std)),
                l_pos=float(((pos - g) ** 2).sum() + IE.LAMBDA * (std**2).sum()))  # fmt: skip


def step(a: dict, st: dict, ev: dict) -> np.ndarray:
    return WI.clip_t(dict(sd=a["sd"]), ev["tb"] - (ev["pos"] - st["goal"]))


def select2(model, m, st: dict) -> tuple[dict, dict, int]:
    g, cands = st["goal"], []
    direct = walk_for(g, AE.X_NOM)
    for standoff in STANDOFFS:
        walk = walk_for(g, standoff)
        for side, sd in (("right", 1.0), ("left", -1.0)):
            for pz in PELVIS:
                for pitch in PITCHES:
                    a = dict(side=side, sd=sd, standoff=standoff, walk=walk, pz=pz, pitch=pitch)
                    ev = score_one(model, m, st, a, np.r_[g[:2] - walk[:2], g[2]])
                    if ev is not None:
                        cands.append((a, ev))
    assert cands, "no feasible action"
    a, ev = min(cands, key=lambda c: 100 * math.sqrt(c[1]["l_pos"]) + AE.STEP_COST * abs(c[0]["walk"][0] - direct[0]))
    return a, ev, len(cands)


def select(model, m, st: dict) -> tuple[dict, dict, int]:
    g, cands, n = st["goal"], [], 0
    direct = walk_for(g, AE.X_NOM)
    for standoff in STANDOFFS:
        walk = walk_for(g, standoff)
        for side, sd in (("right", 1.0), ("left", -1.0)):
            for pz in PELVIS:
                for pitch in PITCHES:
                    a = dict(side=side, sd=sd, standoff=standoff, walk=walk, pz=pz, pitch=pitch)
                    e0 = score_one(model, m, st, a, np.r_[g[:2] - walk[:2], g[2]])
                    n += 1
                    if e0 is None:
                        continue
                    e1 = score_one(model, m, st, a, step(a, st, e0))
                    n += 1
                    best = e0 if e1 is None or e0["l_pos"] <= e1["l_pos"] else e1
                    cands.append((a, best))
    assert cands, "no feasible action"
    key = lambda c: 100 * math.sqrt(c[1]["l_pos"]) + AE.STEP_COST * abs(c[0]["walk"][0] - direct[0])  # noqa: E731
    top, done = sorted(cands, key=key)[:TOP_K], []
    for a, ev in top:
        best = ev
        for _ in range(AE.ITERS):
            ev = score_one(model, m, st, a, step(a, st, ev))
            n += 1
            if ev is None:
                break
            best = min(best, ev, key=lambda e: e["l_pos"])
        done.append((a, best))
    a, best = min(done, key=key)
    return a, best, n


def episode(env, rec, tape, models: dict, c: dict, method: str) -> dict:
    import gen_reach as GR
    import walk_model as WK

    m = env.m
    env.reset(None)
    tape.q, tape.k, tape.on = [], 0, False
    if c["walked"]:
        d0 = c["walk0"]
        m.walk_to(np.array(d0[:2]), yaw=float(d0[2]))
        if m.fell_at is not None:
            return dict(i=c["i"], method=method, ix=c["ix"], iz=c["iz"], fell="walk0")
    first = "right" if c["goal"][1] < 0 else "left"
    rec.start(first)
    m.hold(GR.PRE_S)
    p0, Rp0 = env.pelvis()
    yaw0 = E.yaw_of(Rp0)
    R0 = E.rot_z(yaw0)
    start = np.r_[p0, yaw0].astype(np.float32)
    g = np.asarray(c["goal"], float)
    goal_w = np.r_[p0[:2] + (R0 @ np.r_[g[:2], 0.0])[:2], g[2]]
    k_pre = len(rec.S) - 1
    res = dict(i=c["i"], method=method, ix=c["ix"], iz=c["iz"], goal=g.tolist(), walked=c["walked"])
    tic = time.time()
    x0s = {s: x0_side(rec, m, s, sd, c["walked"], start) for s, sd in (("right", 1.0), ("left", -1.0))}
    if method in ("select", "select2"):
        st = dict(p=p0, Rh=R0, yaw=yaw0, start=start, goal=g, x0=x0s)
        a, ev, n = (select if method == "select" else select2)(models["wm"], m, st)
        res.update(pred_chain=ev["pred_err"], pred_chain_std=ev["pred_std"], n_evals=n)
    else:
        side = first
        a = dict(side=side, sd=1.0 if side == "right" else -1.0, standoff=AE.X_NOM, walk=walk_for(g, AE.X_NOM),
                 pz=AE.pelvis_for(g[2]), pitch=rule_pitch(g[2]) if method == "rule" else 0.0)  # fmt: skip
    res["think_s"] = time.time() - tic
    walk, side, sd, pz = a["walk"], a["side"], a["sd"], a["pz"]
    rb = GR.rot_y(math.radians(a["pitch"]))
    res.update(side=side, standoff=a["standoff"], pz=pz, pitch=a["pitch"], walk=walk.tolist())
    GR.open_walk(m, float(walk[0]), float(walk[1]), float(walk[2]))
    if m.fell_at is not None:
        return dict(res, fell="walk")
    S_walk = np.array(rec.S[k_pre:])
    rec.start(side)
    m.hold(GR.PRE_S)
    p1, Rp1 = env.pelvis()
    yaw1 = E.yaw_of(Rp1)
    Rh1 = E.rot_z(yaw1)
    d1 = R0.T @ np.r_[p1[:2] - p0[:2], 0.0]
    res["walk_err"] = [float(d1[0] - walk[0]), float(d1[1] - walk[1]),
                       float((yaw1 - yaw0 - walk[2] + math.pi) % (2 * math.pi) - math.pi)]  # fmt: skip
    S_post = np.array(rec.S)
    if method == "direct":
        tb = AE.believed_goal(g, walk)
        return finish(env, rec, res, a, tb, p1, Rh1, rb, goal_w, R0)
    x = WK.features(x0s[side], walk, WK.post_feats(S_post[-3:]), WK.seq_feats(np.r_[S_walk, S_post[1:]]))
    mu, _ = models["post_wkp"].predict_np(x)
    tb = AE.believed_goal(g, walk, mu[0])
    tic = time.time()
    rot6 = AE.rot6_of(rb)
    stp = dict(sd=sd, p=p1, yaw=yaw1, Rh=Rh1, goal=tb.astype(np.float32).astype(np.float64), rot_w=Rh1 @ rb, rot6=rot6,
               c2=dict(side=side, dyaw=0.0, pitch=math.radians(a["pitch"])), start=np.r_[p1, yaw1].astype(np.float32),
               walked=True, x0b=x0_side(rec, m, side, sd, True, np.r_[p1, yaw1].astype(np.float32)))  # fmt: skip
    evs = WI.fixed_point(models["post"], m, stp, stp["goal"], pz, None)
    best = min(evs, key=lambda ev: ev["l_pos"])
    tb = np.asarray(best["t"], float)
    res.update(
        pred_err=float(best["pred_err"]), pred_std=float(best["pred_std"]), think_s=res["think_s"] + time.time() - tic
    )
    return finish(env, rec, res, a, tb, p1, Rh1, rb, goal_w, R0)


def finish(env, rec, res: dict, a: dict, tb: np.ndarray, p1, Rh1, rb, goal_w, R0) -> dict:
    """The reach with the final command, the hand error against the goal."""
    import gen_reach as GR

    m, side, pz = env.m, a["side"], a["pz"]
    target_w = np.r_[p1[:2] + (Rh1 @ np.r_[tb[:2], 0.0])[:2], tb[2]]
    res["tb"] = np.asarray(tb, float).tolist()
    out = m.reach(side, target_w, Rh1 @ rb, pelvis_z=None if pz < 0 else float(pz), servo=False)
    if m.fell_at is None:
        m.hold(GR.POST_S)
    rec.on = False
    jaw = m.jaw_centre(side)[0]
    return dict(res, err=100 * float(np.linalg.norm(jaw - goal_w)), err_h=(100 * R0.T @ (jaw - goal_w)).tolist(), ok=bool(out.ok),
                fell=None if m.fell_at is None else "reach")  # fmt: skip


def worker(wid: int, ids: list[int], methods, tag: str) -> None:
    import gen_reach as GR
    import torch
    import walk_model as WK

    torch.set_num_threads(1)
    env = E.Env("open")
    rec, tape = GR.Recorder(env), AE.QTape(env)
    models = {"wm": torch.load(AE.WM, map_location="cpu", weights_only=False).eval(),
              "post": torch.load(AE.WM_POST, map_location="cpu", weights_only=False).eval(),
              "post_wkp": torch.load(WK.WALK_MODEL_POST, map_location="cpu", weights_only=False).eval()}  # fmt: skip
    env.reset(None)  # the first call after start-up differs (scalebfm-first-call-nondeterminism): warm up
    env.m.hold(1.0)
    res = []
    for i in ids:
        c = case(i)
        for meth in methods:
            tic = time.time()
            r = episode(env, rec, tape, models, c, meth)
            res.append(r)
            print(f"w{wid} case {i} {meth}: err {r.get('err', float('nan')):.2f} chain {r.get('pred_chain', float('nan')):.2f} post {r.get('pred_err', float('nan')):.2f} "
                  f"side {r.get('side')} stand {r.get('standoff')} pz {r.get('pz')} pitch {r.get('pitch')} fell {r.get('fell')} "
                  f"think {r.get('think_s', 0):.0f} s | {time.time() - tic:.0f} s", flush=True)  # fmt: skip
        json.dump(
            res,
            open(OUT / f"h_w{wid}{tag}.json", "w"),
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )


def launch(procs: int, gpus: str, n: int, methods: str, tag: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cards, ps = gpus.split(","), []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)])
        log = open(OUT / f"h_w{w}{tag}.log", "w")
        cmd = [
            str(E.SIM_PY),
            __file__,
            "_w",
            "--wid",
            str(w),
            "--ids",
            ",".join(map(str, range(w, n, procs))),
            "--methods",
            methods,
            "--tag",
            tag,
        ]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [p.wait() for p in ps], flush=True)


def report(tag: str) -> None:
    R = []
    for w in range(32):
        p = OUT / f"h_w{w}{tag}.json"
        if p.exists():
            R += json.load(open(p))
    methods = [m for m in METHODS if any(r["method"] == m for r in R)]
    for m in methods:
        d = [r for r in R if r["method"] == m]
        e = np.array([r["err"] for r in d if r.get("err") is not None and not r.get("fell")])
        print(
            f"{m:7s} n {len(d)} fell {sum(1 for r in d if r.get('fell'))} med {np.median(e):.2f} p90 {np.percentile(e, 90):.2f} <2 {(e < 2).mean() * 100:.0f}% <5 {(e < 5).mean() * 100:.0f}%"
        )
    print("\nsuccess under 2 cm by height (rows) and forward distance (columns), fell counts as a miss")
    for m in methods:
        print(m)
        for iz in reversed(range(len(Z_BANDS))):
            row = []
            for ix in range(len(X_BANDS)):
                d = [r for r in R if r["method"] == m and r["ix"] == ix and r["iz"] == iz]
                row.append(f"{sum(1 for r in d if not r.get('fell') and r.get('err', 1e9) < 2):2d}/{len(d):<2d}")
            print(f"  z {Z_BANDS[iz][0]:.2f}-{Z_BANDS[iz][1]:.2f}  " + "  ".join(row))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["h", "_w", "d"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    if a.phase == "h":
        launch(a.procs, a.gpus, a.n, a.methods, a.tag)
    elif a.phase == "_w":
        worker(a.wid, [int(x) for x in a.ids.split(",")], a.methods.split(","), a.tag)
    else:
        report(a.tag)
