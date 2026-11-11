"""Walk over, then reach, eyes closed (2026-11-03, after gen_reach.py REACH_APPROACH): the self model extended to before the
walk, against the direct command.

A case: reset (half of them then walk 0.3-1.2 m to a random pose, as in the data, stop correction on: before the
sighting), hold PRE_S, and see the goal once, in the start frame: 75 % ahead, x 0.75-1.30 m (walk forward), 25 % too
close, x 0.05-0.25 m (walk back); y -0.35..0.15 (mirrored for the left hand), z 0.35-1.20 m; the wanted wrist level and
facing ahead. From then on nothing reads the simulator's truth: the walk runs with the motor's stop correction off
(gen_reach.open_walk), and the reach is commanded in the body frame, as if the walk had gone exactly as commanded.
Every method replays the same reset and the same pre-walk state, so all comparisons are paired.

    direct   walk straight by goal x - X_NOM (the goal ends X_NOM ahead), the goal itself as the reach command
    wm       the same walk; the reach command found before walking with the extended model (models_goalref_approach):
             the position fixed point t <- t + (goal - predicted landing), every round the plan re-solved from the
             pre-walk posture, ITERS rounds, the lowest |miss|^2 + |std|^2 kept
    wm_walk  as wm, also choosing the walk among WALK_OFFSETS around the direct one (the same score, STEP_COST per metre
             off the direct walk)
    post     ablation: the direct walk, then the reach model without the walk (wrist_invert2's) inverted from the state
             the robot really stands in after the walk, for the goal where it believes the goal is (the walk as
             commanded): knows its posture after walking, not where the walk ended
    walk_post  wm_walk's walk (chosen before walking with the extended model), then post's inversion after it
  walk model (walk_model.py, run with --methods post,direct_wk,post_wk,post_true --tag _wk; report d2):
    direct_wk  the direct walk; before walking the walk model predicts where it will stop, and the goal is taken in the
               frame the robot then believes it stands in (commanded walk + predicted stop error); that goal as the
               reach command
    post_wk    the same believed goal, then post's inversion after the walk
    post_true  reference, reads the truth: the believed goal from the walk's real stop error, then post's inversion
    post_wkp   as post_wk, the walk model also given the body after the walk (walk_model.py post), predicted then
    post_wks   as post_wkp, also given the readings during the walk (walk_model.py seq, a learned leg odometry)
    (run with --methods post_wkp,post_wks --tag _wkps; report d2 picks them up)
    post_odo   the direct walk with geometric leg odometry (leg_odo.py: encoders, torques, gyro, gravity at 50 Hz, nothing
               learned) from the sighting to the end of the post-walk hold; the believed goal where it says the walk
               ended, then post's inversion
    post_fuse  the walk model's pre-walk prediction and the odometry averaged per axis, weights by the walk model's own
               spread against ODO_SIGMA (s^2 / (s^2 + sigma^2) to the odometry), then post's inversion
    (run with --methods post_odo,post_fuse --tag _odo; report d3)
    2026-11-09, no pelvis truth at run time (BFM_ROOT=odo, ScaleBFM reads leg odometry; models retrained on _odo data; env
    APPROACH_WM_POST, WALK_DATA, WALK_SUFFIX; tags _odo and _odo2, the 11-03 true-pelvis _odo files are in
    approach/true_root_tag_odo/): hand error median direct 4.32, post 3.25, post_odo 2.19, post_wk 1.69, post_fuse 1.68,
    post_wkp 1.12, post_wks 1.14, post_true 0.71 cm of 300, 0 falls. With the true pelvis the readings after the walk
    leak the stop error, with odometry they do not, so post_wkp is honest here.
The pelvis height is pelvis_for's rule for every method. Phase o: the extended model offline, on data_approach's test
split, against the command itself and against the reach-only model told the walk went as commanded.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    REACHWM_GOAL_REF=1 REACHWM_APPROACH=1 CUDA_VISIBLE_DEVICES=0 $PY scripts/experiments/reachwm_20261030/train.py \
        E3_ensemble --data <the five datasets' all.npz, comma-separated> --out-root outputs/reachwm_20261030/models_goalref_approach
    $PY scripts/experiments/reachwm_20261030/approach_eval.py x                # pickle the model for the workers
    $PY scripts/experiments/reachwm_20261030/approach_eval.py o                # offline
    $PY scripts/experiments/reachwm_20261030/approach_eval.py o2               # how much of that is the walk
    $PY scripts/experiments/reachwm_20261030/approach_eval.py h --procs 6     # cases x methods
    $PY scripts/experiments/reachwm_20261030/approach_eval.py d                # tables
Writes outputs/reachwm_20261030/approach/.
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("REACHWM_APPROACH", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wrist_invert as WI  # noqa: E402

PV = WI.PV
E, IE, PI = PV.E, PV.IE, PV.PI
OUT = IE.OUT.parent / "approach"
DATA = tuple(
    os.environ.get("APPROACH_DATA", "ref,side,wrist,wrist2,approach,approach0").split(",")
)  # approach0: walks with no turn (v2)
MODEL_ROOT = Path(
    os.environ.get("APPROACH_MODEL_ROOT", IE.OUT.parent / "models_goalref_approach")
)  # env: another set (2026-11-09, models_goalref_approach_odo)
WM = Path(os.environ.get("APPROACH_WM", OUT / "wm_approach.pt"))
APPROACH_SPEC = ",".join(str(IE.OUT.parent / f"data_{k}" / "all.npz") for k in ("approach", "approach0"))
WM_POST = Path(
    os.environ.get("APPROACH_WM_POST", IE.OUT / "wrist_invert2" / "wm_ref.pt")
)  # the reach-only model (far-turn data), for the post ablation
N_CASES, SEED = 300, 12000000
X_NOM = 0.45
ITERS = 6
WALK_OFFSETS = ((0.0, 0.0), (0.10, 0.0), (-0.10, 0.0), (0.0, 0.10), (0.0, -0.10))
STEP_COST = 1.0  # cm per metre off the direct walk
METHODS = ("direct", "wm", "wm_walk", "post", "walk_post")
WK_METHODS = ("post", "direct_wk", "post_wk", "post_true")
WKPS_METHODS = ("post_wkp", "post_wks")
ODO_METHODS = ("post_odo", "post_fuse")
ODO_SIGMA = (1.3, 1.3, 0.4)  # cm, cm, deg: the odometry's typical error (leg_odo.py variants on cases 0-59)
QFPS = 2  # qpos saved every QFPS motor ticks (25 fps) for the first N_VIDEO cases
N_VIDEO = 40


def case(i: int) -> dict:
    rng = np.random.default_rng(SEED + i)
    side = "right" if rng.random() < 0.5 else "left"
    sd = 1.0 if side == "right" else -1.0
    back = bool(rng.random() < 0.25)
    g = np.array(
        [
            rng.uniform(0.05, 0.25) if back else rng.uniform(0.75, 1.30),
            sd * rng.uniform(-0.35, 0.15),
            rng.uniform(0.35, 1.20),
        ]
    )
    walked = bool(rng.random() < 0.5)
    ang, d, yaw = rng.uniform(-math.pi, math.pi), rng.uniform(0.3, 1.2), rng.uniform(-math.pi, math.pi)
    return dict(
        i=i, side=side, sd=sd, back=back, goal=g, walked=walked, walk0=[d * math.cos(ang), d * math.sin(ang), yaw]
    )


def direct_walk(c: dict) -> np.ndarray:
    lo, hi = G().APPROACH_BOX
    return np.r_[np.clip(c["goal"][0] - X_NOM, lo[0], hi[0]), 0.0, 0.0]


def G():
    import gen_reach

    return gen_reach


def pelvis_for(z_g: float) -> float:
    """grasp_pilot.pelvis_for: crouch under 0.60 m, the pelvis 5 cm over the goal, within the data's 0.52-0.76."""
    return -1.0 if z_g >= 0.60 else float(np.clip(z_g + 0.05, 0.52, 0.76))


def x0_here(rec, sd: float, walked: bool, start: np.ndarray) -> np.ndarray:
    """The 206 start features of the state the recorder holds now (grasp_pilot.x0_here, chain 0)."""
    import data as DT

    seg = dict(S=np.array(rec.S), W=np.array(rec.W), J=np.array(rec.J), C=np.array(rec.C), k0=len(rec.S) - 1,
               side=sd, target=np.zeros(3, np.float32), dyaw=0.0, pitch=0.0, rot6=np.zeros(6, np.float32),
               pelvis_z=-1.0, start=start, walked=walked, chain=0, fell=False, ok=True, jaw_err_cm=-1.0,
               att_err_deg=-1.0, ik_err_cm=-1.0, reach_s=0.0, reset=0)  # fmt: skip
    D = IE.pack([seg])
    D.update(ref_q=np.zeros((1, 29), np.float32), ref_pel=np.r_[0, 0, 0, 1, 0, 0, 0][None].astype(np.float32),
             ref_s=np.zeros(1, np.float32), walk=np.zeros((1, 3), np.float32), target_b=D["target"])  # fmt: skip
    DT.derive(D)
    return D["x0"][0, :206]


def believed_goal(g: np.ndarray, walk: np.ndarray, e=None) -> np.ndarray:
    """The goal in the frame the robot believes it stands in after the walk: the commanded walk, plus a stop error e
    (forward cm, left cm, turn deg, start frame) when given."""
    pos, yaw = np.asarray(walk[:2], float), float(walk[2])
    if e is not None:
        pos, yaw = pos + np.asarray(e[:2], float) / 100, yaw + math.radians(float(e[2]))
    return np.r_[(E.rot_z(yaw).T @ np.r_[g[:2] - pos, 0.0])[:2], g[2]]


def rot6_of(R: np.ndarray) -> np.ndarray:
    return np.r_[R[:, 0], R[:, 1]].astype(np.float32)


def evaluate(model, m, st: dict, tb: np.ndarray, walk: np.ndarray, pz: float) -> dict:
    """The extended model on one (walk, reach command): tb in the frame the robot believes it will stand in, the
    prediction in the start frame."""
    import data as DT
    import torch

    sd, p0, R0, yaw0, start, x0b = (st[k] for k in ("sd", "p", "Rh", "yaw", "start", "x0b"))
    tb = np.asarray(tb, float)
    pr = G().plan_only(m, st["side"], np.r_[p0[:2] + (R0 @ np.r_[tb[:2], 0.0])[:2], tb[2]], E.rot_z(yaw0),
                       None if pz < 0 else float(pz))  # fmt: skip
    x0 = np.r_[x0b, PI.plan_features(pr, x0b[:29], start), DT.walk_features(walk)[0]]
    Rt = E.rot_z(float(walk[2]))
    t0 = np.r_[walk[:2] + (Rt @ np.r_[tb[:2], 0.0])[:2], tb[2]]
    r6 = rot6_of(Rt)
    u = np.r_[sd, t0, r6, float(pz < 0), 0.0 if pz < 0 else pz]
    F = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None])  # noqa: E731
    pos, std, _, _ = PV.predict_pose(model, dict(x0=F(x0), u=F(u), target=F(t0), rot6=F(r6), side=torch.tensor([sd])))
    g = st["goal"]
    return dict(tb=tb.copy(), pos=pos, pred_err=100 * float(np.linalg.norm(pos - g)), pred_std=100 * float(np.linalg.norm(std)),
                l_pos=float(((pos - g) ** 2).sum() + IE.LAMBDA * (std**2).sum()))  # fmt: skip


def fixed_point(model, m, st: dict, walk: np.ndarray, pz: float) -> list[dict]:
    """t_b <- t_b + R(turn)^T (goal - predicted landing), from the goal where the robot believes it will be."""
    Rt = E.rot_z(float(walk[2]))
    g = st["goal"]
    tb = np.r_[(Rt.T @ np.r_[g[:2] - walk[:2], 0.0])[:2], g[2]]
    clip = dict(sd=st["sd"])
    evs = []
    for k in range(ITERS + 1):
        ev = evaluate(model, m, st, tb, walk, pz)
        ev["k"] = k
        evs.append(ev)
        if k < ITERS:
            d = ev["pos"] - g
            tb = WI.clip_t(clip, tb - np.r_[(Rt.T @ np.r_[d[:2], 0.0])[:2], d[2]])
    return evs


class QTape:
    """qpos every QFPS motor ticks while `on` (for the videos)."""

    def __init__(self, env):
        self.on, self.q, self.k, m = False, [], 0, env.m
        orig = m.tick

        def tick():
            ok = orig()
            if self.on:
                self.k += 1
                if self.k % QFPS == 0:
                    self.q.append(env.sim.d.qpos.copy().astype(np.float32))
            return ok

        m.tick = tick


def episode(env, rec, tape, models: dict, c: dict, method: str, odo=None) -> dict:
    import gen_reach as GR

    m = env.m
    side, sd = c["side"], c["sd"]
    env.reset(None)
    tape.q, tape.k, tape.on = [], 0, c["i"] < N_VIDEO
    if c["walked"]:
        d0 = c["walk0"]
        m.walk_to(np.array(d0[:2]), yaw=float(d0[2]))
        if m.fell_at is not None:
            return dict(i=c["i"], method=method, fell="walk0")
    rec.start(side)
    m.hold(GR.PRE_S)
    p0, Rp0 = env.pelvis()
    yaw0 = E.yaw_of(Rp0)
    R0 = E.rot_z(yaw0)
    start = np.r_[p0, yaw0].astype(np.float32)
    g = c["goal"]
    goal_w = np.r_[p0[:2] + (R0 @ np.r_[g[:2], 0.0])[:2], g[2]]
    pz = pelvis_for(float(g[2]))
    walk = direct_walk(c)
    k_pre = len(rec.S) - 1
    res = dict(i=c["i"], method=method, side=side, back=c["back"], walked=c["walked"], goal=g.tolist(), pz=pz,
               goal_w=goal_w.tolist(), start=[*p0.tolist(), yaw0], k_sight=len(tape.q))  # fmt: skip
    tic = time.time()
    if method in ("direct", "post", "post_true"):
        tb = np.r_[g[:2] - walk[:2], g[2]]
    elif method in ("direct_wk", "post_wk"):
        import walk_model as WK

        mu, sdv = models["walk"].predict_np(WK.features(x0_here(rec, sd, c["walked"], start), walk))
        res.update(walk_pred=mu[0].tolist(), walk_pred_std=sdv[0].tolist())
        tb = believed_goal(g, walk, mu[0])
    elif method in WKPS_METHODS:
        x0_pre = x0_here(rec, sd, c["walked"], start)
        tb = np.r_[g[:2] - walk[:2], g[2]]  # replaced after the walk
    elif method in ODO_METHODS:
        tb = np.r_[g[:2] - walk[:2], g[2]]  # replaced after the walk
        if method == "post_fuse":
            import walk_model as WK

            mu, sdv = models["walk"].predict_np(WK.features(x0_here(rec, sd, c["walked"], start), walk))
            res.update(walk_pred=mu[0].tolist(), walk_pred_std=sdv[0].tolist())
    else:
        st = dict(
            sd=sd, side=side, p=p0, yaw=yaw0, Rh=R0, start=start, goal=g, x0b=x0_here(rec, sd, c["walked"], start)
        )
        lo, hi = GR.APPROACH_BOX
        offs = WALK_OFFSETS if method in ("wm_walk", "walk_post") else ((0.0, 0.0),)
        cands = []
        for dx, dy in offs:
            w = np.r_[np.clip(walk[0] + dx, lo[0], hi[0]), np.clip(walk[1] + dy, lo[1], hi[1]), 0.0]
            evs = fixed_point(models["wm"], m, st, w, pz)
            best = min(evs, key=lambda ev: ev["l_pos"])
            off = float(np.hypot(w[0] - walk[0], w[1] - walk[1]))
            cands.append(dict(walk=w, ev=best, score=100 * math.sqrt(best["l_pos"]) + STEP_COST * off,
                              naive=evs[0], iters=[e["pred_err"] for e in evs]))  # fmt: skip
        pick = min(cands, key=lambda cc: cc["score"])
        walk, tb = pick["walk"], pick["ev"]["tb"]
        if method == "walk_post":  # the believed goal for the chosen walk; the reach is inverted after the walk
            tb = np.r_[g[:2] - walk[:2], g[2]]
        res.update(pred_err=pick["ev"]["pred_err"], pred_std=pick["ev"]["pred_std"], pred_naive=pick["naive"]["pred_err"],
                   pred_pos=pick["ev"]["pos"].tolist(), pred_naive_pos=pick["naive"]["pos"].tolist(),
                   pred_naive_std=pick["naive"]["pred_std"], iters=pick["iters"],
                   cands=[dict(walk=cc["walk"].tolist(), score=cc["score"], pred_err=cc["ev"]["pred_err"],
                               pred_std=cc["ev"]["pred_std"]) for cc in cands])  # fmt: skip
    res["think_s"] = time.time() - tic
    res["k_walk"] = len(tape.q)
    if method in ODO_METHODS:
        odo.begin()
    GR.open_walk(m, float(walk[0]), float(walk[1]), float(walk[2]))
    if m.fell_at is not None:
        return dict(res, fell="walk")
    res["k_walked"] = len(tape.q)
    S_walk = np.array(rec.S[k_pre:])  # the readings from the sighting to the end of the walk
    rec.start(side)
    m.hold(GR.PRE_S)
    p1, Rp1 = env.pelvis()
    yaw1 = E.yaw_of(Rp1)
    Rh1 = E.rot_z(yaw1)
    d1 = R0.T @ np.r_[p1[:2] - p0[:2], 0.0]
    res["walk"], res["walk_err"] = walk.tolist(), [float(d1[0] - walk[0]), float(d1[1] - walk[1]),
                                                     float((yaw1 - yaw0 - walk[2] + math.pi) % (2 * math.pi) - math.pi)]  # fmt: skip
    if method == "post_true":
        we = res["walk_err"]
        tb = believed_goal(g, walk, [100 * we[0], 100 * we[1], math.degrees(we[2])])
    if method in ODO_METHODS:
        odo.on = False
        pz_ = odo.odo.pose()
        e_odo = np.r_[100 * (pz_[0] - walk[0]), 100 * (pz_[1] - walk[1]), math.degrees(pz_[2] - walk[2])]
        res["walk_odo"] = e_odo.tolist()
        e = e_odo
        if method == "post_fuse":
            s2 = np.r_[np.hypot(*res["walk_pred_std"][:2]) ** 2 * np.ones(2), res["walk_pred_std"][2] ** 2]
            a = s2 / (s2 + np.asarray(ODO_SIGMA) ** 2)
            e = a * e_odo + (1 - a) * np.asarray(res["walk_pred"])
            res.update(walk_fused=e.tolist(), odo_weight=a.tolist())
        tb = believed_goal(g, walk, e)
    if method in WKPS_METHODS:
        import walk_model as WK

        S_post = np.array(rec.S)
        x = WK.features(x0_pre, walk, WK.post_feats(S_post[-3:]), WK.seq_feats(np.r_[S_walk, S_post[1:]]))
        mu, sdv = models[method].predict_np(x)
        res.update(walk_pred=mu[0].tolist(), walk_pred_std=sdv[0].tolist())
        tb = believed_goal(g, walk, mu[0])
    if method in ("post", "walk_post", "post_wk", "post_true") + WKPS_METHODS + ODO_METHODS:
        tic = time.time()
        stp = dict(sd=sd, p=p1, yaw=yaw1, Rh=Rh1, goal=tb.astype(np.float32).astype(np.float64), rot_w=Rh1,
                   rot6=rot6_of(np.eye(3)), c2=dict(side=side, dyaw=0.0, pitch=0.0), start=np.r_[p1, yaw1].astype(np.float32),
                   walked=True, x0b=x0_here(rec, sd, True, np.r_[p1, yaw1].astype(np.float32)))  # fmt: skip
        evs = WI.fixed_point(models["post"], m, stp, stp["goal"], pz, None)
        best = min(evs, key=lambda ev: ev["l_pos"])
        tb = np.asarray(best["t"], float)
        res.update(
            pred_err=float(best["pred_err"]),
            pred_std=float(best["pred_std"]),
            think_s=res["think_s"] + time.time() - tic,
        )
    target_w = np.r_[p1[:2] + (Rh1 @ np.r_[tb[:2], 0.0])[:2], tb[2]]
    res["k_reach"], res["target_w"] = len(tape.q), target_w.tolist()
    out = m.reach(side, target_w, Rh1, pelvis_z=None if pz < 0 else float(pz), servo=False)
    if m.fell_at is None:
        m.hold(GR.POST_S)
    rec.on, tape.on = False, False
    jaw = m.jaw_centre(side)[0]
    e = R0.T @ (jaw - goal_w)
    res.update(tb=tb.tolist(), err=100 * float(np.linalg.norm(jaw - goal_w)), err_h=(100 * e).tolist(), ok=bool(out.ok),
               fell=None if m.fell_at is None else "reach")  # fmt: skip
    if tape.q:
        (OUT / "q").mkdir(parents=True, exist_ok=True)
        np.save(OUT / "q" / f"{c['i']}_{method}.npy", np.array(tape.q, np.float32))
    return res


def worker(wid: int, ids: list[int], methods=METHODS, tag: str = "") -> None:
    import gen_reach as GR
    import torch

    torch.set_num_threads(1)
    env = E.Env("open")
    rec = GR.Recorder(env)
    tape = QTape(env)
    import walk_model as WK

    models = {"wm": torch.load(WM, map_location="cpu", weights_only=False).eval(),
              "post": torch.load(WM_POST, map_location="cpu", weights_only=False).eval()}  # fmt: skip
    import leg_odo as LO

    odo = LO.TickFeed(env)
    if any(mm in ("direct_wk", "post_wk", "post_fuse") for mm in methods):
        models["walk"] = torch.load(WK.WALK_MODEL, map_location="cpu", weights_only=False).eval()
    for mm, path in (("post_wkp", WK.WALK_MODEL_POST), ("post_wks", WK.WALK_MODEL_SEQ)):
        if mm in methods:
            models[mm] = torch.load(path, map_location="cpu", weights_only=False).eval()
    env.reset(None)  # the first call after start-up differs (scalebfm-first-call-nondeterminism): warm up
    env.m.hold(1.0)
    res = []
    for i in ids:
        c = case(i)
        for meth in methods:
            tic = time.time()
            r = episode(env, rec, tape, models, c, meth, odo)
            res.append(r)
            print(f"w{wid} case {i} {meth}: err {r.get('err', float('nan')):.2f} cm pred {r.get('pred_err', float('nan')):.2f} "
                  f"+- {r.get('pred_std', float('nan')):.2f} walk_err {np.round(100 * np.array(r.get('walk_err', [np.nan] * 3)[:2]), 1)} "
                  f"fell {r.get('fell')} | {time.time() - tic:.0f} s", flush=True)  # fmt: skip
        json.dump(
            res,
            open(OUT / f"h_w{wid}{tag}.json", "w"),
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )


def launch(procs: int, gpus: str, n: int, methods: str = "", tag: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)])
        log = open(OUT / f"h_w{w}{tag}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "_w", "--wid", str(w), "--ids", ",".join(map(str, range(w, n, procs)))]
        cmd += ["--methods", methods, "--tag", tag] if methods else []
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [p.wait() for p in ps], flush=True)


def spec() -> str:
    return ",".join(str(IE.OUT.parent / f"data_{k}" / "all.npz") for k in DATA)


def export() -> None:
    import data as DT
    import torch
    import train as TR

    assert DT.GOAL_REF and DT.APPROACH
    D = DT.load(spec())
    model = TR.load_model("E3_ensemble", 1.0, D, "cpu", MODEL_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model, WM)
    print(f"{WM} from {MODEL_ROOT} ({len(D['n_steps'])} segments)", flush=True)


def offline() -> None:
    """data_approach's test split: the extended model vs the command itself vs the reach-only model told the walk went
    as commanded (its inputs: the pre-walk posture, the proxy plan, the reach command in the believed frame)."""
    import data as DT
    import torch

    D = DT.load(APPROACH_SPEC)
    sp = DT.splits(D)
    ix = sp["test_iid"]
    ix = ix[~D["fell"][ix].astype(bool)]
    wm = torch.load(WM, map_location="cpu", weights_only=False).eval()
    post = torch.load(WM_POST, map_location="cpu", weights_only=False).eval()
    F = lambda a: torch.as_tensor(np.asarray(a, np.float32))  # noqa: E731
    lab = D["pos_f"][ix].astype(np.float64)
    with torch.no_grad():
        b = dict(
            x0=F(D["x0"][ix]),
            u=F(D["u"][ix]),
            target=F(D["target"][ix]),
            rot6=F(D["rot6"][ix]),
            side=torch.as_tensor(D["side"][ix]),
        )
        p_wm = np.mean([mm.predict(b)["pos"].numpy() for mm in wm.members], 0).astype(np.float64)
        # reach-only: everything in the believed frame, the walk assumed exact, then mapped back through the commanded walk
        wk = D["walk"][ix].astype(np.float64)
        c, s = np.cos(wk[:, 2]), np.sin(wk[:, 2])
        tb = D["target_b"][ix].astype(np.float64)
        r6 = D["rot6"][ix].astype(np.float64)
        Rinv = lambda v: np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)  # noqa: E731
        r6b = np.c_[Rinv(r6[:, :3]), Rinv(r6[:, 3:])]
        u_b = D["u"][ix].astype(np.float64).copy()
        u_b[:, 1:4], u_b[:, 4:10] = tb, r6b
        b2 = dict(
            x0=F(D["x0"][ix][:, : DT.X0_DIM - 4]),
            u=F(u_b),
            target=F(tb),
            rot6=F(r6b),
            side=torch.as_tensor(D["side"][ix]),
        )
        pb = np.mean([mm.predict(b2)["pos"].numpy() for mm in post.members], 0).astype(np.float64)
    p_post = np.stack([wk[:, 0] + c * pb[:, 0] - s * pb[:, 1], wk[:, 1] + s * pb[:, 0] + c * pb[:, 1], pb[:, 2]], -1)
    rows = {"the command itself (walk and reach exact)": D["target"][ix].astype(np.float64), "reach-only model, walk assumed exact": p_post,
            "extended model": p_wm}  # fmt: skip
    werr = 100 * np.linalg.norm(D["walk_err"][ix][:, :2], axis=1)
    L = [f"data_approach + data_approach0 test split, {len(ix)} segments; walk ended off the asked position by median {np.median(werr):.2f} cm "
         f"(p90 {np.percentile(werr, 90):.2f}), turn off by median {np.degrees(np.median(np.abs(D['walk_err'][ix][:, 2]))):.2f} deg", "",
         "| prediction of where the jaw ends, start frame | median cm | p90 cm | < 1 cm | < 2 cm |", "|---|---|---|---|---|"]  # fmt: skip
    out = {}
    for k, p in rows.items():
        e = 100 * np.linalg.norm(p - lab, axis=1)
        out[k] = dict(
            median=float(np.median(e)),
            p90=float(np.percentile(e, 90)),
            lt1=float((e < 1).mean()),
            lt2=float((e < 2).mean()),
        )
        L.append(f"| {k} | {out[k]['median']:.2f} | {out[k]['p90']:.2f} | {out[k]['lt1']:.2f} | {out[k]['lt2']:.2f} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offline.md").write_text("\n".join(L) + "\n")
    json.dump(
        dict(rows=out, n=int(len(ix)), walk_err_med_cm=float(np.median(werr))),
        open(OUT / "offline.json", "w"),
        indent=1,
    )
    print("\n".join(L))


def oracle() -> None:
    """Phase o2: how much of the extended model's error is the walk. The reach-only model predicts each test segment's
    reach from the state the robot really stood in after the walk (data.derive re-run with the start at the reach
    command, the real plan), mapped back to the start frame through the TRUE post-walk pose: the prediction a robot that
    knew exactly where it stopped could make."""
    import data as DT
    import torch

    D = DT.load(APPROACH_SPEC)
    ix = DT.splits(D)["test_iid"]
    ix = ix[~D["fell"][ix].astype(bool)]
    so = D["so"]
    kr = so + D["k_reach"].astype(int)
    P = dict(D)
    wk = D["walk"].astype(np.float64)
    c, s = np.cos(wk[:, 2]), np.sin(wk[:, 2])
    r6 = D["rot6"].astype(np.float64)
    Rinv = lambda v: np.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)  # noqa: E731
    P.update(k0=D["k_reach"], start=np.c_[D["W"][kr, :3], D["W"][kr, 3]].astype(np.float32), ref_q=D["ref_q_post"],
             ref_pel=D["ref_pel_post"], ref_s=D["ref_s_post"], target=D["target_b"], target_b=D["target_b"],
             rot6=np.c_[Rinv(r6[:, :3]), Rinv(r6[:, 3:])].astype(np.float32), walk=np.zeros_like(D["walk"]))  # fmt: skip
    DT.derive(P)
    post = torch.load(WM_POST, map_location="cpu", weights_only=False).eval()
    wm = torch.load(WM, map_location="cpu", weights_only=False).eval()
    F = lambda a: torch.as_tensor(np.asarray(a, np.float32))  # noqa: E731
    with torch.no_grad():
        b = dict(x0=F(P["x0"][ix][:, : DT.X0_DIM - 4]), u=F(P["u"][ix]), target=F(P["target"][ix]), rot6=F(P["rot6"][ix]),
                 side=torch.as_tensor(P["side"][ix]))  # fmt: skip
        pp = np.mean([mm.predict(b)["pos"].numpy() for mm in post.members], 0).astype(np.float64)
        b2 = dict(
            x0=F(D["x0"][ix]),
            u=F(D["u"][ix]),
            target=F(D["target"][ix]),
            rot6=F(D["rot6"][ix]),
            side=torch.as_tensor(D["side"][ix]),
        )
        pw = np.mean([mm.predict(b2)["pos"].numpy() for mm in wm.members], 0).astype(np.float64)
    st0, st1 = D["start"][ix].astype(np.float64), P["start"][ix].astype(np.float64)
    wld = np.c_[st1[:, :2] + np.stack([np.cos(st1[:, 3]) * pp[:, 0] - np.sin(st1[:, 3]) * pp[:, 1],
                                       np.sin(st1[:, 3]) * pp[:, 0] + np.cos(st1[:, 3]) * pp[:, 1]], -1), pp[:, 2]]  # fmt: skip
    p_or = DT.hv(wld - np.c_[st0[:, :2], np.zeros(len(ix))], st0[:, 3])
    lab = D["pos_f"][ix].astype(np.float64)
    e_or, e_wm = (100 * np.linalg.norm(p - lab, axis=1) for p in (p_or, pw))
    e_post = 100 * np.linalg.norm(pp - P["pos_f"][ix], axis=1)  # same thing in the post-walk frame (a check)
    werr = 100 * np.linalg.norm(D["walk_err"][ix][:, :2], axis=1)
    L = [f"data_approach + data_approach0 test split, {len(ix)} segments", "",
         "| prediction | median cm | p90 cm |", "|---|---|---|",
         f"| reach-only model, from the real post-walk state, through the true stop | {np.median(e_or):.2f} | {np.percentile(e_or, 90):.2f} |",
         f"| (check: the same in the post-walk frame) | {np.median(e_post):.2f} | {np.percentile(e_post, 90):.2f} |",
         f"| extended model, from before the walk | {np.median(e_wm):.2f} | {np.percentile(e_wm, 90):.2f} |", "",
         f"walk stop error on these segments: median {np.median(werr):.2f} cm; Spearman(extended model error, walk stop error) "
         f"{__import__('scipy.stats', fromlist=['spearmanr']).spearmanr(e_wm, werr)[0]:.2f}"]  # fmt: skip
    (OUT / "oracle.md").write_text("\n".join(L) + "\n")
    json.dump(dict(oracle_med=float(np.median(e_or)), oracle_p90=float(np.percentile(e_or, 90)), wm_med=float(np.median(e_wm)),
                   wm_p90=float(np.percentile(e_wm, 90))), open(OUT / "oracle.json", "w"), indent=1)  # fmt: skip
    print("\n".join(L))


def report() -> None:
    R = []
    for f in sorted(OUT.glob("h_w[0-9].json")) + sorted(OUT.glob("h_w[0-9][0-9].json")):
        R += json.load(open(f))
    by = {}
    for r in R:
        by.setdefault(r["i"], {})[r["method"]] = r
    ok = [i for i, v in by.items() if all(mm in v and v[mm].get("fell") is None and "err" in v[mm] for mm in METHODS)]
    L = [f"{len(by)} cases, {len(ok)} with every method run and nothing fell; eyes closed after the sighting", "",
         "| method | cases | median cm | < 1 cm | < 2 cm | > 2 cm (n) | p90 | think s |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    summ = {}
    for grp, sel in (
        ("all", ok),
        ("forward", [i for i in ok if not by[i]["direct"]["back"]]),
        ("back", [i for i in ok if by[i]["direct"]["back"]]),
    ):
        for mm in METHODS:
            e = np.array([by[i][mm]["err"] for i in sel])
            if not len(e):
                continue
            th = np.array([by[i][mm].get("think_s", 0.0) for i in sel])
            summ[f"{grp}_{mm}"] = dict(n=len(e), median=float(np.median(e)), lt1=float((e < 1).mean()), lt2=float((e < 2).mean()),
                                        gt2=int((e > 2).sum()), p90=float(np.percentile(e, 90)))  # fmt: skip
            s = summ[f"{grp}_{mm}"]
            L.append(
                f"| {grp} {mm} | {s['n']} | {s['median']:.2f} | {s['lt1']:.2f} | {s['lt2']:.2f} | {s['gt2']} | {s['p90']:.2f} | {np.median(th):.1f} |"
            )
    L += ["", "| paired, all cases | a better | b better | a median / b median |", "|---|---|---|---|"]
    for a, b in (
        ("wm", "direct"),
        ("wm_walk", "wm"),
        ("wm", "post"),
        ("post", "direct"),
        ("walk_post", "post"),
        ("walk_post", "direct"),
    ):
        ea, eb = np.array([by[i][a]["err"] for i in ok]), np.array([by[i][b]["err"] for i in ok])
        L.append(
            f"| {a} vs {b} | {int((ea < eb).sum())} | {int((eb < ea).sum())} | {np.median(ea):.2f} / {np.median(eb):.2f} |"
        )
    pe = np.array([by[i]["wm"]["pred_err"] for i in ok])
    ps = np.array([by[i]["wm"]["pred_std"] for i in ok])
    ew = np.array([by[i]["wm"]["err"] for i in ok])
    pn = np.array([by[i]["wm"]["pred_naive"] for i in ok])
    ed = np.array([by[i]["direct"]["err"] for i in ok])
    wd = 100 * np.array([np.hypot(*by[i]["direct"]["walk_err"][:2]) for i in ok])
    from scipy.stats import spearmanr

    L += ["", f"walk ended off the asked position (direct walk): median {np.median(wd):.2f} cm, p90 {np.percentile(wd, 90):.2f} cm",
          f"wm's prediction for the direct command (before walking): median {np.median(pn):.2f} cm vs actual direct {np.median(ed):.2f} cm; "
          f"per case |pred - actual| median {np.median(np.abs(pn - ed)):.2f} cm, Spearman {spearmanr(pn, ed)[0]:.2f}",
          f"wm's chosen command: predicted miss median {np.median(pe):.2f} cm, std {np.median(ps):.2f}; actual {np.median(ew):.2f}; "
          f"std/error Spearman {spearmanr(ps, ew)[0]:.2f}"]  # fmt: skip
    ch = [by[i]["wm_walk"] for i in ok]
    moved = sum(
        np.hypot(r["walk"][0] - by[r["i"]]["direct"]["walk"][0], r["walk"][1] - by[r["i"]]["direct"]["walk"][1]) > 1e-6
        for r in ch
    )
    L.append(f"wm_walk chose a walk other than the direct one in {moved} of {len(ch)} cases")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    json.dump(dict(summ=summ, n=len(ok)), open(OUT / "results.json", "w"), indent=1)
    print("\n".join(L))


def report_wk() -> None:
    """The walk model's methods (h_w*_wk.json) paired with direct and post of the main run (h_w*.json); the rerun post
    checks that the replay is exact."""
    base, wk = {}, {}
    for f in sorted(OUT.glob("h_w*.json")):
        for r in json.load(open(f)):
            (wk if f.stem.endswith(("_wk", "_wkps")) else base).setdefault(r["i"], {})[r["method"]] = r
    same = [i for i in wk if "post" in wk[i] and "post" in base.get(i, {}) and "err" in wk[i]["post"]]
    exact = sum(abs(wk[i]["post"]["err"] - base[i]["post"]["err"]) < 1e-6 for i in same)
    rows = (("direct", base), ("post", base), ("direct_wk", wk), ("post_wk", wk), ("post_true", wk))
    if all(mm in wk.get(0, {}) for mm in WKPS_METHODS):
        rows = rows[:4] + tuple((mm, wk) for mm in WKPS_METHODS) + rows[4:]
    ok = [
        i
        for i in base
        if all(i in src and mm in src[i] and src[i][mm].get("fell") is None and "err" in src[i][mm] for mm, src in rows)
    ]
    E_ = {mm: np.array([src[i][mm]["err"] for i in ok]) for mm, src in rows}
    L = [f"{len(ok)} cases with every method run and nothing fell; post rerun equal to the main run in {exact} of {len(same)}", "",
         "| method | median cm | < 1 cm | < 2 cm | > 2 cm (n) | p90 |", "|---|---|---|---|---|---|"]  # fmt: skip
    for grp, sel in (
        ("all", ok),
        ("forward", [i for i in ok if not base[i]["direct"]["back"]]),
        ("back", [i for i in ok if base[i]["direct"]["back"]]),
    ):
        idx = [ok.index(i) for i in sel]
        for mm, _ in rows:
            e = E_[mm][idx]
            L.append(
                f"| {grp} {mm} | {np.median(e):.2f} | {(e < 1).mean():.2f} | {(e < 2).mean():.2f} | {int((e > 2).sum())} | {np.percentile(e, 90):.2f} |"
            )
    L += ["", "| paired | a better | b better | a median / b median |", "|---|---|---|---|"]
    pairs = (("direct_wk", "direct"), ("post_wk", "post"), ("post_wk", "direct"), ("post_true", "post_wk"))
    if len(rows) > 5:
        pairs += (
            ("post_wkp", "post_wk"),
            ("post_wks", "post_wk"),
            ("post_wks", "post_wkp"),
            ("post_wks", "direct"),
            ("post_true", "post_wks"),
        )
    for a, b in pairs:
        L.append(
            f"| {a} vs {b} | {int((E_[a] < E_[b]).sum())} | {int((E_[b] < E_[a]).sum())} | {np.median(E_[a]):.2f} / {np.median(E_[b]):.2f} |"
        )
    we = np.array(
        [
            [
                100 * wk[i]["post_wk"]["walk_err"][0],
                100 * wk[i]["post_wk"]["walk_err"][1],
                math.degrees(wk[i]["post_wk"]["walk_err"][2]),
            ]
            for i in ok
        ]
    )
    wp = np.array([wk[i]["post_wk"]["walk_pred"] for i in ok])
    r0, r1 = np.hypot(we[:, 0], we[:, 1]), np.hypot(we[:, 0] - wp[:, 0], we[:, 1] - wp[:, 1])
    L += ["", f"walk stop error (position, cm): median {np.median(r0):.2f}, p90 {np.percentile(r0, 90):.2f}; left after the walk model "
          f"median {np.median(r1):.2f}, p90 {np.percentile(r1, 90):.2f}; smaller in {int((r1 < r0).sum())} of {len(ok)}",
          f"turn error (deg): median {np.median(np.abs(we[:, 2])):.2f}, left {np.median(np.abs(we[:, 2] - wp[:, 2])):.2f}"]  # fmt: skip
    for mm in WKPS_METHODS if len(rows) > 5 else ():
        wq = np.array([wk[i][mm]["walk_pred"] for i in ok])
        r2 = np.hypot(we[:, 0] - wq[:, 0], we[:, 1] - wq[:, 1])
        L.append(
            f"{mm}: walk stop error left median {np.median(r2):.2f} cm, p90 {np.percentile(r2, 90):.2f}; "
            f"turn left {np.median(np.abs(we[:, 2] - wq[:, 2])):.2f} deg"
        )
    (OUT / "results_wk.md").write_text("\n".join(L) + "\n")
    json.dump(
        dict(n=len(ok), exact=exact, medians={mm: float(np.median(E_[mm])) for mm, _ in rows}),
        open(OUT / "results_wk.json", "w"),
        indent=1,
    )
    print("\n".join(L))


def report_odo() -> None:
    """post_odo / post_fuse (h_w*_odo.json) paired with direct and post of the main run and post_wk / post_true of _wk."""
    src = {}
    for f in sorted(OUT.glob("h_w*.json")):
        if f.stem.endswith(("_wkps",)):
            continue
        for r in json.load(open(f)):
            src.setdefault(r["i"], {})[r["method"]] = r
    rows = ("direct", "post", "post_wk", "post_odo", "post_fuse", "post_true")
    ok = [i for i in src if all(mm in src[i] and src[i][mm].get("fell") is None and "err" in src[i][mm] for mm in rows)]
    E_ = {mm: np.array([src[i][mm]["err"] for i in ok]) for mm in rows}
    L = [f"{len(ok)} cases with every method run and nothing fell", "",
         "| method | median cm | < 1 cm | < 2 cm | > 2 cm (n) | p90 |", "|---|---|---|---|---|---|"]  # fmt: skip
    for grp, sel in (
        ("all", ok),
        ("forward", [i for i in ok if not src[i]["direct"]["back"]]),
        ("back", [i for i in ok if src[i]["direct"]["back"]]),
    ):
        idx = [ok.index(i) for i in sel]
        for mm in rows:
            e = E_[mm][idx]
            L.append(
                f"| {grp} {mm} | {np.median(e):.2f} | {(e < 1).mean():.2f} | {(e < 2).mean():.2f} | {int((e > 2).sum())} | {np.percentile(e, 90):.2f} |"
            )
    L += ["", "| paired | a better | b better | a median / b median |", "|---|---|---|---|"]
    for a, b in (
        ("post_odo", "post"),
        ("post_odo", "post_wk"),
        ("post_fuse", "post_wk"),
        ("post_fuse", "post_odo"),
        ("post_fuse", "direct"),
        ("post_true", "post_fuse"),
    ):
        L.append(
            f"| {a} vs {b} | {int((E_[a] < E_[b]).sum())} | {int((E_[b] < E_[a]).sum())} | {np.median(E_[a]):.2f} / {np.median(E_[b]):.2f} |"
        )
    we = np.array([[100 * src[i]["post_odo"]["walk_err"][0], 100 * src[i]["post_odo"]["walk_err"][1]] for i in ok])
    L.append("")
    for nm, key, mm in (
        ("walk model", "walk_pred", "post_wk"),
        ("odometry", "walk_odo", "post_odo"),
        ("fused", "walk_fused", "post_fuse"),
    ):
        est = np.array([src[i][mm][key][:2] for i in ok])
        r = np.hypot(*(est - we).T)
        L.append(f"stop error left after the {nm}: median {np.median(r):.2f} cm, p90 {np.percentile(r, 90):.2f}")
    L.append(f"stop error itself: median {np.median(np.hypot(*we.T)):.2f} cm; odometry weight in the fusion, median "
             f"{np.median([src[i]['post_fuse']['odo_weight'][0] for i in ok]):.2f}")  # fmt: skip
    (OUT / "results_odo.md").write_text("\n".join(L) + "\n")
    json.dump(
        dict(n=len(ok), medians={mm: float(np.median(E_[mm])) for mm in rows}),
        open(OUT / "results_odo.json", "w"),
        indent=1,
    )
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["x", "o", "o2", "h", "_w", "d", "d2", "d3"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--n", type=int, default=N_CASES)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--methods", default="", help="comma-separated subset of METHODS")
    ap.add_argument("--tag", default="", help="suffix of the result files")
    a = ap.parse_args(argv)
    if a.phase == "x":
        export()
    elif a.phase == "o":
        offline()
    elif a.phase == "o2":
        oracle()
    elif a.phase == "h":
        launch(a.procs, a.gpus, a.n, a.methods, a.tag)
    elif a.phase == "_w":
        worker(
            a.wid, [int(x) for x in a.ids.split(",") if x], tuple(a.methods.split(",")) if a.methods else METHODS, a.tag
        )
    elif a.phase == "d2":
        report_wk()
    elif a.phase == "d3":
        report_odo()
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
