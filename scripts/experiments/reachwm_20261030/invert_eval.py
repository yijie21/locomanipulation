"""Eyes-closed reaching by inverting the reach world model (2026-10-31, improvement 3): for a wanted jaw-centre position,
find the command whose predicted landing is that position, with and without preferring commands the ensemble agrees
on, and execute it once in sim, servo off.

A goal is gen_reach.sample_command's draw (edge set excluded): side, wanted jaw centre (heading frame at command time),
wrist yaw offset and pitch, a pelvis height; even goal ids start at the reset stand, odd ones after a random walk. Every
variant of one goal replays the same reset (and walk), so all of them start from the bit-identical state.
    naive      the wanted position as the command
    servo      the same with Motor's jaw-centre servo on (it integrates the measured jaw error)
    aim        the command target moved so the E3 ensemble's mean lands on the wanted position (same pelvis height)
    aim_std    the same, loss + LAMBDA * |ensemble std|^2
    aim_std_p  aim_std over the pelvis heights PELVIS plus the goal's own, the lowest loss wins
The wrist rotation command stays the wanted one. Error = final jaw centre (1 s after the reach returns) to the wanted
position, cm.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/invert_eval.py a --goals 400 --procs 4    # naive + servo, start features
    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/reachwm_20261030/invert_eval.py b      # inverse commands (GPU)
    $PY scripts/experiments/reachwm_20261030/invert_eval.py c --procs 4                   # execute them
    $PY scripts/experiments/reachwm_20261030/invert_eval.py e --procs 4                   # servo on the chosen pelvis
    $PY scripts/experiments/reachwm_20261030/invert_eval.py f --procs 4                   # aim-off feedforward + servo
    $PY scripts/experiments/reachwm_20261030/invert_eval.py g --procs 4                   # servo_p, full settle
    $PY scripts/experiments/reachwm_20261030/invert_eval.py d                             # tables
Writes outputs/reachwm_20261030/invert/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "selfwm_20261029"))
import env as E  # noqa: E402

OUT = E.REPO / "outputs" / "reachwm_20261030" / "invert"
SEED = 9000000
LAMBDA = 1.0
PELVIS = (-1.0, 0.56, 0.62, 0.68, 0.74)  # -1 = stand
ITERS, LR = 300, 3e-3
VARIANTS = ("naive", "servo", "aim", "aim_std", "aim_std_p")
# phase e: the servo on top of the chosen pelvis height, without and with the aim-off
SERVO_VARIANTS = ("servo_p", "aim_std_p_servo", "aim_ff_servo", "servo_p_full")
# aim_ff_servo (phase f): the aim_std_p command as the reference (feedforward), the servo pulling the measured jaw centre
# to the WANTED position (Motor.servo_target), not to the aimed one. Its reach never meets the early stop (the jaw ends
# away from the aimed point), so servo_p_full (phase g) is servo_p with the early stop off: the same servo time.


class KeepTarget(dict):
    """Motor.reach pops servo_target at its start; this one keeps it."""

    def pop(self, k, d=None):
        return self.get(k, d)


def is_edge(c) -> bool:
    t, side = c["target"], 1 if c["side"] == "right" else -1
    return bool(t[0] > 0.85 or t[2] < 0.08 or side * t[1] < -0.50)


def setup(env, gid: int):
    """Reset (and walk) for goal `gid`; returns the goal command. Deterministic per gid."""
    import gen_reach as G

    rng = np.random.default_rng(SEED + gid)
    env.reset(None)
    walked = gid % 2 == 1
    ang, d, yw = rng.uniform(-math.pi, math.pi), rng.uniform(0.3, 1.2), float(rng.uniform(-math.pi, math.pi))
    if walked:
        env.m.walk_to(np.array([d * math.cos(ang), d * math.sin(ang)]), yaw=yw)
    c = G.sample_command(rng)
    while is_edge(c):
        c = G.sample_command(rng)
    return c, walked


def execute(
    env, rec, c: dict, target_h, pelvis_z, servo: bool, walked: bool, servo_goal_h=None, full=False, rot_h=None
) -> dict:
    """gen_reach.segment with the command target replaced by `target_h` and the pelvis height by `pelvis_z`; `rot_h`
    (3x3, heading frame) replaces the wrist rotation c's dyaw and pitch give (pose_invert.py)."""
    import gen_reach as G

    m = env.m
    side = c["side"]
    rec.start(side)
    m.hold(G.PRE_S)
    p, Rp = env.pelvis()
    yaw = E.yaw_of(Rp)
    Rh = E.rot_z(yaw)
    t = np.asarray(target_h, float)
    target_w = np.r_[p[:2] + (Rh @ np.r_[t[:2], 0.0])[:2], t[2]]
    rot_w = E.rot_z(yaw + c["dyaw"]) @ G.rot_y(c["pitch"]) if rot_h is None else Rh @ np.asarray(rot_h, float)
    k0 = len(rec.S) - 1
    rec.t_cmd = m.t
    if servo_goal_h is not None:
        g = np.asarray(servo_goal_h, float)
        m.servo_target = KeepTarget({side: np.r_[p[:2] + (Rh @ np.r_[g[:2], 0.0])[:2], g[2]]})
    cfg = m.cfg
    if full:  # MotorConfig is frozen; the settle always runs reach_settle_s
        import dataclasses

        m.cfg = dataclasses.replace(cfg, reach_tolerance=-1.0)
    out = m.reach(side, target_w, rot_w, pelvis_z=None if pelvis_z < 0 else float(pelvis_z), servo=servo)
    m.cfg = cfg
    if m.fell_at is None:
        m.hold(G.POST_S)
    rec.on = False
    m.servo_target = {}
    rot_h = Rh.T @ rot_w
    return dict(
        S=np.array(rec.S), W=np.array(rec.W), J=np.array(rec.J), C=np.array(rec.C), k0=k0,
        side=1 if side == "right" else -1, target=t.astype(np.float32), dyaw=c["dyaw"], pitch=c["pitch"],
        rot6=np.r_[rot_h[:, 0], rot_h[:, 1]].astype(np.float32), pelvis_z=float(pelvis_z),
        start=np.r_[p, yaw].astype(np.float32), walked=walked, chain=0, fell=m.fell_at is not None,
        ok=bool(out.ok), jaw_err_cm=float((out.info or {}).get("jaw_err_cm", -1.0)),
        att_err_deg=-1.0, ik_err_cm=float((out.info or {}).get("ik_err_cm", -1.0)), reach_s=float(m.t - rec.t_cmd), reset=0,
    )  # fmt: skip


def pack(segs: list[dict]) -> dict:
    out = {k: np.concatenate([s[k] for s in segs]) for k in ("S", "W", "J", "C")}
    out["n_steps"] = np.array([len(s["S"]) for s in segs])
    for k in ("k0", "side", "dyaw", "pitch", "pelvis_z", "walked", "chain", "fell", "ok", "jaw_err_cm", "att_err_deg",
              "ik_err_cm", "reach_s", "reset"):  # fmt: skip
        out[k] = np.array([s[k] for s in segs])
    for k in ("target", "rot6", "start"):
        out[k] = np.stack([s[k] for s in segs])
    return out


def summarize(segs: list[dict], goal_t: np.ndarray) -> dict:
    """data.derive on the executed segments: start features, command, landing and its error to the wanted position."""
    import data as DT

    D = pack(segs)
    DT.derive(D)
    err = 100 * np.linalg.norm(D["pos_f"] - goal_t, axis=1)
    return dict(x0=D["x0"], u=D["u"], pos_f=D["pos_f"], err=err, fell=D["fell"], settled=D["settled"])


def worker(phase: str, wid: int, gids: list[int]) -> None:
    import gen_reach as G

    env = E.Env("open")
    rec = G.Recorder(env)
    cmds = np.load(OUT / "commands.npz") if phase in ("c", "e", "f", "g") else None
    rows = {}
    for gid in gids:
        if phase == "a":
            runs = [("naive", None, None, False), ("servo", None, None, True)]
        elif phase == "e":
            j = int(np.where(cmds["gid"] == gid)[0][0])
            pz = float(cmds["aim_std_p_pz"][j])
            runs = [("servo_p", None, pz, True), ("aim_std_p_servo", cmds["aim_std_p_t"][j], pz, True)]
        elif phase == "f":
            j = int(np.where(cmds["gid"] == gid)[0][0])
            runs = [("aim_ff_servo", cmds["aim_std_p_t"][j], float(cmds["aim_std_p_pz"][j]), True)]
        elif phase == "g":
            j = int(np.where(cmds["gid"] == gid)[0][0])
            runs = [("servo_p_full", None, float(cmds["aim_std_p_pz"][j]), True)]
        else:
            j = int(np.where(cmds["gid"] == gid)[0][0])
            runs = [(v, cmds[f"{v}_t"][j], float(cmds[f"{v}_pz"][j]), False) for v in ("aim", "aim_std", "aim_std_p")]
        for v, t, pz, servo in runs:
            c, walked = setup(env, gid)
            if env.m.fell_at is not None:
                break
            goal_t = c["target"].astype(np.float32)
            pz0 = -1.0 if c["pelvis_z"] is None else float(c["pelvis_z"])
            seg = execute(env, rec, c, goal_t if t is None else t, pz0 if pz is None else pz, servo, walked,
                          servo_goal_h=goal_t if v == "aim_ff_servo" else None, full=v == "servo_p_full")  # fmt: skip
            s = summarize([seg], goal_t[None])
            rows.setdefault(v, []).append(dict(gid=gid, walked=walked, goal_t=goal_t, goal_pz=pz0, rot6=seg["rot6"],
                                               side=seg["side"], **{k: x[0] for k, x in s.items()}))  # fmt: skip
        print(f"{phase}{wid} goal {gid} done", flush=True)
    flat = {f"{v}_{k}": np.array([r[k] for r in rs]) for v, rs in rows.items() for k in rs[0]}
    np.savez(OUT / f"{phase}_w{wid}.npz", **flat)


def launch(phase: str, procs: int, goals: int, gpus: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gids = list(range(goals)) if phase == "a" else np.load(OUT / "commands.npz")["gid"].tolist()
    cards = gpus.split(",")
    jobs = []
    for w in range(procs):
        envv = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[w % len(cards)], OMP_NUM_THREADS="1")
        log = open(OUT / f"{phase}_w{w}.log", "w")
        cmd = [
            str(E.SIM_PY),
            __file__,
            "worker",
            "--phase",
            phase,
            "--wid",
            str(w),
            "--gids",
            ",".join(map(str, gids[w::procs])),
        ]
        jobs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print(phase, "exit codes", [j.wait() for j in jobs], flush=True)


def gather(phase: str) -> dict:
    zs = [dict(np.load(p)) for p in sorted(OUT.glob(f"{phase}_w*.npz"))]
    return {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}


def invert() -> None:
    """Phase b: per goal and pelvis candidate, gradient descent on the command target through the E3 ensemble."""
    import data as DT
    import torch
    import train as TR

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = gather("a")
    assert np.array_equal(A["naive_gid"], A["servo_gid"])  # both runs of a goal or neither (a walk fall skips both)
    keep = np.where((A["naive_fell"] == 0) & (A["servo_fell"] == 0))[0]
    D = DT.load(None)
    model = TR.load_model("E3_ensemble", 1.0, D, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
    x0, side, rot6, goal = (
        T(A["naive_x0"][keep]),
        T(A["naive_side"][keep]),
        T(A["naive_rot6"][keep]),
        T(A["naive_goal_t"][keep]),
    )
    own = A["naive_goal_pz"][keep]
    lo, hi = T([0.05, -0.60, 0.02]), T([1.00, 0.35, 1.50])  # gen_reach.WIDE, right hand; y mirrored for the left
    N = len(keep)

    def solve(pz: np.ndarray, lam: float):
        t = goal.clone().requires_grad_(True)
        opt = torch.optim.Adam([t], lr=LR)
        stand = T((pz < 0).astype(np.float32))
        pzv = T(np.where(pz < 0, 0.0, pz))
        s = side[:, None]

        def pred(tt):
            u = torch.cat([side[:, None], tt, rot6, stand[:, None], pzv[:, None]], -1)
            return model.predict(dict(x0=x0, u=u, target=tt, rot6=rot6, side=side))

        for _ in range(ITERS):
            o = pred(t)
            loss = ((o["pos"] - goal) ** 2).sum(-1) + lam * (o["pos_std"] ** 2).sum(-1)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            with torch.no_grad():  # stay inside the training command box
                y = t[:, 1:2] * s
                t[:, 0].clamp_(lo[0], hi[0])
                t[:, 2].clamp_(lo[2], hi[2])
                t[:, 1:2] = y.clamp(lo[1], hi[1]) * s
        with torch.no_grad():
            o = pred(t)
            l_pos = ((o["pos"] - goal) ** 2).sum(-1)
            l_std = (o["pos_std"] ** 2).sum(-1)
        return (
            t.detach(),
            (l_pos + lam * l_std).cpu().numpy(),
            (100 * l_pos.sqrt()).cpu().numpy(),
            (100 * l_std.sqrt()).cpu().numpy(),
        )

    out = dict(gid=A["naive_gid"][keep])
    with torch.no_grad():
        u0 = torch.cat(
            [
                side[:, None],
                goal,
                rot6,
                T((own < 0).astype(np.float32))[:, None],
                T(np.where(own < 0, 0.0, own))[:, None],
            ],
            -1,
        )
        o0 = model.predict(dict(x0=x0, u=u0, target=goal, rot6=rot6, side=side))
    out["naive_pred_err"] = (100 * (o0["pos"] - goal).norm(dim=-1)).cpu().numpy()
    out["naive_pred_std"] = (100 * o0["pos_std"].norm(dim=-1)).cpu().numpy()
    for v, lam in (("aim", 0.0), ("aim_std", LAMBDA)):
        t, _, pe, ps = solve(own, lam)
        out[f"{v}_t"], out[f"{v}_pz"], out[f"{v}_pred_err"], out[f"{v}_pred_std"] = t.cpu().numpy(), own, pe, ps
    cands = [own] + [np.full(N, p) for p in PELVIS]
    sols = [solve(pz, LAMBDA) for pz in cands]
    best = np.argmin(np.stack([s[1] for s in sols]), 0)
    pick = lambda k: np.stack([np.asarray(s[k].cpu() if hasattr(s[k], "cpu") else s[k]) for s in sols])[
        best, np.arange(N)
    ]  # noqa: E731
    out["aim_std_p_t"], out["aim_std_p_pz"] = pick(0), np.stack(cands)[best, np.arange(N)]
    out["aim_std_p_pred_err"], out["aim_std_p_pred_std"] = pick(2), pick(3)
    np.savez(OUT / "commands.npz", **out)
    print(f"{N} goals inverted; aim-off median {np.median(100 * np.linalg.norm(out['aim_std_t'] - A['naive_goal_t'][keep], axis=1)):.2f} cm, "
          f"pelvis changed on {(out['aim_std_p_pz'] != own).mean():.2f}", flush=True)  # fmt: skip


def report() -> None:
    A, C, K = gather("a"), gather("c"), dict(np.load(OUT / "commands.npz"))
    E_ = gather("e") if list(OUT.glob("e_w*.npz")) else {}
    for ph in ("f", "g"):
        if E_ and list(OUT.glob(f"{ph}_w*.npz")):
            E_.update(gather(ph))
    variants = VARIANTS + (tuple(v for v in SERVO_VARIANTS if f"{v}_gid" in E_) if E_ else ())
    src = lambda v: A if v in ("naive", "servo") else (E_ if v in SERVO_VARIANTS else C)  # noqa: E731
    by = {v: {int(g): i for i, g in enumerate(src(v)[f"{v}_gid"])} for v in variants}
    gids = [int(g) for g in K["gid"] if all(int(g) in by[v] for v in variants)]
    err = {v: np.array([src(v)[f"{v}_err"][by[v][g]] for g in gids]) for v in variants}
    fell = {v: np.array([src(v)[f"{v}_fell"][by[v][g]] for g in gids]) for v in variants}
    walked = np.array([g % 2 == 1 for g in gids])
    kidx = {int(g): i for i, g in enumerate(K["gid"])}
    base = lambda v: "naive" if v in ("naive", "servo") else ("aim_std_p" if v in SERVO_VARIANTS else v)  # noqa: E731
    pred_std = {v: np.array([K[f"{base(v)}_pred_std"][kidx[g]] for g in gids]) for v in variants}
    pred_err = {v: np.array([K[f"{base(v)}_pred_err"][kidx[g]] for g in gids]) for v in variants}
    lines = ["| variant | start | n | median | p90 | <1 cm | <2 cm | fell | predicted miss median | ensemble std median |",
             "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    res = dict(n_goals=len(gids), rows=[])
    for v in variants:
        for nm, m in (("all", np.ones(len(gids), bool)), ("reset stand", ~walked), ("after walk", walked)):
            e = err[v][m & (fell[v] == 0)]
            r = dict(variant=v, start=nm, n=int(len(e)), median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                     lt1=float(np.mean(e < 1)), lt2=float(np.mean(e < 2)), fell=int(fell[v][m].sum()),
                     pred_err=float(np.median(pred_err[v][m])), pred_std=float(np.median(pred_std[v][m])))  # fmt: skip
            res["rows"].append(r)
            lines.append(f"| {v} | {nm} | {r['n']} | {r['median']:.2f} | {r['p90']:.2f} | {r['lt1']:.2f} | {r['lt2']:.2f} | "
                         f"{r['fell']} | {r['pred_err']:.2f} | {r['pred_std']:.2f} |")  # fmt: skip
    # selective: aim_std_p by its predicted std quantile
    v = "aim_std_p"
    s, e = pred_std[v][fell[v] == 0], err[v][fell[v] == 0]
    lines += [
        "",
        "| aim_std_p, goals kept by ensemble std | share | std below cm | median | p90 | <1 cm |",
        "|---|---|---|---|---|---|",
    ]
    for q in (0.1, 0.25, 0.5, 1.0):
        th = np.quantile(s, q)
        m = s <= th
        res.setdefault("selective", []).append(
            dict(
                share=q,
                th=float(th),
                median=float(np.median(e[m])),
                p90=float(np.percentile(e[m], 90)),
                lt1=float(np.mean(e[m] < 1)),
            )
        )
        lines.append(
            f"| lowest {int(q * 100)}% | {q:.2f} | {th:.2f} | {np.median(e[m]):.2f} | {np.percentile(e[m], 90):.2f} | {np.mean(e[m] < 1):.2f} |"
        )
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["a", "b", "c", "d", "e", "f", "g", "worker"])
    ap.add_argument("--goals", type=int, default=400)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--phase", default="a")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--gids", default="")
    a = ap.parse_args(argv)
    if a.what in ("a", "c", "e", "f", "g"):
        launch(a.what, a.procs, a.goals, a.gpus)
    elif a.what == "b":
        invert()
    elif a.what == "d":
        report()
    else:
        worker(a.phase, a.wid, [int(x) for x in a.gids.split(",") if x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
