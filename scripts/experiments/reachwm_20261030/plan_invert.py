"""Plan-aware inversion (2026-10-31, improvement 2 into 3): invert_eval.py's aim with the goal-reference world model
(models_goalref/, REACHWM_GOAL_REF=1), which also sees the plan reach() would track (its final whole-body IK pose). The
plan is not differentiable, so per goal and pelvis candidate the aim is a fixed-point iteration in sim:

    t_0      the plain E3 inversion at this pelvis height (invert_eval's solve, + LAMBDA * |ensemble std|^2)
    t_k+1    t_k + (goal - landing predicted for t_k given plan(t_k)), clamped to the WIDE box

where plan(t) is Motor.reach's own IK for command t from the current state, stopped before anything moves (its _blend
raises). Over ITERS + 1 iterates and the pelvis candidates, the lowest predicted |miss|^2 + LAMBDA * |ensemble std|^2
wins. Every goal replays the same reset (and walk) as invert_eval.py, so starts are bit-identical to its variants:
    ref_aim        the chosen command, servo off                                   (compare aim_std_p)
    ref_ff_servo   the chosen command as feedforward, the servo pulling the measured jaw to the wanted position
                   (compare aim_ff_servo)

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 $PY scripts/experiments/reachwm_20261030/plan_invert.py p    # start aims + portable model
    $PY scripts/experiments/reachwm_20261030/plan_invert.py h --procs 6                # plan iterations + execution
    $PY scripts/experiments/reachwm_20261030/plan_invert.py d                          # tables, with invert_eval's
Writes outputs/reachwm_20261030/invert/plan/.
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

PLAN = IE.OUT / "plan"
ITERS = 4
REF_DIM = 68  # data.py GOAL_REF features
CORE = (np.array([0.15, -0.40, 0.15]), np.array([0.70, 0.15, 1.35]))  # gen_reach.CORE, right hand
WIDE = (np.array([0.05, -0.60, 0.02]), np.array([1.00, 0.35, 1.50]))  # gen_reach.WIDE


class PlanOnly(Exception):
    pass


def plan_of(m, side: str, target_w, rot_w, pz: float, **reach_kw):
    """reach()'s whole-body IK plan (poses, durations) for this command from the current state, without moving: the
    reference is built by _blend after every IK solve, so a _blend that raises stops reach before it touches the motor.
    reach_kw: reach's planner options (speed, start, ik_from_ref), so the plan is the one a reach with them gets."""
    blend, got = m._blend, {}

    def stop(poses, durations):
        got["plan"] = (poses, durations)
        raise PlanOnly

    m._blend = stop
    try:
        m.reach(side, target_w, rot_w, pelvis_z=None if pz < 0 else float(pz), servo=False, **reach_kw)
    except PlanOnly:
        pass
    finally:
        m._blend = blend
    return got["plan"]


def plan_record(plan) -> dict:
    """The arrays gen_reach.segment saves for a plan."""
    goal = plan[0][-1]
    return dict(ref_q=goal.joints.astype(np.float32), ref_pel=np.r_[goal.pos, goal.quat].astype(np.float32),
                ref_s=float(sum(plan[1])))  # fmt: skip


def plan_features(rec: dict, q0: np.ndarray, start: np.ndarray) -> np.ndarray:
    """data.derive's GOAL_REF block for one segment (start = the saved float32 [pelvis xyz, yaw])."""
    import data as DT

    st = start.astype(np.float32).astype(np.float64)
    yaw0, org = st[3:4], np.r_[st[:2], 0.0]
    w, x, y, z = rec["ref_pel"][3:].astype(np.float64)
    c0 = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    c1 = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)])
    h = lambda v: DT.hv(np.asarray(v, np.float64)[None], yaw0)[0]  # noqa: E731
    rq = rec["ref_q"]
    return np.r_[rq, rq - q0, h(rec["ref_pel"][:3].astype(np.float64) - org), h(c0), h(c1), rec["ref_s"]].astype(
        np.float32
    )


def prepare() -> None:
    """Phase p: plain-E3 start aims for every goal and pelvis candidate (GPU), and the goal-reference E3 pickled whole
    (its normalization buffers inside), so sim workers need no training data."""
    import data as DT
    import torch
    import train as TR

    assert not DT.GOAL_REF, "run phase p without REACHWM_GOAL_REF (it loads both models)"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = IE.gather("a")
    keep = np.where((A["naive_fell"] == 0) & (A["servo_fell"] == 0))[0]
    D = DT.load(None)
    model = TR.load_model("E3_ensemble", 1.0, D, dev)
    del D
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
    lo, hi = T(WIDE[0]), T(WIDE[1])
    N = len(keep)

    def solve(pz: np.ndarray):  # invert_eval.invert's solve, lam = LAMBDA
        t = goal.clone().requires_grad_(True)
        opt = torch.optim.Adam([t], lr=IE.LR)
        stand, pzv, s = T((pz < 0).astype(np.float32)), T(np.where(pz < 0, 0.0, pz)), side[:, None]
        for _ in range(IE.ITERS):
            u = torch.cat([side[:, None], t, rot6, stand[:, None], pzv[:, None]], -1)
            o = model.predict(dict(x0=x0, u=u, target=t, rot6=rot6, side=side))
            loss = ((o["pos"] - goal) ** 2).sum(-1) + IE.LAMBDA * (o["pos_std"] ** 2).sum(-1)
            opt.zero_grad()
            loss.sum().backward()
            opt.step()
            with torch.no_grad():
                y = t[:, 1:2] * s
                t[:, 0].clamp_(lo[0], hi[0])
                t[:, 2].clamp_(lo[2], hi[2])
                t[:, 1:2] = y.clamp(lo[1], hi[1]) * s
        return t.detach().cpu().numpy()

    cands = [own] + [np.full(N, p) for p in IE.PELVIS]
    PLAN.mkdir(parents=True, exist_ok=True)
    np.savez(PLAN / "init.npz", gid=A["naive_gid"][keep], pz=np.stack(cands), t=np.stack([solve(pz) for pz in cands]),
             x0=A["naive_x0"][keep], goal=A["naive_goal_t"][keep], side=A["naive_side"][keep], rot6=A["naive_rot6"][keep])  # fmt: skip
    # the goal-reference model: the same data module switched to the plan features (both read at call time)
    DT.GOAL_REF, DT.X0_DIM = True, DT.X0_DIM + REF_DIM
    R = DT.load(str(DT.OUT / "data_ref" / "all.npz"))
    ref = TR.load_model("E3_ensemble", 1.0, R, "cpu", DT.OUT / "models_goalref")
    torch.save(ref, PLAN / "wm_ref.pt")
    print(f"{N} goals x {len(cands)} pelvis candidates; start aims and wm_ref.pt in {PLAN}", flush=True)


def summarize(seg: dict, rec: dict, goal_t: np.ndarray) -> dict:
    """invert_eval.summarize with the plan arrays, so data.derive (GOAL_REF) builds the model's own x0."""
    import data as DT

    D = IE.pack([seg])
    D.update({k: np.asarray(v)[None] for k, v in rec.items()})
    DT.derive(D)
    err = 100 * np.linalg.norm(D["pos_f"] - goal_t, axis=1)
    return dict(x0=D["x0"], pos_f=D["pos_f"], err=err, fell=D["fell"], settled=D["settled"])


def worker(wid: int, gids: list[int]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF and DT.X0_DIM == 206 + REF_DIM
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    I = np.load(PLAN / "init.npz")
    model = torch.load(PLAN / "wm_ref.pt", map_location="cpu", weights_only=False).eval()
    lo, hi = WIDE
    rows = {}
    for gid in gids:
        j = int(np.where(I["gid"] == gid)[0][0])
        x0b, goal, sd, rot6 = I["x0"][j], I["goal"][j].astype(np.float64), float(I["side"][j]), I["rot6"][j]
        # planning: the start state invert_eval.execute reaches (setup, then the PRE_S hold), IK only
        c, walked = IE.setup(env, gid)
        m = env.m
        if m.fell_at is not None:
            continue
        rec.start(c["side"])
        m.hold(G.PRE_S)
        rec.on = False
        p, Rp = env.pelvis()
        yaw = E.yaw_of(Rp)
        Rh = E.rot_z(yaw)
        rot_w = E.rot_z(yaw + c["dyaw"]) @ G.rot_y(c["pitch"])
        start = np.r_[p, yaw].astype(np.float32)
        tic = time.time()
        best, trace = None, []
        for ci in range(len(I["pz"])):
            # every iterate is a float32 command, the one execute() sends: reach()'s IK can switch posture on a
            # 1e-8 m change near a bifurcation (2026-10-31 first run: 23 % of plans differed from the executed one)
            pz, t = float(I["pz"][ci, j]), I["t"][ci, j].astype(np.float32).astype(np.float64)
            for k in range(ITERS + 1):
                tw = np.r_[p[:2] + (Rh @ np.r_[t[:2], 0.0])[:2], t[2]]
                pr = plan_record(plan_of(m, c["side"], tw, rot_w, pz))
                x0 = np.r_[x0b, plan_features(pr, x0b[:29], start)]
                u = np.r_[sd, t, rot6, float(pz < 0), 0.0 if pz < 0 else pz]
                F = lambda a: torch.as_tensor(np.asarray(a, np.float32)[None])  # noqa: E731
                with torch.no_grad():
                    o = model.predict(dict(x0=F(x0), u=F(u), target=F(t), rot6=F(rot6), side=torch.tensor([sd])))
                pos, std = o["pos"][0].numpy().astype(np.float64), o["pos_std"][0].numpy().astype(np.float64)
                loss = float(((pos - goal) ** 2).sum() + IE.LAMBDA * (std**2).sum())
                trace.append([ci, k, loss, 100 * np.linalg.norm(pos - goal), 100 * np.linalg.norm(std)])
                if best is None or loss < best["loss"]:
                    best = dict(loss=loss, ci=ci, k=k, t=t.copy(), pz=pz, pred=pos, std=std, x0=x0, rec=pr)
                if k < ITERS:
                    t = t + (goal - pos)
                    y = np.clip(t[1] * sd, lo[1], hi[1]) * sd
                    t = (
                        np.r_[np.clip(t[0], lo[0], hi[0]), y, np.clip(t[2], lo[2], hi[2])]
                        .astype(np.float32)
                        .astype(np.float64)
                    )
        plan_s = time.time() - tic
        for v, servo in (("ref_aim", False), ("ref_ff_servo", True)):
            c, walked = IE.setup(env, gid)
            rec.plan = None
            seg = IE.execute(env, rec, c, best["t"].astype(np.float32), best["pz"], servo, walked,
                             servo_goal_h=goal.astype(np.float32) if servo else None)  # fmt: skip
            tracked = plan_record(rec.plan)
            same_plan = all(np.array_equal(tracked[k], best["rec"][k]) for k in tracked)
            s = summarize(seg, tracked, goal.astype(np.float32)[None])
            rows.setdefault(v, []).append(dict(
                gid=gid, walked=walked, goal_t=goal.astype(np.float32), side=sd, t=best["t"], pz=best["pz"],
                ci=best["ci"], k=best["k"], pred_err=100 * np.linalg.norm(best["pred"] - goal),
                pred_std=100 * np.linalg.norm(best["std"]), pred_to_land=100 * np.linalg.norm(best["pred"] - s["pos_f"][0]),
                same_plan=same_plan, x0_diff=float(np.abs(s["x0"][0] - best["x0"]).max()), plan_s=plan_s,
                init_t=I["t"][best["ci"], j], err=s["err"][0], fell=s["fell"][0], settled=s["settled"][0]))  # fmt: skip
        rows.setdefault("trace", []).append(dict(gid=gid, trace=np.array(trace)))
        r = rows["ref_aim"][-1]
        print(f"h{wid} goal {gid} plan {plan_s:.1f} s best cand {r['ci']} iter {r['k']} pred {r['pred_err']:.2f} "
              f"std {r['pred_std']:.2f} | ref_aim {r['err']:.2f} ff_servo {rows['ref_ff_servo'][-1]['err']:.2f} "
              f"same_plan {r['same_plan']} x0_diff {r['x0_diff']:.1e}", flush=True)  # fmt: skip
    flat = {f"{v}_{k}": np.array([r[k] for r in rs]) for v, rs in rows.items() if v != "trace" for k in rs[0]}
    flat["trace_gid"] = np.array([r["gid"] for r in rows.get("trace", [])])
    flat["trace"] = np.array([r["trace"] for r in rows.get("trace", [])])
    np.savez(PLAN / f"h_w{wid}.npz", **flat)


def launch(procs: int, gpus: str, limit: int) -> None:
    gids = np.load(PLAN / "init.npz")["gid"].tolist()[: limit or None]
    cards = gpus.split(",")
    jobs = []
    for w in range(procs):
        envv = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[w % len(cards)], OMP_NUM_THREADS="1", REACHWM_GOAL_REF="1")
        log = open(PLAN / f"h_w{w}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "worker", "--wid", str(w), "--gids", ",".join(map(str, gids[w::procs]))]
        jobs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [j.wait() for j in jobs], flush=True)


def report() -> None:
    zs = [dict(np.load(p)) for p in sorted(PLAN.glob("h_w*.npz"))]
    H = {k: np.concatenate([z[k] for z in zs]) for k in zs[0] if not k.startswith("trace")}
    A, C = IE.gather("a"), IE.gather("c")
    F_, G_ = IE.gather("f"), IE.gather("g")
    srcs = {
        "naive": A,
        "servo": A,
        "aim_std_p": C,
        "servo_p_full": G_,
        "aim_ff_servo": F_,
        "ref_aim": H,
        "ref_ff_servo": H,
    }
    by = {v: {int(g): i for i, g in enumerate(S[f"{v}_gid"])} for v, S in srcs.items()}
    gids = [int(g) for g in H["ref_aim_gid"] if all(int(g) in by[v] for v in srcs)]
    col = lambda v, k: np.array([srcs[v][f"{v}_{k}"][by[v][g]] for g in gids])  # noqa: E731
    goal, side, walked = col("ref_aim", "goal_t"), col("ref_aim", "side"), col("ref_aim", "walked").astype(bool)
    tt = np.c_[goal[:, 0], goal[:, 1] * side, goal[:, 2]]
    core = np.all((tt >= CORE[0]) & (tt <= CORE[1]), axis=1)
    err = {v: col(v, "err") for v in srcs}
    ok = np.all([col(v, "fell") == 0 for v in srcs], axis=0)
    lines = [f"goals {len(gids)} (all variants, none fell {int(ok.sum())}), core {int((core & ok).sum())}", "",
             "| method | region | n | median | p90 | <0.5 | <1 cm | <2 cm |", "|---|---|---|---|---|---|---|---|"]  # fmt: skip
    res = dict(n=len(gids), rows=[])
    for v in srcs:
        for nm, msk in (
            ("all", ok),
            ("core", core & ok),
            ("core, reset", core & ok & ~walked),
            ("core, walk", core & ok & walked),
        ):
            e = err[v][msk]
            r = dict(method=v, region=nm, n=int(msk.sum()), median=float(np.median(e)), p90=float(np.percentile(e, 90)),
                     lt05=float(np.mean(e < 0.5)), lt1=float(np.mean(e < 1)), lt2=float(np.mean(e < 2)))  # fmt: skip
            res["rows"].append(r)
            lines.append(
                f"| {v} | {nm} | {r['n']} | {r['median']:.2f} | {r['p90']:.2f} | {r['lt05']:.2f} | {r['lt1']:.2f} | {r['lt2']:.2f} |"
            )
    m = core & ok
    for a, b in (("ref_aim", "aim_std_p"), ("ref_ff_servo", "aim_ff_servo")):
        d = err[a][m] - err[b][m]
        lines.append(f"paired core {a} vs {b}: better on {np.mean(d < 0):.2f}, median diff {np.median(d):+.2f} cm")
        res[f"paired_{a}"] = dict(better=float(np.mean(d < 0)), median_diff=float(np.median(d)))
    # selective by the goal-reference ensemble's std, open loop
    s, e = col("ref_aim", "pred_std")[m], err["ref_aim"][m]
    lines += [
        "",
        "| ref_aim core, goals kept by ensemble std | std below cm | median | p90 | <1 cm |",
        "|---|---|---|---|---|",
    ]
    for q in (0.1, 0.25, 0.5, 1.0):
        th = np.quantile(s, q)
        k = s <= th
        lines.append(
            f"| lowest {int(q * 100)}% | {th:.2f} | {np.median(e[k]):.2f} | {np.percentile(e[k], 90):.2f} | {np.mean(e[k] < 1):.2f} |"
        )
    from scipy.stats import spearmanr

    rho = spearmanr(col("ref_aim", "pred_std")[ok], err["ref_aim"][ok]).correlation
    aim_move = 100 * np.linalg.norm(col("ref_aim", "t") - col("ref_aim", "init_t"), axis=1)
    lines += ["", f"plan matches the executed one: {col('ref_aim', 'same_plan').mean():.3f}; "
              f"max x0 diff {col('ref_aim', 'x0_diff').max():.1e}",
              f"chosen iterate k: {np.bincount(col('ref_aim', 'k').astype(int), minlength=ITERS + 1).tolist()}; "
              f"aim moved from the plain start median {np.median(aim_move):.2f} cm; pelvis candidate "
              f"{np.bincount(col('ref_aim', 'ci').astype(int)).tolist()}",
              f"predicted vs actual landing (ref_aim) median {np.median(col('ref_aim', 'pred_to_land')[ok]):.2f} cm; "
              f"std/error Spearman {rho:.2f}; plan time median {np.median(col('ref_aim', 'plan_s')):.1f} s / goal"]  # fmt: skip
    json.dump(res, open(PLAN / "results.json", "w"), indent=1)
    (PLAN / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["p", "h", "d", "worker"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0, help="first N goals only (smoke test)")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--gids", default="")
    a = ap.parse_args(argv)
    if a.what == "p":
        prepare()
    elif a.what == "h":
        launch(a.procs, a.gpus, a.limit)
    elif a.what == "d":
        report()
    else:
        worker(a.wid, [int(x) for x in a.gids.split(",") if x])
    return 0


if __name__ == "__main__":
    sys.exit(main())
