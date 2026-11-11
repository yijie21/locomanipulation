"""Step, look again, step again (2026-11-02, after the short-step retrain).

restance.py with RESTANCE_DATA=side (invert/restance_side/) moves the feet once: wm scores the step grid with the plain
ensemble from the starting stance (the goal translated), then the robot plans with the plan-aware ensemble and reaches
from where it stands. The plan-aware std at that point ranks the open-loop miss (Spearman 0.71); above 1 cm it flags 38
of the 143 stepped core goals and 17 of the 21 that still miss by more than 2 cm. Here every goal wm stepped whose std
after the step is above TAU2 looks again from the stance it is in: the plain ensemble re-scores the same step grid from
the robot's actual start features (restance.invert_plain, not the starting stance translated), and the robot takes a
second step if it beats standing by MARGIN_CM (wm2), or the smallest step that puts the goal inside COMFORT as seen from
here (geo2, no model). Then restance's plan-aware inversion and the two executions, open loop (aim) and as feedforward
with the servo on the wanted position (ff_servo). The world goal and wrist rotation are kept from the starting stance.

Every goal first replays the first step and inverts there again: that must give restance_side's command and std exactly
(the sim is deterministic), which checks the replay before any second step is counted.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 $PY scripts/experiments/reachwm_20261030/recheck.py x   # pickle the plain ensemble
    $PY scripts/experiments/reachwm_20261030/recheck.py h --procs 6                                  # look again, step, reach
    $PY scripts/experiments/reachwm_20261030/recheck.py d                                            # tables
Writes outputs/reachwm_20261030/invert/recheck/.
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

os.environ["RESTANCE_DATA"] = "side"  # restance reads it at import: models_side/, restance_side/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import restance as RS  # noqa: E402

E, IE, PI = RS.E, RS.IE, RS.PI
assert RS.SIDE
OUT = IE.OUT / "recheck"
WM_PLAIN = OUT / "wm_plain.pt"
TAU2 = 1.0  # plan-aware std (cm) after the first step above which the robot looks again (the report sweeps upward)
METHODS2 = ("wm2", "geo2")
BIG = 99.0  # a fall counts as a miss of BIG cm


def export() -> None:
    """Phase x (no REACHWM_GOAL_REF): models_side's plain E3 ensemble, pickled for the CPU workers."""
    import data as DT
    import torch
    import train as TR

    assert not DT.GOAL_REF, "run phase x without REACHWM_GOAL_REF"
    D = DT.load(RS.DATA_SPEC)
    model = TR.load_model("E3_ensemble", 1.0, D, "cpu", RS.PLAIN_ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(model, WM_PLAIN)
    print(f"wm_plain.pt from {RS.PLAIN_ROOT.name} ({len(D['n_steps'])} segments) -> {WM_PLAIN}", flush=True)


def first() -> dict:
    """restance_side's wm rows (one step), by gid order of its workers."""
    zs = [dict(np.load(p)) for p in sorted(RS.OUT.glob("h_w*.npz"))]
    return {
        k[3:]: np.concatenate([z[k] for z in zs])
        for k in zs[0]
        if k.startswith("wm_aim_") or k.startswith("wm_ff_servo_")
    }


def two_steps(env, rec, gid: int, d1: np.ndarray, d2: np.ndarray):
    """restance.step (reset, walk, hold, the first step), the PRE_S hold the robot looks again from, the second step by
    d2 (heading frame of the first stance). None once fallen."""
    import gen_reach as G

    r = RS.step(env, rec, gid, d1)
    if r is None:
        return None
    m = env.m
    rec.start(r[0]["side"])
    m.hold(G.PRE_S)
    rec.on = False
    out = m.sidestep(left=float(d2[1]), forward=float(d2[0]))
    if m.fell_at is not None:
        return None
    return r, out


def worker(wid: int, gids: list[int]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF and DT.X0_DIM == 206 + PI.REF_DIM
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    C = np.load(RS.OUT / "choice.npz")
    H1 = first()
    ref = torch.load(RS.WM_REF, map_location="cpu", weights_only=False).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # the 378-row look-again inversion; the rest stays on the CPU
    plain = torch.load(WM_PLAIN, map_location=dev, weights_only=False).eval()
    for p in plain.parameters():
        p.requires_grad_(False)
    PZ = np.load(PI.PLAN / "init.npz")["pz"]
    Gd, k_still = RS.grid(), int(C["k_still"])
    K, P = len(Gd), len(PZ)
    rows = {}
    for gid in gids:
        j = int(np.where(C["gid"] == gid)[0][0])
        i1 = int(np.where(H1["aim_gid"] == gid)[0][0])
        d1 = C["wm_d"][j]
        tic = time.time()
        r = RS.step(env, rec, gid, d1)
        if r is None:
            rows.setdefault("fellstep1", []).append(gid)
            continue
        c, walked0, goal_w, yaw0, p0, _ = r
        st1 = RS.post_step(env, rec, c, goal_w, yaw0, p0)
        b1, _ = RS.invert_here(ref, env.m, c, st1, C["wm_t"][:, j], C["wm_goal_s"][j], PZ[:, j])
        std1 = 100 * np.linalg.norm(b1["std"])
        same1 = bool(np.array_equal(b1["t"], H1["aim_t"][i1]) and abs(std1 - H1["aim_pred_std"][i1]) < 1e-9)
        # look again: the plain ensemble scores the step grid from the state the robot is in now
        sd = st1["sd"]
        d_h = np.zeros((K, 3))
        d_h[:, :2] = Gd
        d_h[:, 1] *= sd
        goal_s = st1["goal"][None] - d_h
        ik, ic = np.repeat(np.arange(K), P), np.tile(np.arange(P), K)
        n = K * P
        aims, losses = RS.invert_plain(plain, np.repeat(st1["x0b"][None], n, 0), np.repeat(st1["rot6"][None], n, 0),
                                       np.full(n, sd), goal_s[ik], PZ[ic, j], dev)  # fmt: skip
        bad = 100 * np.sqrt(losses.reshape(K, P).min(-1))
        A = aims.reshape(K, P, 3)
        ks = dict(
            wm2=int(RS.choose(bad[None], Gd, k_still)[0]), geo2=int(RS.geo_k(st1["goal"][None], np.array([sd]), Gd)[0])
        )
        rows.setdefault("look", []).append(dict(gid=gid, same1=same1, std1=std1, bad=bad, wm2_k=ks["wm2"], geo2_k=ks["geo2"],
                                                look_s=time.time() - tic))  # fmt: skip
        msg = [f"h{wid} goal {gid} d1 {100 * d1[0]:+.0f},{100 * d1[1]:+.0f} std1 {std1:.2f} same1 {same1} | "
               f"score still {bad[k_still]:.2f}"]  # fmt: skip
        for meth in METHODS2:
            k2 = ks[meth]
            if k2 == k_still or (meth == "geo2" and k2 == ks["wm2"]):
                msg.append(f"{meth} {'stays' if k2 == k_still else 'as wm2'}")
                continue
            d2 = d_h[k2]
            tic2 = time.time()
            r2 = two_steps(env, rec, gid, d1, d2)
            if r2 is None:
                rows.setdefault(f"{meth}_fellstep", []).append(gid)
                msg.append(f"{meth} fell stepping")
                continue
            st2 = RS.post_step(env, rec, c, goal_w, yaw0, p0)  # world goal and wrist rotation from the starting stance
            moved2 = (E.rot_z(st1["yaw"]).T @ np.r_[st2["p"][:2] - st1["p"][:2], 0.0])[:2]
            b2, trace = RS.invert_here(ref, env.m, c, st2, A[k2], goal_s[k2], PZ[:, j])
            plan_s = time.time() - tic2
            goal2 = st2["goal"].astype(np.float32)
            for v, servo in (("aim", False), ("ff_servo", True)):
                if two_steps(env, rec, gid, d1, d2) is None:
                    break
                seg = IE.execute(env, rec, st2["c2"], b2["t"].astype(np.float32), b2["pz"], servo, True,
                                 servo_goal_h=goal2 if servo else None)  # fmt: skip
                tracked = PI.plan_record(rec.plan)
                same_plan = all(np.array_equal(tracked[k], b2["rec"][k]) for k in tracked)
                s = PI.summarize(seg, tracked, goal2[None])
                rows.setdefault(f"{meth}_{v}", []).append(dict(
                    gid=gid, walked=walked0, k2=k2, d2=d2[:2], moved2=moved2, moved=st2["moved"], std1=std1,
                    dyaw_step=float(np.degrees(E.wrap(st2["yaw"] - yaw0))), goal_t=goal2, side=sd, t=b2["t"], pz=b2["pz"],
                    pred_err=100 * np.linalg.norm(b2["pred"] - st2["goal"]), pred_std=100 * np.linalg.norm(b2["std"]),
                    bad_still=bad[k_still], bad_k2=bad[k2], same_plan=same_plan,
                    x0_diff=float(np.abs(s["x0"][0] - b2["x0"]).max()), plan_s=plan_s, err=s["err"][0], fell=s["fell"][0],
                    settled=s["settled"][0]))  # fmt: skip
            if f"{meth}_ff_servo" not in rows or rows[f"{meth}_ff_servo"][-1]["gid"] != gid:  # a replay fell
                rows[f"{meth}_aim"] = [x for x in rows.get(f"{meth}_aim", []) if x["gid"] != gid]
                rows.setdefault(f"{meth}_fellstep", []).append(gid)
                msg.append(f"{meth} fell replaying")
                continue
            rows.setdefault(f"{meth}_trace", []).append(dict(gid=gid, trace=np.array(trace)))
            a, f = rows[f"{meth}_aim"][-1], rows[f"{meth}_ff_servo"][-1]
            msg.append(f"{meth} d2 {100 * d2[0]:+.0f},{100 * d2[1]:+.0f} moved2 {100 * moved2[0]:+.1f},{100 * moved2[1]:+.1f} "
                       f"std2 {a['pred_std']:.2f} aim {H1['aim_err'][i1]:.2f} -> {a['err']:.2f} ff_servo {H1['ff_servo_err'][i1]:.2f} "
                       f"-> {f['err']:.2f} same_plan {a['same_plan'] and f['same_plan']}")  # fmt: skip
        print(" | ".join(msg) + f" | {time.time() - tic:.0f} s", flush=True)
    flat = {}
    for v, rs in rows.items():
        if v.endswith("fellstep") or v.endswith("fellstep1"):
            flat[v] = np.array(rs)
        elif v.endswith("_trace"):
            flat[f"{v}_gid"] = np.array([r["gid"] for r in rs])
            flat[v] = np.array([r["trace"] for r in rs])
        else:
            flat.update({f"{v}_{k}": np.array([r[k] for r in rs]) for k in rs[0]})
    np.savez(OUT / f"h_w{wid}.npz", **flat)


def gated(tau: float = TAU2) -> list[int]:
    H1 = first()
    return sorted(int(g) for g, s in zip(H1["aim_gid"], H1["aim_pred_std"]) if s > tau)


def launch(procs: int, gpus: str, limit: int) -> None:
    gids = gated()[: limit or None]
    print(f"{len(gids)} goals look again (std after the first step > {TAU2} cm)", flush=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[w % len(cards)], OMP_NUM_THREADS="1", REACHWM_GOAL_REF="1")
        log = open(OUT / f"h_w{w}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "worker", "--wid", str(w), "--gids", ",".join(map(str, gids[w::procs]))]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("h exit codes", [p.wait() for p in ps], flush=True)


def report() -> None:
    """Per goal: restance_side's one-step result, or the second step's when the robot looks again and steps."""
    from scipy.stats import spearmanr

    P1 = dict(np.load(RS.OUT / "policy.npz"))  # restance_side per goal at gate 0.83
    zs = [dict(np.load(p)) for p in sorted(OUT.glob("h_w*.npz"))]
    R = {}
    for z in zs:
        for k, v in z.items():
            R[k] = np.concatenate([R[k], v]) if k in R else v
    gid = P1["gid"].astype(int)
    m = P1["core"]
    one = {"aim": P1["wm_aim"], "ff_servo": P1["wm_ff"]}
    std1 = P1["wm_std"]
    st1 = P1["wm_stepped"]
    look = {int(g): i for i, g in enumerate(R["look_gid"])}
    missing = set(gated()) - set(look) - {int(g) for g in R.get("fellstep1", [])}  # only in a partial run
    run = {meth: {int(g): i for i, g in enumerate(R.get(f"{meth}_aim_gid", []))} for meth in METHODS2}
    fell = {meth: {int(g) for g in R.get(f"{meth}_fellstep", [])} for meth in METHODS2}
    stat = lambda e: dict(median=float(np.median(e)), p90=float(np.percentile(e, 90)), lt1=float(np.mean(e < 1)),  # noqa: E731
                          gt2=float(np.mean(e > 2)))  # fmt: skip

    def policy(meth: str, tau: float):
        """err per goal, the stepped-again mask, std after the second step (nan where it did not step again)."""
        out = {v: one[v].copy() for v in one}
        again, s2 = np.zeros(len(gid), bool), np.full(len(gid), np.nan)
        for i, g in enumerate(gid):
            if not st1[i] or not std1[i] > tau or g not in look:
                continue
            L = look[g]
            k2 = int(R[f"look_{meth}_k"][L])
            if meth == "geo2" and k2 == int(R["look_wm2_k"][L]):
                src = "wm2"
            else:
                src = meth
            if g in fell[src]:
                again[i] = True
                out["aim"][i] = out["ff_servo"][i] = BIG
                continue
            if g not in run[src]:  # stays
                continue
            r = run[src][g]
            again[i] = True
            out["aim"][i], out["ff_servo"][i] = R[f"{src}_aim_err"][r], R[f"{src}_ff_servo_err"][r]
            s2[i] = R[f"{src}_aim_pred_std"][r]
            if R[f"{src}_aim_fell"][r] or R[f"{src}_ff_servo_fell"][r]:
                out["aim"][i] = out["ff_servo"][i] = BIG
        return out, again, s2

    n_runs = sum(len(v) for v in run.values())
    sp = [
        bool(R[f"{meth}_aim_same_plan"][i] and R[f"{meth}_ff_servo_same_plan"][i])
        for meth in METHODS2
        for i in run[meth].values()
    ]
    xd = [
        float(max(R[f"{meth}_aim_x0_diff"][i], R[f"{meth}_ff_servo_x0_diff"][i]))
        for meth in METHODS2
        for i in run[meth].values()
    ]
    lines = [f"{len(look)} goals looked again (wm stepped, std after the step > {TAU2} cm; {int(np.sum(m & st1 & (std1 > TAU2)))} core); "
             f"first step replayed exactly on {int(R['look_same1'].sum())}/{len(look)}; second-step runs {n_runs}, falls stepping "
             f"{sum(len(v) for v in fell.values())}, falls reaching "
             f"{sum(int(R[f'{meth}_aim_fell'][i] or R[f'{meth}_ff_servo_fell'][i]) for meth in METHODS2 for i in run[meth].values())}; "
             f"plan matches the executed one {np.mean(sp) if sp else float('nan'):.3f}, max x0 diff {max(xd) if xd else float('nan'):.1e}; "
             f"goals missing from the results {len(missing)}", ""]  # fmt: skip
    k_still = int(np.load(RS.OUT / "choice.npz")["k_still"])
    for meth in METHODS2:
        k = R[f"look_{meth}_k"]
        lines.append(f"{meth}: steps again on {int(np.sum(k != k_still))}/{len(k)} of the goals that looked again")
    bad = R["look_bad"]
    kw = R["look_wm2_k"]
    mv = kw != k_still
    if mv.any():
        lines.append(f"wm2 predicted score where it steps again: still {np.median(bad[mv, k_still]):.2f} -> "
                     f"{np.median(bad[np.where(mv)[0], kw[mv]]):.2f} cm; where it stays {np.median(bad[~mv, k_still]):.2f} cm")  # fmt: skip
    lines += ["", "| policy | look again if std > cm | stepped again | open loop median | <1 cm | >2 cm | p90 | ff+servo median | "
              "<1 cm | >2 cm | p90 |", "|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    res = dict(rows=[])
    for nm, e in (
        ("no step", {"aim": P1["base_aim"], "ff_servo": P1["base_ff"]}),
        ("one step (restance_side wm)", one),
    ):
        a, f = stat(e["aim"][m]), stat(e["ff_servo"][m])
        lines.append(f"| {nm} | - | 0.00 | {a['median']:.2f} | {a['lt1']:.2f} | {a['gt2']:.2f} | {a['p90']:.2f} | "
                     f"{f['median']:.2f} | {f['lt1']:.2f} | {f['gt2']:.2f} | {f['p90']:.2f} |")  # fmt: skip
        res["rows"].append(dict(policy=nm, aim=a, ff_servo=f))
    for meth in METHODS2:
        for tau in (TAU2, 1.2, 1.5, 2.0):
            out, ag, _ = policy(meth, tau)
            a, f = stat(out["aim"][m]), stat(out["ff_servo"][m])
            res["rows"].append(dict(policy=meth, tau=tau, again=float(ag[m].mean()), aim=a, ff_servo=f))
            lines.append(f"| {meth} | {tau:.2f} | {ag[m].mean():.2f} | {a['median']:.2f} | {a['lt1']:.2f} | {a['gt2']:.2f} | "
                         f"{a['p90']:.2f} | {f['median']:.2f} | {f['lt1']:.2f} | {f['gt2']:.2f} | {f['p90']:.2f} |")  # fmt: skip
    # the goals that step again, paired against stopping after the first step
    lines += ["", "| method | core goals that step again | open loop one / two steps median | better on | worse by > 1 cm | "
              "ff+servo one / two median | better on | std after step 1 / step 2 median | std went down on |",
              "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for meth in METHODS2:
        out, ag, s2 = policy(meth, TAU2)
        b = m & ag
        if not b.any():
            lines.append(f"| {meth} | 0 | | | | | | | |")
            continue
        ea, fa, oa, of = out["aim"][b], out["ff_servo"][b], one["aim"][b], one["ff_servo"][b]
        lines.append(f"| {meth} | {int(b.sum())} | {np.median(oa):.2f} / {np.median(ea):.2f} | {np.mean(ea < oa):.2f} | "
                     f"{int(np.sum(ea > oa + 1))} | {np.median(of):.2f} / {np.median(fa):.2f} | {np.mean(fa < of):.2f} | "
                     f"{np.median(std1[b]):.2f} / {np.nanmedian(s2[b]):.2f} | {np.mean(s2[b] < std1[b]):.2f} |")  # fmt: skip
    # the misses the first step left: stepped core goals, open loop > 2 cm
    tail = m & st1 & (one["aim"] > 2)
    lines += ["", f"open-loop misses > 2 cm after the first step (core): {int(tail.sum())}, flagged by std > {TAU2}: "
              f"{int(np.sum(tail & (std1 > TAU2)))}"]  # fmt: skip
    for meth in METHODS2:
        out, ag, _ = policy(meth, TAU2)
        lines.append(f"  {meth}: steps again on {int((tail & ag).sum())}; open loop now < 2 cm on {int((tail & (out['aim'] < 2)).sum())}, "
                     f"< 1 cm on {int((tail & (out['aim'] < 1)).sum())}; median {np.median(one['aim'][tail]):.2f} -> "
                     f"{np.median(out['aim'][tail]):.2f}; ff+servo > 2 cm {int(np.sum(tail & (one['ff_servo'] > 2)))} -> "
                     f"{int(np.sum(tail & (out['ff_servo'] > 2)))}; new open-loop misses > 2 cm (fine after one step) "
                     f"{int(np.sum(m & ag & (one['aim'] <= 2) & (out['aim'] > 2)))}")  # fmt: skip
    # best of one / two steps per goal (hindsight on the error): what a better second choice could still give
    o_w, _, s_w = policy("wm2", TAU2)
    o_g, _, s_g = policy("geo2", TAU2)
    for v in ("aim", "ff_servo"):
        best = np.minimum(np.minimum(one[v], o_w[v]), o_g[v])
        b = stat(best[m])
        lines.append(
            f"{v}: best of one step / wm2 / geo2 per goal median {b['median']:.2f}, <1 cm {b['lt1']:.2f}, >2 cm {b['gt2']:.2f}"
        )
    # does the std after the second step still rank the error
    for meth in METHODS2:
        ok = [i for i in run[meth].values() if not (R[f"{meth}_aim_fell"][i] or R[f"{meth}_ff_servo_fell"][i])]
        if len(ok) > 3:
            rho = spearmanr(R[f"{meth}_aim_pred_std"][ok], R[f"{meth}_aim_err"][ok]).correlation
            gap = 100 * np.linalg.norm(R[f"{meth}_aim_moved2"][ok] - R[f"{meth}_aim_d2"][ok], axis=1)
            lines.append(f"{meth} after the second step: std/error Spearman {rho:.2f} (n {len(ok)}); the step landed off the asked "
                         f"displacement by median {np.median(gap):.1f} cm (p90 {np.percentile(gap, 90):.1f})")  # fmt: skip
    pol = dict(gid=gid, core=m, one_aim=one["aim"], one_ff=one["ff_servo"], std1=std1)
    for meth in METHODS2:
        out, ag, s2 = policy(meth, TAU2)
        pol.update({f"{meth}_aim": out["aim"], f"{meth}_ff": out["ff_servo"], f"{meth}_again": ag, f"{meth}_std": s2})
    np.savez(OUT / "policy.npz", **pol)
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


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
