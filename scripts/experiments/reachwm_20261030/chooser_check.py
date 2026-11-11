"""The per-goal plan chooser on fresh goals (2026-11-05, after the hybrid step of the hindsight relabelling).

On the 345 core goals the hybrid (relabelled controller r5dag executing, self model retrained on it inverting position
only) lost on position to the frozen controller with its own self model, but picking per goal among three plans by the
lowest predicted position std + predicted wrist miss / 20 deg gave 0.82 cm / 7.9 deg. The 20 was chosen on those goals.
Here the same rule, unchanged, on goals nothing so far has seen: invert_eval.setup's reset (and walk) for gids 400+
(the 345 are gids 0-399), its command sampler and edge rule, kept when the goal is in the core box. No step before the
reach (some of the 345 started after one or two).

Per goal, from the planning state (after the PRE_S hold), IK only, nothing executed:
    pelvis   plan_invert's choice: the wanted height and invert_eval.PELVIS, each by the position fixed point from the
             wanted target with the frozen controller's model (wrist_invert2), lowest predicted miss^2 + std^2
    pos      the frozen controller's model, wrist_invert.refine pos from that iterate
    pose3g   the same refine, pose3g
    hybrid   the student's model (hybrid_r5dag), position fixed point from the same iterate, executed by the student
Executed open loop, all at the chosen pelvis height: naive (frozen controller, wanted command), student_naive (student,
wanted command), pos, pose3g, hybrid. The chooser picks offline from the three plans' predictions, each by its own model.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/chooser_check.py run --procs 6
    $PY scripts/experiments/reachwm_20261030/chooser_check.py report
    CHOOSER_OUT=/tmp/x $PY scripts/experiments/reachwm_20261030/chooser_check.py run --procs 1 --limit 2   # smoke test
Writes outputs/reachwm_20261030/invert/chooser_check/ (c_w*.json, report.md).
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402
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
E, IE, PI, SG = PV.E, PV.IE, PV.PI, PV.SG
OUT = Path(os.environ.get("CHOOSER_OUT", IE.OUT / "chooser_check"))
R = E.REPO / "outputs" / "reachwm_20261030"
WMS = {"frozen": IE.OUT / "wrist_invert2" / "wm_ref.pt", "student": IE.OUT / "hybrid_r5dag" / "wm_ref.pt"}
STUDENT = R / "relabel" / "student_r5dag.pt"
GIDS = range(400, 800)
RUNS = ("naive", "student_naive", "pos", "pose3g", "hybrid")
PLANS = ("hybrid", "pose3g", "pos")  # the chooser's candidates
DEG_PER_CM = 20.0  # the rule as chosen on the 345


def command(gid: int) -> dict:
    """invert_eval.setup's command for gid without the sim: its draws (walk, then the command) from the same seed."""
    import gen_reach as G

    rng = np.random.default_rng(IE.SEED + gid)
    rng.uniform(-math.pi, math.pi), rng.uniform(0.3, 1.2), rng.uniform(-math.pi, math.pi)
    c = G.sample_command(rng)
    while IE.is_edge(c):
        c = G.sample_command(rng)
    return c


def core(c: dict) -> bool:
    t, sd = c["target"], 1.0 if c["side"] == "right" else -1.0
    tt = np.r_[t[0], t[1] * sd, t[2]]
    return bool(np.all((tt >= PI.CORE[0]) & (tt <= PI.CORE[1])))


def worker(wid: int, gids: list[int]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF and DT.X0_DIM == 206 + PI.REF_DIM
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    gate = SG.Gate(env.m)
    models = {k: torch.load(p, map_location="cpu", weights_only=False).eval() for k, p in WMS.items()}
    G.STUDENT, reach = str(STUDENT), env.m.reach

    def student_reach(*a, **kw):
        G.use_weights(env.m, "student")
        return reach(*a, **kw)

    res = []
    for gid in gids:
        tic = time.time()
        c0 = command(gid)

        def replay():
            c, walked = IE.setup(env, gid)
            return None if env.m.fell_at is not None else (c, walked)

        st = PV.state(env, rec, dict(kind="none", goal_t=c0["target"].astype(np.float32)), replay)
        if st is None:
            continue
        assert np.array_equal(st["c2"]["target"], c0["target"]), gid
        walked = st["walked"]
        own = -1.0 if c0["pelvis_z"] is None else float(c0["pelvis_z"])
        best = None
        for pz in dict.fromkeys([own, *IE.PELVIS]):
            for ev in WI.fixed_point(models["frozen"], env.m, st, st["goal"], pz, None):
                if best is None or ev["l_pos"] < best[0]["l_pos"]:
                    best = (ev, pz)
        t0, pz = best[0]["t"], best[1]
        fz = WI.refine(models["frozen"], env.m, st, t0, pz)
        hy = min(WI.fixed_point(models["student"], env.m, st, t0, pz, None), key=lambda ev: ev["l_pos"])
        plans = {"naive": None, "student_naive": None, "pos": fz["pos"][0], "pose3g": fz["pose3g"][0], "hybrid": hy}
        goal = st["goal"].astype(np.float32)
        msg = [f"c{wid} goal {gid} pz {pz:.2f}"]
        for run in RUNS:
            if replay() is None:
                break
            b, student = plans[run], run in ("student_naive", "hybrid")
            gate.mode, gate.radius = "orig", None
            gate.reset()
            cfg = env.m.cfg
            env.m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)
            cmd = None if b is None else b["cmd"]
            c_cmd, rot_h = st["c2"], None
            if isinstance(cmd, tuple):
                c_cmd = dict(st["c2"], dyaw=cmd[0], pitch=cmd[1])
            elif cmd is not None:
                rot_h = cmd
            if student:
                env.m.reach = student_reach
            try:
                seg = IE.execute(
                    env, rec, c_cmd, goal if b is None else b["t"].astype(np.float32), pz, False, walked, rot_h=rot_h
                )
            finally:
                if student:
                    env.m.reach = reach
                    G.use_weights(env.m, "frozen")
                env.m.cfg = cfg
            tracked = PI.plan_record(rec.plan)
            s = PI.summarize(seg, tracked, goal[None])
            R_w = E.rot_z(float(seg["start"][3]) + st["c2"]["dyaw"]) @ G.rot_y(st["c2"]["pitch"])
            att = float(np.degrees(WI.angle(R_w, env.m.jaw_centre(st["c2"]["side"])[1])))
            row = dict(gid=gid, run=run, walked=walked, side=st["c2"]["side"], pz=pz, err=float(s["err"][0]), att_deg=att,
                       fell=bool(s["fell"][0]))  # fmt: skip
            if b is not None:
                row.update(same_plan=all(np.array_equal(tracked[k], b["rec"][k]) for k in tracked), k=int(b["k"]),
                           stage=b.get("stage", "pos"), **{f: float(b[f]) for f in ("pred_err", "pred_std", "pred_att", "att_spread", "dev")})  # fmt: skip
            res.append(row)
            msg.append(
                f"{run} {row['err']:.2f} cm {att:.1f} deg"
                + ("" if b is None else f" (pred std {b['pred_std']:.2f} att {b['pred_att']:.1f})")
            )
        print(" | ".join(msg) + f" | {time.time() - tic:.0f} s", flush=True)
    json.dump(res, open(OUT / f"c_w{wid}.json", "w"))


def launch(procs: int, gpus: str, limit: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gids = [g for g in GIDS if core(command(g))][: limit or None]
    print(f"{len(gids)} fresh core goals (gids {GIDS.start}-{GIDS.stop - 1}) x {RUNS}", flush=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(OUT / f"c_w{w}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "worker", "--wid", str(w), "--gids", ",".join(map(str, gids[w::procs]))]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("c exit codes", [p.wait() for p in ps], flush=True)


def stats(rows: list[dict]) -> str:
    ok = [r for r in rows if not r["fell"]]
    e, a = np.array([r["err"] for r in ok]), np.array([r["att_deg"] for r in ok])
    return (f"{len(rows)} | {np.median(e):.2f} | {int((e > 2).sum())} | {np.percentile(e, 90):.2f} | {np.median(a):.1f} | "
            f"{int((a > 30).sum())} | {len(rows) - len(ok)}")  # fmt: skip


def choose(by: dict[str, dict], score) -> dict:
    """by: plan -> its executed row; the row of the plan with the lowest score."""
    return by[min(PLANS, key=lambda k: score(by[k]))]


def rules() -> list[tuple[str, object]]:
    return [(f"std + att / {d:g} deg" + (" (the rule, chosen on the 345)" if d == DEG_PER_CM else ""),
             lambda r, d=d: r["pred_std"] + r["pred_att"] / d) for d in (10.0, 20.0, 40.0)] + [
            ("std only", lambda r: r["pred_std"]), ("predicted wrist miss only", lambda r: r["pred_att"]),
            ("predicted miss + att / 20 deg (no std)", lambda r: r["pred_err"] + r["pred_att"] / 20)]  # fmt: skip


def old_rows() -> dict[str, dict[int, dict]]:
    """The 345 core goals' open-loop rows of the same five executions (wrist_invert2, hybrid_r5dag, media/naive,
    relabel/eval_r5dag)."""
    J = lambda pat: [r for f in sorted(glob.glob(str(pat))) for r in json.load(open(f))]  # noqa: E731
    inv = [r for r in J(IE.OUT / "wrist_invert2" / "h_w*_pos_pose3g_w20_cap15.json") if r["servo"] == "open"]
    out = {m: {r["gid"]: r for r in inv if r["method"] == m} for m in ("pos", "pose3g")}
    out["hybrid"] = {r["gid"]: r for r in J(IE.OUT / "hybrid_r5dag" / "h_w*_pos.json") if r["servo"] == "open"}
    out["naive"] = {r["gid"]: r for r in J(R / "media" / "naive" / "n_w*.json") if r["servo"] == "open"}
    out["student_naive"] = {r["gid"]: r for r in J(R / "relabel" / "eval_r5dag" / "e_w*.json") if r["run"] == "student"}
    return out


def table(by: dict[str, dict[int, dict]], title: str) -> list[str]:
    gids = sorted(set.intersection(*(set(v) for v in by.values())))
    names = {"naive": "frozen, wanted command", "student_naive": "relabelled, wanted command",
             "pos": "frozen + its self model, position only", "pose3g": "frozen + its self model, wrist first",
             "hybrid": "relabelled + self model retrained on it, position only"}  # fmt: skip
    L = [f"## {title}: {len(gids)} goals", "",
         "| controller, command | n | position median cm | >2 cm (n) | p90 cm | wrist median deg | >30 deg (n) | fell | picks hybrid / pose3g / pos |",
         "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for run in RUNS:
        L.append(f"| {names[run]} | {stats([by[run][g] for g in gids])} | |")
    picks = {}
    for name, sc in rules():
        rows = []
        for g in gids:
            r = choose({k: by[k][g] for k in PLANS}, sc)
            rows.append(r)
            picks.setdefault(name, []).append(next(k for k in PLANS if by[k][g] is r))
        n = {k: picks[name].count(k) for k in PLANS}
        L.append(f"| chooser: {name} | {stats(rows)} | {n['hybrid']} / {n['pose3g']} / {n['pos']} |")
    orc = [
        choose({k: by[k][g] for k in PLANS}, lambda r: np.inf if r["fell"] else r["err"] + r["att_deg"] / DEG_PER_CM)
        for g in gids
    ]
    L.append(f"| best of the three after the fact (actual miss + att / 20) | {stats(orc)} | |")
    L.append("")
    sc = rules()[1][1]
    ch = {g: choose({k: by[k][g] for k in PLANS}, sc) for g in gids}
    for ref in ("pos", "pose3g"):
        ok = [g for g in gids if not ch[g]["fell"] and not by[ref][g]["fell"]]
        dp = np.array([ch[g]["err"] - by[ref][g]["err"] for g in ok])
        da = np.array([ch[g]["att_deg"] - by[ref][g]["att_deg"] for g in ok])
        L.append(f"The rule vs {names[ref]}, per goal: position better on {(dp < 0).mean():.2f} (median {np.median(dp):+.2f} cm), "
                 f"wrist better on {(da < 0).mean():.2f} (median {np.median(da):+.1f} deg).")  # fmt: skip
    for k in PLANS:
        ok = [g for g in gids if not by[k][g]["fell"]]
        s, e = np.array([by[k][g]["pred_std"] for g in ok]), np.array([by[k][g]["err"] for g in ok])
        hi = s >= np.median(s)
        L.append(f"{k}: predicted std median {np.median(s):.2f} cm; executed miss > 2 cm on {(e[~hi] > 2).mean():.2f} of the lower "
                 f"half by std, {(e[hi] > 2).mean():.2f} of the upper half.")  # fmt: skip
    return L + [""]


def report() -> None:
    res = [r for f in sorted(glob.glob(str(OUT / "c_w*.json"))) for r in json.load(open(f))]
    new = {run: {r["gid"]: r for r in res if r["run"] == run} for run in RUNS}
    same = [r["same_plan"] for r in res if "same_plan" in r]
    L = [
        "# Plan chooser on fresh goals, eyes closed",
        "",
        f"Executed plans equal to the planned IK: {int(np.sum(same))} / {len(same)}.",
        "",
    ]
    L += table(new, "fresh goals (gids 400-799, core box, no step)")
    L += table(old_rows(), "the 345 goals the rule was chosen on")
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "report"])
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--gids", default="")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if a.what == "run":
        launch(a.procs, a.gpus, a.limit)
    elif a.what == "worker":
        worker(a.wid, [int(x) for x in a.gids.split(",") if x])
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
