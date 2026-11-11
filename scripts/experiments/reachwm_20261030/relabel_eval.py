"""The relabelled controller on the 345 core goals, eyes closed (2026-11-04, hindsight relabelling step 1).

Same goals, start states and command as report_media.py's naive baseline (phase n): the wanted jaw centre and wrist
rotation sent as reach()'s command, servo off. The only change: from the command to the end of the 1 s hold after the
reach, ScaleBFM runs with the student's weights (relabel_train.py), copied into the policy's own parameters in place, so
the captured CUDA graph runs them too; the walk and steps before it run the frozen controller, so every start state is
the stored one. Error = jaw centre to the wanted position after the hold (cm), wrist = angle to the wanted rotation.

Checks per worker: its first CHECK goals also run the frozen controller and must reproduce the stored naive rows; the
first goal runs the student a second time with the CUDA graph off (eager) and must land on the same point.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/reachwm_20261030/relabel_eval.py run --tag r1 --procs 6
    $PY scripts/experiments/reachwm_20261030/relabel_eval.py report --tag r1
Writes outputs/reachwm_20261030/relabel/eval_<tag>/ (e_w*.json, report.md).
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import relabel_gen as RG  # noqa: E402
import wrist_invert as WI  # noqa: E402

PV = WI.PV
RS, RC, E, IE, PI, SG = PV.RS, PV.RC, PV.E, PV.IE, PV.PI, PV.SG
OUT = E.REPO / "outputs" / "reachwm_20261030" / "relabel"
NAIVE = E.REPO / "outputs" / "reachwm_20261030" / "media" / "naive"
INV = IE.OUT / "wrist_invert2" / "h_w*_pos_pose3g_w20_cap15.json"  # the report's self-model rows (same goals)
CHECK = 2


def worker(tag: str, wid: int, procs: int) -> None:
    import gen_reach as G
    import torch

    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    gate = SG.Gate(env.m)
    pol = env.m.pol
    frozen = {n: p.detach().clone() for n, p in pol.w.named_parameters()}
    student = torch.load(OUT / f"student_{tag}.pt", map_location="cpu", weights_only=False)["params"]
    assert set(student) == set(frozen)
    naive = {}
    for f in glob.glob(str(NAIVE / "n_w*.json")):
        naive.update({r["gid"]: r for r in json.load(open(f)) if r["servo"] == "open"})
    rows = SG.plans()["rows"][wid::procs]
    res, out = [], OUT / f"eval_{tag}"
    for n, p in enumerate(rows):
        gid = p["gid"]
        tic = time.time()
        if p["kind"] != "none":

            def replay():
                r = (
                    RS.step(env, rec, gid, p["d1"])
                    if p["kind"] == "one"
                    else RC.two_steps(env, rec, gid, p["d1"], p["d2"])
                )
                return None if r is None else (r if p["kind"] == "one" else r[0])
        else:

            def replay():
                c, walked = IE.setup(env, gid)
                return None if env.m.fell_at is not None else (c, walked)

        st = PV.state(env, rec, p, replay)
        if st is None:
            continue
        goal = st["goal"].astype(np.float32)
        runs = (
            [("student", True)]
            + ([("frozen", True)] if n < CHECK else [])
            + ([("student_eager", False)] if n == 0 else [])
        )
        for name, graph in runs:
            if replay() is None:
                break
            gate.mode, gate.radius = "orig", None
            gate.reset()
            cfg = env.m.cfg
            env.m.cfg = dataclasses.replace(cfg, servo_ki_rot=0.0)
            reach = env.m.reach

            def reach_with(*a, _w=frozen if name == "frozen" else student, **kw):
                RG.set_params(pol, _w)
                return reach(*a, **kw)

            env.m.reach = reach_with
            pol.graph = graph and pol.cfg.cuda_graph
            try:
                seg = IE.execute(env, rec, st["c2"], goal, p["pz"], False, st.get("walked", True))
            finally:
                env.m.reach = reach
                RG.set_params(pol, frozen)
                pol.graph = pol.cfg.cuda_graph
                env.m.cfg = cfg
            s = PI.summarize(seg, PI.plan_record(rec.plan), goal[None])
            att = float(np.degrees(PV.angle(st["rot_w"], env.m.jaw_centre(st["c2"]["side"])[1])))
            res.append(
                dict(gid=gid, kind=p["kind"], run=name, err=float(s["err"][0]), att_deg=att, fell=bool(s["fell"][0]))
            )
        mine = {r["run"]: r for r in res if r["gid"] == gid}
        if "frozen" in mine:
            ref = naive[gid]
            assert mine["frozen"]["err"] == ref["err"] and mine["frozen"]["att_deg"] == ref["att_deg"], (
                gid,
                mine["frozen"],
                ref,
            )
        if "student_eager" in mine:
            assert abs(mine["student_eager"]["err"] - mine["student"]["err"]) < 0.05, (gid, mine)
        print(f"e{wid} goal {gid} " + " ".join(f"{k} {r['err']:.2f} cm {r['att_deg']:.1f} deg" for k, r in mine.items()) +
              f" | naive {naive[gid]['err']:.2f} cm {naive[gid]['att_deg']:.1f} deg | {time.time() - tic:.0f} s", flush=True)  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"e_w{wid}.json", "w"))


def launch(tag: str, procs: int, gpus: str) -> None:
    out = OUT / f"eval_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        envv = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(out / f"e_w{w}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "worker", "--tag", tag, "--wid", str(w), "--procs", str(procs)]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=envv, cwd=str(E.REPO)))
    print("exit codes", [p.wait() for p in ps], flush=True)


def stats(rows: list[dict]) -> str:
    ok = [r for r in rows if not r["fell"]]
    e, a = np.array([r["err"] for r in ok]), np.array([r["att_deg"] for r in ok])
    return (f"{len(rows)} | {np.median(e):.2f} | {int((e > 2).sum())} | {np.percentile(e, 90):.2f} | {np.median(a):.1f} | "
            f"{(a < 10).mean():.2f} | {int((a > 30).sum())} | {len(rows) - len(ok)}")  # fmt: skip


def report(tag: str) -> None:
    res = []
    for f in sorted(glob.glob(str(OUT / f"eval_{tag}" / "e_w*.json"))):
        res += json.load(open(f))
    naive = []
    for f in glob.glob(str(NAIVE / "n_w*.json")):
        naive += [r for r in json.load(open(f)) if r["servo"] == "open"]
    inv = []
    for f in glob.glob(str(INV)):
        inv += [r for r in json.load(open(f)) if r["servo"] == "open"]
    stud = [r for r in res if r["run"] == "student"]
    gids = {r["gid"] for r in stud}
    L = [f"# Relabelled controller ({tag}) on the core goals, eyes closed", "",
         f"{len(gids)} goals evaluated; every row restricted to them, paired by goal.", "",
         "| controller, command | n | position median cm | >2 cm (n) | p90 cm | wrist median deg | <10 deg | >30 deg (n) | fell |",
         "|---|---|---|---|---|---|---|---|---|",
         f"| frozen, wanted pose as the command | {stats([r for r in naive if r['gid'] in gids])} |",
         f"| frozen, self model inverts: position only | {stats([r for r in inv if r['gid'] in gids and r['method'] == 'pos'])} |",
         f"| frozen, self model inverts: wrist first | {stats([r for r in inv if r['gid'] in gids and r['method'] == 'pose3g'])} |",
         f"| frozen, self model inverts: 1 cm = 20 deg | {stats([r for r in inv if r['gid'] in gids and r['method'] == 'w20'])} |",
         f"| relabelled ({tag}), wanted pose as the command | {stats(stud)} |"]  # fmt: skip
    # RELABEL_HYBRID (2026-11-04, controller plus self model): comma-separated label=dir of wrist_invert.py runs with
    # WRIST_STUDENT; each dir's open-loop rows add one line per inversion method
    inv_pos = {r["gid"]: r for r in inv if r["method"] == "pos"}
    paired = []
    for label, d in (h.split("=", 1) for h in os.environ.get("RELABEL_HYBRID", "").split(",") if h):
        hy = [r for f in sorted(glob.glob(f"{d}/h_w*.json")) for r in json.load(open(f)) if r["servo"] == "open"]
        for meth in dict.fromkeys(r["method"] for r in hy):
            rows = [r for r in hy if r["gid"] in gids and r["method"] == meth]
            L.append(f"| relabelled ({tag}) + {label}: {meth} | {stats(rows)} |")
            if meth == "pos":
                dp = np.array([r["err"] - inv_pos[r["gid"]]["err"] for r in rows if not r["fell"]])
                da = np.array([r["att_deg"] - inv_pos[r["gid"]]["att_deg"] for r in rows if not r["fell"]])
                paired.append(f"{label}, position only, vs frozen + self model position only, per goal: position better on "
                              f"{(dp < 0).mean():.2f} (median {np.median(dp):+.2f} cm), wrist better on {(da < 0).mean():.2f} "
                              f"(median {np.median(da):+.1f} deg).")  # fmt: skip
    L.append("")
    nv = {r["gid"]: r for r in naive}
    d = np.array([r["err"] - nv[r["gid"]]["err"] for r in stud if not r["fell"]])
    da = np.array([r["att_deg"] - nv[r["gid"]]["att_deg"] for r in stud if not r["fell"]])
    L.append(f"Relabelled vs frozen, same command, per goal: position better on {(d < 0).mean():.2f} (median change "
             f"{np.median(d):+.2f} cm), wrist better on {(da < 0).mean():.2f} (median {np.median(da):+.1f} deg).")  # fmt: skip
    L += paired
    (OUT / f"eval_{tag}" / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "report"])
    ap.add_argument("--tag", default="r1")
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    a = ap.parse_args(argv)
    if a.what == "run":
        launch(a.tag, a.procs, a.gpus)
    elif a.what == "worker":
        worker(a.tag, a.wid, a.procs)
    else:
        report(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
