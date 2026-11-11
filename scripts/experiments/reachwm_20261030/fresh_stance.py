"""Stance and pelvis height for the wrist, on the 345 core goals and on fresh goals (2026-11-06).

Two leftovers of the self-model line, both eyes closed, ScaleBFM unchanged, model models_goalref_turn (the stance report's):

    pelvis   the pelvis height has been chosen by the position fixed point alone and handed to the wrist inversion and the
             stance search as a constant, though most of the goals still > 30 deg off want the jaw steeply down at
             z ~1 m. Here, on the goals whose staying prediction is poor (stance_wrist's gate), the same joint fixed
             point scores every invert_eval.PELVIS height at the planning state; a height that beats the position
             height by stance_wrist.MARGIN_CM replaces it for pose3g (run pose3g_pz).
    stance   the rule of the stance report on goals it was not tuned on: stance_wrist's search from the planning state,
             the move, wrist_invert.refine from the real body; look again after the move offline from the rows
             (stance_report.look_again).

Sets: core = the 345 goals in the state servo_gate.plans() gives them; fresh = chooser_check's goals, gids 400-799 in the
core box, no step before the reach, the position pelvis rule as chooser_check's with this model.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    FRESH_SET=core FRESH_RUNS=pelvis FRESH_OUT=pelvis_core $PY scripts/experiments/reachwm_20261030/fresh_stance.py h --procs 6
    FRESH_SET=fresh FRESH_RUNS=pelvis,stance FRESH_OUT=stance_fresh $PY .../fresh_stance.py h --procs 6
    FRESH_OUT=stance_fresh $PY .../fresh_stance.py d
Writes outputs/reachwm_20261030/invert/<FRESH_OUT>/.
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("WRIST_SERVOS", "open")

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

PV, RS, E, IE, PI, SG = WI.PV, WI.RS, WI.E, WI.IE, WI.PI, WI.SG
os.environ.setdefault("STANCE_WM", str(IE.OUT / "wrist_turn" / "wm_ref.pt"))
import chooser_check as CC  # noqa: E402
import stance_wrist as SW  # noqa: E402

SET = os.environ.get("FRESH_SET", "fresh")
RUNS = tuple(os.environ.get("FRESH_RUNS", "pelvis,stance").split(","))
OUT = IE.OUT / os.environ.get("FRESH_OUT", "stance_fresh")


def goals() -> list[int]:
    if SET == "core":
        return [r["gid"] for r in SG.plans()["rows"]]
    return [g for g in CC.GIDS if CC.core(CC.command(g))]


def worker(wid: int, gids: list[int]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    model = torch.load(SW.WM, map_location="cpu", weights_only=False).eval()
    rows = {r["gid"]: r for r in SG.plans()["rows"]} if SET == "core" else {}
    res = []
    for gid in gids:
        tic = time.time()
        p = rows.get(gid) or dict(kind="none", goal_t=CC.command(gid)["target"].astype(np.float32))
        if p["kind"] == "none":

            def replay():
                c, walked = IE.setup(env, gid)
                return None if env.m.fell_at is not None else (c, walked)
        else:

            def replay():
                r = (
                    RS.step(env, rec, gid, p["d1"])
                    if p["kind"] == "one"
                    else WI.RC.two_steps(env, rec, gid, p["d1"], p["d2"])
                )
                return None if r is None else (r if p["kind"] == "one" else r[0])

        st = PV.state(env, rec, p, replay)
        if st is None:
            continue
        walked = st.get("walked", True)
        side = st["c2"]["side"]
        goal_w = np.r_[st["p"][:2] + (st["Rh"] @ np.r_[st["goal"][:2], 0.0])[:2], st["goal"][2]]
        if SET == "core":
            pz0, t0 = p["pz"], p["t"]
        else:
            c0 = CC.command(gid)
            own = -1.0 if c0["pelvis_z"] is None else float(c0["pelvis_z"])
            best = None
            for pz in dict.fromkeys([own, *IE.PELVIS]):
                for ev in WI.fixed_point(model, env.m, st, st["goal"], pz, None):
                    if best is None or ev["l_pos"] < best[0]["l_pos"]:
                        best = (ev, pz)
            t0, pz0 = best[0]["t"], best[1]
        rec_stay = WI.refine(model, env.m, st, t0, pz0)
        s0, b0 = SW.score(model, env.m, st, pz0)
        gated = bool(b0["pred_err"] > SW.GATE_POS_CM or b0["pred_att"] > SW.GATE_ATT_DEG)

        def run(name: str, best: dict, st_r: dict, pz: float, ready, moved: bool, extra=None) -> None:
            """Execute one pick from the state ready() replays to; append its row."""
            if not ready():
                res.append(dict(gid=gid, run=name, fell=True, moved=moved))
                return
            cmd = best["cmd"]
            c_cmd, rot_h = st_r["c2"], None
            if isinstance(cmd, tuple):
                c_cmd = dict(st_r["c2"], dyaw=cmd[0], pitch=cmd[1])
            elif cmd is not None:
                rot_h = cmd
            seg = IE.execute(env, rec, c_cmd, best["t"].astype(np.float32), pz, False, walked or moved, rot_h=rot_h)
            jaw, Rj = env.m.jaw_centre(side)
            res.append(dict(gid=gid, kind=p["kind"], run=name, pz=float(pz), moved=moved, gated=gated, err=100 * float(np.linalg.norm(jaw - goal_w)),
                            att_deg=float(np.degrees(WI.angle(st["rot_w"], Rj))), fell=bool(seg["fell"]),
                            pred_err=float(best["pred_err"]), pred_att=float(best["pred_att"]), pred_std=float(best["pred_std"]),
                            dev=float(best["dev"]), **(extra or {})))  # fmt: skip

        # every plan below comes before the first execution: Motor plans from the body the sim holds now, and PV.state left
        # it at the planning state (a run of this file that scored after executing planned from the arm already out)
        msg = [f"w{wid} goal {gid} {p['kind']} pz {pz0:.2f} stay score {s0:.2f}"]
        pelvis = None
        if "pelvis" in RUNS and gated:
            sc = {pz: SW.score(model, env.m, st, pz)[0] for pz in dict.fromkeys([pz0, *IE.PELVIS]) if pz != pz0}
            pz1 = min(sc, key=sc.get)
            if sc[pz1] < s0 - SW.MARGIN_CM:
                pelvis = (pz1, float(sc[pz1]), WI.refine(model, env.m, st, st["goal"], pz1))
        ranked = []
        if "stance" in RUNS and gated:
            for d in SW.STEPS:
                for tr in SW.TURNS:
                    if d == (0.0, 0.0) and tr == 0.0:
                        continue
                    s, _ = SW.score(model, env.m, SW.as_if(st, d, tr), pz0)
                    ranked.append((s + SW.STEP_COST * math.hypot(*d) + SW.TURN_COST * abs(tr), d, float(tr)))
            ranked.sort(key=lambda x: x[0])

        stay_ready = lambda: replay() is not None  # noqa: E731
        run("pos", rec_stay["pos"][0], st, pz0, stay_ready, False)
        run("pose3g", rec_stay["pose3g"][0], st, pz0, stay_ready, False)
        if pelvis:
            run(
                "pose3g_pz",
                pelvis[2]["pose3g"][0],
                st,
                pelvis[0],
                stay_ready,
                False,
                dict(score=pelvis[1], stay=float(s0)),
            )
            msg.append(f"pelvis {pelvis[0]:.2f} score {pelvis[1]:.2f}")
        if ranked:
            if ranked[0][0] < s0 - SW.MARGIN_CM:
                _, d, tr = ranked[0]

                def to_stance() -> bool:
                    if replay() is None:
                        return False
                    m = env.m
                    rec.start(side)
                    m.hold(G.PRE_S)
                    rec.on = False
                    q, Rq = env.pelvis()
                    yw = E.yaw_of(Rq)
                    m.walk_to(q[:2] + (E.rot_z(yw) @ np.r_[d[0], d[1], 0.0])[:2], yw + tr, face="fixed")
                    return m.fell_at is None

                if to_stance():
                    st_r = RS.post_step(env, rec, st["c2"], goal_w, st["yaw"], st["p"])
                    pl = WI.refine(model, env.m, st_r, st_r["goal"], pz0)
                    ex = dict(
                        d=[float(x) for x in d], turn_deg=math.degrees(tr), score=float(ranked[0][0]), stay=float(s0)
                    )
                    run("move_pos", pl["pos"][0], st_r, pz0, to_stance, True, ex)
                    run("move_pose3g", pl["pose3g"][0], st_r, pz0, to_stance, True, ex)
                    msg.append(f"move d {d} turn {math.degrees(tr):.0f}")
                else:
                    res.append(dict(gid=gid, run="move_pose3g", fell=True, moved=True))
        a = [x for x in res if x["gid"] == gid and "err" in x]
        print(
            " | ".join(msg + [f"{x['run']} {x['err']:.2f} cm {x['att_deg']:.1f} deg" for x in a])
            + f" | {time.time() - tic:.0f} s",
            flush=True,
        )
        json.dump(res, open(OUT / f"h_w{wid}.json", "w"))


def launch(procs: int, gpus: str, limit: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gids = goals()[: limit or None]
    print(f"{len(gids)} goals, set {SET}, runs {RUNS}, model {SW.WM}", flush=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(OUT / f"h_w{w}.log", "w")
        ps.append(subprocess.Popen([str(E.SIM_PY), __file__, "_w", "--wid", str(w), "--gids", ",".join(map(str, gids[w::procs]))],
                                   stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))  # fmt: skip
    print("exit codes", [p.wait() for p in ps], flush=True)


def line(name: str, rs: list[dict]) -> str:
    e, a = np.array([r["err"] for r in rs]), np.array([r["att_deg"] for r in rs])
    return (f"| {name} | {len(rs)} | {np.median(e):.2f} | {(e > 2).sum()} | {np.percentile(e, 90):.2f} | {np.median(a):.1f} | "
            f"{(a <= 10).mean():.0%} | {(a > 30).sum()} | {sum(r['fell'] for r in rs)} |")  # fmt: skip


def report() -> None:
    R = [r for f in sorted(OUT.glob("h_w*.json")) for r in json.load(open(f)) if "err" in r]
    by = {}
    for r in R:
        by.setdefault(r["gid"], {})[r["run"]] = r
    gids = sorted(g for g, v in by.items() if "pos" in v and "pose3g" in v)
    pick = {
        "stay, position only": lambda v: v["pos"],
        "stay, wrist first": lambda v: v["pose3g"],
        "pelvis by wrist, wrist first": lambda v: v.get("pose3g_pz", v["pose3g"]),
    }  # fmt: skip
    if any("move_pose3g" in by[g] for g in gids):
        pick["move, wrist first"] = lambda v: v.get("move_pose3g", v["pose3g"])
        pick["move, look again"] = lambda v: (
            v["move_pos"] if "move_pose3g" in v and v["move_pose3g"]["pred_err"] > v["pose3g"]["pred_err"] + 0.2 else v.get("move_pose3g", v["pose3g"]))  # fmt: skip
    L = [
        f"# {OUT.name}: {len(gids)} goals",
        "",
        "| rule | goals | position cm | > 2 cm | p90 | wrist deg | <= 10 deg | > 30 deg | falls |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    L += [line(k, [f(by[g]) for g in gids]) for k, f in pick.items()]
    sw = [g for g in gids if "pose3g_pz" in by[g]]
    if sw:
        L += [
            "",
            f"{len(sw)} goals changed the pelvis height; on those:",
            "",
            "| rule | goals | position cm | > 2 cm | p90 | wrist deg | <= 10 deg | > 30 deg | falls |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        L += [
            line("position height", [by[g]["pose3g"] for g in sw]),
            line("wrist height", [by[g]["pose3g_pz"] for g in sw]),
        ]
        better = sum(by[g]["pose3g_pz"]["att_deg"] < by[g]["pose3g"]["att_deg"] for g in sw)
        L += ["", f"wrist better on {better} / {len(sw)}"]
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["h", "d", "_w"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--gids", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "h":
        launch(a.procs, a.gpus, a.limit)
    elif a.phase == "d":
        report()
    else:
        worker(a.wid, [int(g) for g in a.gids.split(",") if g])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
