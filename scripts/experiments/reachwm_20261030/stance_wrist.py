"""Stance for the wrist (2026-11-03, the user: for big wrist turns, step and turn first instead of a contorted reach).

restance.py picks a step by the predicted position only, heading kept. Here the self model also scores the wrist and the
robot may turn: from the planning state of each of the 345 core goals (wrist_invert's, the same replays), every candidate
stance in STEPS x TURNS is scored as if the same body stood there (restance's trick, now with a rotation: the goal and
the wanted wrist rotation are re-expressed in the candidate's heading frame and planned from the current body) by
wrist_invert's joint fixed point from the wanted command, best gated iterate by pose3g's loss. The score in cm is
100 sqrt(loss) + STEP_COST per m + TURN_COST per rad. The robot moves only if the best candidate beats staying by
MARGIN_CM, and only goals whose staying prediction is poor are searched (GATE_POS_CM, GATE_ATT_DEG).
A move is Motor.walk_to to the candidate with the heading set (stop correction on, as restance's sidestep), then
restance.post_step re-expresses the goal and the wanted rotation from where the robot stands (the convention of the
report's step-then-reach), wrist_invert.refine there, and the picks of pos, pose3g, w20 run eyes closed.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    REACHWM_GOAL_REF=1 $PY scripts/experiments/reachwm_20261030/stance_wrist.py h --procs 6
    $PY scripts/experiments/reachwm_20261030/stance_wrist.py d
STANCE_WM picks the model (default wrist_invert2's, the report's), STANCE_OUT the results dir, STANCE_BASE the glob of the
same model's no-move run (wrist_invert.py), whose rows the goals that stay take. STANCE_TURNS_DEG sets the turns (0 alone:
steps only). STANCE_PRIOR names an earlier run of the same model with every candidate here and the same costs on them: a goal
whose choice there did not turn chooses the same here (its choice is in the smaller set, and staying stays), so `h` runs only
the goals that turned there and `d` takes the rest from it.
STANCE_GUARD_CM (2026-11-04) is the position guard (stance_report.py's offline one, now before the move): in score order,
each candidate that beats staying by MARGIN_CM gets wrist_invert.refine as if there, and is taken if its pose3g pick's
predicted position error is no more than this many cm above the pose3g pick's staying (refine at the planning state, as the
no-move run); up to GUARD_TRIES candidates, else the robot stays. The score's own iterate is no reference: from the wanted
command alone, staying predicts 1 to 20 cm where the refined pick predicts 0.0 to 1.5 cm (goals 0..42, 2026-11-04), so a
guard on it passes every move. With STANCE_PRIOR (an unguarded run, the same candidates and costs) `h` runs only the goals that
moved there (a guard only removes candidates), and a goal whose guarded choice is the prior's takes the prior's rows.
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
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
OUT = Path(os.environ.get("STANCE_OUT", str(IE.OUT / "stance_wrist")))
WM = Path(os.environ.get("STANCE_WM", str(IE.OUT / "wrist_invert2" / "wm_ref.pt")))
STEPS = ((0.0, 0.0), (0.15, 0.0), (-0.15, 0.0), (0.0, 0.15), (0.0, -0.15))  # forward, left (heading frame), m
TURNS = tuple(np.radians(float(a)) for a in os.environ.get("STANCE_TURNS_DEG", "0,-15,15,-30,30").split(","))
PRIOR = os.environ.get("STANCE_PRIOR", "")
GUARD_CM = float(os.environ["STANCE_GUARD_CM"]) if os.environ.get("STANCE_GUARD_CM") else None
GUARD_TRIES = 6  # refines per goal (~19 s each)
STEP_COST, TURN_COST, MARGIN_CM = 1.0, 0.5, 0.5  # cm per m, cm per rad, cm
GATE_POS_CM, GATE_ATT_DEG = 0.5, 10.0
PICKS = ("pos", "pose3g", "w20")


def as_if(st: dict, d, turn: float) -> dict:
    """st as if the same body stood d (heading frame) away, turned by `turn`: goal and wanted rotation re-expressed in that
    stance's heading frame and handed to the current body (plans and predictions from the current state)."""
    import gen_reach as G

    Rh, p, yaw = st["Rh"], st["p"], st["yaw"]
    goal_w = np.r_[p[:2] + (Rh @ np.r_[st["goal"][:2], 0.0])[:2], st["goal"][2]]
    p1 = p[:2] + (Rh @ np.r_[d[0], d[1], 0.0])[:2]
    R1 = E.rot_z(yaw + turn)
    g1 = R1.T @ np.r_[goal_w[:2] - p1, 0.0]
    goal1 = np.r_[g1[:2], goal_w[2]].astype(np.float32).astype(np.float64)
    rot_h1 = R1.T @ st["rot_w"]  # the wanted rotation seen from the candidate
    rot_w = Rh @ rot_h1  # handed to the current body
    c2 = dict(st["c2"], dyaw=float(st["c2"]["dyaw"] - turn))
    assert np.allclose(rot_w, E.rot_z(yaw + c2["dyaw"]) @ G.rot_y(c2["pitch"]), atol=1e-6)
    return dict(st, goal=goal1, rot_w=rot_w, c2=c2, rot6=np.r_[rot_h1[:, 0], rot_h1[:, 1]].astype(np.float32))


def score(model, m, st: dict, pz: float) -> tuple[float, dict]:
    """The joint fixed point from the wanted command at st, best gated iterate by pose3g's loss: (score cm, iterate)."""
    ev0 = WI.evaluate(model, m, st, st["goal"], pz, None)
    ev0["k"] = 0
    evs = [ev0] + WI.fixed_point3(model, m, st, ev0, pz)
    gated = [ev for ev in evs if ev["att_spread"] <= WI.SPREAD_MAX_DEG] or [ev0]
    best = min(gated, key=lambda ev: ev["l_pos"] + ev["l_att"])
    return 100 * math.sqrt(best["l_pos"] + best["l_att"]), best


def worker(wid: int, gids: list[int]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    model = torch.load(WM, map_location="cpu", weights_only=False).eval()
    rows = {r["gid"]: r for r in SG.plans()["rows"]}
    prior = {}
    if PRIOR and GUARD_CM is not None:
        for r in [r for f in sorted(Path(PRIOR).glob("h_w*.json")) for r in json.load(open(f))]:
            prior.setdefault(r["gid"], []).append(r)
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
                    else WI.RC.two_steps(env, rec, gid, p["d1"], p["d2"])
                )
                return None if r is None else (r if p["kind"] == "one" else r[0])

        st = PV.state(env, rec, p, replay)
        if st is None:
            continue
        walked = st.get("walked", True)
        pz = p["pz"]
        s0, b0 = score(model, env.m, st, pz)
        choice = dict(d=(0.0, 0.0), turn=0.0, score=s0, stay=s0, searched=False)
        cands, ranked = [], []  # every candidate's score before the step and turn costs; (score with them, d, turn)
        if b0["pred_err"] > GATE_POS_CM or b0["pred_att"] > GATE_ATT_DEG:
            choice["searched"] = True
            for d in STEPS:
                for tr in TURNS:
                    if d == (0.0, 0.0) and tr == 0.0:
                        continue
                    s, _ = score(model, env.m, as_if(st, d, tr), pz)
                    cands.append([d[0], d[1], math.degrees(tr), s])
                    ranked.append((s + STEP_COST * math.hypot(*d) + TURN_COST * abs(tr), d, float(tr)))
        ranked.sort(key=lambda x: x[0])  # stable: the first is the unguarded choice
        beat = [x for x in ranked if x[0] < s0 - MARGIN_CM]
        guard = None
        if GUARD_CM is None:
            beat = beat[:1]
        elif beat:
            ref = float(WI.refine(model, env.m, st, p["t"], pz)["pose3g"][0]["pred_err"])
            guard = dict(
                cm=GUARD_CM, stay_pred=ref, free_d=list(beat[0][1]), free_turn_deg=math.degrees(beat[0][2]), tried=[]
            )
            for sc, d, tr in beat[:GUARD_TRIES]:
                sa = as_if(st, d, tr)
                pe = float(WI.refine(model, env.m, sa, sa["goal"], pz)["pose3g"][0]["pred_err"])
                guard["tried"].append([d[0], d[1], math.degrees(tr), sc, pe])
                if pe <= ref + GUARD_CM:
                    beat = [(sc, d, tr)]
                    break
            else:
                beat = []
        if beat:
            choice.update(d=beat[0][1], turn=beat[0][2], score=beat[0][0])
        moved = bool(beat)
        if not moved:
            choice["score"] = min([s0] + [x[0] for x in ranked[:1]])  # the best other, as before the guard
            # staying replays wrist_invert2's run bit for bit (checked on goal 1, 2026-11-03): the report takes its rows
            res.append(dict(gid=gid, kind=p["kind"], moved=False, searched=choice["searched"], stay=s0, best=choice["score"], cands=cands,
                            guard=guard))  # fmt: skip
            print(f"w{wid} goal {gid} {p['kind']} | stay, searched {choice['searched']}, score {s0:.2f}, best other {choice['score']:.2f} | "
                  f"{time.time() - tic:.0f} s", flush=True)  # fmt: skip
            json.dump(res, open(OUT / f"h_w{wid}.json", "w"))
            continue
        old = [r for r in prior.get(gid, []) if "method" in r and r.get("moved")]
        if old and old[0]["d"] == list(choice["d"]) and abs(old[0]["turn_deg"] - math.degrees(choice["turn"])) < 1e-9:
            # the same move as the unguarded run's: deterministic replays (x0_diff 0), so its rows
            res += [dict(r, guard=guard, cands=cands, reused=True) for r in old]
            print(f"w{wid} goal {gid} {p['kind']} | moved d {choice['d']} turn {math.degrees(choice['turn']):.0f}, as the prior run | "
                  f"{time.time() - tic:.0f} s", flush=True)  # fmt: skip
            json.dump(res, open(OUT / f"h_w{wid}.json", "w"))
            continue

        def to_reach_state():
            """Replay to the planning state, and the move if any, up to where IE.execute's hold begins."""
            if replay() is None:
                return False
            if not moved:
                return True
            m = env.m
            rec.start(st["c2"]["side"])
            m.hold(G.PRE_S)
            rec.on = False
            q, Rq = env.pelvis()
            yw = E.yaw_of(Rq)
            Rh = E.rot_z(yw)
            goal_xy = q[:2] + (Rh @ np.r_[choice["d"][0], choice["d"][1], 0.0])[:2]
            m.walk_to(goal_xy, yw + choice["turn"], face="fixed")
            return m.fell_at is None

        if moved:
            if not to_reach_state():
                res.append(dict(gid=gid, kind=p["kind"], fell=True, moved=True))
                continue
            goal_w = np.r_[st["p"][:2] + (st["Rh"] @ np.r_[st["goal"][:2], 0.0])[:2], st["goal"][2]]
            st_r = RS.post_step(env, rec, st["c2"], goal_w, st["yaw"], st["p"])
        else:
            st_r = st
        plans_ = WI.refine(model, env.m, st_r, st_r["goal"], pz) if moved else WI.refine(model, env.m, st, p["t"], pz)
        done = {}
        for meth in PICKS:
            best = plans_[meth][0]
            cmd = best["cmd"]
            key = best["t"].tobytes() + (b"want" if cmd is None else np.asarray(cmd, float).tobytes())
            if key in done:
                res.append(dict(done[key], method=meth))
                continue
            if not to_reach_state():
                break
            c_cmd, rot_h = st_r["c2"], None
            if isinstance(cmd, tuple):
                c_cmd = dict(st_r["c2"], dyaw=cmd[0], pitch=cmd[1])
            elif cmd is not None:
                rot_h = cmd
            seg = IE.execute(env, rec, c_cmd, best["t"].astype(np.float32), pz, False, walked or moved, rot_h=rot_h)
            jaw, Rj = env.m.jaw_centre(st["c2"]["side"])
            goal_w = np.r_[st["p"][:2] + (st["Rh"] @ np.r_[st["goal"][:2], 0.0])[:2], st["goal"][2]]
            res.append(dict(gid=gid, kind=p["kind"], method=meth, servo="open", err=100 * float(np.linalg.norm(jaw - goal_w)),
                            att_deg=float(np.degrees(WI.angle(st["rot_w"], Rj))), fell=bool(seg["fell"]), moved=bool(moved),
                            searched=choice["searched"], d=list(choice["d"]), turn_deg=math.degrees(choice["turn"]),
                            score=choice["score"], stay=choice["stay"], pred_err=float(best["pred_err"]), pred_att=float(best["pred_att"]),
                            pred_std=float(best["pred_std"]), att_spread=float(best["att_spread"]), dev=float(best["dev"]), cands=cands,
                            guard=guard))  # fmt: skip
            done[key] = res[-1]
        a = [x for x in res if x.get("gid") == gid and "err" in x]
        mv = f"moved d {choice['d']} turn {math.degrees(choice['turn']):.0f}" if moved else "stay"
        print(f"w{wid} goal {gid} {p['kind']} | {mv} "
              f"score {choice['stay']:.2f} -> {choice['score']:.2f} | " + " ".join(f"{x['method']} {x['err']:.2f} cm {x['att_deg']:.1f} deg" for x in a)
              + f" | {time.time() - tic:.0f} s", flush=True)  # fmt: skip
        json.dump(res, open(OUT / f"h_w{wid}.json", "w"))
    json.dump(res, open(OUT / f"h_w{wid}.json", "w"))


def launch(procs: int, gpus: str, limit: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gids = [r["gid"] for r in SG.plans()["rows"]][: limit or None]
    if PRIOR:
        old = [r for f in sorted(Path(PRIOR).glob("h_w*.json")) for r in json.load(open(f))]
        if GUARD_CM is None:
            turned = {
                r["gid"] for r in old if r.get("turn_deg", 0.0) != 0.0 or (r.get("moved") and "turn_deg" not in r)
            }
            gids = [g for g in gids if g in turned]
            print(f"{len(gids)} goals turned (or fell) in {PRIOR}: only these run", flush=True)
        else:
            moved = {r["gid"] for r in old if r.get("moved")}
            gids = [g for g in gids if g in moved]
            print(f"{len(gids)} goals moved (or fell) in {PRIOR}: only these run", flush=True)
    cards = gpus.split(",")
    ps = []
    for w in range(procs):
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(OUT / f"h_w{w}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "_w", "--wid", str(w), "--gids", ",".join(map(str, gids[w::procs]))]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))
    print("exit codes", [p.wait() for p in ps], flush=True)


def base_rows(pattern: str) -> dict:
    """A no-move run's eyes-closed rows by (gid, method)."""
    import glob

    return {(r["gid"], r["method"]): r for f in glob.glob(pattern) for r in json.load(open(f)) if r["servo"] == "open"}


def run_rows(out: Path, prior: str, base: dict) -> tuple[dict, list[int]]:
    """A stance run's rows by (gid, method), the goals that stayed filled from `base`, the goals `out` did not run from `prior`."""
    here = [r for f in sorted(Path(out).glob("h_w*.json")) for r in json.load(open(f))]
    ran = {r["gid"] for r in here}
    old = [r for f in sorted(Path(prior).glob("h_w*.json")) for r in json.load(open(f))] if prior else []
    new, gids = {}, set()
    for r in [r for r in old if r["gid"] not in ran] + here:
        gids.add(r["gid"])
        if "method" in r:
            new[(r["gid"], r["method"])] = r
        elif not r.get("moved", True):
            for m in PICKS:  # stayed: the no-move run itself
                if (r["gid"], m) in base:
                    new[(r["gid"], m)] = dict(base[(r["gid"], m)], moved=False)
    return new, sorted(gids)


def report() -> None:
    base = base_rows(os.environ.get("STANCE_BASE", str(IE.OUT / "wrist_invert2" / "h_w*_pos_pose3g_w20_cap15.json")))
    new, gids = run_rows(OUT, PRIOR, base)
    moved = sorted({g for (g, m), r in new.items() if r.get("moved")})
    L = [f"{len(gids)} goals, {len(moved)} moved; eyes closed; base = wrist_invert2 (no wrist-aware stance)", "",
         "| inversion | set | n | base position median | new | base >2 cm | new | base wrist median | new | base >30 deg | new | base <10 deg | new |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for m in PICKS:
        for nm, sel in (("all", gids), ("moved", moved)):
            ok = [g for g in sel if (g, m) in base and (g, m) in new and not new[(g, m)]["fell"]]
            if not ok:
                continue
            eb, en = np.array([base[(g, m)]["err"] for g in ok]), np.array([new[(g, m)]["err"] for g in ok])
            ab, an = np.array([base[(g, m)]["att_deg"] for g in ok]), np.array([new[(g, m)]["att_deg"] for g in ok])
            L.append(f"| {m} | {nm} | {len(ok)} | {np.median(eb):.2f} | {np.median(en):.2f} | {int((eb > 2).sum())} | {int((en > 2).sum())} | "
                     f"{np.median(ab):.1f} | {np.median(an):.1f} | {int((ab > 30).sum())} | {int((an > 30).sum())} | {(ab < 10).mean():.2f} | {(an < 10).mean():.2f} |")  # fmt: skip
    mv = [new[(g, "pose3g")] for g in moved if (g, "pose3g") in new]
    if mv:
        turns = np.array([r["turn_deg"] for r in mv])
        L += ["", f"moves: {len(mv)}; turned {int((np.abs(turns) > 0).sum())} (|turn| median {np.median(np.abs(turns)):.0f} deg), stepped "
              f"{sum(1 for r in mv if any(r['d']))}; fell {sum(1 for g in moved if new.get((g, 'pose3g'), {}).get('fell'))}"]  # fmt: skip
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["h", "_w", "d"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--gids", default="")
    a = ap.parse_args(argv)
    if a.phase == "h":
        launch(a.procs, a.gpus, a.limit)
    elif a.phase == "_w":
        OUT.mkdir(parents=True, exist_ok=True)
        worker(a.wid, [int(x) for x in a.gids.split(",") if x])
    else:
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
