"""Data where the stance search looks (2026-11-03, after stance_wrist.py): the moves it chose with turns lost position
(on the 111 of 345 goals that turned, pose3g 0.81 -> 1.13 cm, > 2 cm 17 -> 25) while the model predicted ~0.3 cm. The
search scores 24 candidate stances and keeps the best, so it favours the stances the model is too optimistic about (the
inversion's optimizer's curse, one level up; onpolicy.py's round 1 covered the commands, not the stances).

A training goal is onpolicy.py's (invert_eval.setup for gid >= GID0, half of them after one short heading-kept step) and
its pelvis height onpolicy.py's (the best position fixed point among invert_eval.PELVIS). From the planning state, the
stance is chosen as stance_wrist.py chooses it with the current model (the gate, STEPS x TURNS, the costs, the margin).
Goals that stay are only logged (onpolicy.py's data covers those). A goal that moves runs its K_STANCES best stances
that beat staying (the first is stance_wrist's choice; the next ones are where the choice goes when the model shifts a
little, and the search, the costly part, is paid once): the same replay, the PRE_S hold, Motor.walk_to the candidate,
restance.post_step, wrist_invert.refine there, then up to N_EXEC distinct picks over onpolicy.PICKS, each from a fresh
replay of the move (deterministic), saved in gen_reach's format (reset id ID0 + gid) with the picks in a json.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    REACHWM_GOAL_REF=1 $PY scripts/experiments/reachwm_20261030/stance_onpolicy.py h --procs 6 --n 480
    $PY scripts/experiments/reachwm_20261030/stance_onpolicy.py merge
SONP_TAG (default onp2) names the data dir outputs/reachwm_20261030/data_<tag>/, SONP_WM the model (default wrist_turn's).
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
import onpolicy as OP  # noqa: E402
import stance_wrist as SW  # noqa: E402

WI, RS, E, IE, PI = SW.WI, SW.RS, SW.E, SW.IE, SW.PI
TAG = os.environ.get("SONP_TAG", "onp2")
OUT = IE.OUT.parent / f"data_{TAG}"
OP.OUT = OUT  # onpolicy.save writes through its module global
WM = Path(os.environ.get("SONP_WM", str(IE.OUT / "wrist_turn" / "wm_ref.pt")))
GID0 = int(os.environ.get("SONP_GID0", "120000"))  # onpolicy.py's round 1 used 100000-100899
ID0 = int(os.environ.get("SONP_ID0", "700000"))
N_EXEC = 3  # picks per stance
K_STANCES = 3  # the best stances that beat staying, the first as stance_wrist chooses


def worker(wid: int, gids: list[int]) -> None:
    import data as DT
    import gen_reach as G
    import torch

    assert DT.GOAL_REF
    torch.set_num_threads(1)
    env = E.Env("open")
    rec = G.Recorder(env)
    model = torch.load(WM, map_location="cpu", weights_only=False).eval()
    env.reset(None)  # the first call after start-up differs (scalebfm-first-call-nondeterminism): warm up
    env.m.hold(1.0)
    segs, picks, log, chunk = [], [], [], 0
    for gid in gids:
        tic = time.time()
        rng = np.random.default_rng([gid, 0x0B1])
        d_h = rng.uniform(*G.SIDESTEP_BOX) if rng.random() < 0.5 else None
        r = OP.plan_state(env, rec, gid, d_h)
        if r is None:
            continue
        c, walked, st = r
        best_pz, best_l = None, np.inf
        for pz in IE.PELVIS:
            fp = WI.fixed_point(model, env.m, st, st["goal"], pz, None)
            b = min(fp, key=lambda ev: ev["l_pos"])
            if b["l_pos"] < best_l:
                best_pz, best_l = pz, b["l_pos"]
        s0, b0 = SW.score(model, env.m, st, best_pz)
        searched = b0["pred_err"] > SW.GATE_POS_CM or b0["pred_att"] > SW.GATE_ATT_DEG
        cands = []  # (score with the step and turn costs, d, turn), stance_wrist's candidate order
        if searched:
            for d in SW.STEPS:
                for tr in SW.TURNS:
                    if d == (0.0, 0.0) and tr == 0.0:
                        continue
                    s, _ = SW.score(model, env.m, SW.as_if(st, d, tr), best_pz)
                    cands.append((s + SW.STEP_COST * math.hypot(*d) + SW.TURN_COST * abs(tr), d, float(tr)))
        cands.sort(key=lambda x: x[0])  # stable: the first is stance_wrist's choice
        moved = bool(cands) and cands[0][0] < s0 - SW.MARGIN_CM
        chosen = [x for x in cands if x[0] < s0][:K_STANCES] if moved else []
        log.append(dict(gid=gid, searched=bool(searched), moved=bool(moved), stay=s0, pz=float(best_pz), stepped=d_h is not None,
                        stances=[dict(score=x[0], d=list(x[1]), turn_deg=math.degrees(x[2])) for x in chosen]))  # fmt: skip
        if not moved:
            print(f"w{wid} goal {gid}: stay (searched {searched}, {s0:.2f}) | {time.time() - tic:.0f} s", flush=True)
            json.dump(log, open(OUT / f"stance_w{wid}.json", "w"))
            continue
        goal_w = np.r_[st["p"][:2] + (st["Rh"] @ np.r_[st["goal"][:2], 0.0])[:2], st["goal"][2]]
        for rank, (sc, dd, tr) in enumerate(chosen):

            def to_reach_state(dd=dd, tr=tr) -> bool:
                if not OP.replay(env, rec, gid, d_h):
                    return False
                m = env.m
                rec.start(st["c2"]["side"])
                m.hold(G.PRE_S)
                rec.on = False
                q, Rq = env.pelvis()
                yw = E.yaw_of(Rq)
                m.walk_to(q[:2] + (E.rot_z(yw) @ np.r_[dd[0], dd[1], 0.0])[:2], yw + tr, face="fixed")
                return m.fell_at is None

            if not to_reach_state():
                log[-1]["stances"][rank]["fell"] = True
                continue
            st_r = RS.post_step(env, rec, st["c2"], goal_w, st["yaw"], st["p"])
            plans_ = WI.refine(model, env.m, st_r, st_r["goal"], best_pz)
            todo, tags = [], {}
            for meth in OP.PICKS:
                b = plans_[meth][0]
                key = b["t"].tobytes() + (b"want" if b["cmd"] is None else np.asarray(b["cmd"], float).tobytes())
                if key not in tags:
                    todo.append((key, b))
                tags.setdefault(key, []).append(meth)
            for key, b in todo[:N_EXEC]:
                if not to_reach_state():
                    break
                cmd = b["cmd"]
                c_cmd, rot_h = st_r["c2"], None
                if isinstance(cmd, tuple):
                    c_cmd = dict(st_r["c2"], dyaw=cmd[0], pitch=cmd[1])
                elif cmd is not None:
                    rot_h = cmd
                seg = IE.execute(env, rec, c_cmd, b["t"].astype(np.float32), best_pz, False, True, rot_h=rot_h)
                seg.update(PI.plan_record(rec.plan))
                seg.update(reset=ID0 + gid, homed=False, sidestep=np.zeros(2, np.float32) if d_h is None else d_h.astype(np.float32),
                           n_sidesteps=0 if d_h is None else 1, wrist_dev=np.zeros(3, np.float32))  # fmt: skip
                jaw, Rj = env.m.jaw_centre(st["c2"]["side"])
                x0d = float(
                    np.abs(
                        PI.summarize(seg, PI.plan_record(rec.plan), st_r["goal"][None].astype(np.float32))["x0"][0]
                        - b["x0"]
                    ).max()
                )
                segs.append(seg)
                picks.append(dict(gid=gid, rank=rank, methods=tags[key], err=100 * float(np.linalg.norm(jaw - goal_w)),
                                  att_deg=float(np.degrees(WI.angle(st["rot_w"], Rj))), pred_err=float(b["pred_err"]),
                                  pred_std=float(b["pred_std"]), pred_att=float(b["pred_att"]), att_spread=float(b["att_spread"]),
                                  dev=float(b["dev"]), pz=float(best_pz), d=list(dd), turn_deg=math.degrees(tr),
                                  stepped=d_h is not None, fell=bool(seg["fell"]), x0_diff=x0d))  # fmt: skip
        print(f"w{wid} goal {gid}: moved " + " || ".join(
            f"d {x[1]} turn {math.degrees(x[2]):.0f}: " + " | ".join(f"{'+'.join(p['methods'])} {p['err']:.2f} cm {p['att_deg']:.0f} deg "
                                                                    f"(pred {p['pred_err']:.2f}, {p['pred_att']:.0f})"
                                                                    for p in picks if p["gid"] == gid and p["rank"] == r)
            for r, x in enumerate(chosen)) + f" | {time.time() - tic:.0f} s", flush=True)  # fmt: skip
        if len(segs) >= 200:
            OP.save(segs, OUT / f"w{wid}_{chunk:03d}.npz")
            segs, chunk = [], chunk + 1
        json.dump(picks, open(OUT / f"picks_w{wid}.json", "w"))
        json.dump(log, open(OUT / f"stance_w{wid}.json", "w"))
    if segs:
        OP.save(segs, OUT / f"w{wid}_{chunk:03d}.npz")
    json.dump(picks, open(OUT / f"picks_w{wid}.json", "w"))
    json.dump(log, open(OUT / f"stance_w{wid}.json", "w"))


def launch(procs: int, gpus: str, n: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cards = gpus.split(",")
    gids = list(range(GID0, GID0 + n))
    ps = []
    for w in range(procs):
        env = dict(os.environ, OMP_NUM_THREADS="1", CUDA_VISIBLE_DEVICES=cards[w % len(cards)], REACHWM_GOAL_REF="1")
        log = open(OUT / f"w{w}.log", "w")
        cmd = [str(E.SIM_PY), __file__, "_w", "--wid", str(w), "--ids", ",".join(map(str, gids[w::procs]))]
        ps.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))
    print("exit codes", [p.wait() for p in ps], flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["h", "_w", "merge"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    if a.phase == "h":
        launch(a.procs, a.gpus, a.n)
    elif a.phase == "_w":
        OUT.mkdir(parents=True, exist_ok=True)
        worker(a.wid, [int(x) for x in a.ids.split(",") if x])
    else:
        OP.merge()
        log = [r for f in sorted(OUT.glob("stance_w*.json")) for r in json.load(open(f))]
        json.dump(log, open(OUT / "stance.json", "w"))
        mv = [r for r in log if r["moved"]]
        sts = [x for r in mv for x in r["stances"]]
        print(f"{len(log)} goals, {sum(r['searched'] for r in log)} searched, {len(mv)} moved; {len(sts)} stances run, "
              f"{sum(1 for x in sts if x['turn_deg'] != 0)} turned, {sum(1 for x in sts if x.get('fell'))} fell")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
