"""Data where the inversion looks (2026-11-03, the user: "在反推挑中的命令上补数据"): the inversion picks the command the
model scores best among ~100 candidates, so it favours the ones the model is too optimistic about (the report's edge
section; wrist_invert2's wrist-first picks missed > 2 cm on 47 of 345 goals where position-only missed 18). Here the
current model inverts fresh goals, the picks are executed eyes closed, and the reaches go back into the training set.

A training goal: invert_eval.setup(gid) for gid >= GID0 (the same generator as the 345 test goals, other ids), and in half
of them one short heading-kept step first (gen_reach.SIDESTEP_BOX; restance.step); the wanted position and wrist
rotation stay in the world. From the planning state (restance.post_step): the pelvis height whose position fixed point
scores best among invert_eval.PELVIS, then wrist_invert.refine there. Executed, each from a fresh replay of the same
state (deterministic): the picks of pos, pose3g and w20 (the report's three), then of wrist_invert's other rules
over the same candidates (cap15, pose3, k3), at most N_EXEC distinct commands. Segments are saved in gen_reach's format (reset id ID0 + gid), the picks in a json.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    REACHWM_GOAL_REF=1 $PY scripts/experiments/reachwm_20261030/onpolicy.py h --procs 6 --n 900   # round 1
    $PY scripts/experiments/reachwm_20261030/onpolicy.py merge
ONP_TAG (default onp1) names the round: data in outputs/reachwm_20261030/data_<tag>/; ONP_WM the model that inverts
(default wrist_invert2's).
"""

from __future__ import annotations

import os

os.environ.setdefault("REACHWM_GOAL_REF", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wrist_invert as WI  # noqa: E402

PV, RS, E, IE, PI = WI.PV, WI.RS, WI.E, WI.IE, WI.PI
TAG = os.environ.get("ONP_TAG", "onp1")
OUT = IE.OUT.parent / f"data_{TAG}"
WM = Path(os.environ.get("ONP_WM", str(IE.OUT / "wrist_invert2" / "wm_ref.pt")))
GID0 = int(os.environ.get("ONP_GID0", "100000"))
ID0 = 600000 if TAG == "onp1" else int(os.environ["ONP_ID0"])
PICKS = ("pos", "pose3g", "w20", "cap15", "pose3", "k3")  # priority order
N_EXEC = 4


def plan_state(env, rec, gid: int, d_h):
    """Goal gid's planning state, after the step d_h (None: no step). Returns (c, walked, st) or None once fallen."""
    import gen_reach as G

    if d_h is None:
        c, walked = IE.setup(env, gid)
        m = env.m
        if m.fell_at is not None:
            return None
        p, Rp = env.pelvis()
        yaw = E.yaw_of(Rp)
        g = c["target"].astype(np.float32).astype(np.float64)
        goal_w = np.r_[p[:2] + (E.rot_z(yaw) @ np.r_[g[:2], 0.0])[:2], g[2]]
        st = RS.post_step(env, rec, c, goal_w, yaw, p)
        st["walked"] = walked
        st["x0b"] = RS.x0_now(rec, st["sd"], walked, st["start"])
        return c, walked, st
    r = RS.step(env, rec, gid, d_h)
    if r is None:
        return None
    c, _, goal_w, yaw0, p0, _ = r
    st = RS.post_step(env, rec, c, goal_w, yaw0, p0)
    st["walked"] = True
    return c, True, st


def replay(env, rec, gid: int, d_h) -> bool:
    """The same reset, walk and step again, up to where IE.execute takes over (its hold is the planning hold)."""
    if d_h is None:
        IE.setup(env, gid)
        return env.m.fell_at is None
    return RS.step(env, rec, gid, d_h) is not None


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
    segs, picks, chunk = [], [], 0
    for gid in gids:
        tic = time.time()
        rng = np.random.default_rng([gid, 0x0B1])
        d_h = rng.uniform(*G.SIDESTEP_BOX) if rng.random() < 0.5 else None
        r = plan_state(env, rec, gid, d_h)
        if r is None:
            continue
        c, walked, st = r
        # the pelvis height: the best position fixed point among the candidates, as plan_invert chose it
        best_pz, best_t, best_l = None, None, np.inf
        for pz in IE.PELVIS:
            fp = WI.fixed_point(model, env.m, st, st["goal"], pz, None)
            b = min(fp, key=lambda ev: ev["l_pos"])
            if b["l_pos"] < best_l:
                best_pz, best_t, best_l = pz, b["t"], b["l_pos"]
        plans_ = WI.refine(model, env.m, st, np.asarray(best_t, float), best_pz)
        todo, tags = [], {}
        for meth in PICKS:
            b = plans_[meth][0]
            key = b["t"].tobytes() + (b"want" if b["cmd"] is None else np.asarray(b["cmd"], float).tobytes())
            if key not in tags:
                todo.append((key, b))
            tags.setdefault(key, []).append(meth)
        for key, b in todo[:N_EXEC]:
            if not replay(env, rec, gid, d_h):
                break
            cmd = b["cmd"]
            c_cmd, rot_h = st["c2"], None
            if isinstance(cmd, tuple):
                c_cmd = dict(st["c2"], dyaw=cmd[0], pitch=cmd[1])
            elif cmd is not None:
                rot_h = cmd
            seg = IE.execute(env, rec, c_cmd, b["t"].astype(np.float32), best_pz, False, walked, rot_h=rot_h)
            seg.update(PI.plan_record(rec.plan))
            seg.update(reset=ID0 + gid, homed=False, sidestep=np.zeros(2, np.float32) if d_h is None else d_h.astype(np.float32),
                       n_sidesteps=0 if d_h is None else 1, wrist_dev=np.zeros(3, np.float32))  # fmt: skip
            jaw, Rj = env.m.jaw_centre(st["c2"]["side"])
            goal_w = np.r_[st["p"][:2] + (st["Rh"] @ np.r_[st["goal"][:2], 0.0])[:2], st["goal"][2]]
            err = 100 * float(np.linalg.norm(jaw - goal_w))
            att = float(np.degrees(WI.angle(st["rot_w"], Rj)))
            x0d = float(
                np.abs(
                    PI.summarize(seg, PI.plan_record(rec.plan), st["goal"][None].astype(np.float32))["x0"][0] - b["x0"]
                ).max()
            )
            segs.append(seg)
            picks.append(dict(gid=gid, methods=tags[key], err=err, att_deg=att, pred_err=float(b["pred_err"]), pred_std=float(b["pred_std"]),
                              pred_att=float(b["pred_att"]), att_spread=float(b["att_spread"]), dev=float(b["dev"]), pz=float(best_pz),
                              stepped=d_h is not None, fell=bool(seg["fell"]), x0_diff=x0d))  # fmt: skip
        print(f"w{wid} goal {gid}: " + " | ".join(f"{'+'.join(p['methods'])} {p['err']:.2f} cm {p['att_deg']:.0f} deg (pred {p['pred_err']:.2f}, "
              f"{p['pred_att']:.0f})" for p in picks if p["gid"] == gid) + f" | {time.time() - tic:.0f} s", flush=True)  # fmt: skip
        if len(segs) >= 200:
            save(segs, OUT / f"w{wid}_{chunk:03d}.npz")
            segs, chunk = [], chunk + 1
        json.dump(picks, open(OUT / f"picks_w{wid}.json", "w"))
    if segs:
        save(segs, OUT / f"w{wid}_{chunk:03d}.npz")
    json.dump(picks, open(OUT / f"picks_w{wid}.json", "w"))


def save(segs: list[dict], path: Path) -> None:
    import gen_reach as G

    G.OUT = OUT
    G.save(segs, path)


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


def merge() -> None:
    zs = [dict(np.load(p)) for p in sorted(OUT.glob("w*_[0-9][0-9][0-9].npz"))]
    keys = zs[0].keys()
    np.savez_compressed(OUT / "all.npz", **{k: np.concatenate([z[k] for z in zs]) for k in keys})
    picks = [p for f in sorted(OUT.glob("picks_w*.json")) for p in json.load(open(f))]
    json.dump(picks, open(OUT / "picks.json", "w"))
    print(
        f"{sum(len(z['n_steps']) for z in zs)} segments from {len(zs)} chunks, {len(picks)} picks -> {OUT / 'all.npz'}"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["h", "_w", "merge"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    if a.phase == "h":
        launch(a.procs, a.gpus, a.n)
    elif a.phase == "_w":
        OUT.mkdir(parents=True, exist_ok=True)
        worker(a.wid, [int(x) for x in a.ids.split(",") if x])
    else:
        merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
