"""Self world model data: the standing robot of env.py, one IK pre-grasp, then 4 s of random jaw-centre commands
at 10 Hz through Motor.hand (no servo: what ScaleBFM itself delivers). The box is parked; the table stays.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/selfwm_20261029/gen_data.py run --procs 6 --episodes 600
    $PY scripts/experiments/selfwm_20261029/gen_data.py merge

Writes outputs/selfwm_20261029/data/w<i>.npz per worker and data/all.npz after merge: per episode S (T+1, 65),
K (T+1, 12), W (T+1, 10), A (T, 6), stacked with `ep` offsets.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import env as E
import numpy as np

T = 40  # 10 Hz steps per episode
VMAX = 0.20  # m/s, command velocity cap


def commands(rng: np.random.Generator, start: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """T (target, velocity) pairs in the stance frame from `start`: an OU walk, waypoints, or a jittered hold."""
    kind = rng.choice(["ou", "way", "hold"], p=[0.4, 0.45, 0.15])
    lo, hi = E.ROAM_LO, E.ROAM_HI
    out, x, v = [], start.copy(), np.zeros(3)
    goal, speed = None, 0.0
    for _ in range(T):
        if kind == "ou":
            v = 0.7 * v + rng.normal(0.0, 0.06, 3)
        elif kind == "way":
            if goal is None or np.linalg.norm(goal - x) < 0.005 or rng.random() < 0.08:
                goal, speed = rng.uniform(lo, hi), rng.uniform(0.03, 0.15)
            d = goal - x
            n = float(np.linalg.norm(d))
            v = d / n * min(speed, n / E.STEP_S) if n > 1e-9 else np.zeros(3)
        else:
            v = rng.normal(0.0, 0.01, 3)
        n = float(np.linalg.norm(v))
        v = v * (VMAX / n) if n > VMAX else v
        nxt = x + v * E.STEP_S
        bounce = (nxt < lo) | (nxt > hi)
        v[bounce] *= -1.0
        x = np.clip(x + v * E.STEP_S, lo, hi)
        out.append((x.copy(), v.copy()))
    return out


def episode(env: E.Env, rng: np.random.Generator) -> dict | None:
    env.reset(None)
    if env.pregrasp(rng.uniform(E.START_LO, E.START_HI)) is None:
        return None
    S, K, W, A = [env.self_state()], [env.kinematics()], [env.world()], []
    seq = commands(rng, env.commanded_world())
    for tgt, vel in seq:
        A.append(env.command(tgt, vel))
        if not env.step():
            break
        S.append(env.self_state())
        K.append(env.kinematics())
        W.append(env.world())
    return dict(S=np.array(S), K=np.array(K), W=np.array(W), A=np.array(A), fell=env.m.fell_at is not None)


def worker(wid: int, episodes: list[int], seed: int) -> None:
    env = E.Env()
    rows, t0 = [], time.time()
    for i in episodes:
        rng = np.random.default_rng(seed + i)
        r = episode(env, rng)
        if r is not None:
            r["id"] = i
            rows.append(r)
        print(f"w{wid} ep {i} {'skip' if r is None else len(r['A'])} {time.time() - t0:.0f}s", flush=True)
    out = E.OUT / "data" / f"w{wid}.npz"
    pack = {k: np.concatenate([r[k] for r in rows]) for k in ("S", "K", "W", "A")}
    pack["n_states"] = np.array([len(r["S"]) for r in rows])
    pack["n_actions"] = np.array([len(r["A"]) for r in rows])
    pack["fell"] = np.array([r["fell"] for r in rows])
    pack["id"] = np.array([r["id"] for r in rows])
    np.savez_compressed(out, **pack)
    print(f"w{wid} done {len(rows)} episodes {time.time() - t0:.0f}s", flush=True)


def run(procs: int, episodes: int, seed: int) -> None:
    (E.OUT / "data").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="1")
    jobs = []
    for w in range(procs):
        ids = list(range(w, episodes, procs))
        log = open(E.OUT / "data" / f"w{w}.log", "w")
        cmd = [
            str(E.SIM_PY),
            __file__,
            "worker",
            "--wid",
            str(w),
            "--seed",
            str(seed),
            "--ids",
            ",".join(map(str, ids)),
        ]
        jobs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))
    codes = [j.wait() for j in jobs]
    json.dump(dict(procs=procs, episodes=episodes, seed=seed, codes=codes), open(E.OUT / "data" / "run.json", "w"))
    print("exit codes", codes)


def merge() -> None:
    parts = sorted((E.OUT / "data").glob("w*.npz"))
    zs = [np.load(p) for p in parts]
    out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    np.savez_compressed(E.OUT / "data" / "all.npz", **out)
    print(f"{len(out['id'])} episodes, {int(out['n_actions'].sum())} steps, fell {int(out['fell'].sum())}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "merge"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=290000)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    if a.what == "run":
        run(a.procs, a.episodes, a.seed)
    elif a.what == "worker":
        worker(a.wid, [int(x) for x in a.ids.split(",") if x], a.seed)
    else:
        merge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
