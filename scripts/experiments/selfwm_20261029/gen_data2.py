"""Generalised self world model data (step 1): the standing reach envelope the ADT skills actually use.

Each episode draws an anchor from the recorded reach targets of the 10 sealed-pool objects
(outputs/scene_rsi_20261016/body/reach_samples.npz: pre / pregrasp / entry / descend rows, heading frame, the left
hand's y mirrored), a hand, a wrist attitude (level for side grasps; down or tilt45 for pinches) and the recorded
pelvis height. 40 % of episodes first walk 0.4-1.0 m, so the reach starts in the post-walk sway. The IK reach takes
the jaw to the anchor (+-3 cm), then 4 s of local commands at 10 Hz drive the wrist through Motor.hand without the
servo: an OU walk, waypoints, an approach along the attitude's axis, or a jittered hold, inside anchor +-(15, 12, 12) cm.
Open floor (walk_mars), nothing to touch.

    PY=third_party/GR00T-WholeBodyControl/.venv_sim/bin/python
    $PY scripts/experiments/selfwm_20261029/gen_data2.py run --procs 8 --episodes 24000 --gpus 0,1
    $PY scripts/experiments/selfwm_20261029/gen_data2.py merge

Writes outputs/selfwm_20261029/data2/w<i>.npz and all.npz: per episode S (T+1, 65), A (T, 13), W (T+1, 13) and the
episode's side, attitude, pelvis target, walked flag and anchor.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

import env as E
import numpy as np

T = 40
VMAX = 0.20
BOX_HALF = np.array([0.15, 0.12, 0.12])
SAMPLES = E.REPO / "outputs" / "scene_rsi_20261016" / "body" / "reach_samples.npz"
OUT = E.OUT / "data2"
ATT = ("level", "tilt45", "down")


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def wrist_rot(yaw: float, att: str) -> np.ndarray:
    return E.rot_z(yaw) @ {"level": np.eye(3), "tilt45": rot_y(math.pi / 4), "down": rot_y(math.pi / 2)}[att]


def anchors() -> np.ndarray:
    a = np.load(SAMPLES)["samples"]
    keep = np.isin(a["step"], ["pre", "pregrasp", "entry", "descend"])
    return a[keep]


def local_commands(rng, start_w, anchor_w, approach_w):
    """T (target, velocity) pairs in the world around anchor_w."""
    kind = rng.choice(["ou", "way", "approach", "hold"], p=[0.35, 0.35, 0.2, 0.1])
    lo, hi = anchor_w - BOX_HALF, anchor_w + BOX_HALF
    out, x, v = [], np.clip(start_w.copy(), lo, hi), np.zeros(3)
    goal, speed = None, 0.0
    if kind == "approach":
        goal, speed = np.clip(x + approach_w * rng.uniform(0.10, 0.16), lo, hi), rng.uniform(0.03, 0.12)
    for _ in range(T):
        if kind == "ou":
            v = 0.7 * v + rng.normal(0.0, 0.06, 3)
        elif kind in ("way", "approach"):
            if kind == "way" and (goal is None or np.linalg.norm(goal - x) < 0.005 or rng.random() < 0.08):
                goal, speed = rng.uniform(lo, hi), rng.uniform(0.03, 0.15)
            d = goal - x
            n = float(np.linalg.norm(d))
            v = d / n * min(speed, n / E.STEP_S) if n > 1e-9 else np.zeros(3)
        else:
            v = rng.normal(0.0, 0.01, 3)
        n = float(np.linalg.norm(v))
        v = v * (VMAX / n) if n > VMAX else v
        nxt = x + v * E.STEP_S
        v[(nxt < lo) | (nxt > hi)] *= -1.0
        x = np.clip(x + v * E.STEP_S, lo, hi)
        out.append((x.copy(), v.copy()))
    return out


def probe_rows_wide(rng, n: int):
    """probe_rows over the wider table range of the posture test (tables 0.50-1.00 m: side grasps at 0.55-1.10 m), with
    the 0.62 m pelvis as well."""
    rows = []
    for _ in range(n):
        pel = float(rng.choice([0.74, 0.70, 0.66, 0.62])) if rng.random() < 0.8 else 0.80
        rows.append(dict(hand=1, pinch=False, pelvis_z0=pel, tx=rng.uniform(0.18, 0.34), ty=rng.uniform(-0.22, -0.08),
                         tz=rng.uniform(0.55, 1.10) - 0.785, step="probe"))  # fmt: skip
    return rows


def probe_rows(rng, n: int):
    """Workspace rows for the posture chooser (item 1): pre-grasp-like targets in front of the probe's tables at every
    candidate pelvis height, standing still, reached the way the probe reaches them (through its via point: the
    chooser borrows these post-reach states, and a direct IK reach ends in a different arm configuration -- from the
    real start the world model called all four heights right, from direct-reach neighbours it called all four wrong).
    Same fields as reach_samples rows (heading frame from the start pelvis)."""
    rows = []
    for _ in range(n):
        pinch = bool(rng.random() < 0.3)
        pel = float(rng.choice([0.74, 0.70, 0.66])) if rng.random() < 0.75 else 0.80  # 0.80 = stays standing
        x = rng.uniform(0.18, 0.34) if not pinch else rng.uniform(0.30, 0.44)
        y = rng.uniform(-0.22, -0.08)
        zw = rng.uniform(0.66, 1.00) if not pinch else rng.uniform(0.70, 0.95)
        rows.append(dict(hand=1, pinch=pinch, pelvis_z0=pel, tx=x, ty=y, tz=zw - 0.785, step="probe"))
    return rows


def servo_commands(env: E.Env, rng, side: str, rot: np.ndarray, anchor_w: np.ndarray):
    """The relative servo's own behaviour as data (item 1): a virtual grasp point 10-16 cm ahead along the approach and
    up to 3 cm off it; each step u = 2/s x (grasp - jaw) in the grasp frame, no advance while sideways off, capped at
    0.15 m/s, plus a little noise; the target integrates u. Where the hand cannot follow, the target runs ahead of it,
    which random commands never produce and the chooser's imagination does."""
    n, c, t = rot[:, 0], rot[:, 1], rot[:, 2]
    grasp = anchor_w + n * rng.uniform(0.10, 0.16) + c * rng.uniform(-0.03, 0.03) + t * rng.uniform(-0.03, 0.03)
    tol = np.array([0.012, rng.uniform(0.005, 0.02), 0.02])
    target = env.m.commanded_jaw(side).copy()

    def step():
        nonlocal target
        e = rot.T @ (grasp - env.m.jaw_centre(side)[0])
        ok = abs(e[1]) < tol[1] and abs(e[2]) < tol[2]
        u = np.clip(2.0 * e * np.array([1.0 if ok else 0.0, 1.0, 1.0]) + rng.normal(0.0, 0.02, 3), -0.15, 0.15)
        target = target + rot @ u * E.STEP_S
        return target.copy(), rot @ u

    return step


def episode(env: E.Env, rng, rows, walk_p: float = 0.4, servo: bool = False, probe_via: bool = False) -> dict | None:
    r = rows[rng.integers(len(rows))]
    side = "right" if int(r["hand"]) == 1 else "left"
    if r["pinch"]:
        att = "down" if rng.random() < 0.6 else "tilt45"
    else:
        att = "level" if rng.random() < 0.9 else "tilt45"
    pel_z = float(r["pelvis_z0"]) if float(r["pelvis_z0"]) < 0.755 else None
    env.reset(None)
    m = env.m
    walked = bool(rng.random() < walk_p)
    if walked:
        ang = rng.uniform(-math.pi, math.pi)
        d = rng.uniform(0.4, 1.0)
        out = m.walk_to(np.array([d * math.cos(ang), d * math.sin(ang)]), yaw=float(rng.uniform(-math.pi, math.pi)))
        if m.fell_at is not None:
            return None
    p, Rp = env.pelvis()
    yaw = E.yaw_of(Rp)
    ty = float(r["ty"]) if side == "right" else -float(r["ty"])
    anchor = p + E.rot_z(yaw) @ np.array([float(r["tx"]), ty, float(r["tz"])]) + rng.uniform(-0.03, 0.03, 3)
    rot = wrist_rot(yaw, att)
    via = []
    if probe_via:  # the grip probe's own pre-grasp route (eval2.py): through a point outside the table edge
        x_via = env.cap.EDGE_M - env.cap.VIA_OUT_M
        if anchor[0] > x_via:
            via = [np.array([x_via, anchor[1], anchor[2]])]
    out = m.reach(side, anchor, rot, via=via, pelvis_z=pel_z)
    if m.fell_at is not None:
        return None
    approach = rot[:, 0]  # the wrist's x: forward for level, down for "down"
    S, W, A = [env.self_state()], [env.world2()], []
    if servo:
        nxt = servo_commands(env, rng, side, rot, anchor)
        seq = (nxt() for _ in range(T))
    else:
        seq = iter(local_commands(rng, m.commanded_jaw(side), anchor, approach))
    for tgt, vel in seq:
        A.append(env.command_side(side, tgt, vel, rot))
        if not env.step():
            break
        S.append(env.self_state())
        W.append(env.world2())
    return dict(S=np.array(S), W=np.array(W), A=np.array(A), fell=m.fell_at is not None, side=side, att=ATT.index(att),
                pel_z=-1.0 if pel_z is None else pel_z, walked=walked, anchor=anchor.astype(np.float32),
                reach_err_cm=float((out.info or {}).get("jaw_err_cm", -1.0)))  # fmt: skip


def worker(wid: int, ids: list[int], seed: int, probe: bool = False, servo: bool = False, wide: bool = False) -> None:
    env = E.Env("open")
    if wide:
        rows = probe_rows_wide(np.random.default_rng(seed), 4000)
    else:
        rows = probe_rows(np.random.default_rng(seed), 4000) if probe else anchors()
    probe = probe or wide
    res, t0 = [], time.time()
    for i in ids:
        r = episode(
            env, np.random.default_rng(seed + i), rows, walk_p=0.0 if probe else 0.4, servo=servo, probe_via=probe
        )
        if r is not None:
            r["id"] = i
            res.append(r)
        if len(res) % 50 == 0:
            print(f"w{wid} {len(res)} episodes, id {i}, {time.time() - t0:.0f}s", flush=True)
    pack = {k: np.concatenate([r[k] for r in res]) for k in ("S", "W", "A")}
    pack["n_states"] = np.array([len(r["S"]) for r in res])
    pack["n_actions"] = np.array([len(r["A"]) for r in res])
    for k in ("fell", "att", "pel_z", "walked", "id", "reach_err_cm"):
        pack[k] = np.array([r[k] for r in res])
    pack["side"] = np.array([1 if r["side"] == "right" else -1 for r in res])
    pack["anchor"] = np.array([r["anchor"] for r in res])
    prefix = ("t" if servo else "q") if wide else ("s" if servo else "p" if probe else "w")
    np.savez_compressed(OUT / f"{prefix}{wid}.npz", **pack)
    print(f"w{wid} done {len(res)} episodes {time.time() - t0:.0f}s", flush=True)


def run(
    procs: int, episodes: int, seed: int, gpus: str, probe: bool = False, servo: bool = False, wide: bool = False
) -> None:
    """Workers round-robin over `gpus`: ScaleBFM's network, not the physics, bounds the rate (one card at 100 % with
    six workers, 3.2 s an episode each)."""
    OUT.mkdir(parents=True, exist_ok=True)
    cards = gpus.split(",")
    jobs = []
    for w in range(procs):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=cards[w % len(cards)], OMP_NUM_THREADS="1")
        ids = list(range(w, episodes, procs))
        prefix = ("t" if servo else "q") if wide else ("s" if servo else "p" if probe else "w")
        log = open(OUT / f"{prefix}{w}.log", "w")
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
        cmd += (["--probe"] if probe else []) + (["--servo"] if servo else []) + (["--wide"] if wide else [])
        jobs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(E.REPO)))
    codes = [j.wait() for j in jobs]
    name = (
        ("run_wide_servo.json" if servo else "run_wide.json")
        if wide
        else "run_servo.json"
        if servo
        else "run_probe.json"
        if probe
        else "run.json"
    )
    json.dump(dict(procs=procs, episodes=episodes, seed=seed, codes=codes), open(OUT / name, "w"))
    print("exit codes", codes)


def merge(probe: bool = False, servo: bool = False, wide: bool = False) -> None:
    if wide:  # everything: envelope, posture workspace (p, s) and the wide posture range (q, t)
        files = [f for pre in "wpsqt" for f in sorted(OUT.glob(f"{pre}[0-9]*.npz"))]
        zs = [dict(np.load(p)) for p in files]
        out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}
        np.savez_compressed(OUT / "all_wide.npz", **out)
        return
    """The envelope set (w*.npz) into all.npz; with probe, also the posture-workspace set (p*.npz) into all_p.npz; with
    servo as well, the servo-behaviour set (s*.npz) into all_ps.npz. Earlier files and their models stay as they were."""
    files = (
        sorted(OUT.glob("w*.npz"))
        + (sorted(OUT.glob("p*.npz")) if probe or servo else [])
        + (sorted(OUT.glob("s*.npz")) if servo else [])
    )
    zs = [dict(np.load(p)) for p in files]
    out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0]}
    np.savez_compressed(OUT / ("all_ps.npz" if servo else "all_p.npz" if probe else "all.npz"), **out)
    print(f"{len(out['id'])} episodes, {int(out['n_actions'].sum())} steps, fell {int(out['fell'].sum())}, "
          f"walked {int(out['walked'].sum())}, left {int((out['side'] < 0).sum())}, att {np.bincount(out['att']).tolist()}")  # fmt: skip


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "worker", "merge"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=2900000)
    ap.add_argument("--wid", type=int, default=0)
    ap.add_argument("--ids", default="")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--probe", action="store_true", help="the posture chooser's workspace rows instead of the records")
    ap.add_argument("--servo", action="store_true", help="the relative servo's own commands instead of random ones")
    ap.add_argument(
        "--wide", action="store_true", help="the wider posture-test range (tables 0.50-1.00 m, pelvis 0.62)"
    )
    a = ap.parse_args(argv)
    if a.what == "run":
        run(a.procs, a.episodes, a.seed, a.gpus, a.probe, a.servo, a.wide)
    elif a.what == "worker":
        worker(a.wid, [int(x) for x in a.ids.split(",") if x], a.seed, a.probe, a.servo, a.wide)
    else:
        merge(a.probe, a.servo, a.wide)
    return 0


if __name__ == "__main__":
    sys.exit(main())
