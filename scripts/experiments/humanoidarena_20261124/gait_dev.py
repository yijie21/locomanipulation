"""Development probe (not reported): which walker settings make HumanoidArena's SONIC walk. Each variant from a fresh
reset of one dev episode: a closed-loop walk of --dist along a direction (heading frame), then IDLE; the distance the
plan and the robot covered every 50 steps, the final error, the overshoot, the lateral drift; and a turn in place.

    GPU=3 bash run_ha.sh gait_dev.py --ha_task doubledesk --out <dir>
-> one JSON line per variant
"""

import json

import ha_sim as H
import numpy as np


def main() -> int:
    p = H.build_parser()
    p.add_argument("--out", required=True)
    p.add_argument("--dist", type=float, default=0.8)
    p.add_argument(
        "--variants",
        default="walk:-1:0,walk:-1:8,walk:-1:15,slow:0.3:8,slow:0.3:15,walk:0.5:8,turn:idle:8,turn:slow:8,side:0.3:8,back:0.3:8",
    )
    a = p.parse_args()
    H.finish_args(a, a.ha_task)
    sim = H.Sim(a)
    W = sim.W
    es = H.episode_seed(a.task, 100, 0)
    for v in a.variants.split(","):
        kind, sp, lead = v.split(":")
        sim.walker.lead = int(lead)
        sim.reset(es)
        for _ in range(50):
            sim.step()
        r0 = sim.root()
        y0 = H.yaw_of(r0[3:])
        fwd = np.array([np.cos(y0), np.sin(y0), 0.0])
        left = np.array([-np.sin(y0), np.cos(y0), 0.0])
        trace, plan = [], []
        p_plan0 = sim.wf["pos"].copy()
        if kind == "turn":
            tgt = y0 + np.pi / 2
            face = np.array([np.cos(tgt), np.sin(tgt), 0.0])
            mv = (
                W.Movement(W.IDLE, -1.0, np.zeros(3), face)
                if sp == "idle"
                else W.Movement(W.SLOW_WALK, 0.0, np.zeros(3), face)
            )
            k_done = None
            for k in range(400):
                sim.move = mv
                sim.step()
                e = np.angle(np.exp(1j * (H.yaw_of(sim.root()[3:]) - tgt)))
                if k % 50 == 49:
                    trace.append(round(float(np.degrees(e)), 1))
                if k_done is None and abs(e) < np.radians(3):
                    k_done = k
            r1 = sim.root()
            d = H.rot_z(y0).T @ (r1[:3] - r0[:3])
            row = dict(v=v, k_done=k_done, yaw_err_trace_deg=trace, drift=d[:2].round(3).tolist(), fell=sim.fell_at)
        else:
            dirv = {"walk": fwd, "slow": fwd, "side": left, "back": -fwd}[kind]
            mode = W.WALK if kind == "walk" else W.SLOW_WALK
            speed = float(sp)
            k_stop = None
            for k in range(500):
                done = (H.rot_z(y0).T @ (sim.root()[:3] - r0[:3]))[:2] @ (H.rot_z(y0).T @ dirv)[:2] >= a.dist - 0.1
                if done and k_stop is None:
                    k_stop = k
                sim.move = (
                    W.Movement(W.IDLE, -1.0, np.zeros(3), fwd)
                    if k_stop is not None
                    else W.Movement(mode, speed, dirv, fwd)
                )
                sim.step()
                if k % 50 == 49:
                    d = H.rot_z(y0).T @ (sim.root()[:3] - r0[:3])
                    trace.append([round(float(d[0]), 3), round(float(d[1]), 3)])
                    plan.append(round(float(np.linalg.norm(sim.wf["pos"][:2] - p_plan0[:2])), 3))
                if k_stop is not None and k > k_stop + 150:
                    break
            r1 = sim.root()
            d = H.rot_z(y0).T @ (r1[:3] - r0[:3])
            row = dict(
                v=v,
                k_stop=k_stop,
                end=d[:2].round(3).tolist(),
                yaw_deg=round(float(np.degrees(H.yaw_of(r1[3:]) - y0)), 1),
                trace=trace,
                plan=plan,
                fell=sim.fell_at,
                min_up=None,
            )
        print("[gait]", json.dumps(row), flush=True)
    print("[gait] DONE", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    H.main_guard(main)
