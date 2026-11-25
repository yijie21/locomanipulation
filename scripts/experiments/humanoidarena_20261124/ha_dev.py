"""Development probe for the harness (not reported): one dev episode of a task, its start look saved, then the motions
the fixed sequences are built from, each logged: a walk ahead, a side step, a turn, a right-arm reach forward, the
right hand closing. The robot's root, joints and palm every step; a small front-camera frame every 0.5 s.

    GPU=3 bash run_ha.sh ha_dev.py --ha_task ppbox --out <dir> [--dev_seed 100 --dev_repeat 0] [--parts look,walk,...]
-> <dir>/<task>_look.npz (rgb, depth, K, cam pose in the pelvis frame ...), <task>_look.png, <task>_dev.npz, one JSON line
"""

import json
from pathlib import Path

import ha_sim as H
import numpy as np
from PIL import Image


def main() -> int:
    p = H.build_parser()
    p.add_argument("--out", required=True)
    p.add_argument("--dev_seed", type=int, default=100)
    p.add_argument("--dev_repeat", type=int, default=0)
    p.add_argument("--parts", default="look,walk,walk2,side,turn,arm,hand")
    p.add_argument("--v", type=float, default=0.3, help="m/s slow-walk speed")
    p.add_argument("--walk_m", type=float, default=0.6)
    a = p.parse_args()
    H.finish_args(a, a.ha_task)
    H.check_seed(a.dev_seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = H.Sim(a)
    parts = a.parts.split(",")
    es = H.episode_seed(a.task, a.dev_seed, a.dev_repeat)
    sim.reset(es)
    rep = dict(task=a.ha_task, seed=a.dev_seed, repeat=a.dev_repeat, episode_seed=es, startup_s=round(sim.startup_s, 1),
               body_names=sim.body_names, root0=sim.root().tolist())  # fmt: skip
    if "look" in parts:
        L = sim.look()
        np.savez_compressed(out / f"{a.ha_task}_look.npz", **L)
        Image.fromarray(L["rgb"]).save(out / f"{a.ha_task}_look.png")
        rep["look"] = dict(cam_p=L["cam_p"].round(4).tolist(), K=L["K"].round(2).tolist(),
                           depth=[float(np.nanmin(L["depth"])), float(np.nanmax(L["depth"]))])  # fmt: skip
    log, frames = [], []
    phase = ["start"]

    def run(n, fn=None):
        for k in range(n):
            if fn is not None:
                fn(k)
            sim.step()
            r = sim.root()
            palm = sim.body("right_hand_palm_link")
            log.append(dict(k=sim.k, phase=phase[0], root=r.tolist(), q=sim.q().tolist(), ref=sim.joints.tolist(),
                            palm=palm.tolist(), dxy=np.asarray(sim.root_dxy).tolist(), reward=sim.reward,
                            up=sim.up()))  # fmt: skip
            if sim.k % 25 == 0:
                cam = sim.env.scene["front_camera"]
                rgb = cam.data.output["rgb"][0].cpu().numpy()[..., :3].astype(np.uint8)
                frames.append(np.asarray(Image.fromarray(rgb).resize((320, 240))))

    def stats(name, k0):
        seg = log[k0:]
        r0, r1 = np.array(seg[0]["root"]), np.array(seg[-1]["root"])
        y0 = H.yaw_of(r0[3:])
        d = H.rot_z(y0).T @ (r1[:3] - r0[:3])
        rep[name] = dict(steps=len(seg), d_ahead=round(float(d[0]), 3), d_left=round(float(d[1]), 3),
                         d_yaw_deg=round(float(np.degrees(H.yaw_of(r1[3:]) - y0)), 1),
                         min_up=round(min(s["up"] for s in seg), 3), fell=sim.fell_at)  # fmt: skip
        print(f"[dev] {name} {json.dumps(rep[name])}", flush=True)

    W = sim.W
    phase[0] = "stand"
    run(50)
    r0 = sim.root()
    yaw0 = H.yaw_of(r0[3:])
    fwd = np.array([np.cos(yaw0), np.sin(yaw0), 0.0])
    left = np.array([-np.sin(yaw0), np.cos(yaw0), 0.0])

    def go(name, move_fn, done_fn, n_max, settle=100):
        phase[0] = name
        k0 = len(log)
        for _ in range(n_max):
            sim.move = move_fn()
            run(1)
            if done_fn():
                break
        sim.move = W.Movement(W.IDLE, -1.0, np.zeros(3), sim.move.facing.copy(), -1.0)
        run(settle)
        stats(name, k0)
        rep[name]["replans"] = sim.walker.replans

    if "walk" in parts:
        p0 = sim.root()[:2].copy()
        go(
            "walk",
            lambda: W.Movement(W.SLOW_WALK, a.v, fwd, fwd),
            lambda: np.linalg.norm(sim.root()[:2] - p0) > a.walk_m,
            400,
        )
    if "walk2" in parts:
        p0 = sim.root()[:2].copy()
        go(
            "walk2",
            lambda: W.Movement(W.WALK, -1.0, fwd, fwd),
            lambda: np.linalg.norm(sim.root()[:2] - p0) > a.walk_m,
            400,
        )
    if "side" in parts:
        p0 = sim.root()[:2].copy()
        go(
            "side",
            lambda: W.Movement(W.SLOW_WALK, 0.2, -left, fwd),
            lambda: np.linalg.norm(sim.root()[:2] - p0) > 0.3,
            400,
        )
    if "turn" in parts:
        f90 = left.copy()
        y_t = yaw0 + np.pi / 2
        go(
            "turn",
            lambda: W.Movement(W.SLOW_WALK, 0.0, np.zeros(3), f90),
            lambda: abs(np.angle(np.exp(1j * (H.yaw_of(sim.root()[3:]) - y_t)))) < np.radians(5),
            400,
        )
        if "look" in parts:
            L = sim.look()
            np.savez_compressed(out / f"{a.ha_task}_look_left.npz", **L)
            Image.fromarray(L["rgb"]).save(out / f"{a.ha_task}_look_left.png")
        go(
            "turn_idle",
            lambda: W.Movement(W.IDLE, -1.0, np.zeros(3), fwd),
            lambda: abs(np.angle(np.exp(1j * (H.yaw_of(sim.root()[3:]) - yaw0)))) < np.radians(5),
            300,
        )
    if "arm" in parts:
        phase[0] = "arm"
        k0 = len(log)
        j = sim.upper.copy()
        goal = j.copy()
        i_sp, i_el = H.ORDER.index("right_shoulder_pitch_joint"), H.ORDER.index("right_elbow_joint")
        goal[i_sp], goal[i_el] = -0.9, 0.2
        n = 75

        def arm(k):
            u = (k + 1) / n
            s = u * u * u * (10 - 15 * u + 6 * u * u)
            sim.upper = j + s * (goal - j)

        run(n, arm)
        run(75)
        stats("arm", k0)
        rep["arm"]["q_err"] = (np.array(log[-1]["q"]) - goal)[[i_sp, i_el]].round(3).tolist()
    if "hand" in parts:
        phase[0] = "hand"
        sim.hands = np.array([0.0, 1.0])
        run(50)
        sim.hands = np.zeros(2)
        run(25)
    np.savez_compressed(out / f"{a.ha_task}_dev.npz", log=json.dumps(log), frames=np.array(frames))
    rep["steps"] = sim.k
    print(json.dumps(rep), flush=True)
    print("[dev] DONE", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    H.main_guard(main)
