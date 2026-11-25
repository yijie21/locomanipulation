"""Development probe (not reported): one dev episode's look after a 50-step settle, saved with SAM 3's masks for the
task's prompts; then a 45 deg turn each way (closed loop) and a look each.

    GPU=3 bash run_ha.sh look_dev.py --ha_task opendoor --out <dir> [--dev_seed 103 --dev_repeat 0]
"""

import os

import ha_sim as H
import numpy as np


def main() -> int:
    p = H.build_parser()
    p.add_argument("--out", required=True)
    p.add_argument("--dev_seed", type=int, default=103)
    p.add_argument("--dev_repeat", type=int, default=0)
    a = p.parse_args()
    H.finish_args(a, a.ha_task)
    sim = H.Sim(a)
    W = sim.W
    os.makedirs(a.out, exist_ok=True)
    sim.reset(H.episode_seed(a.task, a.dev_seed, a.dev_repeat))
    for _ in range(50):
        sim.step()
    y0 = H.yaw_of(sim.root()[3:])
    for name, dy in (("front", 0.0), ("right", -np.pi / 4), ("left", np.pi / 4)):
        if dy:
            tgt = y0 + dy
            off = 0.0
            for k in range(300):
                e = float(np.angle(np.exp(1j * (tgt - H.yaw_of(sim.root()[3:])))))
                if k % 50 == 49 and abs(e) > np.radians(3):
                    off += e
                f = tgt + off
                sim.move = W.Movement(W.IDLE, -1.0, np.zeros(3), np.array([np.cos(f), np.sin(f), 0.0]))
                sim.step()
            print(f"[look] {name} yaw err {np.degrees(e):.1f} deg", flush=True)
        L = sim.look()
        np.savez_compressed(f"{a.out}/{a.ha_task}_{name}.npz", **L)
        print(f"[look] saved {name} k={sim.k} pelvis {L['pelvis_p'].round(3)} cam_p {L['cam_p'].round(3)}", flush=True)
    print("[look] DONE", flush=True)
    sim.close()
    return 0


if __name__ == "__main__":
    H.main_guard(main)
