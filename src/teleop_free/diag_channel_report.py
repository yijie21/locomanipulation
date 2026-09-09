"""Read outputs/diag/channel.npz and print the channel comparison."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from primitives import GRASP_CENTRE_OFFSET  # noqa: E402

NPZ = f"{paths.REPO}/outputs/diag/channel.npz"
ORDER = ["streamed", "planner", "planner_vel", "planner_pin"]


def rot_of(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def ang(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def main() -> None:
    rows = list(np.load(NPZ, allow_pickle=True)["rows"])
    ok = [r for r in rows if r.get("ok")]
    print(f"{len(rows)} samples, {len(ok)} planned\n")

    def stats(cfg, phase):
        pos, orn, dec = [], [], []
        for r in ok:
            if r["cfg"] != cfg:
                continue
            R = rot_of(r[f"{phase}_quat"])
            gc = r[f"{phase}_pos"] + R @ GRASP_CENTRE_OFFSET
            e = gc - r["grip_w"]
            W = rot_of(r["want_quat"])
            pos.append(float(np.linalg.norm(e)))
            orn.append(ang(W, R))
            dec.append([float(e @ W[:, 0]), float(e @ W[:, 1]), float(e @ W[:, 2])])
        return np.array(pos), np.array(orn), np.array(dec)

    for phase in ("open", "closed"):
        print(f"=== {phase} loop ===")
        print(
            f"{'config':12} {'n':>4} {'pos med':>8} {'p90':>7} {'orn med':>8} "
            f"{'entry':>7} {'lat':>7} {'vert':>7} {'v-bias':>7}"
        )
        for cfg in ORDER:
            pos, orn, dec = stats(cfg, phase)
            if not len(pos):
                continue
            a = np.abs(dec)
            print(
                f"{cfg:12} {len(pos):4d} {np.median(pos) * 100:7.1f}cm "
                f"{np.percentile(pos, 90) * 100:6.1f}cm "
                f"{np.median(orn):7.1f}d "
                f"{np.median(a[:, 0]) * 100:6.1f}cm "
                f"{np.median(a[:, 1]) * 100:6.1f}cm "
                f"{np.median(a[:, 2]) * 100:6.1f}cm "
                f"{np.median(dec[:, 2]) * 100:+6.1f}cm"
            )
        print()

    # per-height-stratum closed-loop, to see if any channel is height-sensitive
    print("=== closed loop by height stratum (median pos err, cm) ===")
    print(f"{'config':12} {'z0':>6} {'z1':>6} {'z2':>6}")
    for cfg in ORDER:
        line = f"{cfg:12}"
        for k in (0, 1, 2):
            v = []
            for r in ok:
                if r["cfg"] != cfg or r["k"] != k:
                    continue
                R = rot_of(r["closed_quat"])
                gc = r["closed_pos"] + R @ GRASP_CENTRE_OFFSET
                v.append(float(np.linalg.norm(gc - r["grip_w"])))
            line += f" {np.median(v) * 100:5.1f}" if v else "     -"
        print(line)
    print()

    # commanded vs delivered waist pitch (MuJoCo joint 14)
    print("=== waist pitch, commanded -> delivered (rad) ===")
    for cfg in ORDER:
        c, d = [], []
        for r in ok:
            if r["cfg"] != cfg or "cmd_joints" not in r:
                continue
            c.append(float(r["cmd_joints"][14]))
            d.append(float(r["open_joints"][14]))
        if not c:
            continue
        c, d = np.array(c), np.array(d)
        use = np.abs(c) > 0.02
        ratio = float(np.median(d[use] / c[use])) if use.any() else float("nan")
        print(
            f"{cfg:12} cmd med {np.median(c):+.3f}  delivered med {np.median(d):+.3f}"
            f"  ratio(|cmd|>0.02, n={int(use.sum())}) {ratio:.2f}"
        )


if __name__ == "__main__":
    main()
