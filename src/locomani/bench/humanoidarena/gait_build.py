"""Builds gait_cycle.json: the mean walking cycle of the benchmark's own example motions (TODO 49.5b). Standalone, run under the benchmark's venv:

    third_party/HumanoidArena/venv/bin/python -I gait_build.py [OUT]

The TWIST2 tracker follows a reference pose (action.joint_pos, the root's step): the legs of the reference have to show the gait,
moving the reference root alone leaves the robot standing (measured: 400 steps at 0.35 m/s, the pelvis did not move). The benchmark
ships ten retargeted human walking clips (TWIST2/assets/example_motions/*.pkl, MIT, from its own repository); they wander and turn, so
this script finds the strides in the near-straight stretches (forward speed 0.45-1.1 m/s, small sideways speed and turning), cuts them at
the left hip pitch's peaks, resamples each to N frames and averages them. The cycle is in the canonical 29-joint order of the action.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/workspace/code/locomanipulation/third_party/HumanoidArena/repo")
N = 40


def yaw_of_xyzw(q):
    x, y, z, w = q.T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main(out: str) -> None:
    sys.path.insert(0, str(ROOT / "isaaclab_twist2_g1"))
    from action_provider.vla_smpl_runtime import reorder_twist2_to_canonical_29

    cycles, periods, speeds = [], [], []
    for i in range(1, 11):
        d = pickle.load(
            open(ROOT / f"TWIST2/assets/example_motions/0807_yanjie_walk_{i:03d}.pkl", "rb")
        )  # the benchmark's own file
        p, q, fps, dof = d["root_pos"], d["root_rot"], float(d["fps"]), d["dof_pos"]
        yaw, dt = yaw_of_xyzw(q), 1.0 / float(d["fps"])
        v = np.diff(p[:, :2], axis=0) / dt
        c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
        vx, vy, wz = c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], np.diff(yaw) / dt
        k = np.ones(9) / 9
        vx, vy, wz = (np.convolve(a, k, "same") for a in (vx, vy, wz))
        hip = np.convolve(dof[:, 0], np.ones(5) / 5, "same")  # left hip pitch (twist2 order index 0)
        peaks = [
            j
            for j in range(3, len(hip) - 3)
            if hip[j] == hip[j - 3 : j + 4].max() and hip[j] > hip.mean() + 0.2 * hip.std()
        ]
        for a, b in zip(peaks[:-1], peaks[1:]):
            T = (b - a) * dt
            seg = slice(a, min(b, len(vx)))
            if not (0.8 < T < 1.5):
                continue
            if not (
                0.45 < vx[seg].mean() < 1.1
                and abs(vy[seg].mean()) < 0.12
                and abs(wz[seg].mean()) < 0.2
                and np.abs(wz[seg]).max() < 0.6
            ):
                continue
            t = np.linspace(a, b, N, endpoint=False)
            cyc = np.stack([np.interp(t, np.arange(len(dof)), dof[:, j]) for j in range(29)], axis=1)
            cycles.append(reorder_twist2_to_canonical_29(cyc))
            periods.append(T)
            speeds.append(float(vx[seg].mean()))
    if not cycles:
        raise SystemExit("no near-straight stride found")
    C = np.mean(cycles, axis=0)
    spread = float(np.mean([np.abs(c - C).max() for c in cycles]))
    Path(out).write_text(
        json.dumps(
            dict(
                n=N,
                period_s=float(np.mean(periods)),
                speed=float(np.mean(speeds)),
                strides=len(cycles),
                spread_rad=spread,
                source="TWIST2/assets/example_motions/0807_yanjie_walk_*.pkl",
                cycle=np.round(C, 4).tolist(),
            )
        )
    )
    print(
        f"{len(cycles)} strides, period {np.mean(periods):.2f} s, speed {np.mean(speeds):.2f} m/s, spread {spread:.2f} rad -> {out}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("gait_cycle.json")))
