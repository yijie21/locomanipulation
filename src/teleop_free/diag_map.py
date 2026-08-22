"""Measure the command -> palm mapping over the region the grasp uses, and fit it.

diag_repeat.py established that this is worth doing: the same command issued
from four very different arm configurations lands the palm within 2-7 mm, so
the mapping is a function of the command and can be inverted.

It also established that the inverse currently in use is badly wrong. The
hand-fitted constants say palm_x = 0.55*cmd_x - 0.02 and palm_z = 0.50*cmd_z -
0.07, with y assumed to track one-to-one. Measured:

    cmd [ 0.55 -0.20  0.00] -> palm [ 0.324 -0.194  0.068]
    cmd [ 0.75 -0.35 -0.10] -> palm [ 0.448 -0.219  0.135]
    cmd [ 0.40 -0.10  0.10] -> palm [ 0.225 -0.136  0.094]

x is systematically 2.5-6 cm low, z has the wrong SIGN (predicts -0.07 where
the palm sits at +0.068, off by 11-26 cm), and y is compressed by about a third
rather than tracking. A seed that wrong leaves the whole correction to a 1 Hz
integrator with a second of dead time, which is why the reach either crawls or
rings.

    python src/teleop_free/diag_map.py [--n 4] [--settle 3.5]

Writes outputs/scene/wrist_map.npz — an affine model and its inverse, plus the
raw samples so a better model can be fitted later without re-running the robot.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import LocomotionMode  # noqa: E402
from primitives import DT, Primitives  # noqa: E402
import paths

OUT = f"{paths.REPO}/outputs/scene/wrist_map.npz"
# The region the grasp actually commands, not the whole REACH_BOX. Fitting
# where the arm never goes spends samples on extrapolation nobody needs.
X_RANGE = (0.20, 1.00)
Y_RANGE = (-0.50, 0.15)
Z_RANGE = (-0.35, 0.30)


def fit(cmd: np.ndarray, palm: np.ndarray):
    """Affine palm = A cmd + b, least squares, with residuals reported."""
    design = np.hstack([cmd, np.ones((len(cmd), 1))])
    sol, *_ = np.linalg.lstsq(design, palm, rcond=None)
    a, b = sol[:3].T, sol[3]
    pred = cmd @ a.T + b
    res = np.linalg.norm(pred - palm, axis=1)

    return a, b, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="samples per axis")
    ap.add_argument("--settle", type=float, default=3.5)
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    grid = [np.linspace(*r, args.n) for r in (X_RANGE, Y_RANGE, Z_RANGE)]
    targets = np.array(np.meshgrid(*grid, indexing="ij")).reshape(3, -1).T
    # Snake through the grid so consecutive commands are close: the arm has to
    # travel between samples, and a settle long enough for a 1 m jump is a
    # settle wasted on every other point.
    order = np.lexsort((targets[:, 2], targets[:, 1], targets[:, 0]))
    targets = targets[order]

    print(f"{len(targets)} commands, {args.settle:.1f} s each "
          f"-> about {len(targets) * args.settle / 60:.1f} min\n")

    p = Primitives()
    p.cmd.start_control()
    p.cmd.reset_pose()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)

    cmds, palms, pelvis = [], [], []
    t0 = time.time()
    for i, target in enumerate(targets):
        p.cmd.set_wrists(**{f"{args.side}_pos": target.astype(np.float32)})
        end = time.time() + args.settle
        while time.time() < end:
            s = p.obs()
            if s["fallen"]:
                print("  robot fell; stopping early")
                targets = targets[:i]
                break
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=p._facing(s))
            time.sleep(DT)
        else:
            s = p.obs()
            cmds.append(target)
            palms.append(p.world_to_root_full(s["palms"][args.side]["pos"], s))
            pelvis.append(float(s["pelvis"]["pos"][2]))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(targets)}  "
                      f"({time.time() - t0:.0f} s elapsed)")
            continue
        break

    p.cmd.pause_control()
    p.close()

    cmds, palms = np.array(cmds), np.array(palms)
    pelvis = np.array(pelvis)
    print(f"\ncollected {len(cmds)} samples, pelvis "
          f"{pelvis.min():.3f}-{pelvis.max():.3f} m")

    a, b, res = fit(cmds, palms)
    print(f"\naffine fit  palm = A cmd + b")
    for i, row in enumerate(a):
        print(f"   {'xyz'[i]}: {np.round(row, 3)}  + {b[i]:+.3f}")
    print(f"   residual  mean {res.mean() * 100:.2f} cm  "
          f"p95 {np.percentile(res, 95) * 100:.2f} cm  max {res.max() * 100:.2f} cm")

    # What the constants in use today would have predicted, for comparison.
    old = np.stack([0.55 * cmds[:, 0] - 0.02, cmds[:, 1],
                    0.50 * cmds[:, 2] - 0.07], axis=1)
    old_res = np.linalg.norm(old - palms, axis=1)
    print(f"   the constants in use today: mean {old_res.mean() * 100:.2f} cm  "
          f"max {old_res.max() * 100:.2f} cm")

    a_inv = np.linalg.inv(a)
    np.savez(args.out, A=a, b=b, A_inv=a_inv, cmds=cmds, palms=palms,
             pelvis=pelvis, residual=res)
    print(f"\nwrote {args.out}")
    print("  invert with:  cmd = A_inv @ (palm_wanted - b)")


if __name__ == "__main__":
    main()
