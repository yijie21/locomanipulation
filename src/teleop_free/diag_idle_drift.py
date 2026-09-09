"""How far does the robot wander while commanded to stand still?

The v1.1 policy fails every gym round with "teleport did not land": after
the reset places it and 4 s of IDLE settle, the pelvis is 0.7 m away. IDLE
carries movement=(0,0,0) and a facing vector, so nothing is asking it to
travel. This isolates the drift: teleport, hold IDLE, measure — once with
the facing the pipeline sends (the sampled world yaw), once with the
robot's own measured yaw, once with no facing at all.

    python src/teleop_free/diag_idle_drift.py [--secs 5] [--reps 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stream_reach  # noqa: E402
from agent_server import LocomotionMode  # noqa: E402
from primitives import DT, Primitives  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402


def trial(p, reset, mode: str, secs: float):
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset(label="drift")
    time.sleep(0.3)
    p.cmd.start_control()
    if mode.endswith("+reanchor"):
        # Force the deploy to re-capture its heading anchor at the pose the
        # teleport just created. v1.1's observations are normalized by the
        # robot heading, so a stale anchor is a standing heading error the
        # policy will keep turning to close.
        p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
        stream_reach.reanchor(p, side="right")
        p.cmd.stream_mode = (
            stream_reach.StreamMode.PLANNER_VR_3PT if hasattr(stream_reach, "StreamMode") else p.cmd.stream_mode
        )
    s = p.obs()
    yaw0 = float(p.pelvis_yaw(s))
    xy0 = p.pelvis_xy(s).copy()
    fv = (float(np.cos(yaw0)), float(np.sin(yaw0)), 0.0)
    t0 = time.time()
    while time.time() - t0 < secs:
        if mode.startswith("measured"):
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=fv)
        elif mode.startswith("worldx"):
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(1.0, 0.0, 0.0))
        else:  # no facing argument at all
            p.cmd.frame(mode=LocomotionMode.IDLE)
        time.sleep(DT)
    s = p.obs()
    d = float(np.linalg.norm(p.pelvis_xy(s) - xy0))
    dyaw = np.degrees((float(p.pelvis_yaw(s)) - yaw0 + np.pi) % (2 * np.pi) - np.pi)
    return d, dyaw, bool(s["fallen"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=5.0)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    print(f"\n{'facing':>10} {'rep':>4} {'drift m':>8} {'dyaw deg':>9} {'fallen':>7}")
    for mode in ("measured", "measured+reanchor", "worldx+reanchor"):
        for r in range(a.reps):
            d, dy, f = trial(p, reset, mode, a.secs)
            print(f"{mode:>10} {r + 1:>4} {d:>8.3f} {dy:>9.1f} {str(f):>7}")


if __name__ == "__main__":
    main()
