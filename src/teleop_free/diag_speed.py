"""What does the planner's `speed` field actually do?

`navigate` has never set it — every walk has gone out with speed = -1, the
deploy's default — and the `movement` vector's magnitude is normalised away by
the deploy, so the gain the follower computes only ever flipped the WALK /
SLOW_WALK mode. In other words nothing in the stack has been controlling pace.

Walks a straight line at a series of commanded speeds and reports the pelvis
velocity that came out, so the follower can be given a number rather than a
guess.

    python src/teleop_free/diag_speed.py
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
from privileged_pub import EpisodeControlClient  # noqa: E402

LEG = 4.0          # seconds of walking per setting
WARMUP = 1.5       # ignore the first stretch: the gait takes a moment to build


def leg(p, mode, speed, heading, seconds=LEG):
    """Walk one heading at one setting; return (mean speed, distance)."""
    t0 = time.time()
    samples = []
    start = None
    while time.time() - t0 < seconds:
        s = p.obs()
        if s["fallen"]:
            return None, None
        if time.time() - t0 > WARMUP:
            if start is None:
                start = (p.pelvis_xy(s).copy(), time.time())
            samples.append(np.linalg.norm(s["pelvis"]["vel"][:2]))
        p.cmd.frame(mode=mode, speed=speed,
                    movement=(np.cos(heading), np.sin(heading), 0.0),
                    facing=(np.cos(heading), np.sin(heading), 0.0))
        time.sleep(DT)
    end = p.pelvis_xy()
    dist = float(np.linalg.norm(end - start[0]))

    return float(np.mean(samples)), dist / (time.time() - start[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    p = Primitives(verbose=False)
    reset = EpisodeControlClient()
    p.cmd.pause_control()
    time.sleep(0.5)
    p.cmd.reset_pose()
    reset.reset(label="speed")
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    if p.obs()["pelvis"]["pos"][2] < 0.5:
        sys.exit("robot is not standing")

    print(f"{'mode':>10} {'speed cmd':>10} {'mean |v|':>9} {'net m/s':>9}")
    heading = 0.0
    for mode in (LocomotionMode.SLOW_WALK, LocomotionMode.WALK):
        for speed in (-1.0, 0.2, 0.3, 0.5, 0.8):
            v, net = leg(p, mode, speed, heading)
            if v is None:
                print(f"{mode.name:>10} {speed:>10.2f}      FELL")
                break
            print(f"{mode.name:>10} {speed:>10.2f} {v:>9.3f} {net:>9.3f}")
            # Walk back the way it came, so the robot stays near the middle of
            # the room instead of marching into a wall over eleven legs.
            heading = heading + np.pi

    p.cmd.frame(mode=LocomotionMode.IDLE)
    p.cmd.pause_control()
    reset.close()
    p.close()


if __name__ == "__main__":
    main()
