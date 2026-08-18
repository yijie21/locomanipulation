"""Can the robot stand up from a reset? Watch pelvis height, second by second.

Isolates the reset/startup transient from everything else in an episode: with
the elastic band disabled nothing holds the robot up between `mj_resetData`
restoring the nominal pose and the controller taking authority, and if that
window is fatal then every episode dies before it starts.

    python src/teleop_free/diag_stand.py [--reset] [--seconds 8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import AgentCommander, LocomotionMode  # noqa: E402
from privileged_pub import EpisodeControlClient, PrivilegedSubscriber  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--gap", type=float, default=0.2,
                    help="seconds between the reset and start_control")
    args = ap.parse_args()

    cmd = AgentCommander()
    state = PrivilegedSubscriber()

    if args.reset:
        ctl = EpisodeControlClient()
        ctl.reset(label="diag_stand")
        print(f"reset sent; waiting {args.gap:.2f}s before start_control")
        time.sleep(args.gap)

    cmd.start_control()
    t0 = time.time()
    next_sample = t0
    print("   t     pelvis xyz                  fallen")
    while time.time() - t0 < args.seconds:
        cmd.frame(mode=LocomotionMode.IDLE)
        if time.time() >= next_sample:
            s = state.latest(0)
            if s:
                p = s["pelvis"]["pos"]
                print(f"  {time.time() - t0:4.1f}  "
                      f"[{p[0]:+.3f} {p[1]:+.3f} {p[2]:+.3f}]   {s['fallen']}")
            next_sample += 0.5
        time.sleep(0.02)

    cmd.pause_control()
    cmd.close()
    state.close()


if __name__ == "__main__":
    main()
