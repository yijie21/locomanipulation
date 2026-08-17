"""How low will the controller take the pelvis, and what makes it?

The grasp needs the grip centre above the pelvis — the arm cannot put it any
lower (teleop_free/diag_reach_map.py). That fixes a floor under the table
height *only if the pelvis height is fixed*, and it is not: the whole-body
controller squats to follow a low wrist target. Earlier notes put the floor at
0.65 m, but every reach trace so far shows it dropping only 0.784 -> 0.764,
so whatever drives the squat is not being driven.

Which matters because it is the one thing that lets a low table work. If the
pelvis can be brought to 0.70, a 0.62 m table is equivalent to a 0.70 m table
at the nominal stance, which the reachability map scores 7 of 12 rather than 0.

Commands a descending series of wrist targets, holds each long enough to
settle, and records the pelvis height and the palm height that came with it.

    python teleop_free/diag_squat.py [--side right]
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

SETTLE = 5.0        # the squat is slow; earlier sweeps used 2.5 and saw little
FORWARD = 0.30      # keep the target near the body, where the arm has travel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    args = ap.parse_args()

    p = Primitives()
    # Reset first. The sim starts at qpos0 — straight legs — and with the
    # elastic band disabled the robot is flat on the floor within a second, so
    # a diagnostic that skips this measures a fallen robot and reports a pelvis
    # at 0.137 m as if that were a squat.
    reset = EpisodeControlClient()
    # Stop control before resetting. If a previous run left the deploy driving
    # a fallen robot, teleporting it out from under an active controller
    # leaves the state estimator somewhere else entirely and it never stands
    # up again — which is what "robot is not standing" looked like on the
    # second attempt, with a perfectly healthy stack.
    p.cmd.pause_control()
    time.sleep(0.5)
    p.cmd.reset_pose()
    reset.reset(label="squat")
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    base = p.obs()["pelvis"]["pos"][2]
    if base < 0.5:
        sys.exit(f"robot is not standing (pelvis {base:.3f} m) — nothing below "
                 f"would mean anything")
    print(f"pelvis at rest: {base:.3f} m\n")
    print(f"{'cmd z':>7} {'height':>8} {'pelvis':>8} {'palm z':>8} "
          f"{'palm world':>11}  (all metres)")

    # Both levers at once: the wrist target the controller bends for, and the
    # planner's `height` field, which an earlier sweep found inert — worth
    # re-testing here because it was only ever tried with the arm at rest.
    for cmd_z, height in [(-0.10, -1.0), (-0.25, -1.0), (-0.40, -1.0),
                          (-0.55, -1.0), (-0.55, 0.70), (-0.55, 0.60),
                          (-0.55, 0.50)]:
        target = np.array([FORWARD, -0.20 if args.side == "right" else 0.20,
                           cmd_z])
        end = time.time() + SETTLE
        while time.time() < end:
            p.cmd.set_wrists(**{f"{args.side}_pos": target})
            p.cmd.frame(mode=LocomotionMode.IDLE, height=height)
            time.sleep(DT)
        s = p.obs()
        pelvis = s["pelvis"]["pos"][2]
        palm = np.array(s["palms"][args.side]["pos"])
        local = p.world_to_root_full(palm, s)
        print(f"{cmd_z:7.2f} {height:8.2f} {pelvis:8.3f} {local[2]:8.3f} "
              f"{palm[2]:11.3f}")

    s = p.obs()
    print(f"\nlowest pelvis seen: {s['pelvis']['pos'][2]:.3f} m "
          f"({(s['pelvis']['pos'][2] - base) * 100:+.1f} cm from rest)")
    print("A pelvis at 0.70 would make a 0.62 m table as reachable as a 0.70 m "
          "one is now; at 0.76 it stays out of range whatever the arm does.")
    p.cmd.pause_control()
    reset.close()
    p.close()


if __name__ == "__main__":
    main()
