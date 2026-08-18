"""How accurately does a wrist target actually get tracked, and how far can it go?

Two unknowns block the grasp stage, and both are cheaper to measure than to
reason about:

  * `vr_3pt_compliance` defaults to (0.5, 0.5, 0.0). Whether a larger number
    means "track harder" or "give way more" decides whether raising it helps.
    The neck entry being 0.0 hints at "track harder", but a hint is not a
    measurement.
  * Where the arm can actually put the palm, in pelvis frame. The first grasp
    attempt timed out 0.246 m short of a target 0.70 m ahead, which is either a
    tracking-gain problem or simply out of reach — and those have opposite fixes.

Sweeps a grid of root-frame targets at each compliance and reports the error.

    python src/teleop_free/diag_reach.py [--side right]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import AgentCommander, LocomotionMode  # noqa: E402
from privileged_pub import EpisodeControlClient, PrivilegedSubscriber  # noqa: E402

# Pelvis-frame targets, chosen to bracket where a table-top object sits: out in
# front, at roughly pelvis height, at a few distances and lateral offsets.
# Commanded wrist targets, pelvis frame. The question the grasp stage needs
# answered is how low in front of the body the palm can actually go, so the
# sweep is mostly in z.
TARGETS = [
    (0.60, -0.25, 0.10),
    (0.60, -0.25, 0.00),
    (0.60, -0.25, -0.10),
    (0.60, -0.25, -0.20),
    (0.60, -0.25, -0.35),
    (0.75, -0.25, -0.20),
    (0.90, -0.25, -0.20),
]
COMPLIANCES = [0.5]
# Stance heights to try. -1 is the planner's default (full stand); lower values
# squat, which is the only way to get the palm down to a 0.75 m table when the
# palm cannot go below roughly pelvis height in front of the body.
HEIGHTS = [-1.0, 0.70, 0.60, 0.50]


def root_to_world(s: dict, local) -> np.ndarray:
    root = np.array(s["pelvis"]["pos"])
    yaw = float(s["pelvis"]["yaw"])
    c, sn = np.cos(yaw), np.sin(yaw)
    p = np.asarray(local, dtype=float)

    return root + np.array([c * p[0] - sn * p[1], sn * p[0] + c * p[1], p[2]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    ap.add_argument("--hold", type=float, default=2.5)
    args = ap.parse_args()

    cmd = AgentCommander()
    state = PrivilegedSubscriber()
    EpisodeControlClient().reset(label="diag_reach")
    time.sleep(0.2)
    cmd.start_control()
    cmd.hold(4.0, mode=LocomotionMode.IDLE)

    print(f"\n{'height':>7} {'cmd (pelvis frame)':>22} {'palm (pelvis frame)':>26} "
          f"{'pelvis z':>9}")
    for comp in [HEIGHTS[0]] and HEIGHTS:
        for tgt in TARGETS:
            # The target is already in root frame, which is the frame the wire
            # format expects, so it is a constant for the whole hold.
            local = np.asarray(tgt, dtype=float)
            if args.side == "right":
                cmd.set_wrists(right_pos=local)
            else:
                cmd.set_wrists(left_pos=local)
            deadline = time.time() + args.hold
            while time.time() < deadline:
                cmd.frame(mode=LocomotionMode.IDLE, height=comp)
                time.sleep(0.02)

            s = state.wait()
            palm_world = np.array(s["palms"][args.side]["pos"])
            root = np.array(s["pelvis"]["pos"])
            yaw = float(s["pelvis"]["yaw"])
            c, sn = np.cos(-yaw), np.sin(-yaw)
            d = palm_world - root
            palm_local = np.array([c * d[0] - sn * d[1], sn * d[0] + c * d[1], d[2]])
            print(f"{comp:>7.2f} {str(np.round(tgt, 2)):>22} "
                  f"{str(np.round(palm_local, 3)):>26} {root[2]:>9.3f}")

    cmd.pause_control()
    cmd.close()
    state.close()


if __name__ == "__main__":
    main()
