"""Show the 3D graspable volume riding down through a stepped squat.

The scene carries the full right-arm reach-map voxel cloud glued to the
pelvis (run_sim_locomani.inject_reach_zone). This demo: stand still so the
standing volume reads clearly, then squat in steps to the delivered limit
(~13 cm), pausing at each depth — the cloud sinks with the body, and the
space it sweeps is the total graspable region from standing to full crouch.

Usage: python src/teleop_free/diag_reach_volume.py   (stack must be up)
Output: a 'reach_volume' video in SIM_VIDEO_DIR.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stream_reach  # noqa: E402
from primitives import LocomotionMode, Primitives, StreamMode  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = "/workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json"
STEPS = (0.04, 0.04, 0.03, 0.02)     # incremental drops -> 4/8/11/13 cm


def main() -> None:
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()

    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset()
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        print("robot did not stand"); return
    stream_reach.reanchor(p, side="right")
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    p.tuck_arms()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    reset.start_video("reach_volume")

    arm = stream_reach.StreamedArm(p, "right")
    arm.start()
    try:
        z0 = p.obs()["pelvis"]["pos"][2]
        arm.hold(3.0)                       # standing volume, clearly
        total = 0.0
        for d in STEPS:
            total += d
            got = arm.squat(d)
            arm.hold(2.0)
            z = p.obs()["pelvis"]["pos"][2]
            print(f"asked {total * 100:.0f} cm cumulative, pelvis "
                  f"dropped {(z0 - z) * 100:.1f} cm")
        arm.hold(2.0)
    finally:
        arm.stop()
        reset.end_video()
    print("done")


if __name__ == "__main__":
    main()
