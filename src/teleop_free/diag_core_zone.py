"""Show the two-tone reach cloud: pale envelope + dark sweet-zone core.

Stand still so the standing volume reads clearly, then squat in steps to
the delivered limit, pausing at each depth — the dark core is where a
grasp is COMFORTABLE (servo correction room), the pale shell where it is
merely possible.

Usage: python src/teleop_free/diag_core_zone.py   (stack must be up)
Output: a 'core_zone' video in SIM_VIDEO_DIR.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
import stream_reach  # noqa: E402
from primitives import LocomotionMode, Primitives, StreamMode  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"
STEPS = (0.05, 0.04, 0.04)


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
    if p.obs()["fallen"]:
        print("robot did not stand")
        return
    stream_reach.reanchor(p, side="right")
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    p.tuck_arms()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    reset.start_video("core_zone")

    arm = stream_reach.StreamedArm(p, "right")
    arm.start()
    try:
        z0 = p.obs()["pelvis"]["pos"][2]
        arm.hold(4.0)
        total = 0.0
        for d in STEPS:
            total += d
            arm.squat(d)
            arm.hold(4.0)
            zz = p.obs()["pelvis"]["pos"][2]
            print(f"squat cmd {total * 100:.0f} cm: pelvis -{(z0 - zz) * 100:.1f} cm")
        arm.hold(2.0)
    finally:
        reset.pin_arms()
        arm.stop()
        reset.end_video()
    print("done")


if __name__ == "__main__":
    main()
