"""Live Dex1 smoke: command open/close through the full chain, read back.

Chain under test: commander set_hands (7-dim wire) -> deploy pass-through
-> DDS -> sim bridge (NUM_HAND_MOTORS=2) -> gripper PD (gear=150) ->
privileged hand_joints readback (2 values/side).

Usage: HAND=dex1 python src/teleop_free/diag_dex1.py   (stack up, HAND=dex1)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paths
from agent_server import HAND_CLOSED, HAND_OPEN  # noqa: E402
from primitives import LocomotionMode, Primitives  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"


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
        print("robot did not stand")
        return

    for label, cmd in (("OPEN", HAND_OPEN), ("CLOSED", HAND_CLOSED), ("OPEN", HAND_OPEN)):
        p.cmd.set_hands(left=cmd, right=cmd)
        p.cmd.hold(2.0, mode=LocomotionMode.IDLE)
        s2 = p.obs()
        print(f"cmd {label:6s} ({cmd[0]:+.4f}) delivered q -> ", end="")
        for sd in ("left", "right"):
            v = s2["palms"][sd]["hand_q"]
            print(f"{sd}: {np.round(v, 4).tolist()}  ", end="")
        print()
    print("done")


if __name__ == "__main__":
    main()
