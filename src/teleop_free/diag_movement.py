"""Which frame does the planner's `movement` vector actually walk in?

facing's frame is measured (diag_heading.py): rotated by the yaw at the last
planner re-anchor, compensated in AgentCommander.frame(). Whether `movement`
shares that frame is an assumption the walk-in loops keep dying on — the
robot has now walked AWAY from the can under two different constructions.

Protocol: re-anchor at a known yaw (streamed toggle), then from stand,
command a few movement directions for 3 s each with facing held at the
current yaw, and compare the measured displacement bearing with the request.

    python src/teleop_free/diag_movement.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent_server import LocomotionMode  # noqa: E402
from primitives import DT, Primitives, wrap  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402
from stream_reach import StreamedArm  # noqa: E402


def main() -> None:
    p = Primitives()
    reset = EpisodeControlClient()
    p.cmd.reset_pose()
    reset.reset(label="diag_movement")
    time.sleep(0.2)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        print("did not stand")
        return

    # Re-anchor exactly the way an episode does, then walk to a nonzero yaw
    # so frame rotations are visible.
    arm = StreamedArm(p, "right")
    arm.start()
    arm.hold(0.5)
    arm.stop()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    p.turn_to(wrap(p.pelvis_yaw() + np.deg2rad(60)), timeout=20.0)
    yaw = p.pelvis_yaw()
    print(f"anchored offset {np.rad2deg(p.cmd.heading_offset):.1f} deg, now facing {np.rad2deg(yaw):.1f} deg")

    for label, ang in (("world +x", 0.0), ("world +y", np.pi / 2), ("own heading", yaw)):
        s = p.obs()
        start = p.pelvis_xy(s)
        yaw_now = p.pelvis_yaw(s)
        mv = (np.cos(ang), np.sin(ang), 0.0)
        end = time.time() + 3.0
        while time.time() < end:
            s = p.obs()
            if s["fallen"]:
                print("fell")
                return
            p.cmd.frame(
                mode=LocomotionMode.SLOW_WALK, speed=0.35, movement=mv, facing=(np.cos(yaw_now), np.sin(yaw_now), 0.0)
            )
            time.sleep(DT)
        stop = time.time() + 1.5
        while time.time() < stop:
            p.cmd.frame(mode=LocomotionMode.IDLE, facing=(np.cos(yaw_now), np.sin(yaw_now), 0.0))
            time.sleep(DT)
        s = p.obs()
        d = p.pelvis_xy(s) - start
        got = np.rad2deg(np.arctan2(d[1], d[0]))
        print(
            f"  {label:12s}: asked {np.rad2deg(ang):7.1f}  moved "
            f"{np.linalg.norm(d) * 100:5.1f} cm at {got:7.1f} deg  "
            f"(err {np.rad2deg(wrap(np.deg2rad(got) - ang)):+6.1f})"
        )

    p.cmd.pause_control()
    reset.close()
    p.close()


if __name__ == "__main__":
    main()
