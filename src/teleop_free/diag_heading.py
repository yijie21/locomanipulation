"""Measure the planner-mode heading offset, before and after a streamed toggle.

Hypothesis: when the deploy re-initialises the planner (boot, or any
STREAMED_MOTION -> PLANNER switch), it anchors its heading machinery at the
robot's CURRENT yaw psi0, after which a commanded facing phi is executed as
roughly phi + psi0. At boot psi0 ~ 0 so nothing shows; after a streamed
session ends at some other yaw, every facing/movement command is rotated and
navigation walks confidently in the wrong direction.

    python src/teleop_free/diag_heading.py
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


def settle_facing(p, phi, seconds=6.0):
    end = time.time() + seconds
    while time.time() < end:
        s = p.obs()
        if s["fallen"]:
            return None
        p.cmd.frame(mode=LocomotionMode.IDLE, facing=(np.cos(phi), np.sin(phi), 0.0))
        time.sleep(DT)

    return p.pelvis_yaw(p.obs())


def probe(p, label):
    yaw0 = p.pelvis_yaw(p.obs())
    offs = []
    for dphi in (0.0, np.deg2rad(40), np.deg2rad(-40)):
        phi = wrap(yaw0 + dphi)
        got = settle_facing(p, phi)
        if got is None:
            print(f"  {label}: fell during probe")
            return
        off = np.rad2deg(wrap(got - phi))
        offs.append(off)
        print(f"  {label}: commanded {np.rad2deg(phi):7.1f}  settled {np.rad2deg(got):7.1f}  offset {off:+6.1f} deg")

    return offs


def main() -> None:
    from stream_reach import StreamedArm

    p = Primitives()
    reset = EpisodeControlClient()
    p.cmd.reset_pose()
    reset.reset(label="diag_heading")
    time.sleep(0.2)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        print("did not stand")
        return
    print(f"spawn yaw {np.rad2deg(p.pelvis_yaw(s)):.1f} deg")

    probe(p, "before toggle")

    # Turn to a distinctly nonzero yaw, then toggle streamed mode there.
    target = wrap(p.pelvis_yaw(p.obs()) + np.deg2rad(70))
    p.turn_to(target, timeout=25.0)
    psi0 = p.pelvis_yaw(p.obs())
    print(f"toggling streamed mode at yaw {np.rad2deg(psi0):.1f} deg")
    arm = StreamedArm(p, "right")
    arm.start()
    arm.hold(1.5)
    arm.stop()
    p.cmd.hold(1.5, mode=LocomotionMode.IDLE)

    probe(p, "after toggle ")
    print(f"(hypothesis predicts offset ~ {np.rad2deg(wrap(psi0)):+.1f} deg or its negative)")

    p.cmd.pause_control()
    reset.close()
    p.close()


if __name__ == "__main__":
    main()
