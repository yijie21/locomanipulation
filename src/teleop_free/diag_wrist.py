"""Does the wrist *orientation* command do anything?

Two of the 18-D interface's fields have already turned out to be inert on this
deploy — `vr_3pt_compliance` (0.5 and 1.0 give identical palm positions to the
millimetre) and the planner's `height` (the pelvis stays at 0.78 across every
value). The wrist quaternion has never been commanded at all: `set_wrists`
accepts one, nothing has ever passed one, and the palm's orientation has been
whatever the controller picked.

That matters now because the grasp closes 12-14 cm to one side of the object in
the *palm's own frame*, consistently, across a whole batch — the signature of a
palm pointing somewhere other than where GRASP_CENTRE_OFFSET assumes. The fix
is to command the orientation. But there is no point designing around a field
that turns out to be ignored, so: command a few, measure what the palm does.

    python src/teleop_free/diag_wrist.py [--side right]

Prints the commanded quaternion, the measured palm quaternion, and the palm's
own axes in the pelvis frame — the last is what actually matters, because
GRASP_CENTRE_OFFSET points along the palm's +x.
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

SETTLE = 3.5           # the arm needs a couple of seconds to reach a new target
HOLD_POS = np.array([0.35, -0.22, -0.05])   # a plausible pre-grasp wrist point


def quat_of_axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2)

    return np.array([np.cos(angle / 2), *(axis * s)], dtype=np.float32)


CASES = [
    ("identity (what every reach has sent)", np.array([1.0, 0, 0, 0])),
    ("palm down: -90 deg about y", quat_of_axis_angle([0, 1, 0], -np.pi / 2)),
    ("palm down: +90 deg about y", quat_of_axis_angle([0, 1, 0], np.pi / 2)),
    ("thumb up: -90 deg about x", quat_of_axis_angle([1, 0, 0], -np.pi / 2)),
    ("yawed: +45 deg about z", quat_of_axis_angle([0, 0, 1], np.pi / 4)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    args = ap.parse_args()

    p = Primitives()
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)

    print(f"\n{'case':38s} palm axes in the pelvis frame")
    results = []
    for label, quat in CASES:
        kw = {f"{args.side}_pos": HOLD_POS, f"{args.side}_quat": quat}
        p.cmd.set_wrists(**kw)
        end = time.time() + SETTLE
        while time.time() < end:
            p.cmd.frame(mode=LocomotionMode.IDLE)
            time.sleep(DT)

        s = p.obs()
        palm = s["palms"][args.side]
        # The palm's +x axis expressed in the pelvis frame: the direction
        # GRASP_CENTRE_OFFSET actually points once it is applied.
        rot_pelvis = Primitives._rot(s["pelvis"]["quat"])
        rot_palm = Primitives._rot(palm["quat"])
        # All three palm axes, in the pelvis frame. +x is where the grip
        # offset points; z matters just as much, because the hand hangs 5.6 cm
        # below the grip centre along it (diag_hand.py) and that is what hits
        # the table on a horizontal approach.
        axes = rot_pelvis.T @ rot_palm
        axis_x, axis_y, axis_z = axes[:, 0], axes[:, 1], axes[:, 2]
        local = p.world_to_root_full(palm["pos"], s)
        print(f"{label:38s} x {str(np.round(axis_x, 2)):20s} "
              f"y {str(np.round(axis_y, 2)):20s} z {str(np.round(axis_z, 2)):20s} "
              f"pos {np.round(local, 3)}")
        results.append(axis_x)

    spread = float(np.max(np.linalg.norm(
        np.array(results) - np.array(results)[0], axis=1)))
    print(f"\nlargest change in the palm's +x axis across all commands: "
          f"{spread:.3f}")
    print("  < 0.05  -> the quaternion field is inert, like compliance and "
          "height; the offset has to be calibrated from the measured pose "
          "instead")
    print("  > 0.30  -> it is live and the grasp should command a palm-down "
          "orientation")

    p.cmd.pause_control()
    p.close()


if __name__ == "__main__":
    main()
