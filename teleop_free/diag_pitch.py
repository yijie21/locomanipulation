"""Can the palm be commanded to a given nose-down pitch, and how faithfully?

diag_angled_grasp shows that tilting the palm 35 deg nose-down puts the grip
point 6.3 cm below the wrist, which is exactly enough to reach a 0.62 m table
from a standing pelvis and drop the crouch entirely. That is a geometry result;
whether the controller will actually hold that pitch is a separate question,
and the wrist orientation channel has a bad record: commanding yaw was measured
to drag the palm ~25 deg nose-down as a side effect, which is why the grasp
stopped commanding orientation at all.

So sweep the commanded pitch and measure what the palm does. The number that
matters is the elevation of the palm's +x axis, because GRASP_CENTRE_OFFSET
points along it and the whole scheme is that +x tips down.

    python teleop_free/diag_pitch.py [--side right]
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

SETTLE = 3.5
# A plausible standing pre-grasp point: out in front, to the hand's own side,
# a little above the pelvis — where the angled scheme wants the wrist.
HOLD_POS = np.array([0.55, -0.20, 0.00])
PITCHES = [0.0, 0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0]


def pitch_quat(deg: float) -> np.ndarray:
    """Nose-down rotation about y, same sign convention as diag_angled_grasp."""
    a = np.deg2rad(deg) / 2.0

    return np.array([np.cos(a), 0.0, np.sin(a), 0.0], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", default="right", choices=["left", "right"])
    args = ap.parse_args()

    p = Primitives()
    p.cmd.start_control()
    p.cmd.reset_pose()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)

    print(f"\n{'cmd pitch':>10} {'palm +x elevation':>18} {'yaw drift':>10} "
          f"{'roll':>8}   palm +x in the pelvis frame")
    rows = []
    for deg in PITCHES:
        kw = {f"{args.side}_pos": HOLD_POS, f"{args.side}_quat": pitch_quat(deg)}
        p.cmd.set_wrists(**kw)
        end = time.time() + SETTLE
        while time.time() < end:
            p.cmd.frame(mode=LocomotionMode.IDLE)
            time.sleep(DT)

        s = p.obs()
        axes = (Primitives._rot(s["pelvis"]["quat"]).T
                @ Primitives._rot(s["palms"][args.side]["quat"]))
        ax, ay = axes[:, 0], axes[:, 1]
        # Nose-down elevation: how far the fingers point below horizontal.
        elev = float(np.rad2deg(np.arcsin(np.clip(-ax[2], -1.0, 1.0))))
        yaw = float(np.rad2deg(np.arctan2(ax[1], ax[0])))
        roll = float(np.rad2deg(np.arcsin(np.clip(ay[2], -1.0, 1.0))))
        print(f"{deg:9.0f}d {elev:17.1f}d {yaw:9.1f}d {roll:7.1f}d   "
              f"{np.round(ax, 3)}")
        rows.append((deg, elev))
        np.save(f"/tmp/claude-1002/-workspace-code-locomanipulation/6b8c31c0-160b-43ad-9116-8d471dbde4c1/scratchpad/palm_{int(deg)}.npy", axes)

    got = np.array([r[1] for r in rows])
    print(f"\nrange of achieved pitch: {got.min():.1f}d .. {got.max():.1f}d "
          f"({got.max() - got.min():.1f}d of authority)")
    print("  < 10d of authority -> the channel cannot hold a tilt; the angled "
          "grasp has to come from the arm pose, not the orientation command")
    want = 35.0
    if got.max() - got.min() > 10.0:
        i = int(np.argmin(np.abs(got - want)))
        print(f"  closest to the {want:.0f}d target: command "
              f"{rows[i][0]:.0f}d -> {rows[i][1]:.1f}d")

    p.cmd.pause_control()
    p.close()


if __name__ == "__main__":
    main()
