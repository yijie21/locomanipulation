"""Does the wrist *orientation* command do anything, and how much of it arrives?

RESTORED 2026-09-11 (deleted by refactor step 2, ec0eb1c) for the v1.1
re-calibration's stage D. `primitives_tuning.TRACK_ROT_BOOST = 4.5` is the
INVERSE of a measured delivery ratio -- 105 deg asked, 23 achieved, i.e. 22% --
taken on policy/release. A pre-scale calibrated against one checkpoint's
delivery is exactly the class of constant the gait map already caught
(gait_speeds.py), so it has to be re-measured before it can be trusted on
policy/sonic_v1_1.

So this reports two things per commanded quaternion: the palm's axes in the
pelvis frame (the original question -- is the field inert, like compliance and
height turned out to be), and the ANGLE actually delivered against the angle
asked, which is the number the boost inverts.

    python -m locomani.backends.sonic.diag_wrist [--side right] [--json OUT]

Needs a live stack. Measurement only: it commands wrist poses and reads the
measured palm, and changes no production constant.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from locomani import paths
from locomani.backends.sonic.primitives import DT, LocomotionMode, Primitives

SETTLE = 3.5  # the arm needs a couple of seconds to reach a new target
HOLD_POS = np.array([0.35, -0.22, -0.05])  # a plausible pre-grasp wrist point


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
    ap.add_argument("--json", default=None, help="write the per-case numbers here")
    args = ap.parse_args()

    p = Primitives()
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)

    print(f"\npolicy {paths.SONIC_POLICY}")
    print(f"\n{'case':38s} asked / delivered / ratio, then the palm axes in the pelvis frame")
    results, rows, measured_frames, base = [], [], [], None
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
        # How much of the commanded rotation arrived. The reference is the
        # FIRST case (identity): the delivered rotation is the palm attitude
        # relative to that, and the asked rotation is the commanded quaternion
        # itself, both reduced to an angle.
        if not measured_frames:
            base = axes.copy()
        delivered = float(np.degrees(np.arccos(np.clip((np.trace(base.T @ axes) - 1) / 2, -1, 1))))
        asked = float(np.degrees(2 * np.arccos(np.clip(abs(float(quat[0])), -1, 1))))
        ratio = delivered / asked if asked > 1e-6 else float("nan")
        measured_frames.append(axes)
        print(
            f"{label:38s} asked {asked:5.1f} deg  delivered {delivered:5.1f} deg"
            f"  ratio {ratio:5.2f}   x {str(np.round(axis_x, 2)):18s} pos {np.round(local, 3)}"
        )
        rows.append(
            dict(
                case=label,
                asked_deg=asked,
                delivered_deg=delivered,
                ratio=ratio,
                axis_x=[float(v) for v in axis_x],
                axis_z=[float(v) for v in axis_z],
                palm_root=[float(v) for v in local],
            )
        )
        results.append(axis_x)

    spread = float(np.max(np.linalg.norm(np.array(results) - np.array(results)[0], axis=1)))
    print(f"\nlargest change in the palm's +x axis across all commands: {spread:.3f}")
    print(
        "  < 0.05  -> the quaternion field is inert, like compliance and "
        "height; the offset has to be calibrated from the measured pose "
        "instead"
    )
    print("  > 0.30  -> it is live and the grasp should command a palm-down orientation")

    live = [r for r in rows if r["asked_deg"] > 1e-6]
    if live:
        med = float(np.median([r["ratio"] for r in live]))
        print(
            f"\nmedian delivery ratio {med:.2f}  ->  a pre-scale that inverts it would be {1 / med:.1f}"
            f"   (primitives_tuning.TRACK_ROT_BOOST is 4.5, measured on policy/release)"
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(dict(policy=paths.SONIC_POLICY, side=args.side, rows=rows), fh, indent=1)
        print(f"wrote {args.json}")
    p.cmd.pause_control()
    p.close()


if __name__ == "__main__":
    main()
