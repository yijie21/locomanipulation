"""Sweep the right palm across the body at GRASP height, on camera.

The green reach-zone carpet claims to be 'where a grasp can happen'; this
drives the ACTUAL arm (streamed reference + actuator pin, exact delivery)
through fans of world targets at the same height and lets the video show
how the palm's real envelope compares to the carpet. Targets that IK cannot
solve within 2 cm are skipped — the palm simply does not go there, so the
reachable boundary is visible as where the hand turns back.

Usage: python src/teleop_free/diag_reach_sweep.py   (stack must be up)
Output: a 'reach_sweep' video in SIM_VIDEO_DIR via the episode server.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adaptive_grasp as ag  # noqa: E402
import paths
import stream_reach  # noqa: E402
from g1_joints import ARM_MJ  # noqa: E402
from primitives import LocomotionMode, Primitives, StreamMode  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"
GRIP_Z = 0.894  # same absolute height as the green carpet
# (the READY/elbow level, where the carried arm lives)
RADII = (0.50, 0.42, 0.34, 0.26, 0.20)
AZ_MAX = np.deg2rad(70.0)


def main() -> None:
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()

    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset()  # nominal spawn: ~0.9 m clear of the table
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    s = p.obs()
    if s["fallen"]:
        print("robot did not stand")
        return
    stream_reach.reanchor(p, side="right")
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    p.tuck_arms()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    reset.start_video("reach_sweep")

    arm = stream_reach.StreamedArm(p, "right")
    arm.start()
    try:
        yaw = p.pelvis_yaw(s)
        c, sn = np.cos(yaw), np.sin(yaw)
        px, py = p.pelvis_xy(s)
        reached = skipped = 0
        for r in RADII:
            azs = np.arange(-AZ_MAX, AZ_MAX + 1e-6, np.deg2rad(6.0))
            if r in (RADII[1], RADII[3]):
                azs = azs[::-1]  # serpentine: smooth camera motion
            for az in azs:
                # pelvis-frame target -> world
                tx, ty = r * np.cos(az), r * np.sin(az)
                wx, wy = px + c * tx - sn * ty, py + sn * tx + c * ty
                aim_pf = p.world_to_root_full(np.array([wx, wy, GRIP_Z]), p.obs())
                # CONSTRAINED orientation: a level grasp pose whose approach
                # axis points outward along the sweep azimuth — the same
                # semantics as the reach map. rot=None showed the position-
                # only envelope with the wrist flipping freely at the rim.
                dir_pf = np.array([np.cos(az), np.sin(az), 0.0])
                rot_pf = ag._align(ag.OFFSET_DIR, dir_pf)
                q_sol, res, _ = arm.ik.solve("right", aim_pf, rot_pf, q_mj29=arm.q, use_waist=True, iters=20)
                if res > 0.02:
                    skipped += 1
                    continue
                arm.q = q_sol
                arm.move_to(q_sol, speed=2.0)
                reset.pin_arms(right=[float(v) for v in q_sol[ARM_MJ["right"]]])
                arm.hold(0.12)
                reached += 1
            print(f"radius {r:.2f}: cumulative reached {reached}, skipped {skipped}")
        arm.hold(1.0)
    finally:
        reset.pin_arms()
        arm.stop()
        reset.end_video()
    print(f"sweep done: {reached} targets reached, {skipped} unreachable")


if __name__ == "__main__":
    main()
