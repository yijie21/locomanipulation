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

import adaptive_grasp as ag  # noqa: E402
import stream_reach  # noqa: E402
from g1_joints import ARM_MJ  # noqa: E402
from primitives import LocomotionMode, Primitives, StreamMode  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = "/workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json"
STEPS = (0.05, 0.04, 0.04)          # incremental drops -> ~5/9/13 cm cmd
RADII = (0.44, 0.30)                # swept at every depth
AZ_STEP = np.deg2rad(9.0)
AZ_MAX = np.deg2rad(70.0)
ELBOW_ABOVE_PELVIS = 0.101          # READY palm height over the pelvis (FK)


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

    def sweep():
        """Level-jaw fan sweep at the CURRENT elbow plane."""
        s2 = p.obs()
        zt = float(s2["pelvis"]["pos"][2]) + ELBOW_ABOVE_PELVIS
        yaw = p.pelvis_yaw(s2)
        c, sn = np.cos(yaw), np.sin(yaw)
        px, py = p.pelvis_xy(s2)
        hit = miss = 0
        for i, r in enumerate(RADII):
            azs = np.arange(-AZ_MAX, AZ_MAX + 1e-6, AZ_STEP)
            if i % 2:
                azs = azs[::-1]
            for az in azs:
                tx, ty = r * np.cos(az), r * np.sin(az)
                aim_pf = p.world_to_root_full(
                    np.array([px + c * tx - sn * ty,
                              py + sn * tx + c * ty, zt]), p.obs())
                dir_pf = np.array([np.cos(az), np.sin(az), 0.0])
                rot_pf = ag._align(ag.OFFSET_DIR, dir_pf)
                q_sol, res, _ = arm.ik.solve("right", aim_pf, rot_pf,
                                             q_mj29=arm.q, use_waist=True,
                                             iters=20)
                if res > 0.02:
                    miss += 1
                    continue
                arm.q = q_sol
                arm.move_to(q_sol, speed=2.2)
                reset.pin_arms(
                    right=[float(v) for v in q_sol[ARM_MJ["right"]]])
                arm.hold(0.1)
                hit += 1
        return hit, miss, zt

    try:
        z0 = p.obs()["pelvis"]["pos"][2]
        arm.hold(1.5)
        h, m, zt = sweep()
        print(f"standing (plane z={zt:.3f}): reached {h}, missed {m}")
        total = 0.0
        for d in STEPS:
            total += d
            reset.pin_arms()                # release before the leg move
            arm.squat(d)
            arm.hold(1.5)
            z = p.obs()["pelvis"]["pos"][2]
            h, m, zt = sweep()
            print(f"squat cmd {total * 100:.0f} cm (pelvis "
                  f"-{(z0 - z) * 100:.1f} cm, plane z={zt:.3f}): "
                  f"reached {h}, missed {m}")
        arm.hold(1.5)
    finally:
        reset.pin_arms()
        arm.stop()
        reset.end_video()
    print("done")


if __name__ == "__main__":
    main()
