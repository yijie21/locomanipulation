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
STRIDE_XY = 2                       # visit every 2nd voxel in x and y
                                    # (all z layers): 402 of 1446 targets
                                    # per depth — the full set at 4 depths
                                    # would be a 40-minute, 300 MB video
DWELL = 0.16                        # s per voxel (interpolated transit)


def main() -> None:
    import reach_map
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()

    z = np.load(reach_map.MAP_PATH)
    occ = z["right"]
    ex_q = z["ex_q_right"]
    ex_wp = z["ex_wp_right"]
    # Serpentine order: z-layer by z-layer, snake through y so neighbouring
    # targets are close and the interpolated transit stays smooth.
    targets = []
    nx, ny, nz = occ.shape
    for k in range(nz):
        for j in range(0, ny, STRIDE_XY):
            xs = range(0, nx, STRIDE_XY)
            if (j // STRIDE_XY) % 2:
                xs = reversed(list(xs))
            for i in xs:
                if occ[i, j, k]:
                    targets.append((i, j, k))
    print(f"[vol] visiting {len(targets)} of {int(occ.sum())} voxels "
          f"per depth")

    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset()
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    s0 = p.obs()
    if s0["fallen"]:
        print("robot did not stand"); return
    stream_reach.reanchor(p, side="right")
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    p.tuck_arms()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    reset.start_video("reach_volume")

    arm = stream_reach.StreamedArm(p, "right")
    arm.start()

    def visit_all():
        """Replay every (subsampled) voxel's own exemplar joints — the
        map's evidence, executed by the actuator pin, no IK involved."""
        n = 0
        DT = 0.02
        steps = max(2, int(round(DWELL / DT)))
        for (i, j, k) in targets:
            q_t = arm.q.copy()
            q_t[14] = float(ex_wp[i, j, k])
            q_t[ARM_MJ["right"]] = ex_q[i, j, k]
            q0 = arm.q.copy()
            for t in range(1, steps + 1):
                u = t / steps
                arm._push(q0 + u * (q_t - q0))
            arm.q = q_t
            reset.pin_arms(right=[float(v) for v in q_t[ARM_MJ["right"]]])
            n += 1
        return n

    try:
        z0 = p.obs()["pelvis"]["pos"][2]
        arm.hold(1.5)
        n = visit_all()
        print(f"standing: visited {n} voxels")
        total = 0.0
        for d in STEPS:
            total += d
            reset.pin_arms()
            arm.squat(d)
            arm.hold(1.5)
            zz = p.obs()["pelvis"]["pos"][2]
            n = visit_all()
            print(f"squat cmd {total * 100:.0f} cm "
                  f"(pelvis -{(z0 - zz) * 100:.1f} cm): visited {n}")
        arm.hold(1.5)
    finally:
        reset.pin_arms()
        arm.stop()
        reset.end_video()
    print("done")


if __name__ == "__main__":
    main()
