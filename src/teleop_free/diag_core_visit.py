"""Wrist visits every dark-core (sweet-zone) voxel at standing height.

Each core voxel's own exemplar joints from the reach map are replayed
through the actuator pin (no IK) — the wrist rasters the dark core the
same way diag_reach_volume rastered the full envelope, but only at the
initial standing height. A short stepped squat at the end shows the core
riding down without re-visiting.

Usage: python src/teleop_free/diag_core_visit.py   (stack must be up)
Output: a 'core_visit' video in SIM_VIDEO_DIR.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stream_reach  # noqa: E402
from g1_joints import ARM_MJ  # noqa: E402
from primitives import LocomotionMode, Primitives, StreamMode  # noqa: E402
from privileged_pub import EpisodeControlClient  # noqa: E402

SCENE_META = "/workspace/code/locomanipulation/outputs/scene/sim/scene_meta.json"
STEPS = (0.05, 0.04, 0.04)
DWELL = 0.16                        # s per voxel (interpolated transit)


def main() -> None:
    import reach_map
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()

    z = np.load(reach_map.MAP_PATH)
    sweet = z["sweet_right"]
    ex_q = z["ex_q_right"]
    ex_wp = z["ex_wp_right"]
    targets = []
    nx, ny, nz = sweet.shape
    for k in range(nz):
        for j in range(ny):
            xs = range(nx)
            if j % 2:
                xs = reversed(list(xs))
            for i in xs:
                if sweet[i, j, k]:
                    targets.append((i, j, k))
    print(f"[core] visiting all {len(targets)} core voxels at standing")

    p.cmd.reset_pose()
    reset.pin_arms()
    reset.reset()
    time.sleep(0.3)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    if p.obs()["fallen"]:
        print("robot did not stand"); return
    stream_reach.reanchor(p, side="right")
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    p.tuck_arms()
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    reset.start_video("core_visit")

    arm = stream_reach.StreamedArm(p, "right")
    arm.start()
    try:
        arm.hold(1.5)
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
        print(f"standing: visited {n} core voxels")
        reset.pin_arms()
        total = 0.0
        for d in STEPS:
            total += d
            arm.squat(d)
            arm.hold(2.5)
        print(f"squatted to {total * 100:.0f} cm cmd (no re-visit)")
        arm.hold(1.5)
    finally:
        reset.pin_arms()
        arm.stop()
        reset.end_video()
    print("done")


if __name__ == "__main__":
    main()
