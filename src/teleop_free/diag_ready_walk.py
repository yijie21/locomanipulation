"""Walk-only demo of the READY pose: forearms level, no grasping.

Resets the scene, pins the upper body with PLANNER_FROZEN_UPPER_BODY, and
walks a few legs back and forth in the open area while the sim records
video. Exists to answer one question by eye: does the DELIVERED walking pose
actually hold the forearms level, and how far off is it from the command.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import stream_reach  # noqa: E402
from primitives import (  # noqa: E402
    READY_UPPER,
    LocomotionMode,
    Primitives,
    StreamMode,
)
from privileged_pub import EpisodeControlClient
import paths

SCENE_META = f"{paths.REPO}/outputs/scene/sim/scene_meta.json"


def main() -> None:
    meta = json.load(open(SCENE_META))
    p = Primitives(meta=meta)
    reset = EpisodeControlClient()

    p.cmd.reset_pose()
    reset.end_video()
    reset.reset()
    time.sleep(0.2)
    p.cmd.start_control()
    p.cmd.hold(4.0, mode=LocomotionMode.IDLE)
    stream_reach.reanchor(p)

    # Pin the walking pose twice over: the frozen-upper-body reference keeps
    # the POLICY expecting the arms there (so its balance model is roughly
    # honest), and the sim-side actuator pin makes the pose exact — the
    # policy alone tops out at elbow ~1.5 rad and the forearms sag 15-30 deg.
    # Elbow ZERO is the level forearm: the G1's joint zero pose is already
    # the L-shape (upper arm vertical, forearm horizontal), and POSITIVE
    # elbow angle extends the arm downward — 1.5 rad is hanging straight.
    # Every earlier attempt commanded 1.5-2.09 believing bigger = more fold,
    # i.e. commanded the arms DOWN and then fought the result.
    from primitives import READY_ARM
    PIN = READY_ARM          # the exact pose collect_episode pins
    p.cmd.stream_mode = StreamMode.PLANNER_FROZEN_UPPER_BODY
    from primitives import upper_body_17
    p.cmd.set_upper_body(upper_body_17(PIN["left"], PIN["right"]))
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    reset.pin_arms(left=PIN["left"], right=PIN["right"])
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)

    reset.start_video(f"readywalk_{int(time.time())}")
    home = p.pelvis_xy().copy()
    legs = [home + np.array([-1.6, 0.0]),
            home + np.array([-1.6, 1.2]),
            home + np.array([0.0, 1.2]),
            home]
    from primitives import wrap
    for i, wp in enumerate(legs):
        end = time.time() + 25.0
        while time.time() < end:
            s = p.obs()
            d = wp - p.pelvis_xy(s)
            if np.linalg.norm(d) < 0.22:
                break
            yaw = p.pelvis_yaw(s)
            b = float(np.arctan2(d[1], d[0]))
            face = yaw + float(np.clip(wrap(b - yaw), -0.5, 0.5))
            fv = (np.cos(face), np.sin(face), 0.0)
            p.cmd.frame(mode=LocomotionMode.SLOW_WALK, speed=0.6,
                        movement=fv, facing=fv)
            time.sleep(0.02)
        print(f"leg {i}: at {np.round(p.pelvis_xy(), 2)}")
        p.cmd.hold(0.5, mode=LocomotionMode.IDLE)

    # Log the delivered arm pose vs the command, so the video has numbers.
    s = p.obs()
    q = np.array(s["joints"])
    for side, base in (("left", 15), ("right", 22)):
        print(f"{side}: sh_pitch {q[base]:+.2f} sh_roll {q[base + 1]:+.2f} "
              f"elbow {q[base + 3]:+.2f} (cmd -0.05/±0.20/2.05)")
    p.cmd.hold(1.0, mode=LocomotionMode.IDLE)
    reset.end_video()
    print("done")


if __name__ == "__main__":
    main()
