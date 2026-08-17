"""Third-person video of what the robot is actually doing, one file per episode.

The failure messages say *that* an episode failed and roughly where, but not
what it looked like — "fell while walking" covers tripping on a table leg,
over-balancing in a turn, and being shoved by its own arm. This renders a
tracking camera from inside the sim process, which is the only place with the
full model state.

Rendering costs sim time, so it is off unless a video directory is configured,
and it renders every EVERY_NTH publish rather than all of them.
"""

from __future__ import annotations

import os

import cv2
import mujoco
import numpy as np

WIDTH, HEIGHT = 720, 540
# A close-up inset, because the wide shot cannot answer the question the videos
# exist for. At the tracking camera's 2.5 m the hand is roughly 40x40 px of
# translucent magenta — the ghost's geometry is a correct copy of the gripper
# (verified by overlaying it on the real hand, agreement 14-42 mm) but at that
# size no silhouette survives. The inset watches the wrist from 0.55 m.
INSET_W, INSET_H = 288, 216
INSET_DIST = 0.55
INSET_BODY = "right_wrist_yaw_link"
# Looking straight down. The wide shot already gives the side view; from
# overhead the lateral gap between the hand and the ghost is a distance on the
# image plane rather than something to infer from foreshortening, which is the
# one measurement these videos exist to make. Azimuth matches the wide camera
# so the two views stay oriented the same way as the robot walks around.
INSET_AZIMUTH, INSET_ELEVATION = 135.0, -89.0
EVERY_NTH = 4          # publishes are 50 Hz, so this records at 12.5 fps
FPS = 50.0 / EVERY_NTH


class VideoRecorder:
    """A tracking camera on the pelvis, written straight to mp4."""

    def __init__(self, model, out_dir: str, track_body: str = "pelvis"):
        self.model = model
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body)
        if bid >= 0:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = bid
        # 2.5, not 3.2: this is a diagnostic view, and at 3.2 m the hand and
        # the object are a few dozen pixels each.
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.5, 135.0, -20.0

        self.inset = None
        bid2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, INSET_BODY)
        if bid2 >= 0:
            self.inset = mujoco.Renderer(model, height=INSET_H, width=INSET_W)
            self.icam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.icam)
            self.icam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.icam.trackbodyid = bid2
            self.icam.distance = INSET_DIST
            self.icam.azimuth = INSET_AZIMUTH
            self.icam.elevation = INSET_ELEVATION

        self.writer = None
        self.path = None
        self.n = 0
        self.frames = 0

    def start(self, name: str) -> None:
        self.stop()
        self.path = os.path.join(self.out_dir, f"{name}.mp4")
        self.writer = cv2.VideoWriter(
            self.path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        self.n = self.frames = 0
        print(f"[video] recording {self.path}")

    def frame(self, data) -> None:
        if self.writer is None:
            return
        self.n += 1
        if self.n % EVERY_NTH:
            return
        self.renderer.update_scene(data, camera=self.cam)
        # MuJoCo renders RGB; OpenCV writes BGR.
        frame = self.renderer.render()[:, :, ::-1].copy()
        if self.inset is not None:
            self.inset.update_scene(data, camera=self.icam)
            tile = self.inset.render()[:, :, ::-1]
            # Bottom-right, with a hairline so it reads as a separate view.
            y, x = HEIGHT - INSET_H - 8, WIDTH - INSET_W - 8
            frame[y:y + INSET_H, x:x + INSET_W] = tile
            cv2.rectangle(frame, (x - 1, y - 1), (x + INSET_W, y + INSET_H),
                          (255, 255, 255), 1)
        self.writer.write(np.ascontiguousarray(frame))
        self.frames += 1

    def stop(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        print(f"[video] wrote {self.path} ({self.frames} frames, "
              f"{self.frames / FPS:.1f}s)")
        self.writer = None
