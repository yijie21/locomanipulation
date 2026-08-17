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
# The inset shows the robot's own ego view — the same fixed head camera the
# data exporter records — so the reviewer sees exactly what the VLA will see.
INSET_CAMERA = "head_camera"
EVERY_NTH = 4          # publishes are 50 Hz, so this records at 12.5 fps
FPS = 50.0 / EVERY_NTH

# 6D-pose overlay: RGB axis triads on the bodies whose poses the dataset now
# records (observation.can_pose / plate_pose, pelvis frame), plus a numeric
# readout of their pelvis-frame positions.
POSE_BODIES = ("can", "plate")
AXIS_LEN = 0.12
AXIS_RGBA = ((1.0, 0.25, 0.25, 1.0),   # x
             (0.25, 1.0, 0.25, 1.0),   # y
             (0.35, 0.55, 1.0, 1.0))   # z


def _draw_axes(renderer, data, bids):
    """Inject arrow geoms for each body's frame into an updated scene."""
    scn = getattr(renderer, "scene", None) or renderer._scene
    for bid in bids:
        pos = data.xpos[bid]
        rot = data.xmat[bid].reshape(3, 3)
        for k, rgba in enumerate(AXIS_RGBA):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                np.zeros(9), np.asarray(rgba, dtype=np.float32))
            tip = pos + rot[:, k] * AXIS_LEN
            try:
                mujoco.mjv_connector(
                    g, mujoco.mjtGeom.mjGEOM_ARROW, 0.008, pos, tip)
            except AttributeError:      # older mujoco
                mujoco.mjv_makeConnector(
                    g, mujoco.mjtGeom.mjGEOM_ARROW, 0.008,
                    pos[0], pos[1], pos[2], tip[0], tip[1], tip[2])
            scn.ngeom += 1


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
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, INSET_CAMERA)
        if cid >= 0:
            self.inset = mujoco.Renderer(model, height=INSET_H, width=INSET_W)
            self.icam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.icam)
            self.icam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.icam.fixedcamid = cid

        self.pose_bids = [b for b in (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in POSE_BODIES) if b >= 0]
        self.pose_names = [n for n in POSE_BODIES if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]
        self.pelvis_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

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
        _draw_axes(self.renderer, data, self.pose_bids)
        # MuJoCo renders RGB; OpenCV writes BGR.
        frame = self.renderer.render()[:, :, ::-1].copy()
        if self.inset is not None:
            self.inset.update_scene(data, camera=self.icam)
            _draw_axes(self.inset, data, self.pose_bids)
            tile = self.inset.render()[:, :, ::-1]
            # Bottom-right, with a hairline so it reads as a separate view.
            y, x = HEIGHT - INSET_H - 8, WIDTH - INSET_W - 8
            frame[y:y + INSET_H, x:x + INSET_W] = tile
            cv2.rectangle(frame, (x - 1, y - 1), (x + INSET_W, y + INSET_H),
                          (255, 255, 255), 1)
        # Pelvis-frame position readout — the numbers the dataset records.
        if self.pelvis_bid >= 0:
            rp = data.xmat[self.pelvis_bid].reshape(3, 3)
            pp = data.xpos[self.pelvis_bid]
            for i, (name, bid) in enumerate(zip(self.pose_names,
                                                self.pose_bids)):
                rel = rp.T @ (data.xpos[bid] - pp)
                cv2.putText(
                    frame,
                    f"{name}@pelvis [{rel[0]:+.2f} {rel[1]:+.2f} {rel[2]:+.2f}]",
                    (10, HEIGHT - 12 - 20 * (len(self.pose_bids) - 1 - i)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                    cv2.LINE_AA)
        self.writer.write(np.ascontiguousarray(frame))
        self.frames += 1

    def stop(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        print(f"[video] wrote {self.path} ({self.frames} frames, "
              f"{self.frames / FPS:.1f}s)")
        self.writer = None
