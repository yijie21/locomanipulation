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

# 6D-pose overlay: an oriented wireframe bounding box on the bodies whose
# poses the dataset now records (observation.can_pose / plate_pose, pelvis
# frame), plus a numeric readout of their pelvis-frame positions. A box shows
# both position and orientation without hiding the object under an axis triad.
POSE_BODIES = ("can", "plate")
POSE_RGBA = {"can": (1.0, 0.3, 0.3, 1.0),      # red box
             "plate": (0.3, 0.85, 1.0, 1.0)}   # cyan box
BOX_MARGIN = 0.008
BOX_EDGE_W = 0.004
# The 12 edges of a unit box, as index pairs into the 8 corner list.
_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3),
          (4, 5), (4, 6), (5, 7), (6, 7),
          (0, 4), (1, 5), (2, 6), (3, 7))


def _body_aabb(model, bid):
    """Local-frame AABB of a body from its geoms (axis-aligned primitives)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    adr, num = int(model.body_geomadr[bid]), int(model.body_geomnum[bid])
    for gi in range(adr, adr + num):
        gt, gs = int(model.geom_type[gi]), model.geom_size[gi]
        if gt == mujoco.mjtGeom.mjGEOM_BOX:
            he = np.array(gs[:3])
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            he = np.full(3, gs[0])
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            he = np.array([gs[0], gs[0], gs[1]])
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            he = np.array([gs[0], gs[0], gs[1] + gs[0]])
        else:
            he = np.full(3, max(float(gs[0]), 0.03))
        gp = model.geom_pos[gi]
        lo = np.minimum(lo, gp - he)
        hi = np.maximum(hi, gp + he)
    return lo - BOX_MARGIN, hi + BOX_MARGIN


def _draw_boxes(renderer, data, boxes):
    """Inject wireframe-bbox capsule edges into an updated scene."""
    scn = getattr(renderer, "scene", None) or renderer._scene
    for bid, lo, hi, rgba in boxes:
        pos = data.xpos[bid]
        rot = data.xmat[bid].reshape(3, 3)
        corners = [pos + rot @ np.array([x, y, z])
                   for x in (lo[0], hi[0])
                   for y in (lo[1], hi[1])
                   for z in (lo[2], hi[2])]
        for a, b in _EDGES:
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                np.zeros(9), np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, BOX_EDGE_W,
                corners[a], corners[b])
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

        self.pose_names, self.pose_bids, self.pose_boxes = [], [], []
        for n in POSE_BODIES:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                continue
            lo, hi = _body_aabb(model, b)
            self.pose_names.append(n)
            self.pose_bids.append(b)
            self.pose_boxes.append((b, lo, hi, POSE_RGBA[n]))
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
        _draw_boxes(self.renderer, data, self.pose_boxes)
        # MuJoCo renders RGB; OpenCV writes BGR.
        frame = self.renderer.render()[:, :, ::-1].copy()
        if self.inset is not None:
            self.inset.update_scene(data, camera=self.icam)
            _draw_boxes(self.inset, data, self.pose_boxes)
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
