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
import sim_vis

WIDTH, HEIGHT = 720, 540
# A close-up inset, because the wide shot cannot answer the question the videos
# exist for. At the tracking camera's 2.5 m the hand is roughly 40x40 px of
# translucent magenta — the ghost's geometry is a correct copy of the gripper
# (verified by overlaying it on the real hand, agreement 14-42 mm) but at that
# size no silhouette survives. The inset watches the wrist from 0.55 m.
INSET_W, INSET_H = 288, 216
# Every inset is a camera the DATASET records, rendered with default
# visualisation options — so what a reviewer checks is what the VLA is
# actually trained on, not a debug view of it. Laid out around the edges so
# the robot stays visible in the middle:
#
#     left_wrist          right_wrist         ego_view
#     top-left            bottom-left         bottom-right
#
# The wrist pair is smaller because there are two of them and because the
# tracking shot is the thing being watched; they are there to answer "what
# did the gripper see when it missed", which does not need to be large.
WRIST_W, WRIST_H = 240, 180
INSETS = (
    ("head_camera", "ego", INSET_W, INSET_H, WIDTH - INSET_W - 8, HEIGHT - INSET_H - 8),
    ("left_wrist", "L wrist", WRIST_W, WRIST_H, 8, 8),
    ("right_wrist", "R wrist", WRIST_W, WRIST_H, 8, HEIGHT - WRIST_H - 8),
)
EVERY_NTH = 4  # publishes are 50 Hz, so this records at 12.5 fps
FPS = 50.0 / EVERY_NTH

# 6D-pose overlay: an oriented wireframe bounding box on the bodies whose
# poses the dataset now records (observation.can_pose / plate_pose, pelvis
# frame), plus a numeric readout of their pelvis-frame positions. A box shows
# both position and orientation without hiding the object under an axis triad.
POSE_BODIES = ("can", "plate")
POSE_RGBA = {
    "can": (1.0, 0.3, 0.3, 1.0),  # red box
    "plate": (0.3, 0.85, 1.0, 1.0),
}  # cyan box
BOX_MARGIN = 0.008
BOX_EDGE_W = 0.004
# The 12 edges of a unit box, as index pairs into the 8 corner list.
_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7))


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


def _draw_boxes(scn, data, boxes):
    """Inject wireframe-bbox capsule edges directly into a live mjvScene.

    Takes the scene itself, not a renderer, so this is reusable for both an
    offscreen mujoco.Renderer's .scene and a passive viewer's .user_scn.
    """
    for bid, lo, hi, rgba in boxes:
        pos = data.xpos[bid]
        rot = data.xmat[bid].reshape(3, 3)
        corners = [
            pos + rot @ np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])
        ]
        for a, b in _EDGES:
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3),
                np.zeros(3),
                np.zeros(9),
                np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, BOX_EDGE_W, corners[a], corners[b])
            scn.ngeom += 1


def _draw_label(scn, pos, text: str, rgba) -> None:
    """A borderless, invisible-radius sphere carrying a floating text label —
    mjvGeom.label renders as a 3D billboard at the geom's position regardless
    of vopt.label (that option only auto-labels named MODEL entities; a label
    set on a geom WE added ourselves, via mjv_initGeom, is drawn unconditionally)."""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([1e-6, 0, 0]),
        np.asarray(pos, dtype=float),
        np.eye(3).flatten(),
        np.asarray(rgba, dtype=np.float32),
    )
    # mjvGeom.label's setter only accepts numpy.bytes_ -- a plain str or
    # bytes both raise "TypeError: string argument without an encoding"
    # (mujoco 3.12.0's pybind11 binding), confirmed empirically.
    g.label = np.bytes_(text.encode())
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

        # Debug visuals (ghost hand, grip bead, reach cloud) live in geom
        # group 4, which default options hide: the ego inset and the exporter's
        # ego stream stay clean for VLA data, and only this tracking view opts
        # back in.  One group holds both, so the group is on if EITHER is
        # wanted; they are separated upstream instead — the cloud by not being
        # injected (run_sim_locomani), the ghost by never being placed
        # (grasp_gym.show_ghost), which leaves it parked below the floor.
        self.vopt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.vopt)
        self.vopt.geomgroup[4] = int(sim_vis.enabled("ghost") or sim_vis.enabled("reach"))

        # One renderer per inset. A camera the scene does not have is simply
        # skipped: an older scene, or a run with WRIST_CAMS=0, still records
        # a video with the ego inset alone.
        self.insets = []
        for cam_name, label, w, h, x, y in INSETS:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cid < 0:
                continue
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cid
            self.insets.append((mujoco.Renderer(model, height=h, width=w), cam, label, w, h, x, y))

        self.pose_names, self.pose_bids, self.pose_boxes = [], [], []
        for n in POSE_BODIES if sim_vis.enabled("bbox") or sim_vis.enabled("readout") else ():
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                continue
            lo, hi = _body_aabb(model, b)
            self.pose_names.append(n)
            self.pose_bids.append(b)
            self.pose_boxes.append((b, lo, hi, POSE_RGBA[n]))
        self.pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

        self.writer = None
        self.path = None
        self.n = 0
        self.frames = 0

    def start(self, name: str) -> None:
        self.stop()
        self.path = os.path.join(self.out_dir, f"{name}.mp4")
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
        self.n = self.frames = 0
        print(f"[video] recording {self.path} ({sim_vis.describe()})")

    def frame(self, data) -> None:
        if self.writer is None:
            return
        self.n += 1
        if self.n % EVERY_NTH:
            return
        self.renderer.update_scene(data, camera=self.cam, scene_option=self.vopt)
        if sim_vis.enabled("bbox"):
            scn = getattr(self.renderer, "scene", None) or self.renderer._scene
            _draw_boxes(scn, data, self.pose_boxes)
        # MuJoCo renders RGB; OpenCV writes BGR.
        frame = self.renderer.render()[:, :, ::-1].copy()
        for rend, cam, label, w, h, x, y in self.insets:
            # Default options (no group 4), no bbox overlay: the insets mirror
            # what the VLA will actually see, and debug art stays in the
            # tracking view behind them.
            rend.update_scene(data, camera=cam)
            frame[y : y + h, x : x + w] = rend.render()[:, :, ::-1]
            # Hairline so it reads as a separate view, and a label because
            # three insets are no longer self-evident.
            cv2.rectangle(frame, (x - 1, y - 1), (x + w, y + h), (255, 255, 255), 1)
            for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                cv2.putText(frame, label, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thick, cv2.LINE_AA)
        # Pelvis-frame position readout — the numbers the dataset records.
        if self.pelvis_bid >= 0 and sim_vis.enabled("readout"):
            rp = data.xmat[self.pelvis_bid].reshape(3, 3)
            pp = data.xpos[self.pelvis_bid]
            for i, (name, bid) in enumerate(zip(self.pose_names, self.pose_bids)):
                rel = rp.T @ (data.xpos[bid] - pp)
                # Top strip, right of the left-hand insets. It used to sit
                # bottom-left, which the right-wrist inset now occupies.
                txt = f"{name}@pelvis [{rel[0]:+.2f} {rel[1]:+.2f} {rel[2]:+.2f}]"
                org = (WRIST_W + 20, 22 + 20 * i)
                for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, thick, cv2.LINE_AA)
        self.writer.write(np.ascontiguousarray(frame))
        self.frames += 1

    def stop(self) -> None:
        if self.writer is None:
            return
        self.writer.release()
        print(f"[video] wrote {self.path} ({self.frames} frames, {self.frames / FPS:.1f}s)")
        self.writer = None


class LiveOverlay:
    """SIM_ONSCREEN's live-window equivalent of the recorded video's bbox and
    readout: the same two SIM_VIS items, but injected into the passive
    viewer's user_scn instead of drawn onto an offscreen frame with cv2 --
    that pixel-drawing path has no live-window counterpart, this is a
    from-scratch match against the same POSE_BODIES / boxes the video uses.

    reach/ghost are handled separately (viewer.opt.geomgroup[4] in
    run_sim_locomani.py) because those are real model geoms, not injected
    per frame.
    """

    def __init__(self, model):
        self.pose_names, self.pose_bids, self.pose_boxes = [], [], []
        for n in POSE_BODIES:
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            if b < 0:
                continue
            lo, hi = _body_aabb(model, b)
            self.pose_names.append(n)
            self.pose_bids.append(b)
            self.pose_boxes.append((b, lo, hi, POSE_RGBA[n]))
        self.pelvis_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    def update(self, viewer, data) -> None:
        with viewer.lock():
            scn = viewer.user_scn
            scn.ngeom = 0
            if sim_vis.enabled("bbox"):
                _draw_boxes(scn, data, self.pose_boxes)
            if sim_vis.enabled("readout") and self.pelvis_bid >= 0:
                rp = data.xmat[self.pelvis_bid].reshape(3, 3)
                pp = data.xpos[self.pelvis_bid]
                for name, bid in zip(self.pose_names, self.pose_bids):
                    rel = rp.T @ (data.xpos[bid] - pp)
                    txt = f"{name}@pelvis [{rel[0]:+.2f} {rel[1]:+.2f} {rel[2]:+.2f}]"
                    label_pos = data.xpos[bid] + np.array([0.0, 0.0, 0.12])
                    _draw_label(scn, label_pos, txt, POSE_RGBA[name])
