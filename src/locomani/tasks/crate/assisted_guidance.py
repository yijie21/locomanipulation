"""Simulation pose admitted only when the ordinary basket is in recorded camera view.

Visibility uses camera geometry and scene ray tests, not RGB pose estimation.
These privileged inputs are explicitly identified in each observation record.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.tasks.crate.perception import Camera, CrateEstimate


class AssistedGuidance:
    streams = {"ego_view": "head_camera", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}

    def __init__(self, ik, key="blue_crate", box=None, *, rendered=True):
        """`rendered=False` (backend.render offline): the frames are placeholders, so the blank-stream test is
        skipped; visibility is geometric either way."""
        self.ik = ik
        self.rendered = rendered
        self.key = key
        self.body = ik.model.body(f"{key}_object").id
        self.object_qadr = int(ik.model.jnt_qposadr[ik.model.body_jntadr[self.body]])
        # Surface samples on the ordinary solid rim and bottom; no added labels. The crate's
        # historical grid, or the same grid scaled to another target's box (2026-09-22).
        if box is None:
            xs, ys, zs = (-0.1933, 0, 0.1933), (-0.12, 0, 0.12), (-0.08, 0.07)
        else:
            hx, hy, hz = (float(v) for v in box["half_extent"])
            cz = float(box.get("centre", (0.0, 0.0, 0.0))[2])
            xs, ys, zs = (
                (-(hx - 0.008), 0, hx - 0.008),
                (-(hy - 0.015), 0, hy - 0.015),
                (cz - hz + 0.005, cz + hz - 0.015),
            )
        self.samples = np.array([[x, y, z] for x in xs for y in ys for z in zs])
        self.diagnostics = {}

    def update(self, images, state, timestamp):
        ik = self.ik
        m, d, mj = ik.model, ik.data, ik._mj
        obj = state["objects"][self.key]
        base = Rotation.from_quat(np.asarray(state["pelvis"]["quat"])[[1, 2, 3, 0]]).as_matrix()
        obj_rotation = Rotation.from_quat(np.asarray(obj["quat"])[[1, 2, 3, 0]]).as_matrix()
        d.qpos[:] = m.qpos0
        d.qpos[ik.qadr] = state["joints"]
        d.qpos[ik.base_qadr : ik.base_qadr + 7] = [*state["pelvis"]["pos"], *state["pelvis"]["quat"]]
        d.qpos[self.object_qadr : self.object_qadr + 7] = [*obj["pos"], *obj["quat"]]
        for side, palm in state.get("palms", {}).items():
            for i, value in enumerate(palm["hand_q"][:2], 1):
                d.qpos[m.joint(f"{side}_hand_gripper_{i}_joint").qposadr[0]] = value
        mj.mj_fwdPosition(m, d)
        points = self.samples @ obj_rotation.T + obj["pos"]
        visible = {}
        for stream, name in self.streams.items():
            if stream not in images or (self.rendered and np.std(images[stream]) < 1):
                continue
            cid = m.camera(name).id
            height, width = images[stream].shape[:2]
            focal = height / (2 * np.tan(np.radians(m.cam_fovy[cid]) / 2))
            camera = Camera(
                d.cam_xpos[cid].copy(),
                d.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1]),
                np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]]),
            )
            pixels, depth = camera.project(points)
            in_frame = (depth > 0.03) & np.all(pixels > 2, axis=1) & np.all(pixels < [width - 2, height - 2], axis=1)
            hits = []
            for index in np.flatnonzero(in_frame):
                ray = points[index] - camera.position
                ray /= np.linalg.norm(ray)
                hit = np.array([-1], dtype=np.int32)
                # Match rendered groups: collision proxies (group 3) are hidden.
                mj.mj_ray(
                    m,
                    d,
                    camera.position,
                    ray,
                    np.array([1, 1, 1, 0, 0, 0], np.uint8),
                    True,
                    int(m.cam_bodyid[cid]),
                    hit,
                )
                if hit[0] >= 0 and m.geom_bodyid[hit[0]] == self.body:
                    hits.append(pixels[index])
            if len(hits) >= 3 and np.max(np.ptp(hits, axis=0)) >= 8:
                visible[stream] = np.asarray(hits).tolist()
        self.diagnostics = dict(
            pose_source="simulation", visibility_source="camera_frustum_and_scene_rays", visible=visible
        )
        if not visible:
            return None
        return CrateEstimate(
            base.T @ (np.asarray(obj["pos"]) - state["pelvis"]["pos"]),
            base.T @ obj_rotation,
            timestamp,
            None,
            visible,
        )
