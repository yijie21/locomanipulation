"""Expert-only geometry proxy gated by actual segmented camera visibility.

Use MUJOCO_GL=osmesa and CUDA_VISIBLE_DEVICES='' for CPU diagnostics. This proxy
is NOT a learned RGB detector and must never be called by the deployed VLA.
"""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.tasks.bottle_visual.observation import SceneMeasurement


class BottleProxySensor:
    def __init__(self, sim, width=320, height=240, quantization=0.005):
        self.sim = sim
        self.quantization = quantization
        self.renderer = mujoco.Renderer(sim.m, height=height, width=width)
        self.renderer.enable_segmentation_rendering()
        self.cameras = [sim.m.camera(n).id for n in ("head_camera", "right_wrist")]
        self.object_geoms = np.flatnonzero(sim.m.geom_bodyid == sim.crate_b)
        root = sim.m.body_rootid[sim.bid["pelvis"]]
        self.robot_geoms = np.flatnonzero(sim.m.body_rootid[sim.m.geom_bodyid] == root)
        self.options = mujoco.MjvOption()
        self.last_evidence = None
        self.approach_edge = None

    def reset(self):
        self.approach_edge = None
        self.last_evidence = None

    def close(self):
        self.renderer.close()

    def _mask(self, camera):
        self.renderer.update_scene(self.sim.d, camera=camera, scene_option=self.options)
        segmentation = self.renderer.render()
        return np.isin(segmentation[..., 0], self.object_geoms) & (
            segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        )

    def capture(self, t):
        sim = self.sim
        pixels, fractions, boxes = [], [], []
        saved_groups = sim.m.geom_group[self.robot_geoms].copy()
        saved_group5 = self.options.geomgroup[5]
        for cid in self.cameras:
            mask = self._mask(cid)
            pixels.append(int(mask.sum()))
            yy, xx = np.where(mask)
            boxes.append(None if not len(xx) else [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())])
            try:
                sim.m.geom_group[self.robot_geoms] = 5
                self.options.geomgroup[5] = 0
                potential = int(self._mask(cid).sum())
            finally:
                sim.m.geom_group[self.robot_geoms] = saved_groups
                self.options.geomgroup[5] = saved_group5
            fractions.append(float(min(1.0, pixels[-1] / max(1, potential))))
        self.last_evidence = dict(t=t, pixels=pixels, fractions=fractions, boxes=boxes)
        empty = SceneMeasurement(t, None, None, None, None, tuple(pixels), tuple(fractions))
        if not empty.visible():
            return empty
        # Geometry is read only after the actual images pass the evidence gate.
        # Quantize in the best camera frame, not world axes, to avoid a world-frame
        # preference. This still assumes RGB-estimable geometry and requires VLA
        # closed-loop validation before claiming learnability.
        eligible = [i for i, (p, f) in enumerate(zip(pixels, fractions)) if p >= 32 and f >= 0.2]
        cid = self.cameras[max(eligible, key=lambda i: pixels[i])]
        camera_rotation = sim.d.cam_xmat[cid].reshape(3, 3)
        camera_position = sim.d.cam_xpos[cid]
        relative = camera_rotation.T @ (sim.crate()[0] - camera_position)
        relative = np.round(relative / self.quantization) * self.quantization
        bottle = camera_position + camera_rotation @ relative
        surface = sim.smeta["surfaces"]["A"]
        center = np.asarray(surface["xy"])
        half = np.asarray(surface["flat_half_extent"])
        if self.approach_edge is None:
            # Select the face approached by the robot, then retain that route.
            # Nearest-to-bottle flips 90 degrees near a table corner under tiny
            # pixel quantization errors. Initial robot heading is observable.
            forward = sim.d.xmat[sim.bid["pelvis"]].reshape(3, 3)[:2, 0]
            axis = int(np.argmax(abs(forward)))
            sign = -1 if forward[axis] >= 0 else 1
            self.approach_edge = axis, sign
        axis, sign = self.approach_edge
        edge = bottle.copy()
        edge[axis] = center[axis] + sign * half[axis]
        edge[2] = surface["top"]
        inward = np.zeros(3)
        inward[axis] = -sign
        return SceneMeasurement(
            t, tuple(bottle), float(surface["top"]), tuple(edge), tuple(inward), tuple(pixels), tuple(fractions)
        )
