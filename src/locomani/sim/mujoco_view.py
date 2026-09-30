"""The MuJoCo counterpart of `isaac.view.IsaacView` and `unreal.view.UnrealView` for a run's third-person video.

`locomani.sim.render_third_person` films a round through a view: `render(qpos) -> {stream: image}` draws the dataset
cameras (the insets, default visual options, as the live sim and `pipeline.render_dataset_cameras` do) and
`film(qpos, camera, ghosts)` the third-person shot. MuJoCo is filmed the way the Unreal import is: the scene gains
mocap bodies -- a camera rig whose pose is the shot's (`third_person_shot.camera_poses`, -Z forward, +Y up, which is
MuJoCo's own camera frame) and the two ghost grippers of `isaac.ghost.inject_mujoco` in geom group 4, which only the
third-person picture shows. Mocap bodies add no joints, so the run's qpos fits the model unchanged.

The GL backend and card are fixed before `mujoco` is imported: the caller sets MUJOCO_GL / MUJOCO_EGL_DEVICE_ID.
"""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.sim.dataset_cameras import CAMERA_CONFIGS
from locomani.sim.isaac import ghost as G

RIG = "loco_third_person_rig"
CAMERA = "loco_third_person"


class MujocoView:
    """A run's MuJoCo scene, read through the dataset's cameras and the third-person rig."""

    def __init__(self, scene_xml: str, log=print):
        from locomani.sim import scene_lib

        spec = mujoco.MjSpec.from_file(scene_xml)
        for side in G.SIDES:
            scene_lib.add_ghost_hand(spec, side, alpha=G.GHOST_ALPHA, name=G.NAMES[side])
        rig = spec.worldbody.add_body(name=RIG, pos=[0, 0, -5])
        rig.mocap = True
        rig.add_camera(name=CAMERA)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.nq = mujoco.MjModel.from_xml_path(scene_xml).nq
        if self.model.nq != self.nq:
            raise SystemExit(f"the third-person bodies changed the scene's qpos ({self.nq} -> {self.model.nq})")
        self.rig = int(self.model.body_mocapid[self.model.body(RIG).id])
        self.ghosts = {side: int(self.model.body_mocapid[self.model.body(G.NAMES[side]).id]) for side in G.SIDES}
        self.renderers = {k: mujoco.Renderer(self.model, c["height"], c["width"]) for k, c in CAMERA_CONFIGS.items()}
        self.film_renderer = None
        log(f"mujoco view: {scene_xml}, cameras {[c['mjcf_name'] for c in CAMERA_CONFIGS.values()]}")

    def _set(self, qpos) -> None:
        qpos = np.asarray(qpos, dtype=np.float64)
        if len(qpos) != self.nq:
            raise RuntimeError(f"a qpos of {len(qpos)} values for a scene of {self.nq}")
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def warm_up(self, qpos) -> int:
        """Nothing streams in: MuJoCo draws a state completely the first time. Returns 0."""
        return 0

    def render(self, qpos) -> dict:
        """`{stream: HxWx3 uint8}` of the dataset cameras at this state."""
        self._set(qpos)
        out = {}
        for k, r in self.renderers.items():
            r.update_scene(self.data, camera=CAMERA_CONFIGS[k]["mjcf_name"])
            out[k] = r.render()
        return out

    def film_setup(self, width: int, height: int, fovy: float) -> None:
        """Size the third-person picture and give the rig the shot's field of view; ghosts in view from now on."""
        self.model.cam_fovy[self.model.camera(CAMERA).id] = fovy
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, width)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, height)
        self.film_renderer = mujoco.Renderer(self.model, height, width)
        self.vopt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.vopt)
        self.vopt.geomgroup[4] = 1  # the ghost grippers; the dataset cameras render with default options

    def film_warm_up(self, qpos, camera, ghosts) -> int:
        return 0

    def film(self, qpos, camera, ghosts) -> np.ndarray:
        """The third-person picture of this state: `camera` (pos, wxyz quat, -Z forward), `ghosts` (pos [2,3],
        quat [2,4]) as `ghost_trace.ghost_poses` gives them, left then right."""
        self.data.mocap_pos[self.rig], self.data.mocap_quat[self.rig] = camera
        G.set_mujoco(self.data, self.ghosts, *ghosts)
        self._set(qpos)
        self.film_renderer.update_scene(self.data, camera=CAMERA, scene_option=self.vopt)
        return self.film_renderer.render()

    def close(self) -> None:
        for r in [*self.renderers.values(), self.film_renderer]:
            if r is not None:
                r.close()
