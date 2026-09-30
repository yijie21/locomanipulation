"""The third-person shot of a recorded round, for whichever renderer draws it.

A run's third-person video is one camera over one round, drawn by the renderer its `images.source` names
(`scripts/render_third_person.sh` -> `locomani.sim.view_render.film_rounds`; MuJoCo's replay
`scripts/render_trace_videos.py` films the same shot). The camera is chosen here once, so the renderers agree by
construction:

    round_camera   the scene's shot for one round: a four-edge scene films each round from its edge's camera
                   (the robot's first yaw names the edge), another scene from its `video_camera`, else the recorder's
                   pelvis-tracking view (`sim_video.recording_camera`)
    camera_poses   that camera's world pose at given states, read back from MuJoCo's own `mjv_updateScene` (a
                   tracking camera follows the pelvis), in the USD convention (-Z forward, +Y up) Isaac and the
                   Unreal import's camera body take
    scene_fovy     the vertical field of view of every round's camera in a scene: the Unreal level bakes one

The frames are the recorder's: every `sim_video.EVERY_NTH`-th state-trace row (`film_rows`), `sim_video.WIDTH` x
`HEIGHT` at `sim_video.FPS`, with the dataset cameras as insets (`sim_video.paste_insets`).
"""

from __future__ import annotations

import mujoco
import numpy as np

from locomani.pipeline import sim_video
from locomani.sim.scene_edges import round_edge, scene_edges


def film_rows(n_rows: int) -> range:
    """The state-trace rows the recorder films: every EVERY_NTH-th, as `VideoRecorder.frame` counts them."""
    return range(sim_video.EVERY_NTH - 1, n_rows, sim_video.EVERY_NTH)


def round_camera(model, meta: dict, qpos0) -> mujoco.MjvCamera:
    """The camera that films the round whose first recorded state is `qpos0`."""
    edge = round_edge(meta, qpos0)
    name = edge["video_camera"] if edge is not None else meta.get("video_camera")
    return sim_video.recording_camera(model, camera_name=name)


def camera_fovy(model, cam: mujoco.MjvCamera) -> float:
    """Vertical field of view, degrees: a scene camera's own, else the model's free-camera one (MuJoCo's rule)."""
    if cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
        return float(model.cam_fovy[cam.fixedcamid])
    return float(model.vis.global_.fovy)


def scene_fovy(model, meta: dict) -> float:
    """The one field of view every round of this scene is filmed with; refuses a scene whose shots disagree."""
    names = {e["video_camera"] for e in scene_edges(meta).values()} or {meta.get("video_camera")}
    fovys = {round(camera_fovy(model, sim_video.recording_camera(model, camera_name=n)), 4) for n in names}
    if len(fovys) != 1:
        raise ValueError(f"the scene's third-person cameras {sorted(map(str, names))} differ in fovy: {sorted(fovys)}")
    return fovys.pop()


def gl_camera_pose(scene):
    """World pose of the scene's (monocular average of the two) GL cameras: pos, quat wxyz with -Z forward, +Y up."""
    from scipy.spatial.transform import Rotation as R

    cams = [scene.camera[i] for i in range(2)]
    pos = np.mean([np.asarray(c.pos) for c in cams], axis=0)
    fwd = np.asarray(cams[0].forward, dtype=float)
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(cams[0].up, dtype=float)
    up -= fwd * (up @ fwd)
    up /= np.linalg.norm(up)
    right = np.cross(fwd, up)
    m = np.stack([right, up, -fwd], axis=1)  # columns = camera x, y, z (USD: -Z looks)
    qx, qy, qz, qw = R.from_matrix(m).as_quat()
    return pos, np.array([qw, qx, qy, qz])


def camera_poses(model, qpos_rows, cam: mujoco.MjvCamera) -> tuple[np.ndarray, np.ndarray]:
    """`cam`'s world pose at each state: (pos [N,3], quat [N,4] wxyz). No GL context: the scene alone."""
    data = mujoco.MjData(model)
    scene = mujoco.MjvScene(model, maxgeom=10000)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    pert = mujoco.MjvPerturb()
    rows = np.asarray(qpos_rows)
    pos, quat = np.zeros((len(rows), 3)), np.zeros((len(rows), 4))
    for k, q in enumerate(rows):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        pos[k], quat[k] = gl_camera_pose(scene)
    return pos, quat
