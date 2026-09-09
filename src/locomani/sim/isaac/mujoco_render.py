"""MuJoCo re-render of a state trace with the recorder's tracking camera, plus
the camera's world trajectory for an Isaac render of the SAME shot.

    RENDER_PY -m locomani.sim.isaac.mujoco_render --state-trace <npz> \
        --scene-xml <scene_locomani.xml> --out <dir> [--stride 4]

Writes <out>/tracking_mujoco.mp4 (every `stride`-th recorded state) and
<out>/cam_traj.npz with cam_pos [T,3], cam_quat [T,4] (wxyz, USD camera
convention: -Z forward, +Y up) for EVERY recorded state, plus `fovy`, so
isaac_render --cam-traj/--stride renders frame-for-frame the same view. The
camera is sim_video's: tracking the pelvis at 2.5 m, azimuth 135, elevation
-20 -- MuJoCo's own mjv_updateScene decides the pose, and the pose is read
back from the scene's GL camera, so the two renderers agree by construction.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)  # reach src/ for `python src/locomani/sim/isaac/<mod>.py`

from locomani.sim.isaac.replay import MujocoReplay  # noqa: E402

TRACK = dict(distance=2.5, azimuth=135.0, elevation=-20.0)


def gl_camera_pose(scene):
    """World pose of the scene's (monocular average of the two) GL cameras."""
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


def main(argv=None) -> None:
    import cv2
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--state-trace", required=True)
    ap.add_argument("--scene-xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=50, help="rate of the recorded states")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    npz = np.load(args.state_trace)
    qpos, qvel = npz["qpos"], npz["qvel"]
    replay = MujocoReplay(args.scene_xml)
    model = replay.model
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = TRACK["distance"], TRACK["azimuth"], TRACK["elevation"]
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)
    vopt.geomgroup[4] = 0  # no debug/ghost geoms

    T = len(qpos)
    vw = cv2.VideoWriter(
        os.path.join(args.out, "tracking_mujoco.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps / args.stride,
        (args.width, args.height),
    )
    cam_pos = np.zeros((T, 3))
    cam_quat = np.zeros((T, 4))
    written = 0
    for t in range(T):
        replay.set(qpos[t], qvel[t] if len(qvel) == T else None)
        renderer.update_scene(replay.data, camera=cam, scene_option=vopt)
        cam_pos[t], cam_quat[t] = gl_camera_pose(renderer.scene)
        if t % args.stride == 0:
            vw.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            written += 1
    vw.release()
    np.savez(
        os.path.join(args.out, "cam_traj.npz"), cam_pos=cam_pos, cam_quat=cam_quat, fovy=float(model.vis.global_.fovy)
    )
    print(
        f"wrote {written} frames ({T} states, stride {args.stride}) -> {args.out}; fovy {float(model.vis.global_.fovy)}"
    )


if __name__ == "__main__":
    main()
