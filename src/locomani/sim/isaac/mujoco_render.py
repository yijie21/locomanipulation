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


def _make_insets(model, width: int, height: int):
    """The live recorder's three insets (sim_video.INSETS), scaled from its
    720x540 frame to this render's size; a camera the scene lacks is skipped."""
    import mujoco

    from locomani.pipeline import sim_video as sv

    sx, sy = width / sv.WIDTH, height / sv.HEIGHT
    out = []
    for cam_name, label, w, h, x, y in sv.INSETS:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cid < 0:
            continue
        w, h = int(round(w * sx)), int(round(h * sy))
        x, y = int(round(x * sx)), int(round(y * sy))
        icam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(icam)
        icam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        icam.fixedcamid = cid
        out.append((mujoco.Renderer(model, height=h, width=w), icam, label, w, h, x, y))
    return out


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
    ap.add_argument(
        "--distance", type=float, default=TRACK["distance"], help="tracking camera distance from the pelvis, m"
    )
    ap.add_argument(
        "--azimuth",
        type=float,
        default=TRACK["azimuth"],
        help="tracking camera azimuth, deg (MuJoCo convention; 135 = behind-right of a +y-facing robot, 225 = front-right). The warehouse start is ~1 m from the south partition, so the default 2.5 m / 135 shot sits outside the room and both renderers see a wall (2026-09-24)",
    )
    ap.add_argument("--elevation", type=float, default=TRACK["elevation"], help="tracking camera elevation, deg")
    ap.add_argument(
        "--ghost-trace",
        default="",
        help="<label>_ghost.npz from the bimanual expert: draw the aimed wrist poses as translucent ghost hands",
    )
    ap.add_argument(
        "--insets",
        action="store_true",
        help="picture-in-picture of the dataset cameras (ego / left wrist / right wrist), "
        "same layout as the live sim_video recorder; scaled to this frame size",
    )
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    npz = np.load(args.state_trace)
    qpos, qvel = npz["qpos"], npz["qvel"]
    ghost = ghost_ids = None
    model = None
    if args.ghost_trace:
        from locomani.sim.isaac import ghost as G
        from locomani.tasks.crate.ghost_trace import ghost_poses, load_ghost_trace

        model, ghost_ids = G.inject_mujoco(args.scene_xml)
        ghost = load_ghost_trace(args.ghost_trace)
        seqs = npz["seq"] if "seq" in npz else np.arange(len(qpos))
        print(f"ghost trace: {len(ghost)} rows for {len(qpos)} states")
    replay = MujocoReplay(args.scene_xml, model=model)
    model = replay.model
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation
    vopt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(vopt)
    vopt.geomgroup[4] = 1 if ghost is not None else 0  # ghost hands only when asked (the insets never show them)
    insets = _make_insets(model, args.width, args.height) if args.insets else []

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
        if ghost is not None:
            gpos, gquat = ghost_poses(ghost, int(seqs[t]))
            G.set_mujoco(replay.data, ghost_ids, gpos, gquat)
            mujoco.mj_forward(model, replay.data)
        renderer.update_scene(replay.data, camera=cam, scene_option=vopt)
        cam_pos[t], cam_quat[t] = gl_camera_pose(renderer.scene)
        if t % args.stride == 0:
            frame = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            for rend, icam, label, w, h, x, y in insets:
                rend.update_scene(replay.data, camera=icam)
                frame[y : y + h, x : x + w] = rend.render()[:, :, ::-1]
                cv2.rectangle(frame, (x - 1, y - 1), (x + w, y + h), (255, 255, 255), 1)
                for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                    cv2.putText(
                        frame, label, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thick, cv2.LINE_AA
                    )
            vw.write(frame)
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
