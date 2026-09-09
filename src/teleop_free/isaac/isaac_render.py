"""Re-render a recorded episode through Isaac Sim RTX.

No re-execution and no new recording: the state comes from what collection
already persisted (see :mod:`frames`), and the stage is assembled and driven by
:mod:`stage`, the same functions the live mirror will use.

    RENDER_PY=... ; "$RENDER_PY" -m teleop_free.isaac.isaac_render \
        --session <exporter session dir> --episode 0 \
        --sidecar '<pose_sidecar/p*_round_*_attempt_*.jsonl>' \
        --out outputs/renders_isaac/first_light

Paths come from teleop_free.paths; --scene-xml points at a batch's archived
scene snapshot ($OUT/scene_<seed>/scene_locomani.xml) when re-rendering an
episode whose scene is no longer the working one.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from .. import paths

OBJECT_KEYS = ("water_bottle",)  # fallback when no scene_meta.json sits next to the scene xml


def object_keys(scene_xml: str):
    """Manipulable objects of the scene: from scene_meta.json next to the xml."""
    import json as _json

    meta = os.path.join(os.path.dirname(os.path.abspath(scene_xml)), "scene_meta.json")
    if os.path.exists(meta):
        keys = tuple(_json.load(open(meta)).get("objects") or ())
        if keys:
            return keys
    return OBJECT_KEYS


from . import frames as F
from . import stage as S


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument(
        "--state-trace",
        default="",
        help="SIMPLE-style qpos/qvel npz from collection. The "
        "correct input: one recorded state per frame, so the "
        "robot, the objects and the camera all come out of a "
        "single mj_forward. Without it the renderer falls "
        "back to reconstructing state from the parquet plus "
        "the pose sidecar, which does NOT reproduce the "
        "recorded video (see docs/rendering.md).",
    )
    ap.add_argument(
        "--sidecar",
        default="",
        help="glob over the batch's pose_sidecar jsonl files; only used by the fallback reconstruction path",
    )
    ap.add_argument("--out", default=os.path.join(paths.ISAAC_ASSETS, "first_light"))
    ap.add_argument("--assets", default=paths.ISAAC_ASSETS)
    ap.add_argument(
        "--scene-dir",
        default="",
        help="the built scene for this episode's batch, normally "
        "$ISAAC_ASSETS/scene/<seed>; default <assets>/scene/work",
    )
    ap.add_argument("--robot-usd", default="", help="default <assets>/robot/g1_dex1.usda")
    ap.add_argument(
        "--scene-xml",
        default=paths.SCENE_XML,
        help="the scene this episode was collected in; default is "
        "the CURRENT working scene, which is only right if no "
        "batch has rebuilt it since",
    )
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame (2 halves wall time)")
    ap.add_argument("--torso-prim", default="/World/scene/torso_link", help="prim the head camera parents to")
    ap.add_argument(
        "--cam-traj",
        default="",
        help="npz from compute_cam_traj.py; overrides the USD camera mount with MuJoCo-FK world poses per frame",
    )
    ap.add_argument("--end-frame", type=int, default=0, help="stop after this many source frames (0 = all)")
    ap.add_argument("--debug-cam", action="store_true", help="save a third-person still every 200 frames")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "render.log")
    open(log_path, "w").close()

    def log(msg):
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    if not (args.state_trace or args.sidecar):
        raise SystemExit("give --state-trace (preferred) or --sidecar")
    cam_spec = F.head_camera_spec(args.scene_xml)
    if args.cam_traj:
        # a world camera trajectory (e.g. mujoco_render's tracking shot) may
        # carry its own vertical fov; the ego spec's pos/quat are then unused
        npz = np.load(args.cam_traj)
        if "fovy" in npz.files:
            cam_spec = (cam_spec[0], cam_spec[1], float(npz["fovy"]))
            log(f"cam-traj fovy {cam_spec[2]}")
    log(f"scene_xml={args.scene_xml}")
    log(f"head_camera pos={cam_spec[0]} quat={cam_spec[1]} fovy={cam_spec[2]}")
    joint_map = F.build_joint_map()
    if args.state_trace:
        _seq, qpos, qvel = F.load_state_trace(args.state_trace, args.session, args.episode)
        log(f"replay: {len(qpos)} recorded states from {args.state_trace}")
        joints = pel = objs = None
    else:
        log(
            "WARNING no --state-trace: reconstructing state from parquet + "
            "sidecar. Two provenances, two clocks -- this does not reproduce "
            "the recorded video and is only for data older than the trace."
        )
        joints, pel, objs = F.load_frames(args.session, args.episode, args.sidecar)
        qpos = qvel = None
        log(f"frames={len(joints)} objects={list(objs)}")

    cfg = S.StageCfg(
        assets=args.assets,
        scene_dir=args.scene_dir,
        robot_usd=args.robot_usd,
        robot_mjcf=args.scene_xml,
        width=args.width,
        height=args.height,
        torso_prim=args.torso_prim,
        # The head mesh is hidden only because the EGO camera sits
        # inside it; a world-camera (tracking) shot must show it.
        hidden_visuals=() if args.cam_traj else S.StageCfg.hidden_visuals,
        # A per-frame WORLD camera pose requires a top-level
        # camera. Parented under torso_link, set_world_pose
        # back-computes a local offset against the parent's
        # pre-step (stale) transform and the whole view shifts.
        # The replay path always supplies a world pose, so this
        # must follow it, not just --cam-traj. Caught 2026-09-06:
        # same qpos rendered by MuJoCo and by Isaac disagreed by a
        # whole-image translation until this was tied together.
        absolute_camera=bool(args.cam_traj or args.state_trace),
    )
    app = S.start_app(cfg)
    # SimulationApp.close() terminates the process itself, and it does so with
    # status 0 -- so a raise inside the render used to surface as SUCCESS to
    # whoever ran it. Caught 2026-09-05 when a missing scene USD exited 0.
    # Record the failure, close, then exit non-zero explicitly.
    try:
        _render(args, cfg, joint_map, joints, pel, objs, cam_spec, log, qpos=qpos, qvel=qvel)
    except BaseException:
        import traceback

        log("FAILED:\n" + traceback.format_exc())
        # NOT app.close() first: close() terminates the process itself, with
        # status 0, so anything after it never runs and the failure surfaces as
        # success. Kit teardown is skipped on the failure path on purpose.
        sys.stderr.write(f"render FAILED, see {log_path}\n")
        os._exit(1)
    app.close()


def _render(args, cfg, joint_map, joints, pel, objs, cam_spec, log, qpos=None, qvel=None) -> None:
    import cv2

    stg = S.assemble_stage(cfg, log)
    world = S.make_world()
    robot, jidx, jsrc = S.bind_robot(joint_map, log)
    sensor, cam_path = S.mount_camera(stg, cfg, cam_spec, log)

    # Replay path: one MuJoCo state decoder, and every quantity Isaac needs is
    # read back out of the SAME mj_forward -- joints, object poses, camera.
    replay = dof_idx = None
    if qpos is not None:
        from .replay import MujocoReplay

        replay = MujocoReplay(args.scene_xml, object_keys(args.scene_xml))
        log(f"replay objects: {list(replay._obj)}")
        names = [n for n, _ in joint_map]
        dof_names = list(robot.dof_names)
        keep = [i for i, n in enumerate(names) if n in dof_names]
        dof_idx = [dof_names.index(names[i]) for i in keep]
        replay_names = [names[i] for i in keep]
        root_name = S.root_body(robot)
        log(f"replay: {len(dof_idx)} of {len(names)} joints bound; root link {root_name}")

    debug_sensor = None
    if args.debug_cam:
        from isaacsim.sensors.camera import Camera as SensorCamera

        debug_sensor = SensorCamera(prim_path="/World/debug_cam", resolution=(640, 480))
        debug_sensor.initialize()

    def aim_debug_cam(target):
        """Park the debug camera off the target, looking at it.  Pose goes
        through SensorCamera.set_world_pose (raw USD xform ops get reset by
        the sensor wrapper); quaternion built column-major and passed with
        camera_axes="usd" so there is exactly one convention in play."""
        from scipy.spatial.transform import Rotation as _R

        eye = np.array([target[0] - 1.4, target[1] - 1.4, target[2] + 0.7])
        fwd = np.asarray(target) - eye
        fwd = fwd / np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(fwd, up)
        right /= np.linalg.norm(right)
        cup = np.cross(right, fwd)
        mcol = np.stack([right, cup, -fwd], axis=1)  # columns = camera axes
        qx, qy, qz, qw = _R.from_matrix(mcol).as_quat()
        debug_sensor.set_world_pose(position=eye, orientation=[qw, qx, qy, qz], camera_axes="usd")

    vw = cv2.VideoWriter(
        os.path.join(args.out, "ego_isaac.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps / args.stride,
        (args.width, args.height),
    )

    T = len(qpos) if replay is not None else len(joints)
    if args.end_frame:
        T = min(T, args.end_frame)

    cam_traj = None
    if args.cam_traj:
        npz = np.load(args.cam_traj)
        cam_traj = (npz["cam_pos"], npz["cam_quat"])
        log(f"cam-traj loaded: {len(cam_traj[0])} frames")

    def push(t):
        """One frame into the stage. Returns the camera pose actually used."""
        if replay is not None:
            replay.set(qpos[t], None if qvel is None else qvel[t])
            if cam_traj is not None:  # a world camera (tracking shot)
                cpos, cquat = cam_traj[0][t], cam_traj[1][t]
            else:
                cpos, cquat = replay.camera_pose("head_camera")
            S.sync_replay(
                stg,
                robot,
                dof_idx,
                replay.dof_positions(replay_names),
                replay.body_pose(root_name),
                replay.object_poses(),
                sensor=sensor,
                cam_pose=(cpos, cquat),
            )
            return
        cp = None if cam_traj is None else (cam_traj[0][t], cam_traj[1][t])
        S.sync_frame(
            stg, robot, jidx, jsrc, joints[t], pel[t], {k: v[t] for k, v in objs.items()}, sensor=sensor, cam_pose=cp
        )

    # Burn the render product's empty first frames on frame 0's state, so the
    # written stream starts at source frame 0 and has exactly T/stride frames.
    for _i in range(8):
        push(0)
        if S.step_and_grab(world, sensor) is not None:
            if _i:
                log(f"sensor warm-up: {_i} discarded step(s)")
            break
    else:
        log("WARNING sensor still empty after 8 warm-up steps")
    # A non-empty sensor is not a finished picture: RTX streams the robot's
    # meshes in over the first ~10 rendered frames (pilot 21 r00: 0.8 s of
    # video with only two finger tips visible), and deactivating /World/G1
    # forces a Hydra resync on top. Hold frame 0 until the picture stops
    # changing.
    prev, calm = None, 0
    for _i in range(80):
        push(0)
        cur = S.step_and_grab(world, sensor)
        if cur is None:
            continue
        cur = cur[..., :3].astype(np.float32)
        if prev is not None and float(np.abs(cur - prev).mean()) < 0.5:
            calm += 1
            if calm >= 3:
                log(f"picture settled after {_i + 1} extra warm-up steps")
                break
        else:
            calm = 0
        prev = cur
    else:
        log("WARNING picture still changing after 80 extra warm-up steps")

    written = 0
    for t in range(0, T, args.stride):
        push(t)
        rgba = S.step_and_grab(world, sensor)
        if rgba is None:
            log(f"frame {t}: empty rgba")
            continue
        vw.write(cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR))
        written += 1
        if debug_sensor is not None and t % 200 == 0:
            try:
                cp, _cq = sensor.get_world_pose()
                pelv = replay.body_pose("pelvis")[:3] if replay is not None else pel[t, :3]
                log(f"t={t} ego_cam@{np.round(cp, 3).tolist()} pelvis@{np.round(pelv, 3).tolist()}")
            except Exception as e:  # noqa: BLE001
                log(f"t={t} get_world_pose failed: {e}")
            aim_debug_cam(replay.body_pose("pelvis")[:3] if replay is not None else pel[t, :3])
            world.step(render=True)
            drgba = debug_sensor.get_rgba()
            if drgba is not None and drgba.size:
                cv2.imwrite(
                    os.path.join(args.out, f"debug_{t:04d}.png"), cv2.cvtColor(drgba[..., :3], cv2.COLOR_RGB2BGR)
                )
        if t % 200 == 0:
            log(f"frame {t}/{T}")
    vw.release()
    expect = len(range(0, T, args.stride))
    log(f"wrote {written}/{expect} frames (source frames {T}, stride {args.stride})")
    if written != expect:
        raise SystemExit(
            f"frame count mismatch: wrote {written}, expected {expect}. An "
            f"image stream shorter than its action stream is misaligned for the "
            f"whole episode, so this is fatal rather than a warning."
        )
    log("done")


if __name__ == "__main__":
    main()
