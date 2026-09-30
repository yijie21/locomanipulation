"""Re-render a recorded episode through Isaac Sim RTX.

No re-execution and no new recording: the state comes from what collection
already persisted (see :mod:`frames`), and the stage is assembled and driven by
:mod:`stage`, the same functions the live mirror will use.

    RENDER_PY=... ; "$RENDER_PY" -m locomani.sim.isaac.isaac_render \
        --session <exporter session dir> --episode 0 \
        --sidecar '<pose_sidecar/p*_round_*_attempt_*.jsonl>' \
        --out outputs/renders_isaac/first_light

Paths come from locomani.paths; --scene-xml points at a batch's archived
scene snapshot ($OUT/scene_<seed>/scene_locomani.xml) when re-rendering an
episode whose scene is no longer the working one.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)  # reach src/ for `python src/locomani/sim/isaac/<mod>.py`
from locomani import paths

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


from locomani.sim.isaac import frames as F
from locomani.sim.isaac import stage as S


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--ghost-trace", default="", help="<label>_ghost.npz: draw the aimed wrist poses as ghost hands")
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
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame (2 halves wall time)")
    ap.add_argument("--torso-prim", default="/World/scene/torso_link", help="prim the head camera parents to")
    ap.add_argument(
        "--cam-traj",
        default="",
        help="npz from compute_cam_traj.py; overrides the USD camera mount with MuJoCo-FK world poses per frame",
    )
    ap.add_argument("--end-frame", type=int, default=0, help="stop after this many source frames (0 = all)")
    ap.add_argument(
        "--rtx-setting",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="carb setting applied after the renderer is configured, repeatable; bool/int/float parsed, else string. "
        "E.g. /rtx/post/histogram/enabled=false (no auto-exposure: with it on, light intensities have NO effect on "
        "the picture, the tonemapper renormalises every frame) and /rtx/post/tonemap/op=0 (Clamp, MuJoCo-like linear "
        "response; 5 = ACES default). Measured 2026-09-24 on the warehouse: auto-exposure lifted the robot's 0.2 "
        "greys to 108/255 and pushed 0.7 to 212, a head/torso ratio of 0.51 against MuJoCo's 0.23",
    )
    ap.add_argument(
        "--supersample",
        type=int,
        default=1,
        help="render every camera at N times its size and box-average back (the scene's isaac.supersample)",
    )
    ap.add_argument("--debug-cam", action="store_true", help="save a third-person still every 200 frames")
    ap.add_argument(
        "--insets",
        action="store_true",
        help="picture-in-picture of the ego + left/right wrist dataset cameras, same layout as the "
        "live sim_video recorder and mujoco_render's --insets. All three are their own independently "
        "mounted, independently posed cameras here -- this renderer's own MAIN frame is often (via "
        "--cam-traj) a third-person tracking shot, not the ego view, so the ego inset is not redundant "
        "even though it sounds like it should be. Needs --state-trace: inset poses come from the same "
        "MuJoCo replay FK as the main camera, not from the parquet+sidecar fallback. Rendered as a fully "
        "separate second pass after the main view (its own warm-up, its own stepping loop, ghost hands "
        "and the head mesh hidden once for the whole pass), composited onto the main pass's own frames "
        "read back off disk -- not interleaved with the main view's rendering -- so this roughly doubles "
        "both the render time and the peak disk use of args.out (lossless per-frame PNGs, deleted after)",
    )
    return ap


def dataset_inset_layout(width: int, height: int) -> dict:
    """Scaled `{camera_name: (label, w, h, x, y)}` for every one of the live
    recorder's picture-in-picture cameras (locomani.pipeline.sim_video.INSETS:
    ego + left/right wrist), so this renderer's insets agree with MuJoCo's
    --insets pixel-for-pixel. All three apply here, including ego: an earlier
    version of this function dropped the ego entry on the assumption that
    this renderer's own frame already IS the ego view, which is only true
    without --cam-traj -- render_wall_pilot.sh always passes --cam-traj, so
    in practice the main frame is the third-person tracking shot instead, and
    the ego inset is the only ego view in the output at all."""
    from locomani.pipeline import sim_video as sv

    sx, sy = width / sv.WIDTH, height / sv.HEIGHT
    return {
        cam_name: (label, int(round(w * sx)), int(round(h * sy)), int(round(x * sx)), int(round(y * sy)))
        for cam_name, label, w, h, x, y in sv.INSETS
    }


def ego_only_mesh_paths(cam_traj_given: bool, inset_camera_names, hidden_visual_rels) -> list:
    """Visual prims that must stay hidden from an ego-positioned camera (it
    sits inside them; RTX does not cull backfaces, so left visible they cover
    the view in a black dome -- see StageCfg.hidden_visuals) even when the
    MAIN camera needs them visible.

    That conflict exists only when BOTH hold: `--cam-traj` was given, so the
    main camera follows a third-person tracking shot instead of the ego view
    and needs these meshes visible (main() then zeroes `cfg.hidden_visuals`
    for the stage-assembly-time hide); and the "head_camera" inset was
    requested, which is itself ego-positioned regardless of what the main
    camera is doing. Without --cam-traj the main camera IS the ego view, so
    the existing stage-assembly-time hide already covers both it and the
    inset -- nothing to toggle per frame.
    """
    if not (cam_traj_given and "head_camera" in inset_camera_names):
        return []
    return [f"/visuals/{rel}" for rel in hidden_visual_rels]


def _paste_inset(frame, rgba, label: str, w: int, h: int, x: int, y: int) -> None:
    """Same visual treatment as mujoco_render.py's `_make_insets` loop: the
    sub-image, a white border, and a black-outlined white label -- so a
    reviewer cannot tell which renderer drew a given inset from its chrome."""
    import cv2

    frame[y : y + h, x : x + w] = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
    cv2.rectangle(frame, (x - 1, y - 1), (x + w, y + h), (255, 255, 255), 1)
    for colour, thick in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(frame, label, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, thick, cv2.LINE_AA)


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
    inset_specs, inset_layout = {}, {}
    if args.insets:
        if not args.state_trace:
            log("WARNING --insets needs --state-trace for camera FK; skipping insets")
        else:
            inset_layout = dataset_inset_layout(args.width, args.height)
            for cam_name, (label, w, h, x, y) in inset_layout.items():
                pos, quat, fovy = F.camera_spec(args.scene_xml, cam_name)
                inset_specs[cam_name] = (pos, quat, fovy, w, h)
                log(f"inset [{cam_name}] pos={pos} quat={quat} fovy={fovy} size={w}x{h}@({x},{y}) label={label!r}")
    joint_map = F.build_joint_map()
    if args.state_trace:
        seq, qpos, qvel = F.load_state_trace(args.state_trace, args.session, args.episode)
        log(f"replay: {len(qpos)} recorded states from {args.state_trace}")
        joints = pel = objs = None
    else:
        log(
            "WARNING no --state-trace: reconstructing state from parquet + "
            "sidecar. Two provenances, two clocks -- this does not reproduce "
            "the recorded video and is only for data older than the trace."
        )
        joints, pel, objs = F.load_frames(args.session, args.episode, args.sidecar)
        qpos = qvel = seq = None
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
        supersample=args.supersample,
    )
    app = S.start_app(cfg)
    # SimulationApp.close() terminates the process itself, and it does so with
    # status 0 -- so a raise inside the render used to surface as SUCCESS to
    # whoever ran it. Caught 2026-09-05 when a missing scene USD exited 0.
    # Record the failure, close, then exit non-zero explicitly.
    try:
        _render(
            args,
            cfg,
            joint_map,
            joints,
            pel,
            objs,
            cam_spec,
            log,
            qpos=qpos,
            qvel=qvel,
            seq=seq,
            inset_specs=inset_specs,
            inset_layout=inset_layout,
        )
    except BaseException:
        import traceback

        log("FAILED:\n" + traceback.format_exc())
        # NOT app.close() first: close() terminates the process itself, with
        # status 0, so anything after it never runs and the failure surfaces as
        # success. Kit teardown is skipped on the failure path on purpose.
        sys.stderr.write(f"render FAILED, see {log_path}\n")
        os._exit(1)
    app.close()


def _render(
    args,
    cfg,
    joint_map,
    joints,
    pel,
    objs,
    cam_spec,
    log,
    qpos=None,
    qvel=None,
    seq=None,
    inset_specs=None,
    inset_layout=None,
) -> None:
    import shutil
    import tempfile

    import cv2

    inset_specs, inset_layout = inset_specs or {}, inset_layout or {}
    stg = S.assemble_stage(cfg, log)
    S.apply_rtx_settings(getattr(args, "rtx_setting", None), log)  # tests drive _render with a bare namespace
    ghost = ghost_paths = None
    if args.ghost_trace and qpos is not None:
        from locomani.sim.isaac import ghost as G
        from locomani.sim.isaac.replay import MujocoReplay as _MR
        from locomani.tasks.crate.ghost_trace import ghost_poses, load_ghost_trace

        ghost = load_ghost_trace(args.ghost_trace)
        ghost_paths = G.add_isaac(stg, _MR(args.scene_xml).model, log)
        log(f"ghost trace: {len(ghost)} rows")
    world = S.make_world()
    robot, jidx, jsrc = S.bind_robot(joint_map, log)
    sensor, cam_path = S.mount_camera(stg, cfg, cam_spec, log)
    # inset_specs is only non-empty when --state-trace was given (checked in
    # main()), so `replay` below is guaranteed to end up set whenever these
    # exist -- every inset camera is re-posed from replay FK every frame,
    # independent of the main camera's own pose source.
    inset_sensors = {name: s for name, (s, _path) in S.mount_inset_cameras(stg, cfg, inset_specs, log).items()}

    # Replay path: one MuJoCo state decoder, and every quantity Isaac needs is
    # read back out of the SAME mj_forward -- joints, object poses, camera.
    replay = dof_idx = None
    if qpos is not None:
        from locomani.sim.isaac.replay import MujocoReplay

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

    T = len(qpos) if replay is not None else len(joints)
    if args.end_frame:
        T = min(T, args.end_frame)

    cam_traj = None
    if args.cam_traj:
        npz = np.load(args.cam_traj)
        cam_traj = (npz["cam_pos"], npz["cam_quat"])
        log(f"cam-traj loaded: {len(cam_traj[0])} frames")

    def push(t):
        """One frame into the stage."""
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
            for cam_name, isensor in inset_sensors.items():
                ipos, iquat = replay.camera_pose(cam_name)
                isensor.set_world_pose(position=ipos, orientation=iquat, camera_axes="usd")
            if ghost is not None:
                gpos, gquat = ghost_poses(ghost, int(seq[t]))
                G.set_isaac(ghost_paths, gpos, gquat)
            return
        cp = None if cam_traj is None else (cam_traj[0][t], cam_traj[1][t])
        S.sync_frame(
            stg, robot, jidx, jsrc, joints[t], pel[t], {k: v[t] for k, v in objs.items()}, sensor=sensor, cam_pose=cp
        )

    # === PASS 1: the main view. This is the ORIGINAL single-camera pipeline
    # unchanged from before --insets existed -- no visibility juggling, no
    # extra world.step() calls sharing this pass's state with anything else.
    # A prior design rendered the insets and the main view interleaved within
    # one shared stepping loop, toggling ghost/head-mesh visibility between
    # them; that was tried twice (2026-09-13) and twice left the robot's body
    # adrift in the main view even after re-pushing the frame's pose between
    # the two toggles, most likely PhysX carrying momentum across the insets
    # pass's extra world.step() calls in a way a single same-frame re-push did
    # not fully undo. Keeping this pass exactly as it always worked, and
    # doing the insets afterward as their own fully separate pass (below),
    # removes the failure mode instead of chasing it with more toggles.
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

    final_path = os.path.join(args.out, "ego_isaac.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = args.fps / args.stride
    # Frames the main pass actually produced land as lossless PNGs (not a
    # second lossy video) under args.out, not system /tmp: a shared machine's
    # /tmp can be small and this render's own output directory is already
    # provisioned for this episode's data. Only allocated when insets are
    # requested; without them pass 1 writes directly to the final file, byte-
    # for-byte the original behaviour.
    main_dir = tempfile.mkdtemp(prefix="main_", dir=args.out) if inset_layout else None
    vw = None
    try:
        vw = None if inset_layout else cv2.VideoWriter(final_path, fourcc, fps, (args.width, args.height))
        written_ts = []
        for t in range(0, T, args.stride):
            push(t)
            rgba = S.step_and_grab(world, sensor)
            if rgba is None:
                log(f"frame {t}: empty rgba")
                continue
            frame = cv2.cvtColor(S.downsample(rgba[..., :3], cfg.supersample), cv2.COLOR_RGB2BGR)
            if main_dir is not None:
                cv2.imwrite(os.path.join(main_dir, f"{len(written_ts):06d}.png"), frame)
            else:
                vw.write(frame)
            written_ts.append(t)
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
        if vw is not None:
            vw.release()
        expect = len(range(0, T, args.stride))
        log(f"main pass: wrote {len(written_ts)}/{expect} frames (source frames {T}, stride {args.stride})")

        # === PASS 2: the insets, entirely separate from pass 1 above -- its
        # own warm-up, its own stepping loop, reading pass 1's already-
        # rendered frames back off disk to composite onto rather than sharing
        # any world.step() with the main view. Ghost hands and the head mesh
        # (see ego_only_mesh_paths) are hidden ONCE for this whole pass, not
        # toggled per frame -- the same "set once, leave it" pattern
        # assemble_stage's own hidden_visuals already relies on for the
        # no-insets case, and the only thing about this pass that isn't a
        # verbatim rerun of pass 1's own logic against a different sensor set.
        written = len(written_ts)
        if main_dir is not None:
            from pxr import UsdGeom

            ego_only_meshes = ego_only_mesh_paths(bool(args.cam_traj), inset_sensors, S.StageCfg.hidden_visuals)
            hide_paths = (list(ghost_paths.values()) if ghost is not None else []) + ego_only_meshes
            for path in hide_paths:
                UsdGeom.Imageable(stg.GetPrimAtPath(path)).MakeInvisible()

            for _i in range(8):
                push(0)
                if all(v is not None for v in S.step_and_grab_all(world, inset_sensors).values()):
                    if _i:
                        log(f"inset warm-up: {_i} discarded step(s)")
                    break
            else:
                raise RuntimeError("Inset cameras missing or stale after 8 warm-up steps")
            prev, calm = None, 0
            for _i in range(80):
                push(0)
                grabbed = S.step_and_grab_all(world, inset_sensors)
                if any(v is None for v in grabbed.values()):
                    prev, calm = None, 0
                    continue
                cur = {name: rgba[..., :3].astype(np.float32) for name, rgba in grabbed.items()}
                quiet = prev is not None and all(float(np.abs(cur[n] - prev[n]).mean()) < 0.5 for n in cur)
                calm = calm + 1 if quiet else 0
                if calm >= 3:
                    log(f"inset pictures settled after {_i + 1} extra warm-up steps")
                    break
                prev = cur
            else:
                log("WARNING inset pictures still changing after 80 extra warm-up steps")

            vw = cv2.VideoWriter(final_path, fourcc, fps, (args.width, args.height))
            written = 0
            for i, t in enumerate(written_ts):
                push(t)
                grabbed = S.step_and_grab_all(world, inset_sensors)
                missing = [name for name in inset_layout if grabbed.get(name) is None]
                if missing:
                    raise RuntimeError(f"frame {t}: missing/stale inset cameras: {', '.join(missing)}")
                frame = cv2.imread(os.path.join(main_dir, f"{i:06d}.png"))
                if frame is None:
                    raise RuntimeError(f"frame {t}: missing main-pass image")
                for cam_name, (label, w, h, x, y) in inset_layout.items():
                    _paste_inset(frame, S.downsample(grabbed[cam_name][..., :3], cfg.supersample), label, w, h, x, y)
                vw.write(frame)
                written += 1
                if t % 200 == 0:
                    log(f"inset frame {t}/{T}")
            vw.release()
            for path in hide_paths:
                UsdGeom.Imageable(stg.GetPrimAtPath(path)).MakeVisible()
            log(f"inset pass: wrote {written}/{len(written_ts)} composited frames")
    finally:
        if vw is not None:
            vw.release()
        if main_dir is not None:
            shutil.rmtree(main_dir, ignore_errors=True)

    if written != expect:
        raise SystemExit(
            f"frame count mismatch: wrote {written}, expected {expect}. An "
            f"image stream shorter than its action stream is misaligned for the "
            f"whole episode, so this is fatal rather than a warning."
        )
    log("done")


if __name__ == "__main__":
    main()
