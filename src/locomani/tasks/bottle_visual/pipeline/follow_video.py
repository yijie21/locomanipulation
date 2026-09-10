"""Follow-camera videos from saved qpos; no simulation integration or policy calls."""

import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from locomani.tasks.bottle_visual.harness import atomic_json

CAMERA_VERSION = "right_front_follow_v1"
WIDTH, HEIGHT, HEADER, FPS = 800, 600, 96, 25


def heading(quaternion):
    w, x, y, z = quaternion
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def wrapped(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def camera_values(pelvis, yaw, previous=None):
    """Eye lies 80deg to robot's right of its front direction, never behind it."""
    target = np.array([pelvis[0], pelvis[1], 0.70])
    if previous is None:
        return target, yaw
    look, old_yaw = previous
    yaw_smooth = old_yaw + 0.35 * wrapped(yaw - old_yaw)
    # Limit lag so even during a turn the view stays in the front hemisphere.
    yaw_smooth = yaw + np.clip(wrapped(yaw_smooth - yaw), -np.radians(4), np.radians(4))
    look = look + 0.30 * (target - look)
    # Also bound spatial lag during faster approaches.
    look[:2] = target[:2] + np.clip(look[:2] - target[:2], -0.025, 0.025)
    return look, yaw_smooth


def frame_bounds(renderer, points):
    cam = renderer.scene.camera[0]
    eye = np.mean([c.pos for c in renderer.scene.camera], axis=0)
    delta = points - eye
    forward, up = np.asarray(cam.forward), np.asarray(cam.up)
    right = np.cross(forward, up)
    depth = delta @ forward
    slope = (cam.frustum_top - cam.frustum_bottom) / (2 * cam.frustum_near)
    projected = np.column_stack((delta @ right / (depth * slope * WIDTH / HEIGHT), delta @ up / (depth * slope)))
    return projected, depth, eye


def render_case(case, video, ordinal, parameters, preview_only=False):
    case, video = Path(case), Path(video)
    result = json.loads((case / "result.json").read_text())
    qfile = case / "trace.npz"
    saved = np.load(qfile)
    qpos, dt = saved["qpos"], float(saved["dt"])
    commands = json.loads((case / "commands.json").read_text())
    if len(qpos) != len(commands) or not np.isfinite(qpos).all() or not np.isclose(dt, 0.02):
        raise ValueError("Invalid saved trace")
    model = mujoco.MjModel.from_xml_path(str(Path(result["scene_directory"]) / "scene_locomani.xml"))
    data = mujoco.MjData(model)
    pelvis_id = model.body("pelvis").id
    robot_ids = np.flatnonzero(model.body_rootid == model.body_rootid[pelvis_id])
    wrist = model.body("right_wrist_yaw_link").id
    metadata = video.with_suffix(".json")
    source_hash = hashlib.sha256(qfile.read_bytes()).hexdigest()
    if video.exists() and metadata.exists() and not preview_only:
        prior = json.loads(metadata.read_text())
        if (
            prior["trace_sha256"] == source_hash
            and prior["camera_version"] == CAMERA_VERSION
            and hashlib.sha256(video.read_bytes()).hexdigest() == prior["video_sha256"]
        ):
            return prior
        raise ValueError(f"Existing video differs: {video}")
    # Software rendering: shadows/reflections/MSAA dominate cost and are not
    # needed to inspect motion. These visual settings never enter simulation.
    model.vis.quality.offsamples = 0
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    cam = mujoco.MjvCamera()
    cam.elevation = -12
    indices = list(range(0, len(qpos), 2))  # Include settle, approach, grasp and final outcome.
    checkpoints = sorted(set([0, len(indices) // 3, 2 * len(indices) // 3, len(indices) - 1]))
    snapshots = []
    camera_state = None
    camera_trace = []
    min_front = 1.0
    max_bound = 0.0
    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 19)
    partial = video.with_suffix(".partial.mp4")
    encoder = None
    try:
        if not preview_only:
            encoder = subprocess.Popen(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{WIDTH}x{HEIGHT + HEADER}",
                    "-r",
                    str(FPS),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "1",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "20",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ],
                stdin=subprocess.PIPE,
            )
        distance = 2.65
        for f, i in enumerate(indices):
            data.qpos[:] = qpos[i]
            mujoco.mj_kinematics(model, data)
            yaw = heading(qpos[i, 3:7])
            camera_state = camera_values(qpos[i, :3], yaw, camera_state)
            cam.lookat[:] = camera_state[0]
            cam.azimuth = np.degrees(camera_state[1]) - 80 + 180
            # Extent includes limb bodies, palm tip, and a conservative head cap.
            points = data.xpos[robot_ids].copy()
            pad = data.xpos[wrist] + data.xmat[wrist].reshape(3, 3) @ np.array([0.1515, 0, 0])
            shoulder = (
                data.xpos[model.body("left_shoulder_pitch_link").id]
                + data.xpos[model.body("right_shoulder_pitch_link").id]
            ) / 2
            bottle = data.xpos[model.body("water_bottle_object").id]
            points = np.vstack(
                (
                    points,
                    pad,
                    shoulder + [0, 0, 0.26],
                    qpos[i, :3] * [1, 1, 0],
                    bottle + [0, 0, 0.09],
                    bottle - [0, 0, 0.09],
                )
            )
            cam.distance = distance
            for _ in range(8):
                renderer.update_scene(data, camera=cam)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                bounds, depth, eye = frame_bounds(renderer, points)
                bound = float(np.max(abs(bounds)))
                if np.min(depth) > 0 and bound <= 0.90:
                    break
                cam.distance *= 1.10
            else:
                raise RuntimeError("Cannot fit complete robot in follow camera")
            distance = max(distance, float(cam.distance))
            ray = eye[:2] - qpos[i, :2]
            ray /= np.linalg.norm(ray)
            front = float(ray @ np.array([np.cos(yaw), np.sin(yaw)]))
            if front <= 0:
                raise RuntimeError("Follow camera moved behind robot")
            min_front = min(min_front, front)
            max_bound = max(max_bound, bound)
            camera_trace.append(
                dict(
                    frame=f,
                    t=(i + 1) * dt,
                    lookat=cam.lookat.tolist(),
                    distance=float(cam.distance),
                    azimuth=float(cam.azimuth),
                    front_dot=front,
                    framing_bound=bound,
                )
            )
            if preview_only and f not in checkpoints:
                continue
            frame = Image.new("RGB", (WIDTH, HEIGHT + HEADER), "#f5f7fa")
            frame.paste(Image.fromarray(renderer.render()), (0, HEADER))
            draw = ImageDraw.Draw(frame)
            names = {"high_near": "高桌近瓶", "low_near": "低桌近瓶", "low_deep": "低桌深瓶"}
            phases = {
                "settle": "稳定启动",
                "walk": "走近",
                "reach": "伸手",
                "entry": "对齐进入",
                "close": "合爪",
                "lift": "试提",
                "carry": "回收持瓶",
            }
            phase = commands[i]["phase"]
            outcome = "成功" if result["ok"] else "失败：" + result["why"]
            draw.text(
                (10, 3),
                f"{ordinal:02d}/10  a16专家 | {names[result['stratum']]} | seed {result['seed']} | {result['edge']}",
                font=font,
                fill="#18334f",
            )
            draw.text(
                (10, 27),
                f"瓶子横向{parameters['bottle_along_m'] * 100:+.1f}cm  向内{parameters['inboard_m'] * 100:.1f}cm  起步{parameters['spawn_edge_m']:.2f}m",
                font=font,
                fill="#18334f",
            )
            draw.text(
                (10, 51),
                f"起步横移{parameters['spawn_lat_m'] * 100:+.1f}cm  朝向偏差{parameters['yaw_error_deg']:+.1f}°  t={(i + 1) * dt:.2f}s  {phases.get(phase, phase)}",
                font=font,
                fill="#18334f",
            )
            draw.text((10, 75), f"侧前方全身跟随 | 本集结果：{outcome}", font=font, fill="#18334f")
            if f in checkpoints:
                snap = video.with_name(video.stem + f"_frame{f:04d}.jpg")
                frame.save(snap, quality=88)
                snapshots.append(str(snap))
            if encoder is not None:
                encoder.stdin.write(np.asarray(frame).tobytes())
    finally:
        renderer.close()
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("Video encoding failed")
    if preview_only:
        return dict(snapshots=snapshots, min_front_dot=min_front, max_framing_bound=max_bound)
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(partial),
            ]
        )
    )["streams"][0]
    if (
        probe["r_frame_rate"] != "25/1"
        or int(probe["nb_read_frames"]) != len(indices)
        or (probe["width"], probe["height"]) != (WIDTH, HEIGHT + HEADER)
    ):
        raise RuntimeError("Encoded video integrity mismatch")
    partial.replace(video)
    report = dict(
        video=str(video),
        seed=result["seed"],
        stratum=result["stratum"],
        success=result["ok"],
        why=result["why"],
        parameters=parameters,
        trace_sha256=source_hash,
        camera_version=CAMERA_VERSION,
        frames=len(indices),
        duration_s=len(indices) / FPS,
        probe=probe,
        snapshots=snapshots,
        all_frames_robot_extent_inside_margin=True,
        max_framing_bound=max_bound,
        min_front_dot=min_front,
        note="Projection checks framing, not visibility through occluding objects; inspect contact sheet.",
        render_style="software: no shadows, reflections or multisample antialiasing",
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
    )
    atomic_json(video.with_suffix(".camera.json"), camera_trace)
    atomic_json(metadata, report)
    return report
