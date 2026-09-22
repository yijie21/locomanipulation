"""Reach-envelope reel from a search run: per (h, d) a y × yaw tile grid, each tile the farthest passing trial.

    python -m locomani.tasks.crate.reach_envelope.render_reel <run_dir> <h[,h...]> <name>

Writes <run_dir>/videos/<name>_source.mp4 (encode it under 500,000 bytes before delivering).
Ghost hands are drawn as collision-free mocap overlays on a copy of the run's floor scene.
"""

import json
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from locomani import paths
from locomani.tasks.crate.reach_envelope.ghost_scene import build_ghost_scene, set_ghost_pose

RUN = Path(sys.argv[1]).resolve()
H_SEL = [float(a) for a in sys.argv[2].split(",")]
NAME = sys.argv[3]
TW, TH, HDR, CAP = 256, 192, 64, 34
cfg = json.load(open(RUN / "search.json"))
Y, YAW, D = tuple(cfg["grid"]["y"]), tuple(cfg["grid"]["yaw"]), tuple(cfg["grid"]["d"])
rows = json.load(open(RUN / "boundary_table.json"))
scene_dir = Path(cfg["scene_dir"])
if not scene_dir.is_absolute():
    scene_dir = Path(paths.REPO) / scene_dir
model = build_ghost_scene(scene_dir, RUN)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=TH, width=TW)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0.25, 0, 0.55]
cam.distance, cam.azimuth, cam.elevation = 2.0, 140, -14
opt = mujoco.MjvOption()
opt.geomgroup[:] = 0
opt.geomgroup[1] = opt.geomgroup[4] = 1
wrists = [model.body(s + "_wrist_yaw_link").id for s in ("left", "right")]
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"


def font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def add_connector(scene, start, end, rgba, width=0.003):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, dtype=np.float32)
    )
    mujoco.mjv_connector(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    )
    scene.ngeom += 1


def frames(c):
    n = np.load(c["artifacts"]["state_trace"])
    qpos, seq = n["qpos"], n["seq"]
    centres = np.asarray(c["target"]["fixed_centres_world_m"])
    rot = np.asarray(c["target"]["permitted_tool_rotation"])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    idx = list(range(0, len(seq), 4))
    idx[-1:] = [len(seq) - 1]
    out = []
    set_ghost_pose(model, data, centres, rot)
    for j in idx:
        data.qpos[:] = qpos[j]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        rs = data.xmat[wrists].reshape(2, 3, 3)
        tools = data.xpos[wrists] + np.einsum("nij,j->ni", rs, [0.1515, 0, 0])
        err = np.linalg.norm(tools - centres, axis=1).max() * 100
        renderer.update_scene(data, camera=cam, scene_option=opt)
        for i in range(2):
            add_connector(renderer.scene, tools[i], centres[i], [1, 0.45, 0.1, 0.9])
        out.append((renderer.render().copy(), err))
    return out


W, Hh = len(YAW) * TW, HDR + len(Y) * (TH + CAP)
src = RUN / "videos" / f"{NAME}_source.mp4"
src.parent.mkdir(exist_ok=True)
proc = subprocess.Popen(
    [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{Hh}",
        "-r",
        "12.5",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(src),
    ],
    stdin=subprocess.PIPE,
)
for h in H_SEL:
    for d in D:
        tiles = {}
        for a, y in enumerate(Y):
            for b, yaw in enumerate(YAW):
                r = next(r for r in rows if (r["y"], r["h"], r["d"], r["yaw"]) == (y, h, d, yaw))
                if r["x_pass"] is None:
                    # No passing x: replay the closest attempt so the shortfall is visible.
                    measured = [s for s in r["samples"] if s["pos"] is not None]
                    if not measured:
                        tiles[(a, b)] = None
                        continue
                    sample = min(measured, key=lambda s: s["pos"])
                    c = json.load(open(RUN / sample["case"]))["case"]
                    tiles[(a, b)] = (sample["x"], frames(c), False)
                    continue
                sample = next(s for s in r["samples"] if abs(s["x"] - r["x_pass"]) < 1e-6)
                c = json.load(open(RUN / sample["case"]))["case"]
                tiles[(a, b)] = (r["x_pass"], frames(c), True)
        n = max((len(t[1]) for t in tiles.values() if t), default=12)
        squat = cfg["envelope"].get("pelvis_height_target") or next(
            (r["envelope"].get("pelvis_height_target") for r in cfg.get("envelope_overrides", []) if h < r["h_below"]),
            None,
        )
        for k in range(n + 12):
            im = Image.new("RGB", (W, Hh), "#101d2b")
            dr = ImageDraw.Draw(im)
            dr.text(
                (12, 8),
                f"G1 双臂前伸包络   抓点高度 h = {h:.2f} m   双手间距 d = {d:.2f} m"
                + (f"   下蹲目标 {squat:.2f} m" if squat else "   不下蹲"),
                font=font(22),
                fill="white",
            )
            dr.text(
                (12, 38),
                "每格 = 该 (y, yaw) 下位置误差 ≤ 2 cm 的最远 x（橙字 = 无解，回放最接近的一次）；橙线 = 手到目标；双脚不动",
                font=font(15),
                fill="#9fb3c8",
            )
            for (a, b), t in tiles.items():
                x0, y0 = b * TW, HDR + a * (TH + CAP)
                if t is None:
                    dr.rectangle((x0, y0, x0 + TW, y0 + TH), fill="#1b2a3a")
                    dr.text((x0 + 70, y0 + 80), "无有效尝试", font=font(16), fill="#8a8f98")
                    label, col = "×", "#8a8f98"
                else:
                    xp, fr, passed = t
                    rgb, err = fr[min(k, len(fr) - 1)]
                    im.paste(Image.fromarray(rgb), (x0, y0))
                    if passed:
                        label = f"x {xp:.2f} m   手位 {err:4.1f} cm"
                        col = "#7CFC9A" if err <= 2 else "#d8e9f6"
                    else:
                        label = f"× 最近一次 x {xp:.2f} m   手位 {err:4.1f} cm"
                        col = "#ffb37c"
                dr.text((x0 + 6, y0 + TH + 2), f"y {Y[a]:+.2f}  yaw {YAW[b]:+.0f}°", font=font(13), fill="#9fb3c8")
                dr.text((x0 + 6, y0 + TH + 17), label, font=font(13), fill=col)
            proc.stdin.write(im.tobytes())
proc.stdin.close()
proc.wait()
print(src)
