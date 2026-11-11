"""Offline rendering for the minimal proof's videos: qpos rows recorded at every 50 Hz motor tick are drawn later from
a second MjData of the same (runtime-resized) model, with marker spheres and caption text.

    MUJOCO_GL=egl $PY scripts/experiments/selfwm_20261029/videos.py all
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
W, H = 480, 360


class Recorder:
    """Wrap a Motor's tick so every simulated 50 Hz step keeps a copy of qpos."""

    def __init__(self, motor):
        self.rows: list[np.ndarray] = []
        self.motor = motor
        self._tick = motor.tick
        motor.tick = self.tick

    def tick(self) -> bool:
        ok = self._tick()
        self.rows.append(self.motor.sim.d.qpos.copy())
        return ok

    def mark(self) -> int:
        return len(self.rows)


class View:
    def __init__(
        self, model, lookat=(0.30, -0.16, 0.84), distance=1.05, azimuth=-135.0, elevation=-18.0, width=W, height=H
    ):
        self.m = model
        self.d = mujoco.MjData(model)
        self.w, self.h = width, height
        self.r = mujoco.Renderer(model, height=height, width=width, max_geom=model.ngeom + 500)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = lookat
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.font = ImageFont.truetype(FONT, 17)
        self.small = ImageFont.truetype(FONT, 14)

    def frame(self, qpos, markers=(), title="", lines=(), arrows=()) -> np.ndarray:
        """markers: (position, rgba, radius) spheres; arrows: (start, direction, length, rgba) drawn as MuJoCo arrows."""
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera=self.cam)
        sc = self.r.scene
        for pos, rgba, size in markers:
            if sc.ngeom >= sc.maxgeom:
                break
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]),
                                np.asarray(pos, float), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))  # fmt: skip
            sc.ngeom += 1
        for start, direction, length, rgba in arrows:
            if sc.ngeom >= sc.maxgeom:
                break
            z = np.asarray(direction, float) / np.linalg.norm(direction)
            x = np.cross(z, [0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.cross(z, [1.0, 0.0, 0.0])
            x /= np.linalg.norm(x)
            mat = np.stack([x, np.cross(z, x), z], -1)  # columns: the arrow's local axes, z along the arrow
            mujoco.mjv_initGeom(sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, np.array([0.006, 0.006, length]),
                                np.asarray(start, float), mat.reshape(-1), np.asarray(rgba, np.float32))  # fmt: skip
            sc.ngeom += 1
        img = Image.fromarray(self.r.render())
        dr = ImageDraw.Draw(img)
        if title:
            dr.rectangle([0, 0, self.w, 26], fill=(20, 24, 30))
            dr.text((8, 3), title, font=self.font, fill=(240, 240, 240))
        y = self.h - 20 * len(lines) - 6
        for ln in lines:
            dr.text((8, y), ln, font=self.small, fill=(15, 15, 15), stroke_width=2, stroke_fill=(250, 250, 250))
            y += 20
        return np.asarray(img)


def tile(frames: list[np.ndarray], cols: int) -> np.ndarray:
    rows = [np.concatenate(frames[i : i + cols], 1) for i in range(0, len(frames), cols)]
    return np.concatenate(rows, 0)


def write(frames: list[np.ndarray], path: Path, fps: int = 25) -> None:
    """Lossless-ish source mp4 (kept as the source recording)."""
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(path)],
                         stdin=subprocess.PIPE)  # fmt: skip
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    p.wait()


def preview(src: Path, dst: Path, crf: int = 32, scale: int | None = None) -> int:
    """docs/rendering.md#video-export: full duration and speed, size under 500,000 bytes (raise crf until it fits)."""
    while True:
        vf = f"scale={scale}:-2" if scale else "null"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src), "-map", "0:v:0", "-vf", vf,
                        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(dst)], check=True)  # fmt: skip
        n = dst.stat().st_size
        if n <= 480_000 or crf >= 45:
            return n
        crf += 2
