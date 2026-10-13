"""Render the walk-to-marker brake trials: one video per speed, trials back to back, with subtitles.

Input: a diag_brake output directory (results.json + trace/brake_<speed>_<i>_state_trace.npz). The red
floor line is the marker where IDLE was commanded; subtitles give the speed command, the trial, the
phase, the 0.5 s pelvis speed, the distance past the marker, and -- once settled -- the carry.
A row may carry its own `marker` (the ScaleBFM side jitters the run-up, wbc_compare bfm_brake.py) and the
file its own `phase_labels` {walk, brake, slow} for the subtitles; `target_fwd` rows add the target error.
Also writes summary.json, including how far the frontmost low-body point (geoms reaching below
OBSTACLE_TOP) went past the line at worst and after settling.

    MUJOCO_GL=egl third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/render_brake_marker_20260923.py outputs/diagnostics/brake_marker_v11_20260923
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[2]
SCENE = REPO / "outputs" / "scene" / "walk" / "scene_locomani.xml"
FONT = REPO / "assets" / "blender" / "fonts" / "NotoSansCJK-Regular.ttc"
HZ, EVERY = 50, 2
OBSTACLE_TOP = 0.37  # m: geoms reaching below this count toward the line (the logistics pallet is 0.371 high)
W, H = 960, 540


def yaw_of(q):
    w, x, y, z = q[3:7]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def frontmost(model, data, h, x0, robot):
    """Farthest point along h of any robot geom (world AABB) whose bottom is below OBSTACLE_TOP."""
    best = -np.inf
    for g in robot:
        rot = data.geom_xmat[g].reshape(3, 3)
        centre = data.geom_xpos[g] + rot @ model.geom_aabb[g, :3]
        half = np.abs(rot) @ model.geom_aabb[g, 3:]
        if centre[2] - half[2] < OBSTACLE_TOP:
            best = max(best, float((centre[:2] - x0) @ h + np.abs(h) @ half[:2]))
    return best


def add_marker(scene, centre, heading):
    """A thin red bar across the walking line, lying on the floor."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    rot = np.array([[np.cos(heading), -np.sin(heading), 0], [np.sin(heading), np.cos(heading), 0], [0, 0, 1]])
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_BOX,
        np.array([0.01, 0.6, 0.002]),
        np.r_[centre, 0.003],
        rot.ravel(),
        np.array([0.9, 0.1, 0.1, 1.0], dtype=np.float32),
    )
    scene.ngeom += 1


def compress(full: Path, preview: Path):
    for crf, scale, fps in ((32, 640, 12), (36, 640, 10), (38, 560, 8), (40, 480, 8)):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(full), "-vf", f"scale={scale}:-2,fps={fps}",
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(preview)],
            check=True,
        )  # fmt: skip
        if preview.stat().st_size <= 500_000:
            return


def main() -> int:
    out = Path(sys.argv[1])
    res = json.loads((out / "results.json").read_text())
    marker0 = res["marker"]
    labels = {"walk": "行走 SLOW_WALK", "brake": "刹车 IDLE", "slow": "减速", **res.get("phase_labels", {})}
    rows = [r for r in res["rows"] if r.get("fell") is None and "carry" in r]
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    robot = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0]
    font = ImageFont.truetype(str(FONT), 26)
    summary = {}

    def tag(r):
        return f"{r['speed']:.2f}" if r.get("final_speed") is None else f"{r['speed']:.2f}to{r['final_speed']:.2f}"

    for key in sorted({tag(r) for r in res["rows"]}, reverse=True):
        mine = [r for r in rows if tag(r) == key]
        speed = mine[0]["speed"]
        past = {}  # label -> per-frame frontmost past the line, metres
        for r in mine:
            qpos = np.load(out / "trace" / f"{r['label']}_state_trace.npz")["qpos"]
            heading = yaw_of(qpos[0])
            h, x0 = np.array([np.cos(heading), np.sin(heading)]), qpos[0, :2]
            vals = []
            for q in qpos:
                data.qpos[:] = q
                mujoco.mj_kinematics(model, data)
                vals.append(frontmost(model, data, h, x0, robot) - r.get("marker", marker0))
            past[r["label"]] = np.array(vals)
        worst = np.array([past[r["label"]].max() for r in mine])
        final = np.array([past[r["label"]][-1] for r in mine])
        carry = np.array([r["carry"] for r in mine])
        stop = np.array([r["stop_fwd"] - r.get("marker", marker0) for r in mine])
        summary[key] = dict(
            n=len(mine),
            falls=sum(1 for r in res["rows"] if tag(r) == key and r.get("fell")),
            brake_speed_mean=float(np.mean([r["brake_speed"] for r in mine])),
            carry_mean_cm=float(100 * carry.mean()),
            carry_sd_cm=float(100 * carry.std()),
            carry_range_cm=[float(100 * carry.min()), float(100 * carry.max())],
            lateral_sd_cm=float(100 * np.std([r["stop_lat"] for r in mine])),
            yaw_abs_mean_deg=float(np.mean([abs(r["yaw_deg"]) for r in mine])),
            stop_past_marker_cm=[round(float(100 * v), 1) for v in stop],
            frontmost_max_past_line_cm=[round(float(100 * v), 1) for v in worst],
            frontmost_final_past_line_cm=[round(float(100 * v), 1) for v in final],
            crossed=int((worst > 0).sum()),
            stagger_cm=[round(100 * r["stagger"], 1) for r in mine if "stagger" in r],
            stagger_before_resquare_cm=[round(100 * r["stagger_settled"], 1) for r in mine if "stagger_settled" in r],
            target_err_cm=[
                round(100 * (r["stop_fwd"] - r["target_fwd"]), 1) for r in mine if r.get("target_fwd") is not None
            ],
            still_after_brake_s=[
                round(r["still_after_brake_s"], 2) for r in mine if r.get("still_after_brake_s") is not None
            ],
        )
        tmp = out / f"brake_{key}_raw.mp4"
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), HZ / EVERY, (W, H))
        for r in mine:
            z = np.load(out / "trace" / f"{r['label']}_state_trace.npz")
            qpos = z["qpos"]
            heading = yaw_of(qpos[0])
            h = np.array([np.cos(heading), np.sin(heading)])
            xy0 = qpos[0, :2]
            marker = r.get("marker", marker0)
            fwd = (qpos[:, :2] - xy0) @ h
            v = np.gradient(fwd) * HZ
            v = np.convolve(v, np.ones(HZ // 2) / (HZ // 2), mode="same")
            braked = (
                r["brake_seq"]
                if "brake_seq" in r
                else (np.argmax(fwd >= marker) if (fwd >= marker).any() else len(fwd))
            )
            for i in range(0, len(qpos), EVERY):
                data.qpos[:] = qpos[i]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="walk_follow")
                add_marker(renderer.scene, xy0 + h * marker, heading)
                frame = Image.fromarray(renderer.render())
                d = ImageDraw.Draw(frame)
                slow = r.get("final_speed") is not None and fwd[i] >= marker - r["final_distance"]
                phase = (
                    labels["brake"]
                    if i >= braked
                    else (f"{labels['slow']} {r['final_speed']:.2f}" if slow else labels["walk"])
                )
                lines = [
                    f"指令速度 {key.replace('to', '→')}   第 {r['i'] + 1}/{len(mine)} 次   {phase}",
                    f"实测速度 {v[i]:+.2f} m/s   身体最前点距红线 {100 * past[r['label']][i]:+.1f} cm",
                ]
                if i >= len(qpos) - HZ:  # the last second: the settled result
                    worst_i = 100 * past[r["label"]].max()
                    verdict = "越线！" if worst_i > 0 else "未越线"
                    lines.append(f"全程最前点 {worst_i:+.1f} cm（{verdict}），停稳后 {100 * past[r['label']][-1]:+.1f} cm，"
                                 f"偏航 {r['yaw_deg']:+.1f}°")  # fmt: skip
                    if "stagger" in r:
                        lines.append(f"两脚前后差 {100 * r['stagger']:.1f} cm")
                for k, text in enumerate(lines):
                    d.text(
                        (20, 16 + 36 * k), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0)
                    )
                writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        writer.release()
        full, preview = out / f"brake_{key}.mp4", out / f"brake_{key}_preview.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(tmp), "-c:v", "libx264", "-crf", "20",
             "-pix_fmt", "yuv420p", str(full)],
            check=True,
        )  # fmt: skip
        tmp.unlink()
        compress(full, preview)
        print(f"{preview} {preview.stat().st_size} bytes")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
