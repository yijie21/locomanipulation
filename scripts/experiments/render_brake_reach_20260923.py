"""Render brake_then_reach trials: one video per reach target, trials back to back, red line + target dots + subtitles.

Per trial the brake trace and the reach trace (trace/<label>_{brake,reach}_state_trace.npz) are joined. Measured
from the MuJoCo state, against the line: the frontmost point of any geom on the two hands (wrist-yaw link and the
dex1 fingers, world AABB), and the frontmost low-body point (geoms reaching below OBSTACLE_TOP) -- the part that
must stay behind the line. Writes reach_<x>.mp4, a <=500 kB preview (split into parts if needed), summary.json.

    MUJOCO_GL=egl third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/render_brake_reach_20260923.py outputs/diagnostics/brake_reach_20260923
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
HZ, EVERY, W, H = 50, 2, 960, 540
OBSTACLE_TOP = 0.37
HAND_BODIES = [
    f"{s}_{b}" for s in ("left", "right") for b in ("wrist_yaw_link", "dex1_finger_1_link", "dex1_finger_2_link")
]


def extent(model, data, geoms, h, origin):
    """Farthest point along h over these geoms (world AABB)."""
    best = -np.inf
    for g in geoms:
        rot = data.geom_xmat[g].reshape(3, 3)
        centre = data.geom_xpos[g] + rot @ model.geom_aabb[g, :3]
        half = np.abs(rot) @ model.geom_aabb[g, 3:]
        best = max(best, float((centre[:2] - origin) @ h + np.abs(h) @ half[:2]))
    return best


def low_geoms(model, data, robot):
    out = []
    for g in robot:
        rot = data.geom_xmat[g].reshape(3, 3)
        if (data.geom_xpos[g] + rot @ model.geom_aabb[g, :3])[2] - (np.abs(rot) @ model.geom_aabb[g, 3:])[
            2
        ] < OBSTACLE_TOP:
            out.append(g)
    return out


def add_geom(scene, kind, size, pos, rot, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], kind, np.asarray(size, float), np.asarray(pos, float),
                        np.asarray(rot, float).ravel(), np.asarray(rgba, np.float32))  # fmt: skip
    scene.ngeom += 1


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", *args], check=True)


def previews(full: Path):
    """<=500 kB previews covering the whole clip, split into equal numbered parts when one is not enough."""
    duration = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                     str(full)], capture_output=True, text=True).stdout)  # fmt: skip
    for parts in (1, 2, 3, 4):
        outs = []
        for k in range(parts):
            dst = full.with_name(full.stem + ("_preview.mp4" if parts == 1 else f"_preview_part{k + 1}.mp4"))
            span = ["-ss", f"{k * duration / parts:.3f}", "-i", str(full), "-t", f"{duration / parts:.3f}"]
            ffmpeg(*span, "-vf", "scale=480:-2,fps=8", "-c:v", "libx264", "-preset", "medium", "-crf", "38",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst))  # fmt: skip
            outs.append(dst)
        if all(o.stat().st_size <= 500_000 for o in outs):
            return outs
        for o in outs:
            o.unlink()
    raise RuntimeError(f"{full}: no split under 500 kB")


def main() -> int:
    out = Path(sys.argv[1])
    res = json.loads((out / "results.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    robot = [g for g in range(model.ngeom) if model.geom_bodyid[g] != 0]
    hand_ids = {model.body(n).id for n in HAND_BODIES}
    hands = [g for g in robot if model.geom_bodyid[g] in hand_ids]
    font = ImageFont.truetype(str(FONT), 24)
    summary = {}
    for x in sorted({r["target_x"] for r in res["rows"]}):
        rows = [r for r in res["rows"] if r["target_x"] == x and "status" in r]
        full = out / f"reach_{x:.2f}.mp4"
        tmp = full.with_suffix(".raw.mp4")
        writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), HZ / EVERY, (W, H))
        stats = []
        for r in rows:
            h = np.array([np.cos(r["heading"]), np.sin(r["heading"])])
            line = np.asarray(r["line_xy"])
            targets = json.loads((out / "traces" / f"{r['label']}.json").read_text())["target"]["fixed_centres_world_m"]
            hand_past, low_past = [], []
            parts = [
                q for q in ("brake", "prep", "reach") if (out / "trace" / f"{r['label']}_{q}_state_trace.npz").exists()
            ]
            for part in parts:
                qpos = np.load(out / "trace" / f"{r['label']}_{part}_state_trace.npz")["qpos"]
                for i, q in enumerate(qpos):
                    data.qpos[:] = q
                    mujoco.mj_kinematics(model, data)
                    hp = extent(model, data, hands, h, line)
                    lp = extent(model, data, low_geoms(model, data, robot), h, line)
                    if part != "brake":
                        hand_past.append(hp)
                        low_past.append(lp)
                    if i % EVERY:
                        continue
                    mujoco.mj_forward(model, data)  # walk_follow is trackcom: it needs subtree_com and camlight
                    renderer.update_scene(data, camera="walk_follow")
                    rot = np.array([[h[0], -h[1], 0], [h[1], h[0], 0], [0, 0, 1]])
                    add_geom(
                        renderer.scene,
                        mujoco.mjtGeom.mjGEOM_BOX,
                        [0.01, 0.6, 0.002],
                        [*line, 0.003],
                        rot,
                        [0.9, 0.1, 0.1, 1],
                    )
                    if part != "brake":
                        for c in targets:
                            add_geom(
                                renderer.scene,
                                mujoco.mjtGeom.mjGEOM_SPHERE,
                                [0.02, 0, 0],
                                c,
                                np.eye(3),
                                [1, 0.2, 0.8, 0.8],
                            )
                    frame = Image.fromarray(renderer.render())
                    d = ImageDraw.Draw(frame)
                    lines = [
                        f"目标 x={x:.2f} m（骨盆前方）  第 {r['i'] + 1}/{len(rows)} 次  {dict(brake='行走+守线刹车', prep='抬臂准备+伸出', reach='双臂伸出')[part]}",
                        f"手最前点距红线 {100 * hp:+.1f} cm    下身最前点距红线 {100 * lp:+.1f} cm",
                    ]
                    if part == parts[-1] and i >= len(qpos) - HZ:
                        lines.append(
                            f"本次手最远 {100 * max(hand_past):+.1f} cm，下身最远 {100 * max(low_past):+.1f} cm（{r['reason']}）"
                        )
                    for k, text in enumerate(lines):
                        d.text(
                            (16, 12 + 32 * k),
                            text,
                            font=font,
                            fill=(255, 255, 255),
                            stroke_width=3,
                            stroke_fill=(0, 0, 0),
                        )
                    writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
            stats.append(dict(label=r["label"], status=r["status"], hand_max_past_line_cm=100 * max(hand_past),
                              low_body_max_past_line_cm=100 * max(low_past),
                              tool_max_past_line_cm=r["tool_max_past_line_cm"],
                              target_past_line_cm=r["target_past_line_cm"],
                              pelvis_ready_past_line_cm=r["ready_pelvis_past_line_cm"],
                              best_position_error_cm=r["best_position_error_cm"]))  # fmt: skip
        writer.release()
        ffmpeg("-i", str(tmp), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", str(full))
        tmp.unlink()
        summary[f"{x:.2f}"] = dict(trials=stats, previews=[str(p) for p in previews(full)])
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
