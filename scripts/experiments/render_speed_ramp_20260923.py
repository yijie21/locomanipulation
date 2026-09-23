"""Render the speed-ramp run with commanded and delivered speed as subtitles, plus a per-level table.

Input: a diag_speed_ramp output directory (commands.json + trace/speed_ramp_state_trace.npz), joined on the
privileged `seq`. Delivered speed = pelvis velocity along the start heading, from the MuJoCo qpos trace,
averaged over one second (a stride is ~0.7 s, so a shorter window shows the gait, not the speed).

    MUJOCO_GL=egl third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \
        scripts/experiments/render_speed_ramp_20260923.py outputs/diagnostics/speed_ramp_v11_20260923
Writes speed_ramp.mp4, speed_ramp_preview.mp4 (<= 500 kB), levels.json.
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
HZ, EVERY = 50, 2  # trace rows are control frames; render every 2nd -> 25 fps
W, H, CHART_H = 960, 540, 150
WINDOW_S = 1.0


def smooth(x, n):
    k = np.ones(n) / n
    pad = np.pad(x, (n // 2, n - 1 - n // 2), mode="edge")
    return np.convolve(pad, k, mode="valid")


def main() -> int:
    out = Path(sys.argv[1])
    cmd = json.loads((out / "commands.json").read_text())
    trace = np.load(out / "trace" / "speed_ramp_state_trace.npz")
    seq, qpos = trace["seq"], trace["qpos"]
    ticks = [t for t in cmd["ticks"] if t["seq"] is not None]
    walk_name = next((f"{t.get('mode_name', t['mode'])} ({cmd.get('mode', 1)})" for t in ticks if t["speed"] > 0), "")
    tseq = np.array([t["seq"] for t in ticks])
    tspeed = np.array([t["speed"] for t in ticks])
    # the command in force at each trace row: the latest tick at or before that seq
    idx = np.clip(np.searchsorted(tseq, seq, side="right") - 1, 0, len(ticks) - 1)
    commanded = np.where(seq >= tseq[0], tspeed[idx], 0.0)
    heading = np.array([np.cos(cmd["heading"]), np.sin(cmd["heading"])])
    fwd = qpos[:, :2] @ heading
    t = (seq - seq[0]) / HZ
    raw = np.gradient(fwd, t)
    delivered = smooth(raw, int(WINDOW_S * HZ))
    travelled = fwd - fwd[0]

    # per-level table: each held command, averaged over the second half of its plateau
    table, start = [], 0
    for i in range(1, len(commanded) + 1):
        if i == len(commanded) or commanded[i] != commanded[start] or (i < len(commanded) and seq[i] - seq[i - 1] > 5):
            span = slice(start + (i - start) // 2, i)
            table.append(
                dict(
                    commanded=float(commanded[start]),
                    phase="up" if not table or commanded[start] >= table[-1]["commanded"] else "down",
                    t0=float(t[start]),
                    delivered_mps=float(
                        (fwd[span.stop - 1] - fwd[span.start]) / max(t[span.stop - 1] - t[span.start], 1e-6)
                    ),
                )
            )
            start = i
    (out / "levels.json").write_text(json.dumps(dict(policy=cmd["policy"], levels=table), indent=1))

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H - CHART_H, W)
    font, small = ImageFont.truetype(str(FONT), 26), ImageFont.truetype(str(FONT), 17)
    tmax = t[-1]
    vmax = max(0.9, float(np.nanmax(delivered)) * 1.1, float(commanded.max()) * 1.1)

    def chart_xy(tt, v):
        return 60 + (W - 80) * tt / tmax, CHART_H - 25 - (CHART_H - 40) * v / vmax

    chart = Image.new("RGB", (W, CHART_H), (24, 24, 28))
    cd = ImageDraw.Draw(chart)
    for v in np.arange(0, vmax, 0.2):
        y = chart_xy(0, v)[1]
        cd.line([(60, y), (W - 20, y)], fill=(60, 60, 66))
        cd.text((12, y - 9), f"{v:.1f}", font=small, fill=(150, 150, 150))
    cd.line([chart_xy(tt, v) for tt, v in zip(t[::5], commanded[::5])], fill=(255, 200, 60), width=2)
    cd.line([chart_xy(tt, v) for tt, v in zip(t[::5], delivered[::5])], fill=(90, 200, 255), width=2)
    cd.text((70, 4), "黄：指令速度    蓝：实测速度 (m/s，1 s 平均)", font=small, fill=(220, 220, 220))

    level_of = {round(r["t0"], 3): r for r in table}
    starts = np.array(sorted(level_of))
    tmp = out / "speed_ramp_raw.mp4"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), HZ / EVERY, (W, H))
    for i in range(0, len(seq), EVERY):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="walk_follow")
        frame = Image.new("RGB", (W, H))
        frame.paste(Image.fromarray(renderer.render()), (0, 0))
        c = chart.copy()
        ImageDraw.Draw(c).line([(chart_xy(t[i], 0)[0], 20), (chart_xy(t[i], 0)[0], CHART_H - 20)], fill=(255, 255, 255))
        frame.paste(c, (0, H - CHART_H))
        d = ImageDraw.Draw(frame)
        level = level_of[starts[np.searchsorted(starts, round(t[i], 3), side="right") - 1]]
        mode = "IDLE（停）" if commanded[i] == 0 else walk_name
        lines = [
            f"指令速度 {commanded[i]:.2f}   {mode}",
            f"实测速度 {delivered[i]:+.2f} m/s   （本档稳态 {level['delivered_mps']:+.2f}）",
            f"t = {t[i]:5.1f} s   已走 {travelled[i]:5.2f} m",
        ]
        for k, text in enumerate(lines):
            xy = (20, 16 + 36 * k)
            d.text(xy, text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()
    full, preview = out / "speed_ramp.mp4", out / "speed_ramp_preview.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(tmp),
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(full),
        ],
        check=True,
    )
    tmp.unlink()
    for crf, scale, fps in ((32, 640, 12), (36, 640, 10), (38, 560, 8), (40, 480, 8)):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(full),
                "-vf",
                f"scale={scale}:-2,fps={fps}",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(preview),
            ],
            check=True,
        )
        if preview.stat().st_size <= 500_000:
            break
    print(f"{full} ({t[-1]:.1f} s), {preview} {preview.stat().st_size} bytes")
    for r in table:
        print(f"{r['phase']:>4} cmd {r['commanded']:.2f} -> {r['delivered_mps']:+.3f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
