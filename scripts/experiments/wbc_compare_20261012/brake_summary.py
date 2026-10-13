"""Merge the finish-line brake runs of both controllers (TODO 24.7): one table, one JSON, side-by-side videos.

Reads <out>/<run>/summary.json + results.json for the runs brake_batch.sh produces (each rendered by
render_brake_marker_20260923.py): sonic_marker / bfm_marker (stop command at the tick the pelvis crosses the
line), sonic_guard / sonic_guard_final020 / bfm_plan (each controller's never-cross program). Writes
<out>/brake.md and brake.json, <out>/videos/<program>_<speed>.mp4 (SONIC left, ScaleBFM right, the trials back
to back) and <out>/videos/reel_preview.mp4 (the first seconds of every pair, <= 500 KB).

    third_party/GR00T-WholeBodyControl/.venv_sim/bin/python \\
        scripts/experiments/wbc_compare_20261012/brake_summary.py --out outputs/wbc_compare_20261012/brake
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FONT = REPO / "assets" / "blender" / "fonts" / "NotoSansCJK-Regular.ttc"
RUNS = {  # run dir -> (controller, program)
    "sonic_marker": ("sonic", "marker"),
    "bfm_marker": ("scalebfm", "marker"),
    "sonic_guard": ("sonic", "plan"),
    "sonic_guard_final020": ("sonic", "plan"),
    "bfm_plan": ("scalebfm", "plan"),
}
PAIRS = [("marker", "sonic_marker", "bfm_marker"), ("plan", "sonic_guard", "bfm_plan")]
PROGRAM_NOTE = {
    "marker": "刹车命令在骨盆越线那一拍发出（SONIC：IDLE；ScaleBFM：参考冻结在当时的骨盆位置）；看越线后还走多远",
    "plan": "各自的『不越线』程序（SONIC：落脚点守线刹车 extra 0.20 / margin 0.03；ScaleBFM：梯形参考停在线前 脚尖伸出 + 3 cm）；看身体最前点离线多远",
}
PREVIEW_MAX = 500_000
CLIP_S = 12


def ms(xs, nd=1):
    xs = np.asarray(xs, float)
    if xs.size == 0:
        return "–"
    return f"{xs.mean():+.{nd}f} ± {xs.std():.{nd}f}"


def rng(xs, nd=1):
    xs = np.asarray(xs, float)
    return "–" if xs.size == 0 else f"[{xs.min():+.{nd}f}, {xs.max():+.{nd}f}]"


def stats(run: str, key: str, s: dict) -> dict:
    return dict(
        run=run,
        n=s["n"],
        falls=s["falls"],
        brake_speed=round(s["brake_speed_mean"], 2),
        pelvis_cm=s["stop_past_marker_cm"],
        toes_final_cm=s["frontmost_final_past_line_cm"],
        toes_worst_cm=s["frontmost_max_past_line_cm"],
        crossed=s["crossed"],
        carry_cm=[round(s["carry_mean_cm"], 1), round(s["carry_sd_cm"], 1)],
        lateral_sd_cm=round(s["lateral_sd_cm"], 1),
        yaw_abs_deg=round(s["yaw_abs_mean_deg"], 1),
        still_s=s.get("still_after_brake_s", []),
        target_err_cm=s.get("target_err_cm", []),
    )


def load(out: Path) -> dict:
    table = {}  # program -> speed key -> controller -> stats
    for run, (ctrl, prog) in RUNS.items():
        f = out / run / "summary.json"
        if not f.exists():
            continue
        for key, s in json.loads(f.read_text()).items():
            table.setdefault(prog, {}).setdefault(key, {})[ctrl if run != "sonic_guard_final020" else "sonic*"] = stats(
                run, key, s
            )
    return table


def markdown(table: dict, out: Path) -> str:
    lines = [
        "# 终点线刹车：SONIC 对 ScaleBFM（wbc_compare, TODO 24.7）",
        "",
        "空地（`outputs/scene/walk`），从原点朝 +x 走到 1.5 m 外的线再停，停稳 3 s 后读数。单位 cm，正 = 线的前方（越线）。"
        "『骨盆』= 停稳后骨盆到线；『脚尖』= 身体最前的低位点（底面低于 0.37 m 的 geom）到线，停稳值与全程最远值；"
        "『越线』= 最远值 > 0 的局数。ScaleBFM 每局把线抖动 ±10 cm（仿真确定性，否则 10 局全同）；SONIC 的出生和步态自带散布。",
        "",
    ]
    for prog in ("marker", "plan"):
        if prog not in table:
            continue
        lines += [f"## {prog}：{PROGRAM_NOTE[prog]}", ""]
        lines.append(
            "| 速度 | 控制器 | 局 (摔) | 刹车时速 m/s | 骨盆 mean ± sd | 骨盆范围 | 脚尖停稳 mean ± sd | 脚尖最远（最差局） | 越线 | 横向 sd | \\|yaw\\| ° | 刹车到静止 s（中位） | 目标误差 |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in sorted(table[prog], key=lambda k: -float(k.split("to")[0])):
            for ctrl in ("sonic", "sonic*", "scalebfm"):
                s = table[prog][key].get(ctrl)
                if s is None:
                    continue
                still = np.asarray(s["still_s"], float)
                lines.append(
                    f"| {key.replace('to', '→')} | {ctrl} | {s['n']} ({s['falls']}) | {s['brake_speed']:.2f} | "
                    f"{ms(s['pelvis_cm'])} | {rng(s['pelvis_cm'])} | {ms(s['toes_final_cm'])} | "
                    f"{max(s['toes_worst_cm']):+.1f} | {s['crossed']}/{s['n']} | {s['lateral_sd_cm']} | {s['yaw_abs_deg']} | "
                    f"{'–' if still.size == 0 else f'{np.median(still):.2f}'} | "
                    f"{ms(s['target_err_cm']) if s['target_err_cm'] else '–'} |"
                )
        lines.append("")
    lines += [
        "`sonic*` = SONIC 守线刹车加最后 0.3 m 降到 0.20 的两段式（09-23 验证过的最好设置）。",
        "目标误差只有 ScaleBFM plan 有：停稳骨盆减参考终点（负 = 停在参考之前）。",
        f"视频：`{out.relative_to(REPO) if out.is_relative_to(REPO) else out}/videos/`（左 SONIC、右 ScaleBFM；`reel_preview.mp4` 每对取前 {CLIP_S} s）。",
        "",
    ]
    return "\n".join(lines)


def drawtext(label: str) -> str:
    if not FONT.exists():
        return ""
    return f",drawtext=fontfile={FONT}:text='{label}':x=20:y=h-50:fontsize=30:fontcolor=white:borderw=3"


def side_by_side(out: Path, table: dict) -> list[Path]:
    vids = out / "videos"
    vids.mkdir(exist_ok=True)
    made = []
    for prog, left, right in PAIRS:
        for key in sorted(table.get(prog, {}), key=lambda k: -float(k.split("to")[0])):
            a, b = out / left / f"brake_{key}.mp4", out / right / f"brake_{key}.mp4"
            if not (a.exists() and b.exists()):
                continue
            dst = vids / f"{prog}_{key}.mp4"
            fc = (
                f"[0:v]scale=640:-2{drawtext('SONIC')}[a];[1:v]scale=640:-2{drawtext('ScaleBFM')}[b];"
                "[a][b]hstack=inputs=2[v]"
            )
            subprocess.run(
                ["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(a), "-i", str(b), "-filter_complex", fc,
                 "-map", "[v]", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", str(dst)],
                check=True,
            )  # fmt: skip
            made.append(dst)
            print(f"{dst} {dst.stat().st_size} bytes")
    return made


def reel(clips: list[Path], dst: Path) -> None:
    if not clips:
        return
    ins = [x for c in clips for x in ("-i", str(c))]
    parts = "".join(f"[{i}:v]trim=0:{CLIP_S},setpts=PTS-STARTPTS[c{i}];" for i in range(len(clips)))
    cat = "".join(f"[c{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0,scale=800:-2[v]"
    for crf in (36, 40, 44, 48):
        subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-y", *ins, "-filter_complex", parts + cat, "-map", "[v]",
             "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", str(dst)],
            check=True,
        )  # fmt: skip
        if dst.stat().st_size <= PREVIEW_MAX:
            break
    print(f"{dst} {dst.stat().st_size} bytes (crf {crf})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "wbc_compare_20261012" / "brake")
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args(argv)
    table = load(a.out)
    (a.out / "brake.json").write_text(json.dumps(table, indent=1))
    (a.out / "brake.md").write_text(markdown(table, a.out))
    print((a.out / "brake.md").read_text())
    if not a.no_video:
        clips = side_by_side(a.out, table)
        reel(clips, a.out / "videos" / "reel_preview.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
