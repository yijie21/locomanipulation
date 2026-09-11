#!/usr/bin/env python3
"""Export one attempt's rounds as captioned clips, with the clearance diagnostic.

The image-servo rounds are only readable next to the number that explains them,
so each clip's banner carries the round's failure string AND the minimum grip
stand-off it reached against the clearance it needed (ledger bottle.md §19.10:
a negative stand-off means the grip point went past the bottle along the entry
axis, which is how the dex1 fingers foul it).

Clips are matched to rounds by wall clock, the same way the collection recorder
works: the recorder starts a few seconds into a round, so a video belongs to the
last round whose `execution.started_wall` is at or before its timestamp.

Usage:
  python3 scripts/experiments/image_servo_export_videos_20260911.py ATTEMPT_DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def rounds_with_videos(attempt: Path):
    rows = [json.loads(line) for line in (attempt / "manifest.jsonl").open() if line.strip()]
    rows.sort(key=lambda r: r["execution"].get("started_wall") or 0.0)
    starts = [r["execution"].get("started_wall") or 0.0 for r in rows]
    for r in rows:
        r["video"] = None
    for path in sorted(glob.glob(str(attempt / "videos/gym_*.mp4"))):
        ts = int(Path(path).stem.split("_")[1])
        i = bisect.bisect_right(starts, ts) - 1
        if i >= 0 and rows[i]["video"] is None:
            rows[i]["video"] = Path(path)
    return rows


def diagnostic(row) -> str:
    """min grip stand-off vs what the geometry required, when recorded."""
    samples = row.get("execution", {}).get("metrics", {}).get("image_servo", {}).get("samples", [])
    offs = [s["grip_standoff_m"] for s in samples if s.get("grip_standoff_m") is not None]
    if not offs:
        return ""
    need = next((s.get("clearance_m") for s in samples if s.get("clearance_m") is not None), None)
    tail = f" / need +{need:.3f}" if need else ""
    return f"  |  min grip stand-off {min(offs):+.3f} m{tail}"


def render(src: Path, dst: Path, caption: str, scratch: Path) -> None:
    txt = scratch / (dst.stem + ".txt")
    txt.write_text(caption + "\n")
    draw = f"pad=iw:ih+28:0:28:black,drawtext=fontfile={FONT}:textfile={txt}:x=8:y=6:fontsize=14:fontcolor=white"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-vf",
            draw,
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dst),
        ],
        check=True,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("attempt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    attempt = Path(args.attempt)
    out = Path(args.out or REPO / "outputs/videos" / f"image_servo_{attempt.name}")
    scratch = out / ".captions"
    scratch.mkdir(parents=True, exist_ok=True)

    clips = []
    for row in rounds_with_videos(attempt):
        if row["video"] is None:
            print(f"  r{row['round']}: no video")
            continue
        why = (row["why"] or "ok").replace("[grasp] ", "").replace("[view] ", "view: ")
        caption = f"{attempt.name}  round {row['round']}  |  {why[:74]}{diagnostic(row)}"
        dst = out / f"r{row['round']}.mp4"
        render(row["video"], dst, caption, scratch)
        clips.append(dst)
        print(f"  {dst.name}  {caption}")

    if clips:
        listing = out / "reel.txt"
        listing.write_text("".join(f"file '{c}'\n" for c in clips))
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listing),
                "-c",
                "copy",
                str(out / "reel.mp4"),
            ],
            check=True,
        )
        print(f"  reel -> {out / 'reel.mp4'}")
    for leftover in scratch.glob("*.txt"):
        leftover.unlink()
    scratch.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
