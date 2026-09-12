#!/usr/bin/env python3
"""Caption the squat rounds of run3 with the retreat they delivered.

The number this reel exists to make visible is `squat_delivered_pose.retreat_m`
against the 0.092 m `project_squat` predicted (ledger 32.1): at the bottom of
each squat the pelvis has slid ~3.5 cm further back than the plan assumed.
Rounds map to clips by `video_start_wall`, not by mtime order.

    python scripts/experiments/retreat_reel_20260912.py
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/experiments/retreat_seed10c_20260912"
OUT = ROOT / "outputs/videos/retreat_20260912"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def clip_for(r):
    wall = ((r.get("execution") or {}).get("metrics") or {}).get("video_start_wall")
    if wall is None:
        return None
    for cand in (int(wall), int(wall) + 1, int(wall) - 1):
        p = RUN / "videos" / f"gym_{cand}.mp4"
        if p.is_file():
            return p
    return None


def esc(s):
    return s.replace("\\", "").replace(":", "\\:").replace("'", "").replace(",", " ")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for line in (RUN / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ct = ((r.get("execution") or {}).get("metrics") or {}).get("coupled_transport") or []
        pose = next((s["squat_delivered_pose"] for s in ct if s.get("squat_delivered_pose")), None)
        src = clip_for(r)
        if not pose or src is None:
            continue
        l1 = (
            f"round {r['round']}  case {r['case_id']}  v1.1  squat drop {pose['drop_m']:.3f} m"
            f"  pitch +{pose['pitch_deg']:.0f} deg (predicted 0)"
        )
        l2 = (
            f"retreat DELIVERED {pose['retreat_m']:.4f} m   PREDICTED {pose['predicted_retreat_m']:.4f} m"
            f"   -> plan is {pose['retreat_m'] - pose['predicted_retreat_m']:+.3f} m optimistic"
        )
        dst = OUT / f"r{r['round']}_retreat.mp4"
        draw = (
            "drawbox=y=0:h=54:w=iw:color=black@0.55:t=fill,"
            f"drawtext=fontfile={FONT}:text='{esc(l1)}':x=8:y=6:fontsize=14:fontcolor=white,"
            f"drawtext=fontfile={FONT}:text='{esc(l2)}':x=8:y=30:fontsize=13:fontcolor=#ffd479"
        )
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", draw,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-an", str(dst)],
            check=True,
        )
        parts.append(dst)
        print(f"round {r['round']}: {src.name} -> {dst.name}")
    if len(parts) > 1:
        listing = OUT / "parts.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        reel = OUT / "reel.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", "-movflags", "+faststart", str(reel)],
            check=True,
        )
        print(f"\nreel {reel}  ({reel.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
