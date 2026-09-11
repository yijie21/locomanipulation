#!/usr/bin/env python3
"""Caption the Stage B pilot's rounds and stitch a reel.

Rounds map to files by `execution.metrics.video_start_wall` -- the exporter
names each clip `gym_<epoch>.mp4`, so nothing here guesses from mtime order.
Each clip is captioned with what a reviewer needs to judge the change: the
case, the outcome, the approach's exit reason, and where it braked and parked
(the two numbers Stage B moved).

    python scripts/experiments/stageb_reel_20260911.py [--all]
"""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/experiments/stageb_coarse_pilot_20260911"
VIDEOS = RUN / "videos"
OUT = ROOT / "outputs/videos/stageb_coarse_pilot_20260911"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
# Two successes and two failures: enough to see the arrival is now release-like
# and that what remains is the grasp phase, not the walk.
PICK = (3, 6, 5, 9)


def rounds():
    return [json.loads(x) for x in (RUN / "manifest.jsonl").read_text().splitlines() if x.strip()]


def clip_for(r):
    wall = ((r.get("execution") or {}).get("metrics") or {}).get("video_start_wall")
    if wall is None:
        return None
    # The exporter truncates the epoch to whole seconds.
    for cand in (int(wall), int(wall) + 1, int(wall) - 1):
        p = VIDEOS / f"gym_{cand}.mp4"
        if p.is_file():
            return p
    return None


def caption(r):
    a = (((r.get("execution") or {}).get("metrics") or {}).get("arrival_measurements") or [{}])[0]
    verdict = "OK" if r.get("ok") else "FAIL"
    why = (r.get("why") or "").strip()
    brake, end = a.get("brake_depth_m"), a.get("end_depth_m")
    line1 = f"case {r['case_id']}  {verdict}  {r.get('seconds', 0):.0f}s   v1.1 + APPR_COARSE 0.40"
    line2 = (
        f"approach exit {a.get('reason', '?')}"
        + (f"  brake x {brake:.3f}" if isinstance(brake, float) and brake == brake else "")
        + (f" -> stop {end:.3f}" if isinstance(end, float) and end == end else "")
        + (f"  park {r['park']:.3f}" if isinstance(r.get("park"), float) else "")
    )
    return line1, (why[:74] if why else line2)


def esc(s):
    return s.replace("\\", "").replace(":", "\\:").replace("'", "").replace(",", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="caption every recorded round, not just PICK")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rs = {r["round"]: r for r in rounds()}
    want = sorted(rs) if args.all else [i for i in PICK if i in rs]
    parts = []
    for i in want:
        r = rs[i]
        src = clip_for(r)
        if src is None:
            print(f"round {i}: no clip (produced={r.get('produced')})")
            continue
        l1, l2 = caption(r)
        dst = OUT / f"r{i}_{'ok' if r.get('ok') else 'fail'}.mp4"
        draw = (
            f"drawbox=y=0:h=54:w=iw:color=black@0.55:t=fill,"
            f"drawtext=fontfile={FONT}:text='{esc(l1)}':x=8:y=6:fontsize=15:fontcolor=white,"
            f"drawtext=fontfile={FONT}:text='{esc(l2)}':x=8:y=30:fontsize=13:fontcolor=#ffd479"
        )
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
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-an",
                str(dst),
            ],
            check=True,
        )
        parts.append(dst)
        print(f"round {i}: {src.name} -> {dst.name}   {l1} | {l2}")
    if len(parts) > 1:
        listing = OUT / "parts.txt"
        listing.write_text("".join(f"file '{p}'\n" for p in parts))
        reel = OUT / "reel.mp4"
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
                "-movflags",
                "+faststart",
                str(reel),
            ],
            check=True,
        )
        print(f"\nreel {reel}  ({reel.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
