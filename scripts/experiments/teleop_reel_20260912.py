#!/usr/bin/env python3
"""Caption the teleop-channel rounds with what the channel delivered.

The numbers this reel exists to make visible, per round: the squat retreat the
base actually delivered (NOTHING predicts it on this channel -- the wrist target
is re-expressed in the live pelvis frame every tick, so `project_squat` and its
0.8 constant have no role), the wrist residual at the end of that squat, how
long the local align took, and the lift. Rounds map to clips by
`video_start_wall`, not by mtime order.

    python scripts/experiments/teleop_reel_20260912.py [run_dir ...]
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/videos/teleop_20260912"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
DEFAULT = [ROOT / "outputs/experiments/teleop_pilot_20260912/sonic_v1_1_run5"]


def clip_for(run, r):
    wall = ((r.get("execution") or {}).get("metrics") or {}).get("video_start_wall")
    if wall is None:
        return None
    for cand in (int(wall), int(wall) + 1, int(wall) - 1):
        p = run / "videos" / f"gym_{cand}.mp4"
        if p.is_file():
            return p
    return None


def esc(s):
    return s.replace("\\", "").replace(":", "\\:").replace("'", "").replace(",", " ")


def main():
    runs = [Path(a) for a in sys.argv[1:]] or DEFAULT
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for run in runs:
        arm = run.name
        for line in (run / "manifest.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            metrics = (r.get("execution") or {}).get("metrics") or {}
            ct = (metrics.get("coupled_transport") or [{}])[-1]
            pose = ct.get("squat_delivered_pose") or {}
            src = clip_for(run, r)
            if src is None or not pose:
                continue
            ok = "LIFTED" if r.get("ok") else "FAILED"
            why = "" if r.get("ok") else f"  {str(r.get('why', ''))[:64]}"
            l1 = f"{arm}  round {r['round']}  case {r['case_id']}  {ok}{why}"
            l2 = (
                f"squat delivered drop {pose.get('drop_m', 0):.3f} m  retreat {pose.get('retreat_m', 0):.3f} m"
                f"  (PREDICTED BY NOTHING)   wrist residual {pose.get('residual_m', 0) * 100:.1f} cm"
            )
            lifted = ct.get("lifted_m")
            align = ct.get("alignment_s")
            l3 = "  ".join(
                x
                for x in (
                    "VR 3-point channel, encoder mode teleop: no arm IK, waist free",
                    f"align {align:.1f} s" if align else "",
                    f"lift {lifted * 100:.1f} cm" if lifted else "",
                )
                if x
            )
            dst = OUT / f"{arm}_r{r['round']}.mp4"
            draw = (
                "drawbox=y=0:h=72:w=iw:color=black@0.55:t=fill,"
                f"drawtext=fontfile={FONT}:text='{esc(l1)}':x=8:y=6:fontsize=14:fontcolor=white,"
                f"drawtext=fontfile={FONT}:text='{esc(l2)}':x=8:y=28:fontsize=13:fontcolor=yellow,"
                f"drawtext=fontfile={FONT}:text='{esc(l3)}':x=8:y=50:fontsize=12:fontcolor=0xAADDFF"
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
                    "-preset",
                    "veryfast",
                    "-crf",
                    "24",
                    str(dst),
                ],
                check=True,
            )
            parts.append(dst)
    if not parts:
        raise SystemExit("no clips matched a manifest round")
    listing = OUT / "parts.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts))
    reel = OUT / "reel.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(reel)],
        check=True,
    )
    print(f"{len(parts)} clips -> {reel}")


if __name__ == "__main__":
    main()
