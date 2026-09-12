"""Ten-case third-person reel for the 2026-09-10 no-retract closed loop.

Replays saved qpos only -- no simulation, no policy. Picks the two failure modes and all three
strata so the reel shows what the 41/100 is made of, not just its wins.
"""

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from locomani.tasks.bottle_visual.pipeline.follow_video import render_case  # noqa: E402

EVAL = Path("/workspace/code/locomanipulation/outputs/experiments/scalebfm200_noretract_s17_20260910/eval100")
VIDEO = Path("/workspace/code/locomanipulation/outputs/videos/scalebfm200_s17_20260910")


def pick():
    cases = []
    for path in sorted(EVAL.glob("seed*/result.json")):
        r = json.loads(path.read_text())
        cases.append((path.parent, r))
    chosen, seen = [], set()
    # one success and one failure of each kind per stratum, in seed order
    for stratum in ("high_near", "low_near", "low_deep"):
        for want in (("ok", None), ("fail", "timeout"), ("fail", "physics warning/fall")):
            for d, r in cases:
                if r["stratum"] != stratum or d in seen:
                    continue
                if want[0] == "ok" and r["ok"]:
                    chosen.append((d, r))
                    seen.add(d)
                    break
                if want[0] == "fail" and not r["ok"] and r["why"] == want[1]:
                    chosen.append((d, r))
                    seen.add(d)
                    break
    return chosen[:10]


def main():
    VIDEO.mkdir(parents=True, exist_ok=True)
    parts = []
    for ordinal, (case_dir, r) in enumerate(pick(), start=1):
        scene = r["scene"]
        parameters = dict(
            bottle_along_m=scene["along_center_offset"],
            inboard_m=scene["inboard"],
            spawn_edge_m=scene["spawn_edge"],
            spawn_lat_m=scene["spawn_lat"],
            yaw_error_deg=scene["spawn_yaw_deg"],
        )
        out = VIDEO / f"{ordinal:02d}_{r['stratum']}_seed{r['seed']}_{'ok' if r['ok'] else 'fail'}.mp4"
        if not out.exists():
            render_case(case_dir, out, ordinal, parameters)
        parts.append(out)
        print(
            f"{ordinal:02d} {r['stratum']:9s} seed {r['seed']} {'OK  ' if r['ok'] else 'FAIL'} {r['why']}", flush=True
        )
    listing = VIDEO / "reel_parts.txt"
    listing.write_text("".join(f"file '{p}'\n" for p in parts))
    reel = VIDEO / "reel_10.mp4"
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
    print("reel:", reel, reel.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
