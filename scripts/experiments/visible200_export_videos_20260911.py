#!/usr/bin/env python3
"""Export 10 collection + 10 evaluation third-person clips for the visible200 arm.

Both sets come from the sim's third-person composite recordings that already exist on
disk (main view + L/R wrist + ego insets, 720x540 @ 12.5 fps):

  collection : outputs/experiments/visible200_20260910/collect/videos_<batch>/gym_<ts>.mp4
  evaluation : outputs/experiments/visible200_20260910/eval/videos/gym_<ts>.mp4

Nothing is re-simulated. A clip is matched to its round by wall clock: the recorder
starts ~10 s after the round does, so a video with start timestamp `ts` belongs to the
round whose `execution.started_wall` is the largest one <= ts.

Each exported clip gets a burnt-in caption (batch/round, outcome, reason), and each set
is also concatenated into one reel.

Usage: python3 scripts/experiments/visible200_export_videos_20260911.py
"""

from __future__ import annotations

import bisect
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "outputs/experiments/visible200_20260910"
OUT = REPO / "outputs/videos/visible200_export_20260911"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
N = 10


def video_index(directory: Path) -> list[tuple[int, Path]]:
    """(start_unix_ts, path) for every gym_<ts>.mp4 in `directory`, sorted."""
    out = []
    for p in glob.glob(str(directory / "gym_*.mp4")):
        out.append((int(Path(p).stem.split("_")[1]), Path(p)))
    return sorted(out)


def attach_videos(rounds: list[dict], vids: list[tuple[int, Path]]) -> None:
    """Set rounds[i]['video'] to the recording that started inside round i."""
    starts = [r["execution"].get("started_wall") or 0.0 for r in rounds]
    for ts, path in vids:
        i = bisect.bisect_right(starts, ts) - 1
        if i >= 0 and rounds[i].get("video") is None:
            rounds[i]["video"] = path


def load_collection() -> list[dict]:
    rounds = []
    for manifest in sorted((EXP / "collect").glob("manifest_*.jsonl")):
        batch = int(manifest.stem.split("_")[1])
        batch_rounds = [json.loads(line) for line in manifest.open()]
        for r in batch_rounds:
            r["batch"] = batch
            r["video"] = None
        attach_videos(batch_rounds, video_index(EXP / "collect" / f"videos_{batch:02d}"))
        rounds.extend(batch_rounds)
    return rounds


def load_eval() -> list[dict]:
    rounds = [json.loads(line) for line in (EXP / "eval/closed_loop.manifest.jsonl").open()]
    for r in rounds:
        r["video"] = None
    attach_videos(rounds, video_index(EXP / "eval/videos"))
    return rounds


def pick_collection(rounds: list[dict]) -> list[dict]:
    """N kept (ok + produced) episodes spread evenly over the 34 batches."""
    kept = [r for r in rounds if r["ok"] and r["produced"] and r["video"]]
    by_batch: dict[int, dict] = {}
    for r in kept:
        by_batch.setdefault(r["batch"], r)  # first kept round of each batch
    batches = sorted(by_batch)
    step = len(batches) / N
    return [by_batch[batches[int(i * step)]] for i in range(N)]


def min_dist(why: str) -> float | None:
    m = re.search(r"min ([0-9.]+) m", why or "")
    return float(m.group(1)) if m else None


def pick_eval(rounds: list[dict]) -> list[dict]:
    """Both successes plus 8 failures spread over the stop-distance range.

    The arm scores 2/100 and 97 failures are the same string ("at the table but never
    commanded a close"), so a uniform random 10 would be 10 near-identical failures.
    Sampling the failures along their stop distance keeps the spread visible.
    """
    ok = [r for r in rounds if r["ok"] and r["video"]]
    fails = [r for r in rounds if not r["ok"] and r["video"] and min_dist(r["why"]) is not None]
    fails.sort(key=lambda r: (min_dist(r["why"]), r["round"]))
    k = N - len(ok)
    idx = [round(i * (len(fails) - 1) / (k - 1)) for i in range(k)]
    picked = ok + [fails[i] for i in dict.fromkeys(idx)]
    return sorted(picked, key=lambda r: r["round"])


def caption_collection(r: dict) -> str:
    return f"COLLECT  batch {r['batch']:02d} round {r['round']}  |  KEPT (in the 200-episode dataset)"


def caption_eval(r: dict) -> str:
    if r["ok"]:
        return f"EVAL  round {r['round']:03d}  |  SUCCESS"
    why = re.sub(r"^\[[a-z-]+\]\s*", "", r["why"] or "").strip()
    return f"EVAL  round {r['round']:03d}  |  FAIL: {why[:78]}"


def render(src: Path, dst: Path, caption: str, scratch: Path) -> None:
    txt = scratch / (dst.stem + ".txt")
    txt.write_text(caption + "\n")
    # Pad a banner above the frame rather than drawing over it: the composite's own
    # "L wrist" inset label sits in the top-left corner and an overlay would cover it.
    draw = f"pad=iw:ih+28:0:28:black,drawtext=fontfile={FONT}:textfile={txt}:x=8:y=6:fontsize=15:fontcolor=white"
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


def concat(clips: list[Path], dst: Path) -> None:
    listing = dst.with_suffix(".txt")
    listing.write_text("".join(f"file '{c}'\n" for c in clips))
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(dst)],
        check=True,
    )


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def export(kind: str, picks: list[dict], caption, scratch: Path) -> list[dict]:
    out_dir = OUT / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    clips = []
    for i, r in enumerate(picks, 1):
        if kind == "collect":
            name = f"{i:02d}_b{r['batch']:02d}_r{r['round']}.mp4"
        else:
            name = f"{i:02d}_round{r['round']:03d}_{'ok' if r['ok'] else 'fail'}.mp4"
        dst = out_dir / name
        render(r["video"], dst, caption(r), scratch)
        clips.append(dst)
        index.append(
            {
                "file": str(dst),
                "source": str(r["video"]),
                "round": r["round"],
                "batch": r.get("batch"),
                "case_id": r["case_id"],
                "ok": r["ok"],
                "why": r["why"],
                "min_stop_dist_m": min_dist(r.get("why") or ""),
                "seconds": round(duration(dst), 1),
            }
        )
        print(f"  {name}  {index[-1]['seconds']:>5.1f}s  {caption(r)}")
    reel = OUT / f"reel_{kind}_10.mp4"
    concat(clips, reel)
    print(f"  reel -> {reel}  {duration(reel):.1f}s")
    return index


def main() -> int:
    scratch = OUT / ".captions"
    scratch.mkdir(parents=True, exist_ok=True)

    coll = load_collection()
    ev = load_eval()
    print(
        f"collection: {len(coll)} rounds, {sum(1 for r in coll if r['video'])} with video, "
        f"{sum(1 for r in coll if r['ok'])} kept"
    )
    print(
        f"eval:       {len(ev)} rounds, {sum(1 for r in ev if r['video'])} with video, "
        f"{sum(1 for r in ev if r['ok'])} successes"
    )

    print("\ncollection clips:")
    ci = export("collect", pick_collection(coll), caption_collection, scratch)
    print("\neval clips:")
    ei = export("eval", pick_eval(ev), caption_eval, scratch)

    (OUT / "INDEX.json").write_text(json.dumps({"experiment": str(EXP), "collect": ci, "eval": ei}, indent=2) + "\n")
    print(f"\nindex -> {OUT / 'INDEX.json'}")

    for leftover in scratch.glob("*.txt"):
        leftover.unlink()
    scratch.rmdir()
    return 0


if __name__ == "__main__":
    sys.exit(main())
