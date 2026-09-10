"""Put the two checkpoints' videos for the SAME case side by side.

Rounds are recorded in order, so the Nth video in an arm's video directory is
that arm's Nth manifest row. Pairing by round rather than by filename keeps the
two halves showing the same spawn, the same bottle position and the same
instruction, which is the only way a visual comparison means anything.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("V11_OUT", ROOT / "outputs/experiments/v11_bottle_pilot_20260910"))
ARMS = ["release", "sonic_v1_1"]  # left, right


def rows_of(arm):
    path = OUT / arm / "manifest.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def videos(arm):
    return sorted((OUT / arm / "videos").glob("gym_*.mp4"))


def rounds_with_video(arm):
    """Manifest rows paired with their video, matched by wall clock.

    A round that dies before `reset.start_video` leaves a manifest row and no
    file, and which failure modes do that is not something to guess: the
    pose sidecar writes one jsonl per round with `wall` on every sample, and a
    video is named gym_<int(start wall)>.mp4, so a video belongs to the round
    whose sample span contains its stamp. Rounds with no video are dropped.
    """
    rows = rows_of(arm)
    spans = []
    for f in sorted((OUT / arm / "pose_sidecar").glob("*_round_*.jsonl")):
        rnd = int(f.name.split("_round_")[1].split("_")[0])
        lines = [ln for ln in f.read_text().splitlines() if ln.strip()]
        if not lines:
            continue
        spans.append((rnd, json.loads(lines[0])["wall"], json.loads(lines[-1])["wall"]))
    by_round = {}
    for v in videos(arm):
        stamp = int(v.stem.split("_")[1])
        hit = [r for r, lo, hi in spans if lo - 8 <= stamp <= hi + 8]
        if len(hit) != 1:
            raise SystemExit(f"{arm}: {v.name} matches rounds {hit}; refusing to guess the pairing")
        by_round[hit[0]] = v
    out = [(r, by_round[r["round"]]) for r in rows if r["round"] in by_round]
    print(
        f"{arm}: {len(rows)} rows, {len(videos(arm))} videos, {len(out)} paired"
        f"{'' if len(out) == len(videos(arm)) else ' (some videos unmatched!)'}"
    )
    return out


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return float(out or 0.0)


def label(arm, r):
    if r is None:
        return f"{arm}: (no round)"
    return f"{arm}  {'OK' if r['ok'] else 'FAIL ' + r['why'].split(']')[0].strip('[')}"


def main():
    want = sys.argv[1:] or None  # case ids, default: all shared rounds
    pairs = {a: rounds_with_video(a) for a in ARMS}
    for a in ARMS:
        print(f"{a}: {len(pairs[a])} filmed rounds")
    n = min(len(pairs[a]) for a in ARMS)
    dest = OUT / "video_pairs"
    dest.mkdir(exist_ok=True)
    made = []
    for i in range(n):
        r = {a: pairs[a][i][0] for a in ARMS}
        cid = r[ARMS[0]]["case_id"]
        if r[ARMS[1]]["case_id"] != cid:
            raise SystemExit(f"pair {i} is case {cid} vs {r[ARMS[1]]['case_id']} -- alignment is wrong")
        if want and cid not in want:
            continue
        left, right = (pairs[a][i][1] for a in ARMS)
        mp4 = dest / f"{cid.replace(':', '_')}.mp4"
        # Freeze the shorter clip on its last frame for exactly the difference,
        # so both halves end together without padding the pair to a fixed length.
        dur = [probe(v) for v in (left, right)]
        longest = max(dur)
        pads = [max(0.0, longest - d) for d in dur]
        flt = (
            f"[0:v]scale=640:-2,tpad=stop_mode=clone:stop_duration={pads[0]:.2f},"
            f"drawtext=text='{label(ARMS[0], r[ARMS[0]])}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[l];"
            f"[1:v]scale=640:-2,tpad=stop_mode=clone:stop_duration={pads[1]:.2f},"
            f"drawtext=text='{label(ARMS[1], r[ARMS[1]])}':x=8:y=8:fontsize=20:fontcolor=white:box=1:boxcolor=black@0.6[r];"
            "[l][r]hstack=inputs=2"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            flt,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(mp4),
        ]
        subprocess.run(cmd, check=True)
        made.append(str(mp4))
        print("made", mp4, "|", label(ARMS[0], r[ARMS[0]]), "vs", label(ARMS[1], r[ARMS[1]]))
    print(json.dumps(made, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
