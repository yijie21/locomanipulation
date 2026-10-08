"""Shareable previews of a video under the repository's size cap (AGENTS.md, docs/rendering.md#video-export).

    python -m locomani.pipeline.video_preview <source.mp4> <out_stem> [--width 1280] [--fps 10] [--cap 500000]

Writes `<out_stem>.mp4`, or `<out_stem>_part1of3.mp4` ... when one file cannot hold the whole clip. Every preview keeps
full duration, normal speed and the aspect ratio. Width and frame rate are capped at the source's own, never raised. The
bitrate is lowered step by step (CRF ladder). When the highest CRF still leaves a file over the cap, the clip is split into
more consecutive frame-aligned parts. Each file is checked for size, a clean full decode and its duration against the
span it covers. Earlier previews of the same stem (single or split) are removed first, so a re-run never leaves a stale
mix. Stdlib + ffmpeg/ffprobe only, so any python runs it.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

CAP = 500_000  # bytes, AGENTS.md hard cap
CRFS = (32, 34, 36, 38)  # past 38 the carton print in a 1280-wide side-by-side stops being legible; split instead
# 16: a closed-loop round that runs its whole budget renders ~99 s of side-by-side with insets, which took 11 parts
# of ~9 s (179-454 KB each, warehouse_smoke 2026-09-28); 8 failed every such round. Past 16 parts, lower the width.
MAX_PARTS = 16


def probe(path: Path) -> tuple[float, int, float]:
    """(duration in seconds, width in pixels, frame rate) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,avg_frame_rate:format=duration", "-of", "default=noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    fields = dict(line.split("=", 1) for line in out.split() if "=" in line)
    num, _, den = fields["avg_frame_rate"].partition("/")
    rate = float(num) / float(den or 1) if float(den or 1) else 0.0
    return float(fields["duration"]), int(fields["width"]), rate


def spans(duration: float, parts: int, fps: float) -> list[tuple[float, float]]:
    """Consecutive (start, length) pieces covering [0, duration], cut on output-frame boundaries."""
    frames = max(1, math.ceil(duration * fps - 1e-6))
    cuts = [round(k * frames / parts) for k in range(parts + 1)]
    edges = [c / fps for c in cuts[:-1]] + [duration]
    return [(edges[k], edges[k + 1] - edges[k]) for k in range(parts)]


def stale(out_stem: Path) -> list[Path]:
    """Earlier previews of this stem: the single file and any split set."""
    return [p for p in (out_stem.with_name(out_stem.name + ".mp4"),) if p.exists()] + sorted(
        out_stem.parent.glob(f"{out_stem.name}_part*of*.mp4")
    )


def encode(src: Path, dst: Path, start: float, length: float, width: int, fps: float, crf: int) -> int:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(src),
         "-map", "0:v:0", "-vf", f"scale={width}:-2,fps={fps:g}", "-c:v", "libx264", "-preset", "medium",
         "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)],
        check=True,
    )  # fmt: skip
    return dst.stat().st_size


def verify(dst: Path, length: float, fps: float, cap: int) -> None:
    """The per-file checks of docs/rendering.md#video-export; raises on the first one that fails."""
    size = dst.stat().st_size
    if size > cap:
        raise RuntimeError(f"{dst}: {size} bytes over the {cap} cap")
    errors = subprocess.run(
        ["ffmpeg", "-v", "error", "-xerror", "-i", str(dst), "-f", "null", "-"], capture_output=True, text=True
    )
    if errors.returncode or errors.stderr.strip():
        raise RuntimeError(f"{dst}: does not decode cleanly: {errors.stderr.strip()[:200]}")
    got, _, _ = probe(dst)
    if abs(got - length) > 1.0 / fps + 0.02:
        raise RuntimeError(f"{dst}: {got:.3f} s against the {length:.3f} s it should cover")


def make_previews(
    src: Path, out_stem: Path, *, width: int = 1280, fps: float = 10, cap: int = CAP, crfs=CRFS, max_parts=MAX_PARTS
) -> list[tuple[Path, float, float]]:
    """Write and verify the previews; return (path, start, length) per file, in time order."""
    duration, src_width, src_fps = probe(src)
    width = min(width, src_width) // 2 * 2
    # A stride-16 render runs at 3.125 fps; resampling it up only duplicates frames and puts the cuts off its grid.
    fps = min(fps, src_fps) if src_fps > 0 else fps
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for old in stale(out_stem):
        old.unlink()
    parts = 1
    while parts <= max_parts:
        pieces = spans(duration, parts, fps)
        names = [
            out_stem.with_name(out_stem.name + (".mp4" if parts == 1 else f"_part{k + 1}of{parts}.mp4"))
            for k in range(parts)
        ]
        largest = 0
        for crf in crfs:
            largest = max(encode(src, n, s, t, width, fps, crf) for n, (s, t) in zip(names, pieces))
            if largest <= cap:
                for n, (_, t) in zip(names, pieces):
                    verify(n, t, fps, cap)
                return [(n, s, t) for n, (s, t) in zip(names, pieces)]
        for n in names:
            n.unlink()
        # Skip part counts that cannot fit: size is close to proportional to the span. `largest` is the worst part, so
        # uneven content is already priced in; extra headroom here overshot to 7 parts where 5 fit (96 s round).
        parts = max(parts + 1, math.ceil(parts * largest / cap))
    raise RuntimeError(f"{src}: no split into <= {max_parts} parts fits {cap} bytes at {width} px wide")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("out_stem", type=Path, help="output path without .mp4; split parts get _partKofN")
    ap.add_argument("--width", type=int, default=1280, help="preview width, capped at the source width")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--cap", type=int, default=CAP, help="bytes per file")
    args = ap.parse_args(argv)
    try:
        written = make_previews(args.source, args.out_stem, width=args.width, fps=args.fps, cap=args.cap)
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"preview: FAILED {e}", file=sys.stderr)
        return 1
    for path, start, length in written:
        print(f"preview: {path} {path.stat().st_size} bytes, {start:.1f}-{start + length:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
