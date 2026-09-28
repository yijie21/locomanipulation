"""Preview compression under the size cap (docs/rendering.md#video-export).

The rule is per file: at most `cap` bytes, the full clip covered by consecutive parts, no stale previews left behind.
The encode tests make a small noisy clip with ffmpeg and force a split through a tiny cap, so they run in seconds.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from locomani.pipeline import video_preview as V

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="no ffmpeg")


def test_spans_are_consecutive_frame_aligned_and_cover_the_clip():
    pieces = V.spans(65.84, 3, 10)
    assert pieces[0][0] == 0
    for (s0, t0), (s1, _) in zip(pieces, pieces[1:]):
        assert s0 + t0 == pytest.approx(s1)
        assert round(s1 * 10, 6) == int(round(s1 * 10))  # cut on an output frame
    assert pieces[-1][0] + pieces[-1][1] == pytest.approx(65.84)
    assert V.spans(4.0, 1, 10) == [(0.0, 4.0)]


def _clip(path: Path, seconds: int = 6, size: str = "640x240", rate: int = 25) -> Path:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
         "-t", str(seconds), "-vf", "noise=alls=60:allf=t", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
         str(path)],
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_a_clip_that_fits_stays_one_file(tmp_path):
    src = _clip(tmp_path / "src.mp4", seconds=2, size="320x120")
    written = V.make_previews(src, tmp_path / "out" / "clip", cap=5_000_000)
    assert [p.name for p, _, _ in written] == ["clip.mp4"]
    assert V.probe(written[0][0])[1] == 320  # never upscaled past the source


@needs_ffmpeg
def test_an_oversized_clip_is_split_under_the_cap_and_stale_previews_go(tmp_path):
    src = _clip(tmp_path / "src.mp4")
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip.mp4").write_bytes(b"old single preview")
    (out / "clip_part1of9.mp4").write_bytes(b"old split preview")
    single = V.make_previews(src, out / "clip", cap=10_000_000, crfs=V.CRFS[-1:])[0][0]
    cap = single.stat().st_size * 6 // 10  # the whole clip cannot fit one file even at the ladder's last CRF
    written = V.make_previews(src, out / "clip", cap=cap)
    names = sorted(p.name for p in out.glob("clip*.mp4"))
    assert len(written) >= 2 and names == sorted(p.name for p, _, _ in written)
    assert all(p.stat().st_size <= cap for p, _, _ in written)
    assert sum(V.probe(p)[0] for p, _, _ in written) == pytest.approx(6.0, abs=0.1 * len(written))


@needs_ffmpeg
def test_a_low_frame_rate_source_keeps_its_rate(tmp_path):
    src = _clip(tmp_path / "src.mp4", seconds=4, rate=3)
    written = V.make_previews(src, tmp_path / "clip", cap=5_000_000)
    assert V.probe(written[0][0])[2] == pytest.approx(3.0)


@needs_ffmpeg
def test_an_impossible_cap_fails_loudly(tmp_path):
    src = _clip(tmp_path / "src.mp4", seconds=2, size="320x120")
    with pytest.raises(RuntimeError, match="no split"):
        V.make_previews(src, tmp_path / "clip", cap=100, max_parts=2)
