"""The overview video that ships beside a converted dataset.

Split out of the converter so it can be imported without pandas/pyarrow, the
same reason `recording_density` is its own module.

This is NOT a dataset camera. It never enters the policy's observation and it
does not live under `videos/observation.images.*`: it is the pelvis-tracking
view with the debug visuals on (`pipeline/sim_video.py`), carried into the
dataset so an episode can be watched by index instead of by hunting for a
timestamped filename in `videos_NN/`.
"""

import os
import shutil
from pathlib import Path


def third_person_source(manifest_path, row, position, produced):
    """The tracking-camera mp4 for one episode, or None when it cannot be joined.

    The overview video is not a dataset camera and never enters the policy's
    observation -- it is the pelvis-tracking view with the debug visuals on
    (`pipeline/sim_video.py`), kept so a converted episode can be watched. It
    lands beside the data rather than under `videos/observation.images.*` for
    exactly that reason.

    Since the flat-scripts refactor (0becdd0, 2026-09-21) the stack outlives the
    batches, so every batch films into one `videos/` beside the manifests; older
    runs have `videos_NN/` next to `manifest_NN.jsonl`. Looking only for
    `videos_NN/` left every dataset converted since then without its overview
    videos, silently (2026-09-24). The crate manifest names its clip (`label`
    is the mp4's stem); the bottle manifest records the clip's start stamp
    (`execution.metrics.video_start_wall`, the stem is `gym_<int(stamp)>`).
    Older bottle rows have neither, so that line joins by order -- and only when
    the counts agree exactly, because a silently shifted join would attach the
    wrong video to every episode after the first missing clip.
    """
    manifest = Path(manifest_path)
    videos = manifest.parent / manifest.name.split(".")[0].replace("manifest", "videos")
    per_batch = videos.is_dir()
    if not per_batch:
        videos = manifest.parent / "videos"
        if not videos.is_dir():
            return None
    label = row.get("label")
    if label and (videos / f"{label}.mp4").exists():
        return videos / f"{label}.mp4"
    stamp = ((row.get("execution") or {}).get("metrics") or {}).get("video_start_wall")
    if stamp is not None:
        clip = videos / f"gym_{int(stamp)}.mp4"
        return clip if clip.exists() else None
    if not per_batch:
        return None  # a shared directory holds every batch's clips: order means nothing
    clips = sorted(videos.glob("*.mp4"))
    return clips[position] if len(clips) == produced else None


def link_third_person(src: Path, dst: Path) -> None:
    """Hardlink when the filesystem allows it: these are tens of MB each."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
