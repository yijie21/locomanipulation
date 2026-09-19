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

    `videos_NN/` always sits next to `manifest_NN.jsonl`, on both lines. The
    crate manifest names its clip (`label` is the mp4's stem); the bottle
    manifest does not, so that line joins by order -- and only when the counts
    agree exactly, because a silently shifted join would attach the wrong video
    to every episode after the first missing clip.
    """
    manifest = Path(manifest_path)
    videos = manifest.parent / manifest.name.split(".")[0].replace("manifest", "videos")
    if not videos.is_dir():
        return None
    label = row.get("label")
    if label and (videos / f"{label}.mp4").exists():
        return videos / f"{label}.mp4"
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
