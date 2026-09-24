"""The overview video that ships beside a converted dataset.

It is not a camera stream: it never enters the policy's observation and does not
live under `videos/observation.images.*`. It is the pelvis-tracking view with the
debug visuals on, carried into the dataset so an episode can be watched by index
instead of by hunting for a timestamped filename in `videos_NN/`.
"""

import json
import os

from locomani.pipeline.third_person import link_third_person, third_person_source


def _session(tmp_path, manifest_name, rows, clips):
    manifest = tmp_path / manifest_name
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    videos = tmp_path / manifest_name.split(".")[0].replace("manifest", "videos")
    videos.mkdir()
    for name in clips:
        (videos / name).write_bytes(b"mp4")
    return manifest


def test_the_crate_line_joins_on_the_name_its_manifest_records(tmp_path):
    rows = [{"produced": True, "label": f"visual_crate_s7000_{i:02d}"} for i in range(3)]
    manifest = _session(tmp_path, "manifest_00.jsonl", rows, [f"{r['label']}.mp4" for r in rows])
    for i, row in enumerate(rows):
        assert third_person_source(manifest, row, i, len(rows)).name == f"{row['label']}.mp4"


def test_the_bottle_line_joins_on_order_and_only_when_the_counts_agree(tmp_path):
    rows = [{"produced": True} for _ in range(3)]  # this line's manifest names no clip
    manifest = _session(tmp_path, "manifest_00.jsonl", rows, ["gym_100.mp4", "gym_200.mp4", "gym_300.mp4"])
    assert third_person_source(manifest, rows[1], 1, 3).name == "gym_200.mp4"
    # One clip short: every episode after the gap would get the wrong video, so refuse.
    (manifest.parent / "videos_00" / "gym_200.mp4").unlink()
    assert third_person_source(manifest, rows[1], 1, 3) is None


def test_one_videos_directory_shared_by_every_batch_joins_by_name_or_stamp(tmp_path):
    # Since 0becdd0 the stack outlives the batches: manifest_00/01 sit beside one videos/.
    videos = tmp_path / "videos"
    videos.mkdir()
    for name in ("visual_crate_s7100_02.mp4", "gym_1790257182.mp4", "gym_1790257219.mp4"):
        (videos / name).write_bytes(b"mp4")
    manifest = tmp_path / "manifest_01.jsonl"
    manifest.write_text("")
    crate = {"produced": True, "label": "visual_crate_s7100_02"}
    assert third_person_source(manifest, crate, 0, 1).name == "visual_crate_s7100_02.mp4"
    bottle = {"produced": True, "execution": {"metrics": {"video_start_wall": 1790257219.4}}}
    assert third_person_source(manifest, bottle, 0, 1).name == "gym_1790257219.mp4"
    # No name and no stamp: order across batches means nothing in a shared directory, so refuse.
    assert third_person_source(manifest, {"produced": True}, 0, 1) is None


def test_a_session_without_an_overview_directory_is_not_an_error(tmp_path):
    manifest = tmp_path / "manifest_00.jsonl"
    manifest.write_text(json.dumps({"produced": True, "label": "x"}) + "\n")
    assert third_person_source(manifest, {"produced": True, "label": "x"}, 0, 1) is None


def test_the_clip_is_hardlinked_so_a_dataset_costs_no_extra_disk(tmp_path):
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"mp4" * 100)
    dst = tmp_path / "third_person" / "episode_000000.mp4"
    link_third_person(src, dst)
    assert os.stat(dst).st_ino == os.stat(src).st_ino
    link_third_person(src, dst)  # idempotent: conversion may be re-run over an existing dataset
    assert dst.read_bytes() == src.read_bytes()
