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
