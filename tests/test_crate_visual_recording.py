import json

import pytest

from locomani.pipeline.execution_record import RoundResult
from locomani.tasks.crate.visual_recording import record_attempt


def test_setup_exception_is_in_started_journal_and_failed_manifest(tmp_path):
    def broken_setup():
        raise RuntimeError("reset failed before camera setup")

    manifest = tmp_path / "manifest.jsonl"
    result = record_attempt(manifest, "episode_00", 7, broken_setup)
    assert not result.accepted()
    started = json.loads((tmp_path / "attempts.jsonl").read_text())
    row = json.loads(manifest.read_text())
    assert started["label"] == row["label"] == "episode_00"
    assert started["seed"] == row["seed"] == 7
    assert row["why"] == "runtime_error"
    assert "reset failed" in row["error"]
    assert json.loads((tmp_path / "observations/episode_00/result.json").read_text()) == row


def test_successful_video_is_distinct_from_produced_training_data_and_cannot_be_overwritten(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    execute = lambda: RoundResult(True, "completed", False, {"metrics": {"frames": 10}})
    result = record_attempt(manifest, "episode_00", 0, execute)
    assert result.accepted() and not result.accepted(require_recording=True)
    row = json.loads(manifest.read_text())
    assert row["ok"] and not row["produced"]
    with pytest.raises(FileExistsError):
        record_attempt(manifest, "episode_00", 0, execute)
    assert len(manifest.read_text().splitlines()) == 1
