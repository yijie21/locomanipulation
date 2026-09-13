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


@pytest.mark.parametrize("failure", ["cleanup", "evaluation"])
def test_external_abort_remains_marked_when_finalization_fails(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace

    from geometry_fixture import scene_ik

    from locomani.tasks.crate import expert
    from locomani.tasks.crate.config import CrateTaskConfig
    from locomani.tasks.crate.visual_scene import MARKERS

    def broken():
        raise RuntimeError("finalization failed")

    stopped = []
    arm = SimpleNamespace(stop=lambda: stopped.append(True), q=[0.0] * 29)
    reset = SimpleNamespace(start_video=lambda label: None, end_video=broken if failure == "cleanup" else lambda: None)
    observation = {
        "fallen": True,
        "objects": {"blue_crate": {"pos": [0, 0, 0], "quat": [1, 0, 0, 0], "touching": []}},
    }
    monkeypatch.setattr(expert, "ArmIK", lambda path: scene_ik())
    monkeypatch.setattr(expert, "reset_ready", lambda *args: (arm, {}))
    if failure == "evaluation":
        monkeypatch.setattr(expert, "evaluate", lambda *args, **kwargs: broken())
    meta = {"crate_visual": {"markers": MARKERS, "grasp_points": [[0.19, 0, 0.06], [-0.19, 0, 0.06]]}}
    result = expert.run_attempt(
        SimpleNamespace(obs=lambda: observation), reset, None, meta, CrateTaskConfig(), tmp_path, "abort", None
    )
    assert not result.accepted() and result.why == "runtime_error"
    assert result.execution["metrics"]["intervention"] is True
    assert result.execution["metrics"]["control_reason"] == "external_fall_abort"
    assert stopped == [True]


def test_low_reach_rejects_reset_drift_before_starting_a_demonstration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import numpy as np
    from geometry_fixture import scene_ik

    from locomani.tasks.crate import expert
    from locomani.tasks.crate.config import CrateTaskConfig
    from locomani.tasks.crate.visual_scene import MARKERS

    arm = SimpleNamespace(q=np.zeros(29), stop=Mock())
    reset = SimpleNamespace(start_video=Mock())
    monkeypatch.setattr(expert, "ArmIK", lambda path: scene_ik())
    monkeypatch.setattr(expert, "reset_ready", lambda *a, **kw: (arm, {"start_edge_distance": 1.4}))
    meta = dict(
        crate_visual=dict(markers=MARKERS, grasp_points=[[0.19, 0, 0.06], [-0.19, 0, 0.06]]),
        objects={"blue_crate": {"surface": "A"}},
        surfaces={"A": {"xy": [0, 0], "flat_half_extent": [0.5, 0.5]}},
    )
    result = expert.run_attempt(
        SimpleNamespace(obs=lambda: {"pelvis": {"pos": [-1.4, 0, 0.78]}}),
        reset,
        None,
        meta,
        CrateTaskConfig.load("configs/tasks/crate_visual_low_reach.json"),
        tmp_path,
        "drift",
        None,
    )
    assert result.why == "invalid_initial_distance" and not result.accepted()
    reset.start_video.assert_not_called()
    arm.stop.assert_called_once()


def test_attempt_evaluates_the_observed_end_state_before_cleanup_can_change_it(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from geometry_fixture import scene_ik

    from locomani.tasks.crate import expert
    from locomani.tasks.crate.config import CrateTaskConfig
    from locomani.tasks.crate.visual_scene import MARKERS

    calls, cleaned = [0], [False]

    def observe():
        calls[0] += 1
        return {
            "fallen": calls[0] > 1,
            "objects": {
                "blue_crate": {
                    "pos": [0, 0, 0.1 if calls[0] > 1 and not cleaned[0] else 0],
                    "quat": [1, 0, 0, 0],
                    "touching": [],
                }
            },
        }

    arm = SimpleNamespace(stop=lambda: cleaned.__setitem__(0, True), q=[0.0] * 29)
    reset = SimpleNamespace(start_video=lambda label: None, end_video=lambda: None)
    monkeypatch.setattr(expert, "ArmIK", lambda path: scene_ik())
    monkeypatch.setattr(expert, "reset_ready", lambda *args: (arm, {}))
    meta = {"crate_visual": {"markers": MARKERS, "grasp_points": [[0.19, 0, 0.06], [-0.19, 0, 0.06]]}}
    result = expert.run_attempt(
        SimpleNamespace(obs=observe), reset, None, meta, CrateTaskConfig(), tmp_path, "end_state", None
    )
    assert result.why == "external_fall_abort"
    assert result.execution["metrics"]["rise"] == pytest.approx(0.1)
