import json

import numpy as np
import pytest

from locomani.tasks.crate.motion_recording import MotionRecording
from scripts.experiments.crate_motion_report_20260913 import attempt


def test_report_joins_exact_state_window_and_keeps_failed_attempt(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    qvel = np.zeros((6, 9))
    qvel[:, 3] = [99, 1, 2, 3, 99, 99]
    np.savez(trace / "failed_state_trace.npz", seq=np.arange(6), qvel=qvel)
    rows = [dict(t=1 + k * 0.02, phase="grasp", velocity=[0.04 * k] * 17, seq=k + 1) for k in range(3)]
    (tmp_path / "failed_motion.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = attempt(
        tmp_path, dict(video="failed", ok=False, why="lost contact", reach_start_s=1, initial_edge_distance=1.19)
    )
    assert not result["ok"]
    assert not result["initial_distance_admissible"]
    assert result["early_base_angular_speed"]["n"] == 3
    assert result["early_base_angular_speed"]["max"] == 3
    np.testing.assert_allclose(result["sent_acceleration"]["max"], 2)


def test_existing_motion_evidence_cannot_be_overwritten(tmp_path):
    path = tmp_path / "motion.jsonl"
    path.write_text("original evidence\n")
    with pytest.raises(FileExistsError):
        MotionRecording(path, 0)
    assert path.read_text() == "original evidence\n"
