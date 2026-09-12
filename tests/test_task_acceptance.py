"""Every task can inspect a completed result without invoking its runtime."""

import subprocess
import sys

import pytest

from locomani.pipeline.execution_record import RoundResult
from locomani.tasks.registry import TASKS


@pytest.mark.parametrize("name", sorted(TASKS))
def test_task_success_and_recording_acceptance_are_distinct(name):
    task = TASKS[name]
    success = RoundResult.from_mapping(dict(ok=True, why="", produced=False))
    assert task.acceptance(success)
    assert not task.acceptance(success, require_recording=True)
    assert task.acceptance(RoundResult(True, "", True, {}), require_recording=True)
    assert not task.acceptance(RoundResult(False, "[hold] dropped", True, {}))
    with pytest.raises(TypeError, match="completed RoundResult"):
        task.acceptance("path-or-runtime-input")


def test_acceptance_imports_no_simulator_policy_or_task_runtime():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'src'); "
            "from locomani.tasks.registry import TASKS; from locomani.pipeline.execution_record import RoundResult; "
            "assert all(t.acceptance(RoundResult(True, '', False, {})) for t in TASKS.values()); "
            "assert not ({'mujoco', 'torch', 'locomani.tasks.bottle.grasp_gym', "
            "'locomani.tasks.crate.bimanual_carry', 'locomani.tasks.bottle_visual.round'} & sys.modules.keys())",
        ],
        check=True,
        capture_output=True,
    )


def test_result_normalization_rejects_missing_or_string_verdicts_and_copies_evidence():
    for row in ({}, {"ok": "false"}, {"ok": True, "produced": "false"}):
        with pytest.raises(ValueError):
            RoundResult.from_mapping(row)
    row = dict(ok=True, why="", execution={"metrics": {"lift": 0.1}})
    result = RoundResult.from_mapping(row)
    row["execution"]["metrics"]["lift"] = 0.0
    assert result.execution["metrics"]["lift"] == 0.1
