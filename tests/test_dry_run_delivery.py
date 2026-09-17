"""Every value the experiment file declares reaches the process that consumes it.

This is the test the 2026-09-17 defects would have failed. Each of them was a field declared in a
config group and then lost at the stage->subprocess seam: an evaluation spawning at 0.7 m while its
config said 1.1-1.3 m, a stack told to render nothing while its config asked for video. A stage
script's `--dry-run` prints every command it would run together with that command's own settings,
so the question "is this value delivered" has an answer that costs no GPU.

The check is deliberately literal -- the number in the file has to appear in the command line or the
environment. A test that reimplemented the assembly would pass on exactly the bugs it exists for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"


def dry_run(script: str, experiment: str, *extra: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), str(EXPERIMENTS / experiment), "--dry-run", *extra],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def gate8():
    cfg = yaml.safe_load((EXPERIMENTS / "bottle_simple_gate8.yaml").read_text())
    return cfg, dry_run("collect_bottle.py", "bottle_simple_gate8.yaml", "--gpu", "1", "--name", "delivery_check")


def test_the_spawn_distribution_is_delivered(gate8):
    cfg, out = gate8
    spawn = cfg["task"]["spawn"]
    assert f"--spawn-edge {spawn['edge_range'][0]} {spawn['edge_range'][1]}" in out
    assert f"--spawn-yaw-err {spawn['yaw_err_deg']}" in out
    assert f"--spawn-lateral {spawn['lateral']}" in out
    assert f"--edge-jitter {spawn['edge_jitter']}" in out


def test_the_collection_budget_is_delivered(gate8):
    cfg, out = gate8
    assert f"--rounds {cfg['collect']['batch_rounds']}" in out
    assert f"--seed {cfg['collect']['seed0']}" in out  # batch 0; later batches step by 100


def test_the_case_bank_and_the_target_are_delivered(gate8):
    cfg, out = gate8
    assert "--cases" in out and "cases_00.json" in out  # this batch's slice of the fixed bank
    assert f"--targets {cfg['task']['target']}" in out


def test_the_stack_is_told_where_to_record(gate8):
    _, out = gate8
    assert "SIM_VIDEO_DIR=" in out and "collect/videos_00" in out
    assert "SIM_STATE_TRACE_DIR=" in out and "state_trace_00" in out


def test_the_encoder_setting_reaches_the_children(gate8):
    """Collection needs the deploy's encoder; without it the planner's arm references are dropped."""
    _, out = gate8
    assert '"no_encoder": false' in out


def test_the_run_uses_its_own_scene_copy(gate8):
    _, out = gate8
    assert "LOCOMANI_SIM_DIR=" in out and "runs/delivery_check/scene" in out


def test_a_fresh_sample_run_asks_for_successes_not_cases():
    cfg = yaml.safe_load((EXPERIMENTS / "bottle_simple_200.yaml").read_text())
    out = dry_run("collect_bottle.py", "bottle_simple_200.yaml", "--gpu", "1", "--name", "delivery_check_200")
    assert f"--until-ok {cfg['collect']['successes']}" in out  # keep going until this many are kept
    assert "--cases" not in out  # every round drawn from task.spawn
    assert f"--seed {cfg['collect']['seed0']}" in out
