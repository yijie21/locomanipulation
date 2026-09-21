"""What a round driver does when the terminal that should hold the stack is not there.

Since 2026-09-21 the stack and the agent that drives it are separate processes, started by
separate scripts. That seam has exactly one new way to go wrong -- the driver runs without its
stack -- and the failure has to name the command that fixes it, because the alternative is a
`FileNotFoundError` on a session path that says nothing about which terminal was forgotten.

The admission check runs BEFORE the attach, so on a loaded machine that is the message the user
gets instead; the order is asserted here so a later edit cannot quietly put a stack lookup in
front of the resource guard.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locomani.pipeline import run_collect_single_arm, run_context, run_eval_single_arm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "configs/experiments/bottle_simple_200.yaml"


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    """A claimed run directory with no stack anywhere, and every resource check satisfied."""
    monkeypatch.setattr(run_context, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(run_context, "require_headroom", lambda *a, **k: {})
    monkeypatch.setattr(run_context, "require_deploy_patches", lambda cfg: None)
    cfg = yaml.safe_load(EXPERIMENT.read_text())
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return experiment, tmp_path / "runs" / cfg["name"]


def test_collection_without_a_stack_names_the_script_that_starts_one(run_dir, monkeypatch):
    experiment, _ = run_dir
    monkeypatch.setattr(run_collect_single_arm, "line_of", _line_that_prepares_nothing)
    monkeypatch.setattr(run_collect_single_arm.Stack, "attach", Mock(side_effect=FileNotFoundError("no state")))
    with pytest.raises(SystemExit) as refusal:
        run_collect_single_arm.main([str(experiment), "--gpu", "0"])
    assert "scripts/collect_data.sh" in str(refusal.value)


def test_evaluation_without_a_stack_names_the_script_that_starts_one(run_dir, monkeypatch, tmp_path):
    experiment, run = run_dir
    run_context.open_run(experiment)  # claim the directory the way the collection stage would
    (run / "summary.json").write_text('{"train": {"checkpoint": "%s"}}' % (tmp_path / "ck"))
    monkeypatch.setattr(run_eval_single_arm.Stack, "attach", Mock(side_effect=FileNotFoundError("no state")))
    monkeypatch.setattr(run_eval_single_arm, "wait_for_server", Mock(side_effect=AssertionError("waited first")))
    with pytest.raises(SystemExit) as refusal:
        run_eval_single_arm.main([str(run), "--gpu", "0"])
    assert "scripts/start_stack.sh" in str(refusal.value)


def test_the_driver_is_not_charged_for_the_stack_it_attaches_to(run_dir, monkeypatch):
    """The admission check subtracts what the live stack already holds.

    This assertion used to be the opposite -- guard first, stack second -- and the 2026-09-21
    bimanual gate overturned it: both processes asked for the whole stage budget, so the driver
    was refused for resources the stack it was about to use had already taken (pids 1864 + 1800 >
    3640 while that stack held 313 of the 1864). The driver cannot subtract what it has not yet
    looked up, so the order is attach, then ask for the rest. A machine with no stack at all is
    told to start one, which is the thing that would have to happen anyway.
    """
    experiment, _ = run_dir
    seen = {}
    monkeypatch.setattr(run_collect_single_arm, "line_of", _line_that_prepares_nothing)
    monkeypatch.setattr(
        run_collect_single_arm.Stack, "attach", Mock(return_value=_stack_with(pids={"sim": 11, "deploy": 12}))
    )
    monkeypatch.setattr(run_context, "group_threads", lambda pids: 313)
    monkeypatch.setattr(run_context, "require_headroom", lambda *a, **k: seen.update(k) or {})
    monkeypatch.setattr(run_collect_single_arm, "run_batch", Mock(side_effect=SystemExit("stop here")))
    with pytest.raises(SystemExit):
        run_collect_single_arm.main([str(experiment), "--gpu", "0"])
    assert seen.get("already") == 313


def _stack_with(pids):
    stack = Mock()
    stack.pids = pids
    return stack


def _line_that_prepares_nothing(cfg):
    from locomani.pipeline.lines import Line

    return Line(
        name="single_arm",
        prepare=lambda c, r: r / "scene",
        collect_root=lambda r: r / "collect",
        trace_root=lambda r: r / "collect" / "state_trace",
        scene_name=lambda c, r: r / "scene",
    )
