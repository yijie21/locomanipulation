"""What went wrong when starting a stack and driving it became two processes (review of 2026-09-23).

Each test pins one of the defects that review found. None of them showed up in the gates, which
exercise one uninterrupted run on a fresh instance:

  * the bimanual eval stack started without SONIC_AUTO_START while its rounds ran clean, so
    nothing would have engaged the deploy;
  * an interrupted batch was appended to the bookkeeping twice on resume, shifting the
    by-position session/manifest pairing `convert` relies on;
  * batch 0 took the newest exporter session by mtime from a directory every run shares;
  * a driver could attach to a stack another run, or the other mode, had started;
  * in-process stages wrote no log, and a failed bimanual batch was timed as rc 0;
  * the stack starter's children got no host environment, and a stray SONIC_AUTO_START could leak.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from locomani.pipeline import batches, run_context, run_stack
from locomani.pipeline.stages import Stage
from locomani.stack import StackSpec

ROOT = Path(__file__).resolve().parents[1]


def _experiment(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs/experiments" / name).read_text())


# -- the eval stack's backend is the line's answer ------------------------------------------------


def test_the_bimanual_eval_stack_engages_itself_because_its_rounds_run_clean(tmp_path):
    cfg = _experiment("crate_table30_record.yaml")
    assert cfg["eval"]["client"] == "blocking"  # the pin that made EvalConfig read it as "not clean"
    spec = run_stack.eval_spec(cfg, tmp_path, tmp_path / "scene", "1")
    assert spec.auto_start is True
    assert spec.encoder is False
    assert spec.settings()["SONIC_AUTO_START"] == "1"
    assert spec.default_motion == "neutral_kick_R_001__A543"


def test_the_single_arm_eval_stack_still_follows_its_client(tmp_path):
    cfg = _experiment("bottle_simple_200.yaml")
    spec = run_stack.eval_spec(cfg, tmp_path, tmp_path / "scene", "1")
    assert spec.auto_start is (cfg["eval"].get("client", "async") == "async")
    assert (spec.default_motion is not None) is spec.auto_start


def test_the_collect_stack_is_not_a_clean_deploy(tmp_path):
    cfg = _experiment("bottle_simple_200.yaml")
    spec = run_stack.collect_spec(cfg, tmp_path, tmp_path / "scene", tmp_path / "collect", tmp_path / "trace", "1")
    assert spec.auto_start is False and spec.default_motion is None


# -- resume -------------------------------------------------------------------------------------


def test_an_interrupted_batch_is_rerun_not_appended_twice(tmp_path):
    finished, unfinished = tmp_path / "manifest_00.jsonl", tmp_path / "manifest_01.jsonl"
    finished.write_text('{"ok": true}\n')
    unfinished.write_text('{"ok": true}\n')  # one round got through before the interruption
    inputs = tmp_path / "inputs.json"
    inputs.write_text(
        json.dumps(dict(sessions=["s0", "s1"], manifests=[str(finished), str(unfinished)], kept=1, batches=1))
    )

    state = batches.resume_state(inputs)

    assert state["sessions"] == ["s0"] and state["manifests"] == [str(finished)]
    assert json.loads(inputs.read_text()) == state
    assert not unfinished.exists()  # the next attempt starts an empty manifest
    (aside,) = tmp_path.glob("interrupted_manifest_01_*.jsonl")
    assert aside.read_text() == '{"ok": true}\n'
    assert sorted(tmp_path.glob("manifest_*.jsonl")) == [finished]  # gate globs must not see it


def test_a_fresh_run_starts_empty(tmp_path):
    assert batches.resume_state(tmp_path / "inputs.json") == dict(sessions=[], manifests=[], kept=0, batches=0)


# -- which exporter session a batch records into -----------------------------------------------


def _recording_stack(session, exporter_pid):
    stack = Mock()
    stack.spec = StackSpec(record=True)
    stack.session = session
    stack.pids = {"exporter": exporter_pid}
    stack.env = {}
    return stack


def test_batch_zero_takes_the_session_the_stack_named(tmp_path):
    stack = _recording_stack(tmp_path / "named", os.getpid())
    (tmp_path / "newer_by_mtime").mkdir()  # what the mtime guess would have picked
    assert batches.open_exporter_session(stack, tmp_path, 0) == str(tmp_path / "named")
    stack.restart_exporter.assert_not_called()


def test_a_stack_without_a_named_session_restarts_rather_than_guesses(tmp_path):
    stack = _recording_stack(None, os.getpid())

    def restart():
        stack.session = tmp_path / "fresh"

    stack.restart_exporter.side_effect = restart
    assert batches.open_exporter_session(stack, tmp_path, 0) == str(tmp_path / "fresh")
    assert stack.env["TF_FRAME_AUDIT_PATH"].endswith("frame_audit_00.jsonl")


# -- a batch's episodes are on disk before its exporter is restarted ------------------------------


def _session_with(tmp_path, n_indexed, n_parquet):
    session = tmp_path / "session"
    (session / "meta").mkdir(parents=True, exist_ok=True)
    (session / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (session / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": e}) + "\n" for e in range(n_indexed))
    )
    for e in range(n_parquet):
        (session / "data" / "chunk-000" / f"episode_{e:06d}.parquet").write_bytes(b"PAR1")
    return session


def _manifest(tmp_path, produced):
    manifest = tmp_path / "manifest_00.jsonl"
    manifest.write_text("".join(json.dumps({"ok": True, "produced": p}) + "\n" for p in produced))
    return manifest


def test_the_last_episode_counts_once_its_parquet_is_written(tmp_path):
    # save_episode appends meta/episodes.jsonl BEFORE it writes the parquet
    assert batches.episodes_on_disk(_session_with(tmp_path, 3, 2)) == 2
    assert batches.episodes_on_disk(_session_with(tmp_path, 3, 3)) == 3


def test_wait_saved_waits_for_the_last_save_and_ignores_unproduced_rounds(tmp_path):
    session = _session_with(tmp_path, 2, 1)
    manifest = _manifest(tmp_path, [True, False, True])  # the middle round never started recording
    timer = threading.Timer(0.3, lambda: _session_with(tmp_path, 2, 2))
    timer.start()
    try:
        assert batches.wait_saved(str(session), manifest, timeout=5.0, poll=0.05) == 2
    finally:
        timer.cancel()


def test_wait_saved_refuses_a_session_the_exporter_never_finished(tmp_path):
    # box_room_grasp_200_cap120_20260925 batch 79: 10 produced, the exporter restarted mid-save, 9 on disk
    session = _session_with(tmp_path, 9, 9)
    with pytest.raises(SystemExit, match="9 episode.* produced 10"):
        batches.wait_saved(str(session), _manifest(tmp_path, [True] * 10), timeout=0.2, poll=0.05)
    assert batches.wait_saved(None, _manifest(tmp_path, [True])) == 0  # nothing records


# -- whose stack ---------------------------------------------------------------------------------


def _owned(run_dir, mode):
    return SimpleNamespace(run_dir=run_dir, mode=mode, spec=StackSpec(instance=2))


def test_a_driver_accepts_its_own_runs_stack(tmp_path):
    run_context.require_stack_of(_owned(tmp_path / "run", "collect"), tmp_path / "run", "collect")


@pytest.mark.parametrize(
    "owner, mode, says",
    [
        ("other", "collect", "serves"),  # a --keep-stack left over, or two different --name
        ("run", "eval", "started for 'eval'"),  # the deploy has no encoder
        (None, None, "does not say which run"),  # state written before stacks recorded their run
    ],
)
def test_a_driver_refuses_a_stack_it_does_not_own(tmp_path, owner, mode, says):
    stack = _owned(None if owner is None else tmp_path / owner, mode)
    with pytest.raises(SystemExit) as refusal:
        run_context.require_stack_of(stack, tmp_path / "run", "collect")
    assert says in str(refusal.value) and "stop_stack.sh --instance 2" in str(refusal.value)


def test_the_state_file_carries_the_run_the_mode_and_the_session(tmp_path, monkeypatch):
    from locomani.stack import Stack, manager

    monkeypatch.setattr(manager, "session_root", lambda instance: tmp_path)
    stack = Stack(StackSpec(instance=3), run_dir=tmp_path / "run", mode="eval")
    stack.session = tmp_path / "outputs" / "2026-09-23-00-00-00"
    stack._save()
    again = Stack.attach(3)
    assert (again.run_dir, again.mode, again.session) == (tmp_path / "run", "eval", stack.session)


# -- environment -----------------------------------------------------------------------------------


def test_children_inherit_the_host_but_not_its_auto_start(monkeypatch):
    monkeypatch.setenv("SONIC_AUTO_START", "1")
    monkeypatch.setenv("TMPDIR", "/workspace/tmp")
    env = StackSpec(auto_start=False).child_env()
    assert "SONIC_AUTO_START" not in env
    assert env["TMPDIR"] == "/workspace/tmp" and env["PATH"] == os.environ["PATH"]


# -- in-process stages leave a log and an honest rc -----------------------------------------------


def test_an_in_process_stage_logs_python_c_level_and_child_output(tmp_path, capfd):
    stage = Stage(tmp_path)
    with capfd.disabled(), stage.timed("collect_00"):  # pytest's own capture swaps the same fds
        print("from python")
        os.write(1, b"from the fd\n")
        subprocess.run(["echo", "from a child"], check=True)
    log = (tmp_path / "collect_00.log").read_text()
    assert "from python" in log and "from the fd" in log and "from a child" in log
    (row,) = [json.loads(r) for r in (tmp_path / "timings.jsonl").read_text().splitlines()]
    assert row["rc"] == 0


def test_a_failed_bimanual_batch_is_timed_as_a_failure_and_the_run_goes_on(tmp_path, monkeypatch, capfd):
    from locomani.tasks.crate import collect, expert

    cfg = dict(
        collect=dict(seed0=7000, batch_rounds=2, record=False),
        task=dict(expert="visual"),
        scene=dict(dir="assets/scenes/crate_table30"),
    )
    monkeypatch.setattr(collect, "profile_of", lambda *a, **k: Mock(to_dict=lambda: {}))
    monkeypatch.setattr(expert, "run", Mock(side_effect=SystemExit("[crate] scene is not painted")))
    stack = _recording_stack(None, os.getpid())
    stack.spec = StackSpec(record=False)
    state = dict(sessions=[], manifests=[], kept=0, batches=0)

    with capfd.disabled():  # pytest's own capture swaps the fds the stage log tees
        manifest = collect.run_batch(cfg, tmp_path, stack, state)  # does not raise

    assert manifest == tmp_path / "manifest_00.jsonl"
    (row,) = [json.loads(r) for r in (tmp_path / "timings.jsonl").read_text().splitlines()]
    assert row["stage"] == "visual_00" and row["rc"] == 1
    assert "scene is not painted" in (tmp_path / "visual_00.log").read_text()


def test_the_eval_stack_writes_state_traces_for_offline_renders(tmp_path):
    """backend.render offline writes no video; the per-round qpos trace is what a round is rendered from."""
    cfg = _experiment("crate_table30_record.yaml")
    spec = run_stack.eval_spec(cfg, tmp_path, tmp_path / "scene", "1")
    assert spec.env["SIM_STATE_TRACE_DIR"] == str(tmp_path / "eval" / "state_trace")
