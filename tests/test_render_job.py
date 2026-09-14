"""Owned sleeping processes exercise launch/stop without a renderer or shared jobs."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from locomani.pipeline import render_job as jobs


def test_selection_requires_one_job_even_with_a_pattern():
    population = [jobs.Process(1, 1, "a", "/out/run_A"), jobs.Process(2, 1, "b", "/out/run_B")]
    for pattern in ["", "run"]:
        with pytest.raises(RuntimeError, match="Ambiguous"):
            jobs.select_job(population, pattern)
    assert jobs.select_job(population, "run_A") == ("a", "/out/run_A")
    assert jobs.select_job(population, "missing") is None


def test_reused_pid_is_never_signalled(monkeypatch):
    old = jobs.Process(100, 1, "job", "/out")
    population = iter([[old], []])
    monkeypatch.setattr(jobs, "processes", lambda: next(population))
    monkeypatch.setattr(jobs, "_open_pidfd", lambda pid: 123)
    monkeypatch.setattr(jobs, "_read_process", lambda pid: jobs.Process(100, 2, "job", "/out"))
    closed, signals = [], []
    monkeypatch.setattr(jobs.os, "close", closed.append)
    monkeypatch.setattr(jobs, "_signal", lambda *args: signals.append(args))
    assert jobs.stop_job(("job", "/out"), grace_seconds=0) == 0
    assert closed == [123] and signals == []


PAYLOAD = """
import json,os,subprocess,sys,time
from pathlib import Path
sleep = [sys.executable, '-c', 'import time;time.sleep(120)']
managed = subprocess.Popen(sleep, start_new_session=True)
env = {k:v for k,v in os.environ.items() if not k.startswith('LOCOMANI_RENDER_')}
unrelated = subprocess.Popen(sleep, env=env)
Path(sys.argv[1]).write_text(json.dumps([os.getpid(),managed.pid,unrelated.pid]))
time.sleep(120)
"""


@pytest.fixture
def launched_job(tmp_path):
    records = []

    def launch(name):
        output = tmp_path / name
        info = tmp_path / f"{name}.json"
        command = [
            sys.executable,
            str(Path(jobs.__file__)),
            "run",
            "--out",
            str(output),
            "--",
            sys.executable,
            "-c",
            PAYLOAD,
            str(info),
        ]
        supervisor = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        records.append((supervisor, info))
        deadline = time.monotonic() + 10
        while not info.exists() and supervisor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert info.exists(), supervisor.communicate(timeout=1)
        ids = json.loads(info.read_text())
        deadline = time.monotonic() + 5
        while len([p for p in jobs.processes() if p.output == str(output)]) < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        return output, ids, supervisor

    yield launch
    for supervisor, info in records:
        if info.exists():
            driver, managed, unrelated = json.loads(info.read_text())
            # These groups were created by the fixture, never discovered globally.
            for pgid in [driver, managed]:
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if supervisor.poll() is None:
            supervisor.terminate()
        supervisor.communicate(timeout=5)


def test_stop_preserves_other_jobs_and_unmarked_process_in_the_same_group(launched_job):
    out_a, (driver, worker, unrelated), supervisor = launched_job("review_A")
    out_b, _, _ = launched_job("review_B")
    population = jobs.processes()
    with pytest.raises(RuntimeError, match="Ambiguous"):
        jobs.select_job(population, "review_")
    assert jobs._read_process(unrelated) is None
    assert os.getpgid(driver) == os.getpgid(unrelated)
    assert os.getpgid(driver) != os.getpgid(worker)
    stop = Path(__file__).resolve().parents[1] / "scripts/stack/stop_render.sh"
    result = subprocess.run(["bash", str(stop), str(out_a)], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    supervisor.wait(timeout=5)
    os.kill(unrelated, 0)  # would fail with the old group-wide signalling
    assert any(p.output == str(out_b) for p in jobs.processes())
    assert not any(p.output == str(out_a) for p in jobs.processes())


def test_orphan_worker_keeps_its_job_identity_and_directory_lock(launched_job):
    output, (driver, worker, unrelated), supervisor = launched_job("orphan")
    os.kill(driver, signal.SIGKILL)
    supervisor.wait(timeout=5)
    matches = [p for p in jobs.processes() if p.output == str(output)]
    assert [p.pid for p in matches] == [worker]
    with pytest.raises(RuntimeError, match="owns"):
        jobs.run_job([sys.executable, "-c", "pass"], output)
    assert jobs.stop_job(jobs.select_job(matches), grace_seconds=0.1) == 1
    os.kill(unrelated, 0)


def test_batch_driver_propagates_renderer_failure_before_compositing(tmp_path):
    pilot, output = tmp_path / "pilot", tmp_path / "output"
    (pilot / "trace").mkdir(parents=True)
    (pilot / "trace/wall_s0_00_state_trace.npz").touch()
    renderer = tmp_path / "renderer"
    renderer.write_text(
        '#!/bin/bash\ncase "$*" in\n  *isaac_render*) exit 42;;\n  *) echo "mujoco completed";;\nesac\n'
    )
    renderer.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, RENDER_PY=str(renderer), LOCOMANI_ROOT=str(root))
    result = subprocess.run(
        ["bash", str(root / "scripts/render_wall_pilot.sh"), str(pilot), str(output), "1"],
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 42, result.stdout + result.stderr
    assert "RENDER_ALL_DONE" not in result.stdout
    assert not (output / "sbs_round00.mp4").exists()
