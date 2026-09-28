"""`locomani.stack.Stack`: admission lock, failed-start cleanup and process-group stop, without GPUs."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from locomani import paths
from locomani.stack import manager
from locomani.stack.manager import Stack, StackSpec


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """Session root and lock dir under tmp; children are `sleep` processes started through `_spawn`."""
    monkeypatch.setattr(paths, "TMP", str(tmp_path / "runtime"))
    monkeypatch.setattr(manager, "session_root", lambda instance: tmp_path / "runtime" / f"p03_i{instance}")
    return tmp_path


def _sleeping_child(stack: Stack, name: str) -> int:
    # a child that itself forks: the grandchild must die with the group too
    return stack._spawn(name, ["bash", "-c", "sleep 120 & wait"], cwd=str(stack.root), env=dict(os.environ))


def _group_alive(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_start_failure_stops_its_children_and_releases_the_lock(runtime, monkeypatch):
    spec = StackSpec(instance=137, dds_domain=17, record=False)
    stack = Stack(spec)
    monkeypatch.setattr(stack, "_start_sim", lambda: _sleeping_child(stack, "sim"))

    def deploy_fails():
        raise stack._fail("deploy", "never reached 'Init Done'")

    monkeypatch.setattr(stack, "_start_deploy", deploy_fails)
    assert stack.env["SONIC_DDS_DOMAIN"] == "17"  # explicit config survives the environment build
    with pytest.raises(RuntimeError, match="never reached"):
        stack.up()
    assert not stack.pids and not stack.state_path.exists()
    lock = Path(paths.TMP) / "locomani_locks" / "instance_137.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)  # released: we can take it


def test_admission_refusal_does_not_launch_or_stop_another_owner(runtime):
    lock = Path(paths.TMP) / "locomani_locks" / "instance_137.lock"
    lock.parent.mkdir(parents=True)
    with subprocess.Popen(["sleep", "30"]) as other, lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stack = Stack(StackSpec(instance=137, record=False))
            with pytest.raises(RuntimeError, match="held by a live stack"):
                stack.up()
            assert other.poll() is None and not stack.pids
        finally:
            other.terminate()


def test_down_kills_the_whole_process_group_and_attach_reads_the_state_file(runtime, monkeypatch):
    spec = StackSpec(instance=138, record=False)
    stack = Stack(spec)
    monkeypatch.setattr(stack, "_start_sim", lambda: _sleeping_child(stack, "sim"))
    monkeypatch.setattr(stack, "_start_deploy", lambda: _sleeping_child(stack, "deploy"))
    monkeypatch.setattr(manager, "_wait", lambda pred, timeout, interval=1.0: True)  # no readiness probes here
    stack.up()
    state = json.loads(stack.state_path.read_text())
    assert set(state["pids"]) == {"sim", "deploy"} and state["ports"]["TF_CAMERA_PORT"] == 5555 + 10 * 138
    other = Stack.attach(138)
    assert other.pids == stack.pids
    pids = dict(stack.pids)
    monkeypatch.setattr(manager, "_wait", _real_wait)
    other._procs = stack._procs  # the attached view reaps through the owner's handles in this one-process test
    other.down()
    deadline = time.time() + 5
    while time.time() < deadline and any(_group_alive(p) for p in pids.values()):
        time.sleep(0.1)
    assert not any(_group_alive(p) for p in pids.values())
    assert not stack.state_path.exists()


def _real_wait(predicate, timeout, interval=1.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(min(interval, 0.1))
    return predicate()


def test_spec_builds_the_deploy_command_from_explicit_fields(tmp_path):
    root = tmp_path / "deploy"
    (root / "policy" / "p").mkdir(parents=True)
    (root / "policy" / "p" / "model_decoder.onnx").write_bytes(b"")
    spec = StackSpec(
        instance=2,
        gpu="1",
        deploy_root=str(root),
        policy="policy/p",
        encoder=False,
        lockstep=True,
        default_motion="clip",
    )
    argv, cwd, env = Stack(spec)._deploy_argv()
    assert cwd == str(root) and argv[0].endswith("target/release/g1_deploy_onnx_ref")
    assert (
        "--encoder-file" not in argv and "--sim-lockstep" in argv and argv[argv.index("--default-motion") + 1] == "clip"
    )
    assert argv[argv.index("--zmq-port") + 1] == "5576" and env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "1" and env["LOCOMANI_SIM_DIR"].startswith("/")
    # a lockstep deploy exits at init when handed a planner model ("--sim-lockstep does not permit a planner model")
    assert "--planner-file" not in argv
    wall, _, _ = Stack(StackSpec(deploy_root=str(root), policy="policy/p"))._deploy_argv()
    assert "--planner-file" in wall and "--sim-lockstep" not in wall
    with pytest.raises(RuntimeError, match="model_decoder.onnx"):
        Stack(StackSpec(deploy_root=str(root), policy="policy/missing"))._deploy_argv()


def test_a_camera_port_someone_else_holds_is_refused_not_adopted(runtime, monkeypatch):
    """An orphaned publisher satisfies "is anything listening" and must not be adopted.

    Found 2026-09-17: an aborted run left its image publisher on :5575; the probe
    saw the port open, this sim's own publisher died on bind, the sim ran on
    without images, and the crate expert reported `camera_unavailable` a full
    round later as though the task had failed.
    """
    stack = Stack(StackSpec(instance=139, record=False))
    monkeypatch.setattr(manager, "_port_open", lambda port: True)
    spawned = []
    monkeypatch.setattr(stack, "_spawn", lambda *a, **kw: spawned.append(a[0]))
    with pytest.raises(RuntimeError, match="already in use"):
        stack._start_sim()
    assert spawned == []  # refused before launching anything


def test_the_image_marker_is_read_from_this_run_s_section_of_the_shared_log(runtime, monkeypatch):
    """sim.log is shared per instance and only grows, so the marker needs this run's offset.

    Without it the readiness check finds a previous run's "Image publish
    frequency" line and passes while this sim is publishing nothing.
    """
    stack = Stack(StackSpec(instance=140, record=False))
    stack.root.mkdir(parents=True, exist_ok=True)
    stale = stack.root / "sim.log"
    stale.write_text(f"{manager.READY_SIM_IMAGE}: 30.0 -- from a previous run\n")

    publishes = False

    def start_sim():
        if publishes:  # what a healthy sim does: the marker appears AFTER up() took its offset
            with stale.open("a") as f:
                f.write(f"{manager.READY_SIM_IMAGE}: 31.0 -- this run\n")

    monkeypatch.setattr(stack, "_start_sim", start_sim)
    monkeypatch.setattr(stack, "_start_deploy", lambda: None)
    monkeypatch.setattr(manager, "_wait", lambda pred, timeout, interval=1.0: bool(pred()))
    for name in ("_acquire_lock", "_release_lock", "record_settings", "_save"):
        monkeypatch.setattr(stack, name, lambda: None)

    with pytest.raises(RuntimeError, match="not published a single image"):
        stack.up()  # the stale line alone must not satisfy it

    publishes = True
    stack.up()


def test_down_also_kills_a_deploy_another_process_restarted(runtime, monkeypatch):
    """2026-09-28: the evaluator attached before the rounds; the gym's per-round restart (its own attached Stack)
    recorded a new deploy only in the state file, and the evaluator's down() orphaned it."""
    spec = StackSpec(instance=139, record=False)
    owner = Stack(spec)
    monkeypatch.setattr(owner, "_start_sim", lambda: _sleeping_child(owner, "sim"))
    monkeypatch.setattr(owner, "_start_deploy", lambda: _sleeping_child(owner, "deploy"))
    monkeypatch.setattr(manager, "_wait", lambda pred, timeout, interval=1.0: True)
    owner.up()
    evaluator = Stack.attach(139)  # holds the first deploy's pid from here on
    restarter = Stack.attach(139)
    restarter._procs = owner._procs
    monkeypatch.setattr(restarter, "_start_deploy", lambda: _sleeping_child(restarter, "deploy"))
    monkeypatch.setattr(manager, "_wait", _real_wait)
    assert restarter.restart_deploy()
    fresh = json.loads(owner.state_path.read_text())["pids"]["deploy"]
    assert fresh != evaluator.pids["deploy"] and _group_alive(fresh)
    evaluator.down()
    deadline = time.time() + 5
    while time.time() < deadline and _group_alive(fresh):
        for child in restarter._procs.values():
            child.poll()  # reap: in this one-process test the killed deploy is our zombie until then
        time.sleep(0.1)
    assert not _group_alive(fresh) and not owner.state_path.exists()
