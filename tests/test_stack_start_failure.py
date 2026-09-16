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
    with pytest.raises(RuntimeError, match="model_decoder.onnx"):
        Stack(StackSpec(deploy_root=str(root), policy="policy/missing"))._deploy_argv()
