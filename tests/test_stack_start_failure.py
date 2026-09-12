"""Exercise actual shell admission, DDS delivery and failed-start cleanup without GPUs."""

import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def launcher(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo"
    (root / "scripts/stack").mkdir(parents=True)
    for source in ["scripts/paths.sh", "scripts/env_deploy.sh", "scripts/stack/start_stack.sh"]:
        shutil.copy2(repo / source, root / source)
    fake_python = root / "third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"
    fake_python.parent.mkdir(parents=True)
    marker = tmp_path / "child.json"
    fake_python.write_text(
        "#!/usr/bin/python3\nimport os,json,time\n"
        f'open({str(marker)!r}, "w").write(json.dumps({{"pid":os.getpid(), "dds":os.environ["SONIC_DDS_DOMAIN"]}}))\n'
        "time.sleep(120)\n"
    )
    fake_python.chmod(0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    ss = bindir / "ss"
    ss.write_text(f'#!/bin/sh\nwhile [ ! -f "{marker}" ]; do sleep 0.05; done\necho "LISTEN :6925 "\n')
    ss.chmod(0o755)
    env = dict(
        os.environ,
        LOCOMANI_ROOT=str(root),
        LOCOMANI_TMP=str(tmp_path / "runtime"),
        PATH=f"{bindir}:/usr/bin:/bin",
        INSTANCE="137",
        SONIC_DDS_DOMAIN="17",
        SONIC_DEPLOY_ROOT=str(tmp_path / "missing-deploy"),
    )
    yield root / "scripts/stack/start_stack.sh", env, marker
    if marker.exists():
        pid = json.loads(marker.read_text())["pid"]
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass


def test_start_failure_stops_its_child_and_releases_lock(launcher):
    script, env, marker = launcher
    run = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
    assert run.returncode == 1, run.stdout + run.stderr
    child = json.loads(marker.read_text())
    assert child["dds"] == "17"  # real launcher preserves explicit config
    assert not Path(f"/proc/{child['pid']}").exists()
    lock = Path(env["LOCOMANI_TMP"]) / "locomani_locks/instance_137.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_admission_refusal_does_not_launch_or_stop_another_owner(launcher):
    script, env, marker = launcher
    lock = Path(env["LOCOMANI_TMP"]) / "locomani_locks/instance_137.lock"
    lock.parent.mkdir(parents=True)
    with subprocess.Popen(["sleep", "30"]) as other, lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=10)
            assert run.returncode == 3
            assert other.poll() is None
            assert not marker.exists()
        finally:
            other.terminate()
