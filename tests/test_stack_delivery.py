"""Stack and expert processes receive the same explicit settings and instance."""

import json
import sys
from types import SimpleNamespace

import pytest

from locomani.backends.sonic.config import SonicBackendConfig
from locomani.pipeline import stack
from locomani.pipeline.stages import Stage


def test_resolved_environment_does_not_inherit_another_instances_ports_or_optional_settings():
    inherited = dict(
        PATH="/bin",
        INSTANCE="0",
        TF_SONIC_PORT="5556",
        SONIC_DDS_DOMAIN="19",
        TF_OBS_DUMP="stale",
        TF_INIT_TOKEN_FILE="stale",
        MUJOCO_EGL_DEVICE_ID="0",
    )
    env = stack.resolve_env(SonicBackendConfig(no_encoder=False), instance=1, gpu="1", inherit=inherited)
    assert inherited["INSTANCE"] == "0"
    assert env["INSTANCE"] == "1" and env["NO_ENCODER"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"] == "1"
    for name in ("TF_SONIC_PORT", "SONIC_DDS_DOMAIN", "TF_OBS_DUMP", "TF_INIT_TOKEN_FILE"):
        assert name not in env


def test_stack_and_child_receive_identical_settings_through_real_processes(tmp_path, monkeypatch):
    received = tmp_path / "stack.json"
    script = tmp_path / "start.sh"
    script.write_text(
        f"#!/bin/sh\nexec {sys.executable} -c 'import os,json; print(json.dumps(dict(os.environ)))' > {received}\n"
    )
    monkeypatch.setattr(stack, "START_STACK", str(script))
    backend = SonicBackendConfig(no_encoder=False, obs_dump="observations", sonic_dds_domain=7)
    env = stack.start_stack(
        backend,
        SimpleNamespace(prompt="pick", init_token_file="token.json"),
        instance=1,
        gpu="1",
        inherit={"PATH": "/bin"},
        runner=Stage(tmp_path).run,
    )
    launched = json.loads(received.read_text())
    Stage(tmp_path).check(
        "expert", [sys.executable, "-c", "import os,json; print(json.dumps(dict(os.environ)))"], env=env
    )
    child = json.loads((tmp_path / "expert.log").read_text().splitlines()[-1])
    for name in ("INSTANCE", "NO_ENCODER", "TF_OBS_DUMP", "SONIC_DDS_DOMAIN", "TF_INIT_TOKEN_FILE"):
        assert launched[name] == child[name] == env[name]
    records = [json.loads(line) for line in (tmp_path / "commands.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["settings"] == records[1]["settings"]


def test_stage_retains_failed_exit_and_excludes_host_credentials(tmp_path):
    stage = Stage(tmp_path)
    with pytest.raises(RuntimeError, match="exited 3"):
        stage.check(
            "failure", [sys.executable, "-c", "raise SystemExit(3)"], env={"HF_TOKEN": "private", "NO_ENCODER": "0"}
        )
    command = json.loads((tmp_path / "commands.jsonl").read_text())
    assert command["settings"] == {"NO_ENCODER": "0"}
    assert json.loads((tmp_path / "timings.jsonl").read_text())["rc"] == 3


def test_full_backend_config_and_legacy_overrides_reach_python_consumers(tmp_path):
    from dataclasses import replace
    from pathlib import Path

    from locomani.backends.sonic.config import LockstepConfig

    backend = SonicBackendConfig(lockstep=LockstepConfig(ack_timeout=2.5, warmup_ticks=9, jitter_seed=17))
    env = stack.resolve_env(
        backend,
        inherit={"PYTHONPATH": str(Path("src").resolve())},
        extra_env={"TF_DATASET_FPS": "25", "NO_ENCODER": "0"},
    )
    Stage(tmp_path).check(
        "consumer",
        [
            sys.executable,
            "-c",
            "import json; "
            "from locomani.backends.sonic.config import DEFAULT_BACKEND; "
            "print(json.dumps(DEFAULT_BACKEND.to_dict()))",
        ],
        env=env,
    )
    actual = json.loads((tmp_path / "consumer.log").read_text().splitlines()[-1])
    assert actual == replace(backend, dataset_fps=25.0, no_encoder=False).to_dict()


def test_legacy_string_settings_round_trip_without_quoting_paths():
    backend = SonicBackendConfig()
    env = stack.resolve_env(backend, inherit={}, extra_env={"TF_OBS_DUMP": "/tmp/dump with spaces", "HAND": "dex3"})
    actual = SonicBackendConfig.from_dict(json.loads(env["LOCOMANI_BACKEND_JSON"]))
    assert actual.obs_dump == env["TF_OBS_DUMP"] == "/tmp/dump with spaces"
    assert actual.hand == env["HAND"] == "dex3"
