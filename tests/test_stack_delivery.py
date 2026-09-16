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
    assert env["INSTANCE"] == "1" and '"no_encoder": false' in env["LOCOMANI_BACKEND_JSON"]
    assert env["CUDA_VISIBLE_DEVICES"] == env["MUJOCO_EGL_DEVICE_ID"] == "1"
    assert env["TF_SONIC_PORT"] == "5566"
    for name in ("SONIC_DDS_DOMAIN", "TF_INIT_TOKEN_FILE"):
        assert name not in env


def test_stack_and_child_receive_identical_settings_through_real_processes(tmp_path, monkeypatch):
    from locomani.stack import manager

    received = tmp_path / "stack.json"

    def fake_up(self):
        received.write_text(json.dumps(self.env))
        return dict(self.env)

    monkeypatch.setattr(manager.Stack, "up", fake_up)
    monkeypatch.setattr(manager, "session_root", lambda instance: tmp_path / f"p03_i{instance}")
    backend = SonicBackendConfig(no_encoder=False, obs_dump="observations", sonic_dds_domain=7)
    env = stack.start_stack(
        backend,
        SimpleNamespace(prompt="pick", init_token_file="token.json"),
        instance=1,
        gpu="1",
        inherit={"PATH": "/bin"},
        runner=Stage(tmp_path).run,
    )
    stack.LIVE.pop(1, None)
    launched = json.loads(received.read_text())
    Stage(tmp_path).check(
        "expert", [sys.executable, "-c", "import os,json; print(json.dumps(dict(os.environ)))"], env=env
    )
    child = json.loads((tmp_path / "expert.log").read_text().splitlines()[-1])
    for name in ("INSTANCE", "LOCOMANI_BACKEND_JSON", "SONIC_DDS_DOMAIN", "TF_INIT_TOKEN_FILE"):
        assert launched[name] == child[name] == env[name]
    assert launched["MUJOCO_EGL_DEVICE_ID"] == "1" and launched["TF_SONIC_PORT"] == "5566"
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
