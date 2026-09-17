"""One expanded config in, one stack out -- and one place that decides what the children are told.

`spec_from_config` is how the flat stage scripts will start a stack: hand it the experiment file and
the block that stack uses, and it answers with the whole specification. `StackSpec.settings()` is
then the only function that decides a child's environment, and `Stack` writes it to the run as
`stack_env.json`. The bug this shape removes had one variable resolved in three places, so the
exporter received the video directory and the sim did not (2026-09-17).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from locomani.stack import Stack, spec_from_config

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT = "configs/experiments/bottle_simple_200.yaml"


@pytest.fixture
def cfg():
    return yaml.safe_load(open(EXPERIMENT))


def collect_spec(cfg, run_dir, **kw):
    return spec_from_config(cfg, run_dir=run_dir, gpu="1", video_dir=run_dir / "collect" / "videos_00", **kw)


def test_settings_are_deterministic(cfg, tmp_path):
    assert collect_spec(cfg, tmp_path).settings() == collect_spec(cfg, tmp_path).settings()


def test_settings_carry_what_a_child_cannot_guess(cfg, tmp_path):
    settings = collect_spec(cfg, tmp_path).settings()
    assert settings["SIM_VIDEO_DIR"] == str(tmp_path / "collect" / "videos_00")  # the sim's video writer
    assert settings["SIM_VIS"] == cfg["backend"]["sim_vis"]
    assert settings["LOCOMANI_SIM_DIR"] == str(tmp_path / "scene")  # the run's own scene copy, not the shared one
    assert settings["CUDA_VISIBLE_DEVICES"] == "1" and settings["MUJOCO_EGL_DEVICE_ID"] == "1"
    assert settings["INSTANCE"] == str(cfg["collect"]["instance"])
    assert int(settings["TF_CAMERA_PORT"]) == 5555 + 10 * cfg["collect"]["instance"]
    assert json.loads(settings["LOCOMANI_BACKEND_JSON"])["hand"] == cfg["backend"]["hand"]


def test_the_backend_block_chooses_the_encoder(cfg, tmp_path):
    """Collection needs the deploy's encoder, evaluation refuses to run with it."""
    assert collect_spec(cfg, tmp_path).encoder is True
    evaluation = spec_from_config(cfg, run_dir=tmp_path, gpu="1", backend=cfg["eval"]["backend"], record=False)
    assert evaluation.encoder is False
    assert evaluation.record is False
    assert json.loads(evaluation.settings()["LOCOMANI_BACKEND_JSON"])["no_encoder"] is True


def test_an_inherited_video_directory_cannot_leak_in(cfg, tmp_path):
    """A stack that records nowhere must not pick up the last run's directory from the environment."""
    spec = spec_from_config(cfg, run_dir=tmp_path, gpu="0")
    assert "SIM_VIDEO_DIR" not in spec.child_env({"SIM_VIDEO_DIR": "/tmp/somewhere_else"})


def test_the_run_keeps_what_its_children_were_told(cfg, tmp_path):
    spec = collect_spec(cfg, tmp_path)
    path = Stack(spec, base_env={}, run_dir=tmp_path).record_settings()
    assert json.loads(path.read_text()) == spec.settings()


def test_without_a_run_directory_nothing_is_written(cfg, tmp_path):
    assert Stack(collect_spec(cfg, tmp_path), base_env={}).record_settings() is None
