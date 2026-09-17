"""Serving a checkpoint whose recorded VLM backbone lives elsewhere (torch tier: imports gr00t)."""

import json

import pytest

# torch tier (tests/conftest.py): the GR00T venv's packages, skipped elsewhere.
pytest.importorskip("transformers")

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import get_backbone_cls  # noqa: E402
from gr00t.policy.gr00t_policy import _backbone_override  # noqa: E402


class _Config:
    def __init__(self, model_name):
        self.model_name = model_name


@pytest.mark.parametrize(
    "model_name",
    [
        "nvidia/Cosmos-Reason2-2B",
        "Qwen/Qwen3-VL-4B-Instruct",
        "/group-volume/henry_data/Checkpoints/Cosmos-Reason2-2B",
        "/workspace/huggingface_cache/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a1",
    ],
)
def test_backbone_class_is_chosen_by_model_family_not_by_hub_prefix(model_name):
    assert get_backbone_cls(_Config(model_name)).__name__ == "Qwen3Backbone"


def test_unknown_backbone_family_is_still_rejected():
    with pytest.raises(ValueError, match="Unsupported model name"):
        get_backbone_cls(_Config("/data/Checkpoints/Llama-3"))


def test_backbone_override_replaces_a_recorded_path_and_fails_early_on_a_missing_one(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"model_name": "/elsewhere/Cosmos-Reason2-2B"}))
    local = tmp_path / "Cosmos-Reason2-2B"
    local.mkdir()

    assert _backbone_override(ckpt, str(local)) == {"model_name": str(local)}
    with pytest.raises(ValueError, match="does not exist here"):
        _backbone_override(ckpt, None)
    with pytest.raises(ValueError, match="backbone directory does not exist"):
        _backbone_override(ckpt, str(tmp_path / "missing"))

    (ckpt / "config.json").write_text(json.dumps({"model_name": "nvidia/Cosmos-Reason2-2B"}))
    assert _backbone_override(ckpt, None) == {}
    (ckpt / "config.json").write_text(json.dumps({"model_name": str(local)}))
    assert _backbone_override(ckpt, None) == {}
