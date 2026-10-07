"""Storage/model layouts retain their pinned meaning and identities hash actual weights."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from locomani.pipeline.provenance import sonic_policy_identity
from locomani.schema import FIELD_WIDTHS, SONIC_BOTTLE, SONIC_CRATE, SONIC_MODALITY, VISUAL_BOTTLE


def test_sonic_raw_storage_layout_matches_the_pre_migration_snapshot():
    previous = json.loads((Path(__file__).parent / "fixtures/sonic_modality_v1.json").read_text())
    assert SONIC_MODALITY == previous
    assert (SONIC_BOTTLE.dimension("state"), SONIC_BOTTLE.dimension("action")) == (46, 65)
    assert (SONIC_CRATE.dimension("state"), SONIC_CRATE.dimension("action")) == (46, 66)
    assert (VISUAL_BOTTLE.dimension("state"), VISUAL_BOTTLE.dimension("action")) == (91, 29)
    assert SONIC_MODALITY["state"]["right_arm"]["start"] == 29
    assert SONIC_BOTTLE.state_keys.index("right_arm") < SONIC_BOTTLE.state_keys.index("left_hand")


def _official_sonic_state_keys() -> tuple[str, ...]:
    """The state groups of Isaac-GR00T's built-in unitree_g1_sonic config, read from the vendored source."""
    source = Path(__file__).resolve().parents[1] / "src/gr00t/configs/data/embodiment_configs.py"
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "unitree_g1_sonic":
                    state = dict(zip((k.value for k in value.keys), value.values))["state"]
                    return tuple(ast.literal_eval(next(kw.value for kw in state.keywords if kw.arg == "modality_keys")))
    raise AssertionError("unitree_g1_sonic not found in embodiment_configs.py")


def test_sonic_tasks_select_the_official_state_in_its_order():
    """NVIDIA's SONIC + GR00T recipe: 43 joints + projected gravity, no wrist pose (DECISIONS 2026-10-07)."""
    official = _official_sonic_state_keys()
    assert SONIC_BOTTLE.state_keys == SONIC_CRATE.state_keys == official
    assert sum(FIELD_WIDTHS["state"][k] for k in official) == 46


def test_observation_contract_rejects_missing_cameras_and_wrong_state_width():
    obs = {f"video.{k}": np.zeros((1, 2, 2, 3)) for k in SONIC_BOTTLE.video_keys}
    obs.update({f"state.{k}": np.zeros((1, FIELD_WIDTHS["state"][k])) for k in SONIC_BOTTLE.state_keys})
    SONIC_BOTTLE.validate_observation(obs)
    with pytest.raises(ValueError, match="missing observation field video.right_wrist"):
        SONIC_BOTTLE.validate_observation({k: v for k, v in obs.items() if k != "video.right_wrist"})
    with pytest.raises(ValueError, match="wrong observation width for state.right_arm"):
        SONIC_BOTTLE.validate_observation(dict(obs, **{"state.right_arm": np.zeros((1, 6))}))


def test_checkpoint_identity_detects_equal_size_weight_replacement_and_resolved_policy(tmp_path):
    policy = tmp_path / "policy/custom"
    policy.mkdir(parents=True)
    (policy / "model_decoder.onnx").write_bytes(b"aaaa")
    env = {"SONIC_DEPLOY_ROOT": str(tmp_path), "SONIC_POLICY": "policy/custom"}
    before = sonic_policy_identity(env)
    (policy / "model_decoder.onnx").write_bytes(b"bbbb")
    after = sonic_policy_identity(env)
    assert before["model_decoder.onnx_bytes"] == after["model_decoder.onnx_bytes"]
    assert before["model_decoder.onnx_sha256"] != after["model_decoder.onnx_sha256"]
    assert after["policy"] == "policy/custom"


def test_checkpoint_identity_hashes_shards_and_normalization(tmp_path):
    from locomani.pipeline.provenance import checkpoint_identity

    (tmp_path / "model-00001.safetensors").write_bytes(b"aaa")
    (tmp_path / "processor").mkdir()
    stats = tmp_path / "processor/statistics.json"
    stats.write_text('{"scale": 1}')
    first = checkpoint_identity(tmp_path)
    stats.write_text('{"scale": 2}')
    second = checkpoint_identity(tmp_path)
    assert first["files"]["model-00001.safetensors"] == second["files"]["model-00001.safetensors"]
    assert first["files"]["processor/statistics.json"] != second["files"]["processor/statistics.json"]


def test_deploy_receipt_records_foreign_cwd_and_keeps_each_launch(tmp_path, monkeypatch):
    from locomani.pipeline.provenance import record_deploy

    normal = tmp_path / "normal"
    foreign = tmp_path / "foreign"
    for root, content in [(normal, b"aaaa"), (foreign, b"bbbb")]:
        (root / "policy/custom").mkdir(parents=True)
        (root / "policy/custom/model_decoder.onnx").write_bytes(content)
        (root / "target/release").mkdir(parents=True)
        (root / "target/release/g1_deploy_onnx_ref").write_bytes(b"executable")
    env = dict(
        SONIC_DEPLOY_ROOT=str(normal), CLEAN_DEPLOY="1", CLEAN_DEPLOY_ROOT=str(foreign), SONIC_POLICY="policy/custom"
    )
    predicted = sonic_policy_identity(env)
    monkeypatch.chdir(foreign)
    receipt = tmp_path / "deploy_receipts.jsonl"
    first = record_deploy(receipt, "./target/release/g1_deploy_onnx_ref", "policy/custom")
    (foreign / "policy/custom/model_decoder.onnx").write_bytes(b"cccc")
    second = record_deploy(receipt, "./target/release/g1_deploy_onnx_ref", "policy/custom")
    assert first["model_decoder.onnx_sha256"] == predicted["model_decoder.onnx_sha256"]
    assert first["model_decoder.onnx_sha256"] != second["model_decoder.onnx_sha256"]
    assert first["cwd"] == str(foreign)
    assert len(receipt.read_text().splitlines()) == 2
