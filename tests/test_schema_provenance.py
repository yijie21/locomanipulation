"""Storage/model layouts retain their pinned meaning and identities hash actual weights."""

import json
from pathlib import Path

import numpy as np
import pytest

from locomani.pipeline.provenance import sonic_policy_identity
from locomani.schema import FIELD_WIDTHS, SONIC_BOTTLE, SONIC_CRATE, SONIC_MODALITY, VISUAL_BOTTLE


def test_sonic_raw_storage_layout_matches_the_pre_migration_snapshot():
    previous = json.loads((Path(__file__).parent / "fixtures/sonic_modality_v1.json").read_text())
    assert SONIC_MODALITY == previous
    assert (SONIC_BOTTLE.dimension("state"), SONIC_BOTTLE.dimension("action")) == (53, 65)
    assert (SONIC_CRATE.dimension("state"), SONIC_CRATE.dimension("action")) == (60, 66)
    assert (VISUAL_BOTTLE.dimension("state"), VISUAL_BOTTLE.dimension("action")) == (91, 29)
    assert SONIC_MODALITY["state"]["right_arm"]["start"] == 29
    assert SONIC_BOTTLE.state_keys.index("right_arm") < SONIC_BOTTLE.state_keys.index("left_hand")


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
