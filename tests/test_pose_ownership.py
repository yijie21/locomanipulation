import pytest

from locomani.pipeline.provenance import require_pose_ownership


def test_pose_takeover_requires_the_selected_binary_capability(tmp_path):
    binary = tmp_path / "deploy/target/release/g1_deploy_onnx_ref"
    binary.parent.mkdir(parents=True)
    env = {"SONIC_DEPLOY_BIN": str(binary)}
    binary.write_bytes(b"old binary")
    with pytest.raises(RuntimeError, match="sonic_pose_ownership.patch"):
        require_pose_ownership(env)
    binary.write_bytes(b"new binary POSE_UPPER_OWNERSHIP_V2")
    require_pose_ownership(env)
    # LOCKSTEP selects a different executable, even if the custom binary is patched.
    with pytest.raises(RuntimeError):
        require_pose_ownership({**env, "LOCKSTEP": "1", "LOCKSTEP_DEPLOY_ROOT": str(tmp_path / "missing")})
