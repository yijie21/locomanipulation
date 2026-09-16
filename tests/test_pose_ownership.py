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


@pytest.mark.parametrize("expert", ["visual", "carry"])
def test_whole_body_launch_rejects_an_old_binary_before_starting_resources(tmp_path, monkeypatch, expert):
    from unittest.mock import Mock

    from locomani import conf
    from locomani.stages import collect

    run, cfg = conf.load_run(
        [
            "task=crate_visual_low_reach",
            "embodiment=g1_bimanual",
            "scene=crate_table30_depth04",
            f"task.expert={expert}",
        ]
    )
    monkeypatch.setattr("locomani.tasks.crate.prepare.prepare_scene", lambda *a, **k: tmp_path)
    monkeypatch.setattr(collect, "Stack", Mock(side_effect=AssertionError("resources started before admission")))
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(tmp_path / "unpatched"))
    with pytest.raises(RuntimeError, match="sonic_pose_ownership.patch"):
        collect.run(run, cfg, tmp_path / "out")


def test_standalone_whole_body_carry_checks_the_same_capability(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from locomani.tasks.crate import bimanual_carry

    monkeypatch.setattr("sys.argv", ["carry", "--config", "configs/tasks/crate_low_reach.json"])
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(tmp_path / "unpatched"))
    monkeypatch.setattr(bimanual_carry.ports, "instance", Mock(side_effect=AssertionError("admission was bypassed")))
    with pytest.raises(RuntimeError, match="sonic_pose_ownership.patch"):
        bimanual_carry.main()


def test_loaded_handoff_requires_hand_ownership_without_restricting_existing_pose_users(tmp_path):
    binary = tmp_path / "deploy"
    env = {"SONIC_DEPLOY_BIN": str(binary)}
    binary.write_bytes(b"POSE_UPPER_OWNERSHIP_V2")
    require_pose_ownership(env)
    with pytest.raises(RuntimeError, match="sonic_hand_ownership.patch"):
        require_pose_ownership(env, hand_handoff=True)
    binary.write_bytes(b"POSE_UPPER_OWNERSHIP_V2 PLANNER_HAND_OWNERSHIP_V1")
    require_pose_ownership(env, hand_handoff=True)
