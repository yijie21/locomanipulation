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

    from locomani.pipeline import run_crate_pilot
    from locomani.tasks.crate.config import CrateTaskConfig

    cfg = CrateTaskConfig.load("configs/tasks/crate_visual_low_reach.json")
    monkeypatch.setattr(run_crate_pilot, "prepare", lambda args: (tmp_path, cfg, tmp_path / "config", tmp_path))
    monkeypatch.setattr(
        run_crate_pilot, "Stage", Mock(side_effect=AssertionError("resources started before admission"))
    )
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(tmp_path / "unpatched"))
    with pytest.raises(RuntimeError, match="sonic_pose_ownership.patch"):
        run_crate_pilot.main(["--out", str(tmp_path), "--expert", expert])


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


def test_bottle_whole_body_rejects_missing_hand_patch_before_stack(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from locomani.pipeline import run_learn_pilot

    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "scene_meta.json").write_text("{}")
    binary = tmp_path / "deploy"
    binary.write_bytes(b"POSE_UPPER_OWNERSHIP_V2")
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(binary))
    start = Mock(side_effect=AssertionError("stack started before admission"))
    monkeypatch.setattr(run_learn_pilot.stack, "start_stack", start)
    with pytest.raises(RuntimeError, match="sonic_hand_ownership.patch"):
        run_learn_pilot.main(
            [
                "--task",
                "configs/tasks/bottle_whole_body.json",
                "--scene",
                str(scene),
                "--out",
                str(tmp_path / "out"),
                "--set",
                "through=collect",
            ]
        )
    start.assert_not_called()
