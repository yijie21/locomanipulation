from pathlib import Path

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
def test_collection_rejects_an_old_binary_before_starting_resources(tmp_path, monkeypatch, expert):
    """The check runs before the stack, not after: an unpatched deploy must cost nothing to find out."""
    import sys
    from unittest.mock import Mock

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from locomani.pipeline import lines, run_collect_bimanual, run_context

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/experiments/crate_table30_record.yaml").read_text()
    )
    cfg["task"]["expert"] = expert
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Patch where the value is READ, not where it is re-exported: `open_run` resolves RUNS in
    # run_context, so patching a name that merely points at it leaves the real outputs/runs in
    # play -- which is how this test wrote into a live run directory on 2026-09-21.
    monkeypatch.setattr(run_context, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(run_context, "require_headroom", lambda *a, **k: {})
    # Attaching is a READ of stack.json and starts nothing; what must cost nothing to find out is
    # STARTING a stack, and that check lives in run_stack (asserted below). The driver only has to
    # refuse before it runs a round.
    attached = Mock()
    attached.pids = {}
    monkeypatch.setattr(run_collect_bimanual.Stack, "attach", Mock(return_value=attached))
    monkeypatch.setattr(run_context, "group_threads", lambda pids: 0)
    monkeypatch.setattr(
        run_collect_bimanual, "run_batch", Mock(side_effect=AssertionError("a round ran before admission"))
    )
    monkeypatch.setattr(
        run_collect_bimanual,
        "line_of",
        lambda cfg: lines.Line(
            name="bimanual",
            prepare=Mock(side_effect=AssertionError("the scene was copied before admission")),
            collect_root=lambda run_dir: run_dir,
            trace_root=lambda run_dir: run_dir / "trace",
        ),
    )
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(tmp_path / "unpatched"))
    with pytest.raises(RuntimeError, match="sonic_pose_ownership.patch"):
        run_collect_bimanual.main([str(experiment), "--gpu", "0"])


def test_standalone_whole_body_carry_checks_the_same_capability(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from locomani import profiles
    from locomani.tasks.crate import bimanual_carry

    # bimanual_carry's --config only reads JSON; task=crate_low_reach (whole_body reach_control)
    # dumped to a file stands in for the retired configs/tasks/crate_low_reach.json.
    config_path = tmp_path / "crate_low_reach.json"
    profiles.load("task", "crate_low_reach").dump(config_path)
    monkeypatch.setattr("sys.argv", ["carry", "--config", str(config_path)])
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


def test_starting_a_stack_checks_the_binary_before_it_starts_anything(tmp_path, monkeypatch):
    """The expensive half of the old assertion, now in the process that actually starts the stack."""
    import sys
    from unittest.mock import Mock

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from locomani.pipeline import run_context, run_stack

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/experiments/crate_table30_record.yaml").read_text()
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(yaml.safe_dump(cfg, sort_keys=False))

    monkeypatch.setattr(run_context, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(run_context, "require_headroom", lambda *a, **k: {})
    monkeypatch.setattr(run_stack, "Stack", Mock(side_effect=AssertionError("a stack started before admission")))
    monkeypatch.setenv("SONIC_DEPLOY_BIN", str(tmp_path / "unpatched"))
    with pytest.raises(RuntimeError, match="sonic_pose_ownership.patch"):
        run_stack.main([str(experiment), "--mode", "collect", "--gpu", "0"])
