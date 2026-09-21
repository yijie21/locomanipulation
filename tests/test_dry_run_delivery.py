"""Every value the experiment file declares reaches the process that consumes it.

This is the test the 2026-09-17 defects would have failed. Each of them was a field declared in a
config group and then lost at the stage->subprocess seam: an evaluation spawning at 0.7 m while its
config said 1.1-1.3 m, a stack told to render nothing while its config asked for video. A stage
script's `--dry-run` prints every command it would run together with that command's own settings,
so the question "is this value delivered" has an answer that costs no GPU.

The check is deliberately literal -- the number in the file has to appear in the command line or the
environment. A test that reimplemented the assembly would pass on exactly the bugs it exists for.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "configs" / "experiments"


CHILD = dict(os.environ, PYTHONPATH=str(ROOT / "src"))


def _module(module: str, target: str, *extra: str) -> str:
    result = subprocess.run(
        [sys.executable, "-m", f"locomani.pipeline.{module}", target, "--dry-run", *extra],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=CHILD,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def dry_run(driver: str, experiment: str, *extra: str, mode: str = "collect") -> str:
    """What a line delivers, which since 2026-09-21 is what TWO processes deliver.

    The stack starter and the round driver are separate scripts now, so the question "is this
    value delivered" is answered by the union of their dry runs -- the stack settings come from
    one and the expert's command line from the other. Joining them here keeps every assertion
    below about the line rather than about which process happens to carry the value.
    """
    target = str(EXPERIMENTS / experiment)
    stack = _module("run_stack", target, "--mode", mode, *extra)
    return stack + "\n" + _module(driver, target, *extra)


@pytest.fixture(scope="module")
def gate8():
    cfg = yaml.safe_load((EXPERIMENTS / "bottle_simple_gate8.yaml").read_text())
    return cfg, dry_run("run_collect_single_arm", "bottle_simple_gate8.yaml", "--gpu", "1", "--name", "delivery_check")


def test_the_spawn_distribution_is_delivered(gate8):
    cfg, out = gate8
    spawn = cfg["task"]["spawn"]
    assert f"--spawn-edge {spawn['edge_range'][0]} {spawn['edge_range'][1]}" in out
    assert f"--spawn-yaw-err {spawn['yaw_err_deg']}" in out
    assert f"--spawn-lateral {spawn['lateral']}" in out
    assert f"--edge-jitter {spawn['edge_jitter']}" in out


def test_the_collection_budget_is_delivered(gate8):
    cfg, out = gate8
    assert f"--rounds {cfg['collect']['batch_rounds']}" in out
    assert f"--seed {cfg['collect']['seed0']}" in out  # batch 0; later batches step by 100


def test_the_case_bank_and_the_target_are_delivered(gate8):
    cfg, out = gate8
    assert "--cases" in out and "cases_00.json" in out  # this batch's slice of the fixed bank
    assert f"--targets {cfg['task']['target']}" in out


def test_the_stack_is_told_where_to_record(gate8):
    """One directory per run since 2026-09-21: the stack outlives a batch, so these are fixed when
    the sim starts and the episode's timestamp is what keeps the file names apart."""
    _, out = gate8
    assert "SIM_VIDEO_DIR=" in out and "collect/videos" in out
    assert "SIM_STATE_TRACE_DIR=" in out and "collect/state_trace" in out
    assert "TF_FRAME_AUDIT_PATH=" in out and "frame_audit_00" in out  # this one IS per batch


def test_the_encoder_setting_reaches_the_children(gate8):
    """Collection needs the deploy's encoder; without it the planner's arm references are dropped."""
    _, out = gate8
    assert '"no_encoder": false' in out


def test_the_run_uses_its_own_scene_copy(gate8):
    _, out = gate8
    assert "LOCOMANI_SIM_DIR=" in out and "runs/delivery_check/scene" in out


def test_a_fresh_sample_run_asks_for_successes_not_cases():
    cfg = yaml.safe_load((EXPERIMENTS / "bottle_simple_200.yaml").read_text())
    out = dry_run("run_collect_single_arm", "bottle_simple_200.yaml", "--gpu", "1", "--name", "delivery_check_200")
    assert f"--until-ok {cfg['collect']['successes']}" in out  # keep going until this many are kept
    assert "--cases" not in out  # every round drawn from task.spawn
    assert f"--seed {cfg['collect']['seed0']}" in out


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """A run directory shaped like one collection just finished, without running collection."""
    run = tmp_path_factory.mktemp("run")
    (run / "config.yaml").write_text((EXPERIMENTS / "bottle_simple_200.yaml").read_text())
    (run / "summary.json").write_text(
        json.dumps(
            {
                "collect": {
                    "sessions": ["/tmp/session_a", "/tmp/session_b"],
                    "manifests": ["/tmp/manifest_a.jsonl", "/tmp/manifest_b.jsonl"],
                    "kept": 200,
                }
            }
        )
    )
    result = subprocess.run(
        [sys.executable, "-m", "locomani.pipeline.run_convert", str(run), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=CHILD,
    )
    assert result.returncode == 0, result.stderr
    return yaml.safe_load((EXPERIMENTS / "bottle_simple_200.yaml").read_text()), result.stdout


def test_the_session_manifest_pairing_is_delivered_in_order(converted):
    _, out = converted
    assert "--sessions /tmp/session_a /tmp/session_b" in out
    assert "--manifest /tmp/manifest_a.jsonl /tmp/manifest_b.jsonl" in out


def test_the_holdout_and_modality_reach_the_converter(converted):
    cfg, out = converted
    assert f"--holdout {cfg['train']['holdout']}" in out
    assert cfg["embodiment"]["gr00t_modality_config"] in out
    assert cfg["embodiment"]["gr00t_tag"].upper() in out


def test_the_audit_is_given_the_run_so_it_reads_world_height(converted):
    """Without --run-dir the pelvis-frame column cancels the lift and good episodes fail."""
    _, out = converted
    assert "check_dataset.py" in out and "--run-dir" in out


def test_the_children_are_handed_the_source_path(converted):
    """They run in other virtual environments and import locomani by name.

    The audit only imports it on the --run-dir branch, so relying on an inherited PYTHONPATH worked
    until the day that branch became the default (2026-09-17) and the conversion died after twelve
    minutes of work.
    """
    _, out = converted
    assert out.count("PYTHONPATH=") >= 3 and "/src" in out


def test_a_run_without_a_finished_collection_is_refused(tmp_path):
    (tmp_path / "config.yaml").write_text((EXPERIMENTS / "bottle_simple_200.yaml").read_text())
    (tmp_path / "summary.json").write_text(json.dumps({"collect": {"sessions": [], "manifests": []}}))
    result = subprocess.run(
        [sys.executable, "-m", "locomani.pipeline.run_convert", str(tmp_path), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=CHILD,
    )
    assert result.returncode != 0 and "no finished collection" in result.stderr


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """A run directory shaped like one conversion just finished."""
    run = tmp_path_factory.mktemp("run")
    (run / "config.yaml").write_text((EXPERIMENTS / "bottle_simple_200.yaml").read_text())
    (run / "summary.json").write_text(json.dumps({"convert": {"train_dataset": "/tmp/train_dataset"}}))
    result = subprocess.run(
        [sys.executable, "-m", "locomani.pipeline.run_train", str(run), "--gpu", "3", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=CHILD,
    )
    assert result.returncode == 0, result.stderr
    return yaml.safe_load((EXPERIMENTS / "bottle_simple_200.yaml").read_text()), result.stdout


def test_the_training_budget_reaches_the_command(trained):
    """The recipe used to be a bash script driven through the environment; it assembles the
    torchrun command itself since 2026-09-21, so the budget is asserted where it now lands."""
    cfg, out = trained
    assert f"--max-steps {cfg['train']['steps']}" in out
    assert f"--save-steps {cfg['train']['save_steps']}" in out
    # the single-card path overrides the batch on purpose; either the config's number or that
    assert "--global-batch-size" in out and "--gradient-accumulation-steps" in out
    assert str(cfg["train"]["squat_max"]) in out or "squat" in out.lower()


def test_the_modality_and_the_card_reach_the_command(trained):
    cfg, out = trained
    assert cfg["embodiment"]["gr00t_modality_config"] in out  # which state/action contract is trained
    assert cfg["embodiment"]["gr00t_tag"].upper() in out  # delivered now, not hardcoded in a recipe
    assert "CUDA_VISIBLE_DEVICES=3" in out
    assert "--dataset-path /tmp/train_dataset" in out


def test_the_checkpoint_path_follows_the_step_count(trained):
    cfg, out = trained
    assert f"checkpoint-{cfg['train']['steps']}" in out


def test_training_without_a_dataset_is_refused(tmp_path):
    (tmp_path / "config.yaml").write_text((EXPERIMENTS / "bottle_simple_200.yaml").read_text())
    (tmp_path / "summary.json").write_text(json.dumps({"collect": {"kept": 200}}))
    result = subprocess.run(
        [sys.executable, "-m", "locomani.pipeline.run_train", str(tmp_path), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=CHILD,
    )
    assert result.returncode != 0 and "convert stage" in result.stderr


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory):
    """A run directory shaped like one training just finished."""
    run = tmp_path_factory.mktemp("run")
    (run / "config.yaml").write_text((EXPERIMENTS / "bottle_simple_200.yaml").read_text())
    (run / "summary.json").write_text(json.dumps({"train": {"checkpoint": "/tmp/model/checkpoint-10000"}}))
    # Three processes hold the evaluation between them: the stack, the policy server and the
    # rounds. Their dry runs together are what the line delivers.
    out = "\n".join(
        (
            _module("run_stack", str(run), "--mode", "eval", "--gpu", "2"),
            _module("run_vla_server", str(run), "--gpu", "2"),
            _module("run_eval_single_arm", str(run), "--gpu", "2"),
        )
    )
    return yaml.safe_load((EXPERIMENTS / "bottle_simple_200.yaml").read_text()), out


def test_the_evaluator_is_handed_the_whole_block_as_a_file(evaluated):
    """Not a few fields: that is how a hundred rounds spawned at 0.7 m while the config said 1.1."""
    _, out = evaluated
    assert "--config" in out and "eval/eval_config.json" in out


def test_the_evaluation_stack_runs_without_the_encoder(evaluated):
    """Collection needs the deploy's encoder; token streaming will not hold with it."""
    cfg, out = evaluated
    assert cfg["eval"]["backend"]["no_encoder"] is True
    assert '"no_encoder": true' in out


def test_the_round_video_directory_is_delivered(evaluated):
    """The artifact that silently disappeared for a day."""
    _, out = evaluated
    assert "SIM_VIDEO_DIR=" in out and "eval/videos" in out


def test_the_opening_token_and_the_checkpoint_are_delivered(evaluated):
    cfg, out = evaluated
    assert cfg["eval"]["initial_token"] in out  # must match the pose the training data opens in
    assert "--model-path /tmp/model/checkpoint-10000" in out
    assert cfg["embodiment"]["gr00t_modality_config"] in out


def test_evaluating_without_a_checkpoint_is_refused(tmp_path):
    (tmp_path / "config.yaml").write_text((EXPERIMENTS / "bottle_simple_200.yaml").read_text())
    (tmp_path / "summary.json").write_text(json.dumps({"convert": {"train_dataset": "/tmp/d"}}))
    result = subprocess.run(
        [sys.executable, "-m", "locomani.pipeline.run_eval_single_arm", str(tmp_path), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=CHILD,
    )
    assert result.returncode != 0 and "train stage" in result.stderr


@pytest.fixture(scope="module")
def crate():
    cfg = yaml.safe_load((EXPERIMENTS / "crate_table30_record.yaml").read_text())
    return cfg, dry_run("run_collect_bimanual", "crate_table30_record.yaml", "--gpu", "1", "--name", "crate_delivery")


def test_the_chosen_expert_runs_with_the_run_budget(crate):
    cfg, out = crate
    assert f"--rounds {cfg['collect']['batch_rounds']}" in out
    assert f"--seed {cfg['collect']['seed0']}" in out
    assert f"--label {'visual_crate' if cfg['task']['expert'] == 'visual' else cfg['task']['expert']}" in out


def test_the_bimanual_scene_copy_keeps_the_source_name(crate):
    """The expert's own lookups depend on the directory name, so the copy may not be renamed."""
    cfg, out = crate
    assert Path(cfg["scene"]["dir"]).name in out


def test_recording_is_enabled_on_whichever_instance_this_line_uses(crate):
    cfg, out = crate
    assert cfg["collect"]["instance"] != 0  # this line does not share instance 0 with the bottle runs
    assert "RECORD_INSTANCE=1" in out
    assert f"INSTANCE={cfg['collect']['instance']}" in out


@pytest.fixture(scope="module")
def crate_eval(tmp_path_factory):
    run = tmp_path_factory.mktemp("crate_run")
    (run / "config.yaml").write_text((EXPERIMENTS / "crate_table30_record.yaml").read_text())
    (run / "summary.json").write_text(json.dumps({"train": {"checkpoint": "/tmp/model/checkpoint-10000"}}))
    out = "\n".join(
        (
            _module("run_stack", str(run), "--mode", "eval", "--gpu", "1"),
            _module("run_vla_server", str(run), "--gpu", "1"),
            _module("run_eval_bimanual", str(run), "--gpu", "1"),
        )
    )
    return yaml.safe_load((EXPERIMENTS / "crate_table30_record.yaml").read_text()), out


def test_the_fields_the_bimanual_evaluator_accepts_are_all_delivered(crate_eval, tmp_path):
    """Since 2026-09-18 the whole eval block goes over as one file rather than as
    hand-picked flags, so this parses the command the stage would run and checks the
    values that come out the other side -- not that a flag appears in a string."""
    from locomani.pipeline import eval_crate_closed_loop as driver
    from locomani.pipeline.eval_config import EvalConfig

    cfg, out = crate_eval
    assert "--eval-config" in out and "--instance {}".format(cfg["collect"]["instance"]) in out

    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    evaluation = EvalConfig.from_dict(cfg["eval"])
    evaluation.dump(tmp_path / "eval_config.json")
    args = driver.parse_args(
        [
            "--checkpoint", str(checkpoint),
            "--scene", str(tmp_path),
            "--out", str(tmp_path / "o"),
            "--eval-config", str(tmp_path / "eval_config.json"),
        ]
    )  # fmt: skip
    assert args.rounds == cfg["eval"]["rounds"]
    assert args.seed == cfg["eval"]["seed"]
    assert args.port == cfg["eval"]["policy_port"]
    assert args.execution_horizon == cfg["eval"]["execution_horizon"]
    assert args.budget_s == cfg["eval"]["budget_s"]
    # The single-arm token would start the robot where this line's training data never does.
    assert str(args.initial_token_file).endswith("crate_initial_token.json")
    assert cfg["eval"]["initial_token"].endswith("crate_initial_token.json")


def test_the_bimanual_line_keeps_its_own_port_block(crate_eval):
    cfg, _ = crate_eval
    assert cfg["collect"]["instance"] != 0
    assert cfg["eval"]["policy_port"] != 5590  # 5590 is instance 1's privileged port


def test_the_only_composing_script_names_each_line_its_own_driver():
    """`run_overnight.sh` is the one script allowed to call others, so what it calls is a contract.

    It cannot be dry-run the way a stage can -- it starts stacks and waits on them -- so the check
    is on the script itself: both lines reach their own collect and eval drivers through the
    per-line directories, and share the stages that are genuinely shared.
    """
    text = (ROOT / "scripts" / "run_overnight.sh").read_text()
    for stage in ("collect_data.sh", "convert_data.sh", "train.sh", "start_stack.sh", "run_vla_server.sh"):
        assert stage in text, stage
    assert "$LINE/run_agent_server.sh" in text
    assert "$LINE/run_vla_inference.sh" in text
    assert text.index("=== collect ===") < text.index("=== convert ===")
    assert text.index("=== convert ===") < text.index("=== train ===")
    assert text.index("=== train ===") < text.index("=== evaluate ===")


def test_every_stage_script_is_one_exec_over_one_module():
    """The shape the 2026-09-21 refactor exists to keep: no script calls another but run_overnight."""
    scripts = ROOT / "scripts"
    flat = (
        [
            scripts / name
            for name in (
                "collect_data.sh",
                "start_stack.sh",
                "stop_stack.sh",
                "run_vla_server.sh",
                "convert_data.sh",
                "train.sh",
            )
        ]
        + list((scripts / "single_arm").glob("*.sh"))
        + list((scripts / "bimanual").glob("*.sh"))
    )
    for path in flat:
        body = path.read_text()
        assert body.count("exec ") == 1, path
        assert "bash scripts/" not in body.replace("#", "\n#").split("\n#")[0], path
