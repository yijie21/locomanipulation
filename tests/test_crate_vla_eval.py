"""The crate closed-loop evaluator: action decode, failure strings, driver arguments."""

import json
from pathlib import Path

import numpy as np
import pytest

from locomani.backends.sonic.sonic_token import GRIPPER_MAX, GRIPPER_MIN, expand_hands_bimanual, hand_from_gripper
from locomani.pipeline import eval_crate_closed_loop as driver
from locomani.tasks.crate.vla_rounds import classify


def test_bimanual_hands_mirror_each_scalar_onto_its_own_jaws():
    left, right = expand_hands_bimanual(-0.05, 0.1)  # out of range on both sides -> clipped
    np.testing.assert_allclose(left[:2], GRIPPER_MIN)
    np.testing.assert_allclose(right[:2], GRIPPER_MAX)
    assert not left[2:].any() and not right[2:].any()
    assert hand_from_gripper(0.0).shape == (7,) and hand_from_gripper(0.0).dtype == np.float32


def test_failure_strings_follow_the_single_arm_line():
    assert classify(0.75, False, 0.0) == "[vla] walk: never closer than 0.75 m"
    assert classify(0.31, False, 0.0) == "[vla] reach: at the table (min 0.31 m) but never commanded a close"
    assert classify(0.31, True, 0.042) == "[vla] close: commanded a grasp but no lift (object moved 4 cm laterally)"
    assert driver.failure_stage(classify(0.31, True, 0.0)) == "close"
    assert driver.failure_stage("[stand] robot did not stabilise after reset") == "pre-vla"


def test_aggregate_counts_successes_stages_and_wilson(tmp_path):
    m = tmp_path / "closed_loop.manifest.jsonl"
    rows = [
        dict(ok=True, why="ok"),
        dict(ok=False, why=classify(0.7, False, 0)),
        dict(ok=False, why=classify(0.3, False, 0)),
    ]
    m.write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = driver.aggregate(m)
    assert (summary["rounds"], summary["successes"]) == (3, 1)
    assert summary["failure_stages"] == {"walk": 1, "reach": 1}
    lo, hi = summary["wilson95"]
    assert 0 <= lo < 1 / 3 < hi <= 1


def test_driver_refuses_instance_0_and_missing_checkpoints(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    base = ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path / "o")]
    args = driver.parse_args(base)
    assert (args.instance, args.port, args.gpu, args.serve_gpu) == (1, 5700, "1", "1")
    with pytest.raises(SystemExit):
        driver.parse_args(base + ["--instance", "0"])
    with pytest.raises(SystemExit):
        driver.parse_args(["--checkpoint", str(tmp_path / "none"), "--scene", str(tmp_path), "--out", str(tmp_path)])


def test_backbone_path_reaches_serve_policy_only_when_given(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    backbone = tmp_path / "Cosmos-Reason2-2B"
    backbone.mkdir()
    (backbone / "config.json").write_text("{}")
    base = ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path / "o")]
    assert "--backbone-path" not in driver.serve_command(driver.parse_args(base))
    cmd = driver.serve_command(driver.parse_args(base + ["--backbone-path", str(backbone)]))
    assert cmd[cmd.index("--backbone-path") + 1] == str(backbone)
    assert cmd[cmd.index("--model-path") + 1] == str(ckpt)
    with pytest.raises(SystemExit):
        driver.parse_args(base + ["--backbone-path", str(tmp_path / "missing")])


def test_serve_env_pins_the_gpu_and_unbuffers_the_listening_line(tmp_path, monkeypatch):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    env = driver.serve_env(
        driver.parse_args(
            ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path), "--serve-gpu", "0"]
        )
    )
    assert (env["CUDA_VISIBLE_DEVICES"], env["PYTHONUNBUFFERED"]) == ("0", "1")


def test_the_batch_warms_up_before_the_first_recorded_round(monkeypatch, tmp_path):
    """No recorded round may be the deploy's first.

    The deploy's first round in a process passes through its reference motion's
    frame-0 crouch and steps out of it at the mode switch: 0.5 rad of knee and
    3.3 cm of pelvis height inside 0.3 s, the lurch at the start of every
    evaluation video. Every later round is transient-free, so the batch spends
    one unrecorded prelude on it (docs/handoff/crate-eval-prelude-twitch-20260917.md).
    """
    from locomani.tasks.crate import vla_rounds

    order, placements = [], []

    def fake_prelude(p, reset, meta, cfg, rng):
        placements.append(float(rng.random()))
        order.append("prelude")
        return {}, None

    monkeypatch.setattr(vla_rounds, "prelude", fake_prelude)

    rng_a, rng_b = np.random.default_rng([6000, 0xB1A5]), np.random.default_rng(6000)
    vla_rounds.warm_up(None, None, None, None, rng_a)
    order.append("round 0")
    fake_prelude(None, None, None, None, rng_b)

    assert order == ["prelude", "round 0", "prelude"]
    # The warm-up draws from its own stream, so round 0 still gets seed 6000's placement.
    assert placements[0] != placements[1]
    assert placements[1] == float(np.random.default_rng(6000).random())


def test_the_whole_eval_block_is_delivered_or_refused_but_never_silently_dropped(tmp_path):
    """2026-09-18: eval_config.json recorded `client: async` while the loop ran
    blocking, because the driver hand-picked thirteen flags and ignored the rest."""
    from locomani.pipeline.eval_config import EvalConfig

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text("{}")
    base = ["--checkpoint", str(ckpt), "--scene", str(tmp_path), "--out", str(tmp_path / "o")]

    honoured = tmp_path / "eval_config.json"
    EvalConfig(rounds=7, seed=41, client="blocking", policy_port=5701, budget_s=42.0).dump(honoured)
    args = driver.parse_args(base + ["--eval-config", str(honoured)])
    assert (args.rounds, args.seed, args.port, args.budget_s) == (7, 41, 5701, 42.0)
    # What it cannot implement is named, not dropped.
    assert "spawn_edge" in args.ignored_eval_fields and "backend" in args.ignored_eval_fields
    assert not set(driver.HONOURED) & set(args.ignored_eval_fields)

    lying = tmp_path / "lying.json"
    EvalConfig(client="async").dump(lying)
    with pytest.raises(SystemExit):
        driver.parse_args(base + ["--eval-config", str(lying)])


def test_the_crate_trace_carries_the_two_posture_channels_it_was_missing():
    """2026-09-18: the crate trace had pelvis_xy/pelvis_yaw/fallen and nothing else,
    so "did it lurch or fall at the start" could not be answered from its artifacts."""
    from locomani.tasks.crate.vla_rounds import posture

    upright = posture({"pelvis": {"pos": [0.0, 0.0, 0.788], "quat": [1.0, 0.0, 0.0, 0.0]}})
    assert upright["pelvis_z"] == pytest.approx(0.788)
    assert upright["torso_pitch"] == pytest.approx(0.0, abs=1e-9)

    # +ve pitch is leaning forward, the same sign convention as the single-arm trace.
    half = np.sin(np.radians(10) / 2), np.cos(np.radians(10) / 2)
    leaning = posture({"pelvis": {"pos": [0.0, 0.0, 0.75], "quat": [half[1], 0.0, half[0], 0.0]}})
    assert np.degrees(leaning["torso_pitch"]) == pytest.approx(10.0, abs=0.01)

    blank = posture({})
    assert blank == {"pelvis_z": None, "torso_pitch": None}


def test_each_batch_numbers_its_traces_globally_so_they_do_not_overwrite():
    """A 100-round evaluation kept 10 traces: the batch loop runs one process per batch,
    each numbering its rounds from zero into one shared `traces/` directory, so batch 1
    overwrote batch 0. Videos were always per-batch directories; traces were not."""
    from locomani import paths

    rounds = (Path(paths.REPO) / "src/locomani/tasks/crate/vla_rounds.py").read_text()
    assert '"--round-offset",' in rounds
    assert 'f"round_{index + args.round_offset:03d}.json"' in rounds

    driver_source = (Path(paths.REPO) / "src/locomani/pipeline/eval_crate_closed_loop.py").read_text()
    assert '"--round-offset",' in driver_source and "b * args.batch_rounds," in driver_source


def test_clean_mode_reaches_both_the_round_loop_and_the_stack():
    """The transient was not one thing: leaving streamed mode at the end of every round
    put the NEXT round's 3.4 s settle in planner mode, where IDLE drifts. Both halves of
    the fix -- the keepalive and SONIC_AUTO_START -- hang off this one flag."""
    from locomani import paths

    rounds = (Path(paths.REPO) / "src/locomani/tasks/crate/vla_rounds.py").read_text()
    assert '"--no-clean",' in rounds and "CleanClock(config=vla_cfg)" in rounds
    assert "clean_commander_class(clock)()" in rounds

    driver_source = (Path(paths.REPO) / "src/locomani/pipeline/eval_crate_closed_loop.py").read_text()
    assert "clean_deploy=args.clean" in driver_source
    assert '*([] if args.clean else ["--no-clean"]),' in driver_source
    assert "ap.set_defaults(clean=True)" in driver_source  # clean is the default, not the option
