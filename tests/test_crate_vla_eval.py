"""The crate closed-loop evaluator: action decode, failure strings, driver arguments."""

import json

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
