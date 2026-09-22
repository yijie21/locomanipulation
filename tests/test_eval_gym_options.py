"""The evaluation's gym options are the ones it declares, not the gym's defaults.

Until 2026-09-22 `eval_closed_loop` assembled an argv list and handed it to `grasp_gym.main`,
which parsed it straight back -- for a process that was never started, because the gym runs in the
evaluator's own (docs/handoff/flat-functional-scripts-20260922.md, decision 11). The argv is gone;
the option set is now built directly and resolved through the same merge the gym's own CLI uses.

That removes the stage->subprocess seam, but it leaves one way to go wrong: dropping a key from
the option set. Nothing would fail -- the gym would quietly use its own default, and a 100-round
evaluation would run at `spawn_lateral` 0.35 instead of the 0.20 the collection ran. These tests
pin the values that differ from the gym's defaults, which is exactly the set that can go silent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from locomani.pipeline.eval_closed_loop import gym_options
from locomani.pipeline.eval_config import EvalConfig
from locomani.tasks.bottle.cli_args import GymOptions

MANIFEST = Path("/tmp/closed_loop.manifest.jsonl")


@pytest.fixture
def options():
    cfg = EvalConfig()
    return cfg, gym_options(cfg, MANIFEST, rounds=cfg.rounds, seed=cfg.seed)


def test_the_spawn_distribution_is_the_collection_distribution(options):
    """These four differ from the gym's own defaults; a dropped key reverts each silently."""
    _, resolved = options
    defaults = GymOptions()
    assert resolved.spawn_lateral == 0.20 != defaults.spawn_lateral
    assert resolved.spawn_yaw_err == 20.0 != defaults.spawn_yaw_err
    assert resolved.edge_jitter == 0.20 != defaults.edge_jitter
    assert resolved.face_object is True and defaults.face_object is False


def test_the_eval_config_reaches_the_gym(options):
    cfg, resolved = options
    assert resolved.rounds == cfg.rounds
    assert resolved.seed == cfg.seed
    assert resolved.targets == cfg.target
    assert resolved.stand_retries == cfg.stand_retries
    assert tuple(resolved.spawn_edge) == tuple(float(v) for v in cfg.spawn_edge)
    assert resolved.manifest == str(MANIFEST)


def test_an_evaluation_writes_no_dataset_and_never_retries_past_a_failure(options):
    """Both were absent from the old argv on purpose; absence has to stay deliberate."""
    _, resolved = options
    assert resolved.record is False
    assert resolved.until_ok is None


def test_every_declared_option_is_marked_as_this_evaluations_own(options):
    """`sources` is how a run answers where a value came from; these must not read as defaults."""
    _, resolved = options
    for name in ("rounds", "seed", "targets", "spawn_edge", "spawn_lateral", "spawn_yaw_err", "edge_jitter"):
        assert resolved.sources[name] == "cli", name


def test_a_resumed_run_offsets_the_seed_and_shortens_the_run():
    """The top-up must not redraw the rounds already measured."""
    cfg = EvalConfig()
    resumed = gym_options(cfg, MANIFEST, rounds=cfg.rounds - 17, seed=cfg.seed + 17)
    assert resumed.seed == cfg.seed + 17
    assert resumed.rounds == cfg.rounds - 17


def test_a_fixed_case_bank_is_delivered_and_keeps_success_stopping_off():
    cfg = EvalConfig()
    resolved = gym_options(cfg, MANIFEST, rounds=cfg.rounds, seed=cfg.seed, cases="/tmp/cases.json")
    assert resolved.cases == "/tmp/cases.json"
    assert resolved.until_ok is None  # resolve_options refuses the combination outright


def test_vla_keeps_its_own_reach_calibration_unless_one_is_named(options):
    """The gym would otherwise inherit the expert's calibration and measure the wrong thing."""
    _, resolved = options
    assert resolved.sources["reach"].startswith("default")
