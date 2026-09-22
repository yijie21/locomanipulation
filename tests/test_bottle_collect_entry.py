"""The in-process collection entry configures the gym for the run it was given.

`locomani.paths` resolves `LOCOMANI_SIM_DIR` when it is IMPORTED, and it is imported by the first
`locomani.*` import in the process. That was always correct for a subprocess the stack launched
with the run's environment already in it, and it is wrong for a process that attaches to a stack
and only then learns which run it is serving: the gym read the scene the interpreter started with,
not the run's own copy.

That defect reached a GPU run on 2026-09-22 and cost a stack cycle. It surfaced as a case-bank
fingerprint mismatch, which is luck -- the fixed case bank carries a scene fingerprint. A run
without one would have collected against the wrong scene and said nothing. So the delivery is
pinned here, where it costs no GPU (docs/handoff/flat-functional-scripts-20260922.md).
"""

from __future__ import annotations

import yaml

from locomani.pipeline import run_context
from locomani.tasks.bottle import collect
from locomani.tasks.bottle.config import BottleTaskConfig

EXPERIMENT = run_context.ROOT / "configs/experiments/bottle_simple_200.yaml"


def options_for(tmp_path, batch=0, kept=0):
    cfg = yaml.safe_load(EXPERIMENT.read_text())
    run_dir = tmp_path / "run"
    (run_dir / "collect").mkdir(parents=True)
    profile = BottleTaskConfig.from_dict(cfg["task"])
    resolved = collect.gym_options(
        cfg,
        run_dir,
        run_dir / "collect",
        batch,
        kept,
        run_dir / "collect" / f"manifest_{batch:02d}.jsonl",
        profile=profile,
        reach=profile.reach,
        write=False,
    )
    return cfg, run_dir, resolved


def test_the_gym_is_told_the_runs_own_scene_not_the_interpreters(tmp_path):
    cfg, run_dir, options = options_for(tmp_path)
    assert options.scene_meta == str(run_dir / "scene" / "scene_meta.json")
    # Not the module-level default, which is whatever the process started with.
    from locomani import paths

    assert options.scene_meta != paths.SCENE_META


def test_the_case_bank_is_fingerprinted_against_that_same_scene(tmp_path):
    """One scene answer. Two would fail exactly where this bug was caught, and only sometimes."""
    cfg, run_dir, options = options_for(tmp_path)
    assert str(collect.prepare_scene_meta(cfg, run_dir)) == options.scene_meta


def test_the_batch_seed_and_budget_advance_with_the_batch(tmp_path):
    cfg, _, first = options_for(tmp_path, batch=0, kept=0)
    seed0, rounds = cfg["collect"]["seed0"], cfg["collect"]["batch_rounds"]
    assert (first.seed, first.rounds) == (seed0, rounds)
    assert first.until_ok == cfg["collect"]["successes"]  # nothing kept yet


def test_a_later_batch_asks_only_for_what_is_still_missing(tmp_path):
    cfg, _, later = options_for(tmp_path, batch=3, kept=17)
    assert later.seed == cfg["collect"]["seed0"] + 300
    assert later.until_ok == cfg["collect"]["successes"] - 17


def test_the_spawn_distribution_comes_from_the_experiment(tmp_path):
    cfg, _, options = options_for(tmp_path)
    spawn = cfg["task"]["spawn"]
    assert tuple(options.spawn_edge) == tuple(float(v) for v in spawn["edge_range"])
    assert options.spawn_yaw_err == float(spawn["yaw_err_deg"])
    assert options.spawn_lateral == float(spawn["lateral"])
    assert options.edge_jitter == float(spawn["edge_jitter"])
    assert options.face_object == bool(spawn["face_object"])


def test_recording_is_delivered_because_the_experiment_asks_for_it(tmp_path):
    cfg, _, options = options_for(tmp_path)
    assert options.record is bool(cfg["collect"]["record"])
