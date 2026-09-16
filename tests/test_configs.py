"""Canonical config files reproduce the pre-refactor argv (decision 8).

outputs/experiments/ready_visible200_20260909/launch.sh is the last argv-style
invocation of the official ready-visible protocol, before the CLI shrink:

    --out $E --reach-config configs/collection/ready_grasp_visible.json
    --scene ... --successes 200 --holdout 10 --max-batches 40 --seed0 30000
    --steps 3000 --save-steps 3000 --train-gpu 0 --eval-rounds 100 --eval-seed 6000
    --gym-env GYM_SHUFFLE_BAND=0
    --stack-env CUDA_VISIBLE_DEVICES=1 --stack-env MUJOCO_EGL_DEVICE_ID=1

The new equivalent is:

    python src/locomani/pipeline/run_learn_pilot.py \
        --config configs/pilot/learn_200.json \
        --task configs/tasks/bottle_visible.json \
        --out $E --scene ... \
        --set train_gpu=0 \
        --set gym_env=GYM_SHUFFLE_BAND=0 \
        --set stack_env=CUDA_VISIBLE_DEVICES=1,MUJOCO_EGL_DEVICE_ID=1

These tests assert the canonical files plus those --set overrides produce the
same effective values the old argv did, without actually running the pipeline.
"""

import unittest
import warnings
from dataclasses import dataclass
from pathlib import Path

from locomani.backends.sonic.config import SonicBackendConfig
from locomani.config_base import apply_overrides
from locomani.pipeline.eval_config import EvalConfig
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle.reach_config import ReachConfig

REPO = Path(__file__).resolve().parents[1]


class OptionalOverrideTests(unittest.TestCase):
    def test_optional_scalars_parse_from_the_declared_type_and_can_be_cleared(self):
        @dataclass
        class Config:
            height: float | None = None
            count: int | None = None
            enabled: bool | None = None
            label: str | None = None

        cfg = apply_overrides(Config(), ["height=0.53", "count=2", "enabled=false", "label=fixture"])
        self.assertEqual(cfg, Config(0.53, 2, False, "fixture"))
        self.assertEqual(apply_overrides(cfg, ["height=null", "count=null", "enabled=null", "label=null"]), Config())

    def test_null_does_not_clear_a_required_number(self):
        @dataclass
        class Config:
            height: float = 0.53

        with self.assertRaises(ValueError):
            apply_overrides(Config(), ["height=null"])


class BottleVisibleTaskTests(unittest.TestCase):
    def test_reach_matches_ready_grasp_visible_json(self):
        task = BottleTaskConfig.load(REPO / "configs/tasks/bottle_visible.json")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # brake_preplan, retired (step 2)
            want = ReachConfig.load(REPO / "configs/collection/ready_grasp_visible.json")
        self.assertEqual(task.reach, want)
        self.assertEqual(task.reach.visible_brake_depth, 0.31)  # brake depth

    def test_spawn_matches_the_official_ready_visible_distribution(self):
        task = BottleTaskConfig.load(REPO / "configs/tasks/bottle_visible.json")
        self.assertEqual(task.spawn.edge_range, (1.1, 1.3))
        self.assertTrue(task.spawn.face_object)
        self.assertEqual(task.spawn.yaw_err_deg, 20.0)
        self.assertEqual(task.spawn.lateral, 0.20)
        self.assertEqual(task.spawn.edge_jitter, 0.20)


class CanonicalConfigsRoundTripTests(unittest.TestCase):
    """Every canonical file loads as the dataclass it names and round-trips."""

    def test_bottle_visible(self):
        task = BottleTaskConfig.load(REPO / "configs/tasks/bottle_visible.json")
        self.assertEqual(BottleTaskConfig.from_dict(task.to_dict()), task)

    def test_sonic_backend(self):
        backend = SonicBackendConfig.load(REPO / "configs/backends/sonic.json")
        self.assertEqual(backend, SonicBackendConfig())

    def test_closed_loop_100(self):
        cfg = EvalConfig.load(REPO / "configs/eval/closed_loop_100.json")
        self.assertEqual(
            (cfg.seed, cfg.client, cfg.rounds, cfg.budget_s, cfg.replan_interval, cfg.ensemble_n),
            (
                6000,
                "async",
                100,
                90.0,
                0.4,
                1,
            ),
        )
        self.assertEqual(cfg.spawn_edge, (1.1, 1.3))


if __name__ == "__main__":
    unittest.main()
