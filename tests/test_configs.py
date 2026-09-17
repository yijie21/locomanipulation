"""Canonical config files reproduce the pre-refactor argv (decision 8).

outputs/experiments/ready_visible200_20260909/launch.sh is the last argv-style
invocation of the official ready-visible protocol, before the CLI shrink:

    --out $E --reach-config configs/collection/ready_grasp_visible.json
    --scene ... --successes 200 --holdout 10 --max-batches 40 --seed0 30000
    --steps 3000 --save-steps 3000 --train-gpu 0 --eval-rounds 100 --eval-seed 6000
    --gym-env GYM_SHUFFLE_BAND=0
    --stack-env CUDA_VISIBLE_DEVICES=1 --stack-env MUJOCO_EGL_DEVICE_ID=1

The new equivalent is:

    python -m locomani.cli collect \
        task=bottle_visible name=<run_name> gpu=0 \
        collect.gym_env=GYM_SHUFFLE_BAND=0 \
        collect.stack_env=CUDA_VISIBLE_DEVICES=1,MUJOCO_EGL_DEVICE_ID=1

These tests assert the canonical `configs/task|backend|eval` YAML groups
(the JSON twins under `configs/tasks|backends|eval` are retired -- see
`locomani.conf.load_group`) plus those overrides produce the same effective
values the old argv did, without actually running the pipeline.
"""

import unittest
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

from locomani import profiles
from locomani.backends.scalebfm.config import ScaleBfmBackendConfig
from locomani.backends.sonic.config import SonicBackendConfig
from locomani.config_base import apply_overrides
from locomani.pipeline.eval_config import EvalConfig
from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle.reach_config import ReachConfig
from locomani.tasks.bottle_visual.config import BottleVisualTaskConfig
from locomani.tasks.crate.config import CrateTaskConfig

REPO = Path(__file__).resolve().parents[1]

# Every group name migrated off the retired JSON twins, with the dataclass it composes to.
TASK_GROUPS = [
    ("bottle_simple", BottleTaskConfig),
    ("bottle_visible", BottleTaskConfig),
    ("bottle_visible_teleop", BottleTaskConfig),
    ("bottle_visual", BottleVisualTaskConfig),
    ("bottle_whole_body", BottleTaskConfig),
    ("crate_low_reach", CrateTaskConfig),
    ("crate_table30", CrateTaskConfig),
    ("crate_visual_low_reach", CrateTaskConfig),
    ("crate_wall", CrateTaskConfig),
    ("pick_place", BottleTaskConfig),
]
BACKEND_GROUPS = [("sonic", SonicBackendConfig), ("scalebfm", ScaleBfmBackendConfig)]
EVAL_GROUPS = [("closed_loop_100", EvalConfig)]


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
        task = profiles.load("task", "bottle_visible")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # brake_preplan, retired (step 2)
            want = ReachConfig.load(REPO / "configs/collection/ready_grasp_visible.json")
        self.assertEqual(task.reach, want)
        self.assertEqual(task.reach.visible_brake_depth, 0.31)  # brake depth

    def test_spawn_matches_the_official_ready_visible_distribution(self):
        task = profiles.load("task", "bottle_visible")
        # load_group composes through OmegaConf, which materializes a tuple-typed field as a
        # list; tuple() normalizes before comparing against the dataclass's own tuple default.
        self.assertEqual(tuple(task.spawn.edge_range), (1.1, 1.3))
        self.assertTrue(task.spawn.face_object)
        self.assertEqual(task.spawn.yaw_err_deg, 20.0)
        self.assertEqual(task.spawn.lateral, 0.20)
        self.assertEqual(task.spawn.edge_jitter, 0.20)


class CanonicalConfigsRoundTripTests(unittest.TestCase):
    """Every canonical YAML group loads as the dataclass it names and round-trips."""

    def test_every_group_round_trips(self):
        groups = (
            [("task", name, cls) for name, cls in TASK_GROUPS]
            + [("backend", name, cls) for name, cls in BACKEND_GROUPS]
            + [("eval", name, cls) for name, cls in EVAL_GROUPS]
        )
        for group, name, cls in groups:
            with self.subTest(group=group, name=name):
                cfg = profiles.load(group, name)
                self.assertIsInstance(cfg, cls)
                # Compare at the to_dict() level, not object equality: OmegaConf (load_group's
                # composition) materializes a tuple-typed field as a list, while from_dict
                # restores the dataclass's own tuple type -- to_dict() normalizes both to lists.
                self.assertEqual(cls.from_dict(cfg.to_dict()).to_dict(), cfg.to_dict())

    def test_sonic_backend(self):
        backend = profiles.load("backend", "sonic")
        # collect runs the deploy WITH its encoder (memory: no-encoder-breaks-planner-arms);
        # the retired JSON twin carried the eval-era no_encoder=True default instead.
        self.assertEqual(backend, replace(SonicBackendConfig(), no_encoder=False))

    def test_closed_loop_100(self):
        cfg = profiles.load("eval", "closed_loop_100")
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
        self.assertEqual(tuple(cfg.spawn_edge), (1.1, 1.3))


if __name__ == "__main__":
    unittest.main()
