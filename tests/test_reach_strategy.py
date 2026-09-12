"""Versioned selection keeps legacy control predicates and rejects ambiguous input."""

import dataclasses
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from locomani.tasks.bottle.config import BottleTaskConfig
from locomani.tasks.bottle.policy import make_grasp_policy
from locomani.tasks.bottle.reach_config import LEGACY_STRATEGY_FIELDS, ReachConfig

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CASES = [
    ("legacy", {}),
    ("ready", {"ready_grasp_baseline": True}),
    ("recovery", {"ready_grasp_baseline": True, "ready_grasp_recovery": True}),
    ("coupled", {"ready_grasp_baseline": True, "ready_grasp_recovery": True, "coupled_squat": True}),
    ("teleop", {"ready_grasp_baseline": True, "ready_grasp_recovery": True, "coupled_squat": True, "teleop_arm": True}),
    (
        "image_servo",
        {
            "ready_grasp_baseline": True,
            "ready_grasp_recovery": True,
            "coupled_squat": True,
            "image_servo": True,
            "visible_expert": True,
            "timely_capture": True,
        },
    ),
]


@pytest.mark.parametrize("strategy,flags", LEGACY_CASES)
def test_v1_migration_preserves_effective_control_predicates(strategy, flags):
    cfg = ReachConfig.from_dict({"schema_version": 1, "grip_below_center": 0.02, **flags})
    assert cfg.strategy == strategy
    assert cfg.grip_below_center == 0.02
    for name in LEGACY_STRATEGY_FIELDS:
        assert getattr(cfg, name) == flags.get(name, False)
    assert ReachConfig.from_dict(cfg.to_dict()) == cfg
    assert cfg.to_dict()["schema_version"] == 2
    assert not LEGACY_STRATEGY_FIELDS.intersection(cfg.to_dict())


def test_teleop_is_one_choice_without_independent_scaffolding_flags():
    cfg = ReachConfig(strategy="teleop")
    assert cfg.teleop_arm and cfg.coupled_squat and cfg.ready_grasp_recovery
    assert not LEGACY_STRATEGY_FIELDS.intersection(f.name for f in dataclasses.fields(cfg))
    assert dataclasses.replace(cfg, strategy="coupled").teleop_arm is False


@pytest.mark.parametrize(
    "data",
    [
        {"schema_version": True},
        {"schema_version": 3},
        {"schema_version": 1, "teleop_arm": True},
        {"schema_version": 1, "ready_grasp_baseline": "false"},
        {"schema_version": 1, "strategy": "teleop"},
        {"schema_version": 2, "strategy": "teleop", "teleop_arm": False},
        {"schema_version": 2, "strategy": "unknown"},
        {"schema_version": 2, "strategy": []},
        {"schema_version": 2, "strategy": "teleop", "transport_guard": False},
        {"schema_version": 2, "strategy": "image_servo"},
        {"schema_version": 2, "brake_preplan": False},
    ],
)
def test_ambiguous_or_invalid_formats_fail_before_execution(data):
    with pytest.raises(ValueError):
        ReachConfig.from_dict(data)


def test_current_task_overrides_change_the_selected_strategy():
    task = BottleTaskConfig.load(ROOT / "configs/tasks/bottle_visible.json")
    changed = task.with_overrides(["reach.strategy=teleop"])
    assert changed.reach.teleop_arm and not task.reach.teleop_arm
    with pytest.raises(ValueError):
        task.with_overrides(["reach.teleop_arm=true"])


def test_all_collection_profiles_and_legacy_fixture_keep_their_declared_values():
    paths = list((ROOT / "configs/collection").glob("*.json"))
    paths.append(ROOT / "tests/fixtures/reach_config_ready07d_200.json")
    for path in paths:
        old = json.loads(path.read_text())
        cfg = ReachConfig.load(path)
        for name, value in old.items():
            if name == "schema_version" or name == "brake_preplan":
                continue
            assert getattr(cfg, name) == value, (path, name)


def test_factory_constructs_the_real_controller_with_the_given_dependencies():
    from locomani.tasks.bottle.coupled_grasp_policy import CoupledGraspPolicy
    from locomani.tasks.bottle.image_servo_policy import ImageServoPolicy
    from locomani.tasks.bottle.ready_grasp_policy import ReadyGraspPolicy
    from locomani.tasks.bottle.recoverable_grasp_policy import RecoverableGraspPolicy
    from locomani.tasks.bottle.teleop_grasp_policy import TeleopGraspPolicy

    controllers = dict(
        ready=ReadyGraspPolicy,
        recovery=RecoverableGraspPolicy,
        coupled=CoupledGraspPolicy,
        image_servo=ImageServoPolicy,
        teleop=TeleopGraspPolicy,
    )
    robot, task, ramp, record = object(), object(), object(), object()
    assert make_grasp_policy(robot, task, ReachConfig(), ramp, record) is None
    for strategy, controller in controllers.items():
        cfg = ReachConfig(
            strategy=strategy, **({"visible_expert": True, "timely_capture": True} if strategy == "image_servo" else {})
        )
        with patch.object(controller, "__init__", return_value=None) as init:
            result = make_grasp_policy(robot, task, cfg, ramp, record)
        assert type(result) is controller
        init.assert_called_once_with(robot, task, cfg, ramp, record)
