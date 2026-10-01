import unittest

import pytest

from locomani.tasks.crate.config import CrateTaskConfig


class CrateDyRangeTests(unittest.TestCase):
    def test_default_matches_the_historical_hard_coded_range(self):
        self.assertEqual(CrateTaskConfig().crate_dy_range_m, (-0.08, 0.08))

    def test_set_override_parses_a_comma_separated_pair(self):
        cfg = CrateTaskConfig().with_overrides(["crate_dy_range_m=-0.1,0.12"])
        self.assertEqual(cfg.crate_dy_range_m, (-0.1, 0.12))

    def test_round_trips_through_to_dict_and_from_dict(self):
        cfg = CrateTaskConfig().with_overrides(["crate_dy_range_m=-0.1,0.12"])
        self.assertEqual(CrateTaskConfig.from_dict(cfg.to_dict()).crate_dy_range_m, (-0.1, 0.12))

    def test_rejects_a_low_bound_above_the_high_bound(self):
        with self.assertRaises(ValueError):
            CrateTaskConfig().with_overrides(["crate_dy_range_m=0.1,-0.1"])

    def test_rejects_a_range_that_is_not_a_pair(self):
        with self.assertRaises(ValueError):
            CrateTaskConfig().with_overrides(["crate_dy_range_m=-0.1,0.0,0.1"])


if __name__ == "__main__":
    unittest.main()


def test_bounds_reject_nonfinite_or_nonnumeric_values():
    for bounds in [(float("nan"), 0.1), (-float("inf"), float("inf")), ("a", "b"), (False, True), None]:
        with pytest.raises(ValueError, match="crate_dy_range_m"):
            CrateTaskConfig.from_dict({"crate_dy_range_m": bounds})


@pytest.mark.parametrize("config_source", ["cli", "profile"])
def test_configured_range_reaches_the_actual_reset_without_module_mutation(tmp_path, monkeypatch, config_source):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    from locomani.tasks.crate import bimanual_wall as wall

    class ResetObserved(Exception):
        pass

    received = []

    def reset_scene(**kwargs):
        received.append(kwargs["objects"]["blue_crate"]["dxy"][1])
        raise ResetObserved()

    reset = SimpleNamespace(pin_arms=Mock(), set_ghost=Mock(), reset=reset_scene, close=Mock())
    p = SimpleNamespace(cmd=Mock(), log=Mock(), close=Mock())
    meta = {
        "target_object": "blue_crate",
        "surfaces": {"A": {"flat_half_extent": [0.5, 0.5], "xy": [0, 0]}},
        "objects": {"blue_crate": {"init_pos": [0, 0, 0.4], "init_quat": [1, 0, 0, 0]}},
        "robot_start": {"xy": [-1, 0]},
    }
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(meta))
    monkeypatch.setattr(wall.paths, "SCENE_META", str(scene))
    monkeypatch.setattr(wall, "Primitives", lambda **kwargs: p)
    monkeypatch.setattr(wall, "EpisodeControlClient", lambda: reset)
    monkeypatch.setattr(wall.round_runner, "prime", lambda *args: None)

    def rounds(p, reset, meta, **kwargs):
        kwargs["run_round"](p, reset, meta, 0, kwargs["rng"], p.log, "sample")

    monkeypatch.setattr(wall.round_runner, "run_rounds", rounds)
    for value in [0.035, -0.025, 0.035]:
        if config_source == "profile":
            profile = tmp_path / "crate_profile.json"
            profile.write_text(json.dumps({"crate_dy_range_m": [value, value]}))
            argv = ["wall", "--config", str(profile)]
        else:
            argv = ["wall", "--set", f"crate_dy_range_m={value},{value}"]
        monkeypatch.setattr("sys.argv", argv)
        with pytest.raises(ResetObserved):
            wall.main()
    assert received == [0.035, -0.025, 0.035]
    # Three batches in one process: each closes the clients it opened (TODO 1.17).
    assert (reset.close.call_count, p.close.call_count) == (3, 3)
    assert not hasattr(wall, "CRATE_DY_RANGE")


def test_reach_continuity_weight_reaches_the_whole_body_solve(monkeypatch):
    """TODO 2.4.10.20: the knob is the solver's damping prior around the previous solution; 0 keeps the old solve."""
    import numpy as np

    from locomani.tasks.crate.visual_config import CrateVisualConfig

    with pytest.raises(ValueError, match="reach_continuity_weight"):
        CrateVisualConfig(reach_continuity_weight=0.3)
    with pytest.raises(ValueError, match="reach_continuity_weight"):
        CrateVisualConfig(reach_continuity_weight=-0.1)
    assert CrateVisualConfig().reach_continuity_weight == 0.0
    from locomani.tasks.crate import coordinated_reach

    source = coordinated_reach.__loader__.get_source(coordinated_reach.__name__)
    assert "continuity_reference=self.x," in source and "continuity_weight=cfg.reach_continuity_weight," in source
    assert np.isfinite(CrateVisualConfig(reach_continuity_weight=0.05).reach_continuity_weight)


def test_transit_rotation_weight_is_the_expert_default():
    """TODO 2.4.10.13: 0.2 while travelling to the pregrasp is the default since 2026-10-01; -1 keeps one weight."""
    from locomani.tasks.crate import coordinated_reach
    from locomani.tasks.crate.visual_config import CrateVisualConfig

    assert CrateVisualConfig().reach_rotation_weight_transit == 0.2
    assert CrateVisualConfig(reach_rotation_weight_transit=-1.0).reach_rotation_weight_transit == -1.0
    source = coordinated_reach.__loader__.get_source(coordinated_reach.__name__)
    assert 'if self.phase == "reach" and cfg.reach_rotation_weight_transit >= 0' in source


def test_arm_posture_weight_only_reweights_the_arm_joints():
    """TODO 2.4.10.20: 0 keeps the solver's uniform 0.04; a value applies to the 14 arm joints only."""
    from locomani.tasks.crate import coordinated_reach
    from locomani.tasks.crate.visual_config import CrateVisualConfig

    assert CrateVisualConfig().reach_arm_posture_weight == 0.0
    with pytest.raises(ValueError, match="reach_arm_posture_weight"):
        CrateVisualConfig(reach_arm_posture_weight=-0.1)
    source = coordinated_reach.__loader__.get_source(coordinated_reach.__name__)
    assert "np.r_[np.full(15, 0.04), np.full(14, cfg.reach_arm_posture_weight)]" in source
