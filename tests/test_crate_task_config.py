import unittest

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
    import pytest

    for bounds in [(float("nan"), 0.1), (-float("inf"), float("inf")), ("a", "b"), (False, True), None]:
        with pytest.raises(ValueError, match="crate_dy_range_m"):
            CrateTaskConfig.from_dict({"crate_dy_range_m": bounds})


def test_cli_range_reaches_the_actual_reset_without_module_mutation(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock

    import pytest

    from locomani.tasks.crate import bimanual_wall as wall

    class ResetObserved(Exception):
        pass

    received = []

    def reset_scene(**kwargs):
        received.append(kwargs["objects"]["blue_crate"]["dxy"][1])
        raise ResetObserved()

    reset = SimpleNamespace(pin_arms=Mock(), set_ghost=Mock(), reset=reset_scene)
    p = SimpleNamespace(cmd=Mock(), log=Mock())
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
        monkeypatch.setattr("sys.argv", ["wall", "--set", f"crate_dy_range_m={value},{value}"])
        with pytest.raises(ResetObserved):
            wall.main()
    assert received == [0.035, -0.025, 0.035]
    assert not hasattr(wall, "CRATE_DY_RANGE")
