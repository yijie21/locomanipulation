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
