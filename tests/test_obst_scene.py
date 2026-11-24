"""selfmodel_via_20261120/obst_scene.py: the margin a way of acting needs, which the sixth step's second question rests on."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "selfmodel_via_20261120"))
import obst_scene as OS  # noqa: E402


def test_needed_margin_is_the_smallest_one_from_which_every_larger_one_stays_under_the_rate():
    assert OS.needed({0: 0.60, 1: 0.20, 2: 0.04, 3: 0.05, 4: 0.00, 6: 0.01}) == 2


def test_a_margin_over_the_rate_above_a_clean_one_moves_the_needed_margin_past_it():
    assert OS.needed({0: 0.60, 1: 0.03, 2: 0.08, 3: 0.02, 4: 0.00, 6: 0.00}) == 3  # 1 cm is clean, 2 cm is not


def test_needed_margin_is_infinite_when_the_largest_margin_is_over_the_rate_or_has_no_reach():
    assert np.isinf(OS.needed({0: 0.5, 4: 0.02, 6: 0.06}))
    assert np.isinf(OS.needed({0: 0.5, 4: 0.02, 6: float("nan")}))
