"""Boundary and reset checks for SONIC-aligned ScaleBFM scoring."""

from types import SimpleNamespace

import pytest

from locomani.backends.scalebfm.runtime import Sim
from locomani.tasks.bottle_visual.success import LiftSuccess


def test_six_centimetres_is_not_sufficient():
    score = LiftSuccess()
    assert not score.update(0, 0.06)
    assert not score.update(3, 0.06)
    assert not score.update(4, 0.061)
    assert not score.update(6, 0.061)
    assert score.update(6.02, 0.061)


def test_drop_resets_hold_timer():
    score = LiftSuccess()
    assert not score.update(0, 0.07)
    assert not score.update(1.9, 0.06)
    assert not score.update(2, 0.07)
    assert not score.update(4, 0.07)
    assert score.update(4.02, 0.07)


def test_invalid_simulation_cannot_succeed():
    score = LiftSuccess()
    assert not score.update(0, 0.07)
    assert not score.update(3, 0.07, valid=False)
    assert not score.update(3.02, 0.07)
    assert score.update(5.04, 0.07)


@pytest.mark.parametrize("height,fallen", [(0.399, True), (0.4, False), (0.44, False)])
def test_fall_threshold_matches_sonic(height, fallen):
    sim = SimpleNamespace(d=SimpleNamespace(qpos=[0, 0, height]))
    assert Sim.fallen(sim) == fallen
