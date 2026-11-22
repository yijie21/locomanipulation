"""selfmodel_via_20261120/via_model.py: the two geometric pieces every reported number rests on -- a sweep sampled between
recorded moments, and the swept-region error of a predicted sweep."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experiments" / "selfmodel_via_20261120"))
import via_model as VM  # noqa: E402


def test_interp_is_linear_between_samples_and_held_beyond_the_ends():
    T = np.array([0.0, 1.0, 3.0])
    X = np.array([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[1.0, 4.0, 0.0]]])  # (3 times, 1 sphere, 3)
    got = VM.interp(T, X, np.array([-1.0, 0.5, 2.0, 9.0]))
    assert got.shape == (4, 1, 3)
    assert np.allclose(got[:, 0], [[0, 0, 0], [0.5, 0, 0], [1, 2, 0], [1, 4, 0]])


def test_swept_region_error_ignores_lag_along_the_path_and_counts_the_distance_beside_it():
    line = np.linspace([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 5)  # a straight 1 m sweep
    P = np.stack([line, line + [0.0, 1.0, 0.0]], 1)  # two spheres, the second 1 m to the left
    lagging = np.stack([np.array([[0.1, 0.0, 0.0], [0.33, 0.0, 0.0]]), np.array([[0.2, 1.0, 0.0], [0.9, 1.0, 0.0]])], 1)
    assert np.allclose(VM.sweep_err(lagging, P), 0.0)
    beside = lagging + np.array([0.0, 0.0, 0.03])
    assert np.allclose(VM.sweep_err(beside, P), 0.03)
    past = np.array([[[1.2, 0.0, 0.0], [0.5, 1.0, 0.04]]])  # the first sphere 20 cm past its sweep's end
    assert np.allclose(VM.sweep_err(past, P), [0.2, 0.04])


def test_a_predicted_sweep_that_matches_the_recording_has_no_error():
    rng = np.random.default_rng(0)
    ref = np.cumsum(rng.normal(0, 0.02, (VM.Q, 24, 3)), 0)
    off = rng.normal(0, 0.05, (VM.Q + 1, 24, 3))
    P = np.concatenate([ref, ref[-1:]]) + off
    assert VM.sweep_err(P[3:20], P).max() < 1e-12
    assert VM.sweep_err(P[3:20], np.concatenate([ref, ref[-1:]])).max() > 0.02
