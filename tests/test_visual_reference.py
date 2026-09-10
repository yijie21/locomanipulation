"""Actual reach-switch discontinuity and stateful future sampling contracts."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from locomani.backends.scalebfm.reference import MotionLimits, PoseReference, PoseTransition


class VisualReferenceTests(unittest.TestCase):
    def test_reach_switch_limits_translation_rotation_and_acceleration(self):
        ref = PoseReference([0, 0, 0], [1, 0, 0, 0])
        # Measured bug: a ~12 cm /60 deg command jump in one 20 ms tick.
        target = np.array([0.12, 0, 0])
        quat = Rotation.from_euler("y", 60, degrees=True).as_quat(scalar_first=True)
        previous_p, previous_r = ref.position.copy(), ref.rotation
        previous_v, previous_w = ref.velocity.copy(), ref.angular_velocity.copy()
        for _ in range(250):
            p, q = ref.futures(np.tile(target, (6, 1)), np.tile(quat, (6, 1)), 0.02)
            assert np.linalg.norm(p[0] - previous_p) <= 0.012 + 1e-10
            assert (ref.rotation * previous_r.inv()).magnitude() <= 0.05 + 1e-10
            assert np.linalg.norm(ref.velocity - previous_v) <= 0.04 + 1e-10
            assert np.linalg.norm(ref.angular_velocity - previous_w) <= 0.24 + 1e-10
            previous_p, previous_r = ref.position.copy(), ref.rotation
            previous_v, previous_w = ref.velocity.copy(), ref.angular_velocity.copy()
        np.testing.assert_allclose(ref.position, target, atol=1e-5)
        assert (ref.rotation * Rotation.from_quat(quat, scalar_first=True).inv()).magnitude() < 1e-5

    def test_preview_commits_one_tick_regardless_of_chunk_length(self):
        a, b = [PoseReference([0, 0, 0], [1, 0, 0, 0]) for _ in range(2)]
        p = np.tile([0.3, 0, 0], (6, 1))
        q = np.tile([1, 0, 0, 0], (6, 1))
        future_p, _ = a.futures(p, q, 0.02)
        b.advance(p[0], q[0], 0.02)
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.velocity, b.velocity)
        assert future_p[-1, 0] > future_p[0, 0]  # ScaleBFM receives moving futures.

    def test_future_prediction_agrees_with_subsequent_committed_ticks(self):
        a = PoseReference([0, 0, 0], [1, 0, 0, 0])
        b = a.copy()
        p = np.array([[i * 0.01, 0, 0] for i in range(6)])
        q = Rotation.from_euler("z", np.arange(6) * 0.01).as_quat(scalar_first=True)
        ps, qs = a.futures(p, q, 0.02)
        values = [b.advance(pi, qi, 0.02) for pi, qi in zip(p, q)]
        np.testing.assert_allclose(ps, np.stack([v[0] for v in values]), atol=1e-12)
        np.testing.assert_allclose(qs, np.stack([v[1] for v in values]), atol=1e-12)

    def test_world_rotation_equivariance_and_quaternion_sign(self):
        world = Rotation.from_euler("xyz", [0.2, -0.1, 1.6])
        shift = np.array([1, 2, 3])
        a = PoseReference([0, 0, 0], [1, 0, 0, 0])
        b = PoseReference(shift, -world.as_quat(scalar_first=True))
        target_r = Rotation.from_euler("xyz", [0.5, 0.3, -0.2])
        target_p = np.array([0.3, -0.2, 0.1])
        for _ in range(50):
            a.advance(target_p, target_r.as_quat(scalar_first=True), 0.02)
            b.advance(world.apply(target_p) + shift, -(world * target_r).as_quat(scalar_first=True), 0.02)
            np.testing.assert_allclose(b.position, world.apply(a.position) + shift, atol=1e-10)
            assert (b.rotation * (world * a.rotation).inv()).magnitude() < 1e-10

    def test_bad_future_does_not_mutate_reference(self):
        ref = PoseReference([0, 0, 0], [1, 0, 0, 0])
        q = np.tile([1.0, 0, 0, 0], (6, 1))
        q[-1] = np.nan
        with self.assertRaises(ValueError):
            ref.futures(np.ones((6, 3)), q, 0.02)
        np.testing.assert_array_equal(ref.position, np.zeros(3))

    def test_invalid_limits_are_rejected(self):
        with self.assertRaises(ValueError):
            MotionLimits(linear_speed=-1)

    def test_transition_exactly_joins_previous_command_and_new_path(self):
        previous = (np.array([0.1, -0.2, 0.7]), np.array([1.0, 0, 0, 0]))
        raw = (np.array([0.23, -0.22, 0.8]), Rotation.from_euler("y", 60, degrees=True).as_quat(scalar_first=True))
        transition = PoseTransition(previous, raw, 2.5)
        p, q = transition.sample(*raw, 0)
        np.testing.assert_allclose(p, previous[0], atol=1e-12)
        np.testing.assert_allclose(q, previous[1], atol=1e-12)
        end = (raw[0] + np.array([0.3, 0, 0.1]), raw[1])
        p, q = transition.sample(*end, 2.5)
        np.testing.assert_allclose(p, end[0], atol=1e-12)
        np.testing.assert_allclose(q, end[1], atol=1e-12)

    def test_transition_preview_is_pure(self):
        raw = ([0.12, 0, 0], [0.8660254, 0, 0.5, 0])
        transition = PoseTransition(([0, 0, 0], [1, 0, 0, 0]), raw, 2.5)
        first = transition.sample(*raw, 0.5)
        transition.sample(*raw, 2.5)
        again = transition.sample(*raw, 0.5)
        np.testing.assert_array_equal(first[0], again[0])
        np.testing.assert_array_equal(first[1], again[1])


if __name__ == "__main__":
    unittest.main()
