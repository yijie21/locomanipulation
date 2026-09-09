import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
import coupled_grasp_policy as policy


class TransportIKFallbackTests(unittest.TestCase):
    def test_reachable_target_is_not_shortened_a_second_time(self):
        q = np.zeros(29)
        ik = Mock()
        ik.solve.return_value = (q, 0.005, 0.1)
        aim = np.array([0.08, 0.0, 0.0])
        *_, retried = policy.solve_transport_ik(ik, "right", aim, np.eye(3), q, np.zeros(3), 0.025)
        ik.solve.assert_called_once()
        np.testing.assert_array_equal(ik.solve.call_args.args[1], aim)
        self.assertFalse(retried)

    def test_unreachable_target_gets_a_bounded_retry_under_same_gates(self):
        q = np.zeros(29)
        ik = Mock()
        ik.solve.side_effect = [(q, 0.042, 0.29), (q, 0.005, 0.18)]
        result = policy.solve_transport_ik(ik, "right", np.array([0.075, 0.0, 0.0]), np.eye(3), q, np.zeros(3), 0.025)
        self.assertEqual(ik.solve.call_count, 2)
        np.testing.assert_allclose(ik.solve.call_args.args[1], [0.025, 0.0, 0.0])
        self.assertEqual(result[1:], (0.005, 0.18, True))

    def test_disabled_fallback_preserves_failed_original_solution(self):
        q = np.zeros(29)
        ik = Mock()
        ik.solve.return_value = (q, 0.1, 0.3)
        result = policy.solve_transport_ik(ik, "right", np.ones(3), np.eye(3), q, np.zeros(3), 0.0)
        ik.solve.assert_called_once()
        self.assertEqual(result[1:], (0.1, 0.3, False))
