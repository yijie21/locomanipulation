import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from locomani.sim.privileged_pub import StateTrace


class StateTraceBoundaryTests(unittest.TestCase):
    def test_reset_retry_preserves_previous_attempt_without_explicit_stop(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = StateTrace(folder)
            state = SimpleNamespace(qpos=np.array([1.0, 2.0]), qvel=np.array([3.0]))
            trace.start("failed_attempt")
            trace.frame(state, 123)
            trace.start("next_attempt")
            previous = Path(folder) / "failed_attempt_state_trace.npz"
            self.assertTrue(previous.is_file())
            np.testing.assert_array_equal(np.load(previous)["seq"], [123])
            trace.frame(state, 124)
            trace.stop()
            np.testing.assert_array_equal(np.load(Path(folder) / "next_attempt_state_trace.npz")["seq"], [124])
