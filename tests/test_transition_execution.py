import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/teleop_free'))
from transition_execution import CommitWindow


class CommitTests(unittest.TestCase):
    def test_replans_cannot_postpone_committed_transition(self):
        window = CommitWindow(25)
        window.arm(100., first_crouch=30, horizon=40)
        self.assertAlmostEqual(window.until, 101.4)
        # Preserve the same absolute endpoint through three replans.
        for t in [100.4, 100.8, 101.2]:
            n = window.frozen(t, 2)
            self.assertGreaterEqual(t+n/25, window.until-1e-9)
            self.assertLess(t+n/25, window.until+.041)
        self.assertEqual(window.frozen(102., 2), 2)

    def test_commit_never_extends_beyond_available_prediction(self):
        window = CommitWindow(25)
        window.arm(0., first_crouch=38, horizon=40)
        self.assertEqual(window.until, 1.6)


if __name__ == '__main__': unittest.main()
