import json
import tempfile
import unittest
from pathlib import Path

from locomani.pipeline.recording_density import check_recording_density, recording_density


class RecordingDensityTests(unittest.TestCase):
    def session(self, episodes):
        """A session directory holding just the priv_seq sidecars the check reads."""
        root = Path(tempfile.mkdtemp())
        (root / "priv_seq").mkdir()
        for ep, seqs in episodes.items():
            (root / "priv_seq" / f"episode_{ep:06d}.json").write_text(
                json.dumps(dict(episode_index=ep, n_rows=len(seqs), priv_seq=seqs))
            )
        return root

    def test_density_is_rows_over_the_span_they_cover(self):
        self.assertAlmostEqual(recording_density([100, 150, 200], 110), 1.1)

    def test_density_is_unknown_when_the_span_cannot_say(self):
        for seqs, rows in (([100], 1), ([], 0), ([100, 100], 2)):
            self.assertIsNone(recording_density(seqs, rows))

    def test_accepts_a_session_recorded_end_to_end(self):
        # Every healthy episode measured on 2026-09-17 sat between 1.04 and 1.15.
        check_recording_density(self.session({0: list(range(0, 1000)), 1: list(range(0, 900))}), [0, 1])

    def test_rejects_the_stub_that_would_otherwise_convert_silently(self):
        # The shape that reached the dataset: a 37 s round holding one 1.2 s burst
        # plus three stray frames, whose boundary still landed inside its range.
        stub = [3580] + list(range(4063, 4105)) + [4504, 4800]
        with self.assertRaises(SystemExit) as caught:
            check_recording_density(self.session({0: stub}), [0])
        message = str(caught.exception)
        self.assertIn("episode 0", message)
        self.assertIn("0.04 rows/unit", message)

    def test_names_every_thin_episode_and_leaves_the_rest_alone(self):
        sess = self.session({0: list(range(0, 1000)), 1: [0, 500, 999], 2: list(range(0, 800)), 3: [0, 400]})
        with self.assertRaises(SystemExit) as caught:
            check_recording_density(sess, [0, 1, 2, 3])
        message = str(caught.exception)
        self.assertIn("2 of 4 episodes", message)
        for named in ("episode 1", "episode 3"):
            self.assertIn(named, message)
        self.assertNotIn("episode 0:", message)
        self.assertNotIn("episode 2:", message)

    def test_ignores_episodes_without_a_sidecar(self):
        check_recording_density(self.session({}), [0, 1])


if __name__ == "__main__":
    unittest.main()
