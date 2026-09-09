import sys
import unittest
from pathlib import Path

from locomani.pipeline.verification_tail import training_prefix_length


class VerificationTailTests(unittest.TestCase):
    def row(self, end=106, ok=True):
        return dict(ok=ok, execution=dict(metrics=dict(training_tail=dict(end_seq=end))))

    def test_preserves_contiguous_prefix_with_missing_or_duplicate_sequence_samples(self):
        seqs = [100, 102, 102, 104, 108, 110]
        self.assertEqual(training_prefix_length(seqs, 6, self.row()), 4)

    def test_refuses_wrong_join_missing_marker_and_failed_episode(self):
        for seqs, size, row in (
            ([100, 102], 3, self.row()),
            ([100, 99, 110], 3, self.row()),
            ([100, 102, 110], 3, self.row(200)),
            ([100, 102, 110], 3, self.row(ok=False)),
            ([100, 102, 110], 3, dict(ok=True)),
        ):
            with self.assertRaises(ValueError):
                training_prefix_length(seqs, size, row)
