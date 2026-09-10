import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/teleop_free/scalebfm'))
from visual_chunks import ChunkScheduler


class ChunkTests(unittest.TestCase):
    def test_new_request_keeps_old_chunk_until_delayed_replacement(self):
        scheduler = ChunkScheduler()
        first = np.repeat(np.arange(40)[:, None], 29, axis=1)
        scheduler.submit(80, first)
        self.assertIsNone(scheduler.row(82))
        self.assertEqual(scheduler.row(84)[0], 2)
        scheduler.submit(100, first + 200)
        self.assertEqual(scheduler.row(100)[0], 10)
        self.assertEqual(scheduler.row(102)[0], 11)
        self.assertEqual(scheduler.row(104)[0], 202)

    def test_invalid_prediction_does_not_discard_current_chunk(self):
        scheduler = ChunkScheduler()
        scheduler.submit(0, np.ones((40, 29)))
        scheduler.row(4)
        with self.assertRaises(ValueError):
            scheduler.submit(20, np.full((40, 29), np.nan))
        self.assertEqual(scheduler.row(20)[0], 1)


if __name__ == '__main__':
    unittest.main()
