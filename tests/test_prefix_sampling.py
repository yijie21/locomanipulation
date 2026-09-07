import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/teleop_free/tools'))
from prefix_sampling import boundary_starts, repeat_indices

class PrefixSamplingTests(unittest.TestCase):
    def test_event_is_in_prefix_and_full_horizon_is_retained(self):
        starts = boundary_starts(100, [30, 95, None])
        self.assertEqual(starts, list(range(21,31)))
        for start in starts:
            self.assertLessEqual(start+39,99)
            self.assertIn(30-start,range(10))

    def test_no_cross_episode_padding_or_duplicate_overlap(self):
        self.assertEqual(boundary_starts(20,[5]),[])
        self.assertEqual(boundary_starts(40,[0,1]),[0])
        self.assertEqual(boundary_starts(100,[30,30]),boundary_starts(100,[30]))

    def test_control_is_identical_and_experiment_keeps_all_windows(self):
        source=np.array([9,2,7,1])
        np.testing.assert_array_equal(repeat_indices(source,[2,7],1),source)
        weighted=repeat_indices(source,[2,7],4)
        self.assertEqual(weighted.tolist(),[9,2,2,2,2,7,7,7,7,1])
        np.testing.assert_array_equal(source,[9,2,7,1])

if __name__=='__main__': unittest.main()
