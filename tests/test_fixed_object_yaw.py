import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from grasp_gym import sample_object_yaw
from experiment_cases import validate

class FixedYawTests(unittest.TestCase):
    def test_fixed_yaw_does_not_draw_random_orientation(self):
        with patch('grasp_gym.np.random.uniform') as random:
            self.assertEqual(sample_object_yaw(0.),0.)
            self.assertEqual(sample_object_yaw(-1.2),-1.2)
        random.assert_not_called()
    def test_old_callers_still_sample(self):
        with patch('grasp_gym.np.random.uniform',return_value=.7) as random:
            self.assertEqual(sample_object_yaw(),.7)
        random.assert_called_once()
    def test_invalid_yaw_rejected(self):
        for yaw in (float('nan'),float('inf'),True,'0'):
            with self.assertRaises(ValueError):sample_object_yaw(yaw)
        c=dict(case_id='a',target='water_bottle',appearance=42,can_xy=[.1,.2],stand=[.3,.4],
               rel=[.2,-.2],psi=1.,spawn_off=1.2,spawn_lat=.01,dpsi=.1,squat_hint=.12)
        self.assertEqual(validate([dict(c,object_yaw=0.)])[0]['object_yaw'],0.)
        with self.assertRaises(ValueError):validate([dict(c,object_yaw=float('nan'))])
