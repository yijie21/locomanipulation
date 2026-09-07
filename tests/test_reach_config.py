import sys
from pathlib import Path
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from reach_config import ReachConfig, DEFAULT_REACH
import stream_reach as sr
from test_grasp_transition import Robot, Clock


class ReachConfigTests(unittest.TestCase):
    def test_default_keeps_old_band(self):
        self.assertEqual(DEFAULT_REACH.approach_band(),(.18,.27))

    def test_invalid_calibration_fails_before_actuation(self):
        for args in [dict(post_max=.25),dict(post_min=float('nan')),
                     dict(transport_guard='false'),dict(sweep_joint_step=0.),dict(max_drop=.2)]:
            with self.assertRaises(ValueError):ReachConfig(**args)

    def test_real_approach_uses_explicit_bounds(self):
        config=ReachConfig(pre_min=.20,pre_max=.24,azimuth_deg=25.,approach_timeout=.1)
        with patch.object(sr,'time',Clock()),patch.object(sr,'make_arm') as factory:
            result=sr.execute_grasp(Robot(), 'water_bottle',[.6,0.,.7],None,config=config)
        self.assertEqual(result.execution['transition']['pre_band'],[.20,.24])
        self.assertEqual(result.execution['metrics']['reach_config']['azimuth_deg'],25.)
        factory.assert_not_called()


if __name__=='__main__':unittest.main()
