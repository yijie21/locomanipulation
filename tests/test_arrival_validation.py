import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/teleop_free'))
from arrival_validation import ArrivalAdvance,squat_handover
from reach_config import ReachConfig
class ArrivalTests(unittest.TestCase):
    def test_advance_from_first_reachable_not_premature_map_gate(self):
        g=ArrivalAdvance(.03)
        self.assertFalse(g.stop(.6,False))
        self.assertFalse(g.stop(.4,True))
        self.assertFalse(g.stop(.38,True))
        self.assertTrue(g.stop(.369,True))
        self.assertEqual(g.first_reachable,.4)
    def test_squat_uses_measured_height_and_holds_at_requested_drop(self):
        self.assertEqual(squat_handover(dict(ready=False,world_height_error_m=-.01),.07,.12),'grip_height_aligned')
        self.assertIsNone(squat_handover(dict(ready=False,world_height_error_m=-.04),.07,.12))
        self.assertEqual(squat_handover(dict(ready=False,world_height_error_m=-.04),.12,.12),'requested_drop_delivered')
    def test_experiment_is_opt_in(self):
        self.assertEqual(ReachConfig().arrival_advance,0)
        self.assertFalse(ReachConfig().measured_squat_handover)
        with self.assertRaises(ValueError):ReachConfig(arrival_advance=.03)
