"""Measured stop requires sustained fresh low-speed observations."""

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free/tools"))
import diag_stance_grid as diag


class StandingSettleTests(unittest.TestCase):
    def state(self):
        st = json.loads((Path(__file__).parent / "fixtures/crouched_pregrasp.json").read_text())
        st["fallen"] = False
        st["objects"]["water_bottle"]["quat"] = [1.0, 0.0, 0.0, 0.0]
        st["pelvis"]["vel"] = [0.0, 0.0, 0.0]
        st["pelvis"]["yaw"] = 0.0
        return st

    def test_velocity_spike_restarts_continuous_settle(self):
        template = self.state()
        last = [None]
        index = [0]

        def obs():
            i = index[0]
            index[0] += 1
            st = copy.deepcopy(template)
            st["wall"] = 1000.0 + i * 0.1
            if i == 4:
                st["pelvis"]["vel"] = [0.2, 0.0, 0.0]
            last[0] = st
            return st

        p = SimpleNamespace(obs=obs, cmd=SimpleNamespace(frame=Mock()))
        with patch.object(diag.time, "time", side_effect=lambda: last[0]["wall"]):
            ok, st, _ = diag.wait_for_settle(p, 0.0)
        self.assertTrue(ok)
        self.assertGreaterEqual(st["wall"], 1001.1)
        self.assertLess(st["wall"], 1001.4)

    def test_crouch_wait_preserves_arm_channel_and_requires_height(self):
        template = self.state()
        last = [None]
        index = [0]

        def obs():
            i = index[0]
            index[0] += 1
            st = copy.deepcopy(template)
            st["wall"] = 1000.0 + i * 0.1
            st["pelvis"]["pos"][2] = 0.8 if i < 8 else 0.65
            last[0] = st
            return st

        p = SimpleNamespace(obs=obs, cmd=SimpleNamespace(frame=Mock()))
        reference = [0.0] * 29
        arm = SimpleNamespace(height=0.63, q=reference, _push=Mock())
        with patch.object(diag.time, "time", side_effect=lambda: last[0]["wall"]):
            ok, st, _ = diag.wait_for_settle(p, 0.0, arm=arm)
        self.assertTrue(ok)
        self.assertGreaterEqual(st["wall"], 1001.4)
        p.cmd.frame.assert_not_called()
        self.assertIs(arm._push.call_args.args[0], reference)
        self.assertIn("observed", arm._push.call_args.kwargs)

    def test_stale_state_cannot_qualify_or_publish(self):
        st = self.state()
        st["wall"] = 1000.0
        p = SimpleNamespace(obs=lambda: st, cmd=SimpleNamespace(frame=Mock()))
        with patch.object(diag.time, "time", return_value=1001.0):
            ok, _, why = diag.wait_for_settle(p, 0.0)
        self.assertFalse(ok)
        self.assertEqual(why, "stale settle observation")
        p.cmd.frame.assert_not_called()


if __name__ == "__main__":
    unittest.main()
