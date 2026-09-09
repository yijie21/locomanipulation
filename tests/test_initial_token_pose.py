import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/teleop_free"))
import initial_pose


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, dt):
        self.now += dt


class InitialTokenPoseTests(unittest.TestCase):
    def test_explicit_file_is_exact_and_legacy_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "token.json"
            token = np.linspace(-0.5, 0.5, 64, dtype=np.float32)
            p.write_text(json.dumps(token.tolist()))
            with patch.dict(os.environ, {"TF_INIT_TOKEN_FILE": str(p)}):
                np.testing.assert_array_equal(initial_pose.initial_token(), token)
            with patch.dict(os.environ, {"TF_INIT_TOKEN_FILE": "", "TF_INIT_TOKEN": "ours"}):
                np.testing.assert_array_equal(initial_pose.initial_token(), initial_pose.OURS_INITIAL_MOTION_TOKEN)

    def test_invalid_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "token.json"
            for token in ([0.0] * 63, [float("nan")] * 64, [2.0] * 64):
                with self.subTest(token=token[0]):
                    p.write_text(json.dumps(token))
                    with self.assertRaises(ValueError):
                        initial_pose.load_token_file(p)

    def prepare(self, fallen=False, arm_error=0.0):
        p = Mock()
        p.obs.return_value = {"fallen": fallen, "joints": [0.0] * 15 + [arm_error] * 14}
        events = []
        p.cmd.token_frame.side_effect = lambda *a, **kw: events.append(("token", np.array(a[0]).copy(), kw))
        p.cmd.enter_streamed.side_effect = lambda: events.append(("enter",))
        token = np.arange(64, dtype=np.float32) / 128
        with (
            patch.dict(os.environ, {"TF_INIT_TOKEN_FILE": "explicit.json"}),
            patch.object(initial_pose, "initial_token", return_value=token),
            patch.object(initial_pose, "time", Clock()),
        ):
            result = initial_pose.prepare_token_pose(p, np.zeros(14), hold_seconds=0.4)
        return result, events, token

    def test_fixed_token_is_primed_then_held_with_open_hands(self):
        result, events, token = self.prepare()
        self.assertTrue(result)
        self.assertEqual(events[30], ("enter",))
        self.assertTrue(all(e[0] == "token" for e in events[:30]))
        for event in events:
            if event[0] == "token":
                np.testing.assert_array_equal(event[1], token)
                self.assertAlmostEqual(event[2]["right_hand"][0], 0.0245, places=6)

    def test_fall_or_undelivered_arms_reject_initialization(self):
        self.assertFalse(self.prepare(fallen=True)[0])
        self.assertFalse(self.prepare(arm_error=0.31)[0])
