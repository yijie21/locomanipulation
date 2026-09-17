import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from locomani.backends.sonic import initial_pose


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
            with patch.object(initial_pose, "INIT_TOKEN_FILE", str(p)):
                np.testing.assert_array_equal(initial_pose.initial_token(), token)
            with (
                patch.object(initial_pose, "INIT_TOKEN_FILE", None),
                patch.object(initial_pose, "INIT_TOKEN_OFFICIAL", False),
            ):
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

        def enter_streamed(prime=None):
            """The real commander's order (agent_server.enter_streamed).

            The motion switch first, then the caller's packet, then the POSE
            manager state -- the deploy drops anything that arrived before the
            switch, so where the first packet falls is the whole point.
            """
            events.append(("switch",))
            if prime is not None:
                prime()
            events.append(("manager",))

        p.cmd.enter_streamed.side_effect = enter_streamed
        token = np.arange(64, dtype=np.float32) / 128
        with (
            patch.object(initial_pose, "INIT_TOKEN_FILE", "explicit.json"),
            patch.object(initial_pose, "initial_token", return_value=token),
            patch.object(initial_pose, "time", Clock()),
        ):
            result = initial_pose.prepare_token_pose(p, np.zeros(14), hold_seconds=0.4)
        return result, events, token

    def test_the_first_token_lands_inside_the_switch_and_nothing_precedes_it(self):
        # A packet sent before the switch is dropped by the deploy's ZMQ toggle,
        # and it also makes the post-switch packet an ordinary extension, so the
        # default clip keeps playing -- the twitch at the start of a round.
        result, events, token = self.prepare()
        self.assertTrue(result)
        kinds = [e[0] for e in events]
        self.assertEqual(kinds[:3], ["switch", "token", "manager"])
        self.assertGreater(kinds.count("token"), 1)  # the hold keeps republishing it
        for event in events:
            if event[0] == "token":
                np.testing.assert_array_equal(event[1], token)
                self.assertAlmostEqual(event[2]["right_hand"][0], 0.0245, places=6)

    def test_fall_or_undelivered_arms_reject_initialization(self):
        self.assertFalse(self.prepare(fallen=True)[0])
        self.assertFalse(self.prepare(arm_error=0.31)[0])
