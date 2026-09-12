"""The crate line's shared round loop: self-heal, the exporter keys, the manifest.

Before 2026-09-11 each crate expert had its own copy of this loop, they had
drifted (only the wall one self-healed), and neither could record.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from locomani.backends.sonic.deploy import SelfHeal
from locomani.tasks.crate import round_runner


class FakeCmd:
    def __init__(self):
        self.keys = []
        self.calls = []

    def key(self, k):
        self.keys.append(k)

    def reset_pose(self):
        self.calls.append("reset_pose")

    def start_control(self):
        self.calls.append("start_control")

    def hold(self, *a, **k):
        self.calls.append("hold")


class FakeP:
    def __init__(self):
        self.cmd = FakeCmd()

    def log(self, *a, **k):
        pass


class FakeReset:
    def __init__(self):
        self.calls = []

    def pin_arms(self):
        self.calls.append("pin_arms")

    def reset(self):
        self.calls.append("reset")

    def end_video(self):
        self.calls.append("end_video")


def _rounds(outcomes, **kw):
    """Drive run_rounds over a scripted list of (ok, why) and return (p, rows)."""
    p, reset = FakeP(), FakeReset()
    seq = list(outcomes)

    def run_round(p_, reset_, meta, rnd, rng, log, video_label, record=False):
        ok, why = seq[rnd]
        m = {"round": rnd, "ok": ok, "why": why, "produced": False}
        if record:
            round_runner.start_recording(p_, m)
        return m

    with tempfile.TemporaryDirectory() as d:
        manifest = str(Path(d) / "m.jsonl")
        with mock.patch("time.sleep"):
            round_runner.run_rounds(
                p,
                reset,
                {},
                run_round=run_round,
                rng=None,
                rounds=len(seq),
                label="x",
                tag="t",
                manifest=manifest,
                **kw,
            )
        rows = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
    return p, rows


class RecordingTests(unittest.TestCase):
    def test_no_keys_at_all_without_record(self):
        """Every crate pilot to date ran this path; it must stay byte-identical."""
        p, rows = _rounds([(True, ""), (False, "[grasp] no contact")])
        self.assertEqual(p.cmd.keys, [])
        self.assertEqual([r["produced"] for r in rows], [False, False])

    def test_success_saves_and_failure_discards(self):
        p, rows = _rounds([(True, ""), (False, "[lift] rise 0.01 m")], record=True)
        # 'c' starts each round; then 'c' saves the win, 'x' discards the loss
        self.assertEqual(p.cmd.keys, ["c", "c", "c", "x"])
        self.assertEqual([r["produced"] for r in rows], [True, True])

    def test_a_round_that_never_settled_sends_nothing(self):
        """start_recording is called by run_round only after the reset phase
        clears; a round that fails in the reset produces no episode and must not
        toggle the exporter -- that is what put grasp_gym's recorder out of phase."""
        p, reset = FakeP(), FakeReset()

        def run_round(*a, **k):
            return {"round": 0, "ok": False, "why": "[stand] did not stabilise after reset", "produced": False}

        with mock.patch("time.sleep"):
            round_runner.run_rounds(
                p, reset, {}, run_round=run_round, rng=None, rounds=1, label="x", tag="t", record=True
            )
        self.assertEqual(p.cmd.keys, [])


class SelfHealTests(unittest.TestCase):
    def test_two_consecutive_stabilise_failures_fire_once(self):
        h = SelfHeal(2)
        self.assertFalse(h.observe("[stand] did not stabilise after reset"))
        self.assertTrue(h.observe("[stand] did not stabilise after reset"))
        h.fired()
        self.assertFalse(h.observe("[stand] did not stabilise after reset"))

    def test_any_other_failure_resets_the_count(self):
        h = SelfHeal(2)
        h.observe("[stand] did not stabilise after reset")
        self.assertFalse(h.observe("[grasp] closed without bilateral contact"))
        self.assertFalse(h.observe("[stand] did not stabilise after reset"))

    def test_threshold_zero_disables_it(self):
        h = SelfHeal(0)
        for _ in range(5):
            self.assertFalse(h.observe("[stand] did not stabilise after reset"))

    def test_the_loop_restarts_the_deploy_and_re_primes(self):
        with mock.patch("locomani.backends.sonic.deploy.restart", return_value=True) as r:
            p, rows = _rounds([(False, "[stand] did not stabilise after reset")] * 2, self_heal_rounds=2, instance=1)
        r.assert_called_once_with(1)
        self.assertEqual(p.cmd.calls.count("reset_pose"), 1)  # the re-prime after the restart
        # Both rows say 0: the field is stamped BEFORE the self-heal runs, so it
        # counts restarts that happened before this round -- the restart triggered
        # by round 1 shows up on round 2. That is bimanual_wall.py's original
        # ordering, kept deliberately so manifests stay comparable across the
        # extraction; nothing in scripts/gate.py reads the field.
        self.assertEqual([row["deploy_restarts"] for row in rows], [0, 0])


class PadLevelTests(unittest.TestCase):
    """C (2026-09-11): one pad-hardening implementation, two named calibrations.
    bimanual_scene.py used to carry a second implementation with its own
    constants, applied on top of this one."""

    def test_the_crate_level_is_the_one_bimanual_scene_used_to_apply(self):
        from locomani.sim.build_sim_scene import PAD_LEVELS

        self.assertEqual(PAD_LEVELS["crate"]["solref"], [0.002, 1.0])
        self.assertEqual(PAD_LEVELS["crate"]["solimp"], [0.995, 0.9999, 0.001, 0.5, 2.0])
        self.assertEqual(PAD_LEVELS["object"]["solref"], [0.004, 1.0])
        self.assertEqual(PAD_LEVELS["object"]["solimp"], [0.99, 0.999, 0.001, 0.5, 2.0])

    def test_an_unknown_level_is_refused(self):
        from locomani.sim.build_sim_scene import harden_finger_pads

        with self.assertRaises(SystemExit):
            harden_finger_pads(object(), "firm")


if __name__ == "__main__":
    unittest.main()
