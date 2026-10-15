"""TODO 26.3's bottle adapter (docs/handoff/scalebfm-bottle-rsi-20261015.md §3, §6): the round's spec (keys, merge, dotted
task.reach overrides, field-read log) and the gate's bottle rules (frozen fields, fell / dropped)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "scalebfm_bottle_rsi_20261015"))
import accept_gate as gate  # noqa: E402
import bottle_round as br  # noqa: E402


class Spec(unittest.TestCase):
    def test_defaults_and_unknown_keys(self):
        s = br.load_spec("mode: 6")
        self.assertEqual(s["mode"], 6)
        self.assertEqual(s["set"], {})
        with self.assertRaises(ValueError):
            br.load_spec("stance: near")  # a warehouse key
        with self.assertRaises(ValueError):
            br.load_spec("reference: {pelvis_floor: 0.7}")
        with self.assertRaises(ValueError):
            br.load_spec("extensions: {nobody_wrote_this: {}}")

    def test_merge_overlays_dicts_and_replaces_scalars(self):
        base = br.load_spec("mode: 4\nset: {simple.walk_speed: 0.3}")
        out = br.merge(base, {"mode": 6, "set": {"whole_body.tracking_gain": 1.0}})
        self.assertEqual(out["mode"], 6)
        self.assertEqual(out["set"], {"simple.walk_speed": 0.3, "whole_body.tracking_gain": 1.0})
        self.assertNotEqual(br.spec_hash(base), br.spec_hash(out))

    def test_dotted_sets_reach_the_config_and_reads_are_logged(self):
        log = br.FieldLog()
        rc = br.reach_config({"simple.walk_speed": 0.3, "recovery_timeout": 50.0}, log)
        self.assertAlmostEqual(rc.simple.walk_speed, 0.3)
        self.assertAlmostEqual(rc.recovery_timeout, 50.0)
        log.read.clear()
        _ = rc.simple.guard_margin_m
        _ = rc.whole_body.waist_pitch_max
        # the block attributes themselves (simple, whole_body) are logged too; the gate only looks up set fields
        self.assertLessEqual({"simple.guard_margin_m", "whole_body.waist_pitch_max"}, log.read)
        self.assertNotIn("simple.walk_speed", log.read)


class Gate(unittest.TestCase):
    def test_the_success_rule_is_frozen(self):
        self.assertEqual(gate.FROZEN, {"simple.min_rise_m", "simple.disturb_limit_m", "recovery_timeout"})

    def test_bottle_fell_and_dropped(self):
        def make(d, name, rise, pelvis_min=0.7):
            t = np.arange(len(rise)) * 0.02
            bottle = np.zeros((len(rise), 7))
            bottle[:, 2] = 0.8 + np.asarray(rise)
            pelvis = np.zeros((len(rise), 7))
            pelvis[:, 2] = 0.75
            pelvis[-1, 2] = pelvis_min
            np.savez(d / f"{name}.npz", t=t, bottle=bottle, pelvis=pelvis)
            (d / f"{name}.json").write_text(json.dumps(dict(ok=False, why="x", events={"end": float(t[-1])})))
            return gate.outcome(d / f"{name}.npz")

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            held = make(d, "held", [0, 0.05, 0.06, 0.06])
            dropped = make(d, "dropped", [0, 0.05, 0.06, 0.0])
            never = make(d, "never", [0, 0.01, 0.0, 0.0])
            fell = make(d, "fell", [0, 0.0, 0.0, 0.0], pelvis_min=0.3)
        self.assertEqual((held["dropped"], dropped["dropped"], never["dropped"]), (False, True, False))
        self.assertEqual((held["fell"], fell["fell"]), (False, True))


if __name__ == "__main__":
    unittest.main()
