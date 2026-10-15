"""TODO 25.9.1's acceptance gate and candidate specs (docs/handoff/scalebfm-warehouse-rsi-20261014.md §6.4): the one-sided
exact McNemar, the paired tally, the frozen-field and unknown-key refusals, fork merging, and the field-read log the gate
uses to reject settings that never reach the code."""

import sys
import tempfile
import unittest
from math import comb
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "scalebfm_warehouse_rsi_20261014"))
import accept_gate as gate  # noqa: E402
from bfm_round import BASE_SPEC, FieldLog, load_delta, load_spec, merge, spec_hash, task_config  # noqa: E402

CONFIG = REPO / "configs" / "experiments" / "controller_compare_l2_warehouse_sonic_20261013.yaml"


def outcome(ok, fell=False, dropped=False):
    return dict(ok=ok, why="" if ok else "reach_stalled", end=1.0, fell=fell, dropped=dropped, fields_read=[])


class McNemar(unittest.TestCase):
    def test_one_sided_tail(self):
        self.assertEqual(gate.mcnemar_one_sided(0, 0), 1.0)
        self.assertAlmostEqual(gate.mcnemar_one_sided(5, 0), 1 / 32)
        self.assertAlmostEqual(gate.mcnemar_one_sided(6, 1), sum(comb(7, i) for i in (6, 7)) / 2**7)
        self.assertGreater(gate.mcnemar_one_sided(1, 6), 0.9)

    def test_the_frozen_threshold_needs_more_than_three_net_wins_when_losses_occur(self):
        # b - c >= 3 alone is not enough: 6 wins against 3 losses is p = 0.25
        self.assertGreater(gate.mcnemar_one_sided(6, 3), gate.ALPHA)
        self.assertLessEqual(gate.mcnemar_one_sided(5, 0), gate.ALPHA)


class Tally(unittest.TestCase):
    def test_pairs_by_start_and_counts_guards(self):
        inc = {"a": outcome(True), "b": outcome(False), "c": outcome(False, fell=True), "d": outcome(True)}
        cand = {"a": outcome(True), "b": outcome(True), "c": outcome(True), "e": outcome(True, dropped=True)}
        t = gate.tally(cand, inc)
        self.assertEqual(t["n"], 3)  # e and d are unpaired
        self.assertEqual((t["b"], t["c"]), (2, 0))
        self.assertEqual((t["candidate_ok"], t["incumbent_ok"]), (3, 1))
        self.assertEqual((t["candidate_fell"], t["incumbent_fell"]), (0, 1))


class Specs(unittest.TestCase):
    def test_defaults_and_unknown_keys(self):
        self.assertEqual(load_spec("{}"), BASE_SPEC)
        with self.assertRaises(ValueError):
            load_spec("{stance_offset: 0.02}")
        with self.assertRaises(ValueError):
            load_spec("{reference: {knee_gain: 0.5}}")
        with self.assertRaises(ValueError):
            load_spec("{extensions: {not_registered: {}}}")

    def test_the_incumbent_file_is_25_7(self):
        inc = load_spec(str(REPO / "scripts/experiments/scalebfm_warehouse_rsi_20261014/incumbent.yaml"))
        self.assertEqual((inc["mode"], inc["stance"], inc["servo"]), (2, "near", "none"))
        self.assertEqual(inc["set"], {"visual.tracking_gain": 0.5})

    def test_fork_merge_keeps_the_base_settings(self):
        base = load_spec("{set: {visual.tracking_gain: 0.5}, mode: 2}")
        fork = merge(base, {"set": {"visual.walk_speed": 0.35}, "servo": "pad"})
        self.assertEqual(fork["set"], {"visual.tracking_gain": 0.5, "visual.walk_speed": 0.35})
        self.assertEqual((fork["mode"], fork["servo"]), (2, "pad"))
        self.assertEqual(base["servo"], "none")
        self.assertNotEqual(spec_hash(base), spec_hash(fork))
        self.assertEqual(spec_hash(base), spec_hash(load_spec("{mode: 2, set: {visual.tracking_gain: 0.5}}")))

    def test_a_fork_delta_from_a_file_does_not_reset_the_base(self):
        base = load_spec("{servo: pad, mode: 6, stance_shift: 0.02}")
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "delta.json"
            f.write_text('{"set": {"visual.tracking_gain": 1.0}}')
            fork = merge(base, load_delta(str(f)))
        self.assertEqual((fork["servo"], fork["mode"], fork["stance_shift"]), ("pad", 6, 0.02))
        self.assertEqual(fork["set"], {"visual.tracking_gain": 1.0})
        with self.assertRaises(ValueError):
            load_delta("{servo_gain: 2}")

    def test_frozen_fields_are_task_fields(self):
        cfg = task_config(CONFIG, {}, FieldLog())
        for key in gate.FROZEN:
            block, field = key.split(".", 1)
            self.assertTrue(hasattr(getattr(cfg, block), field), key)

    def test_the_gate_refuses_a_frozen_field_without_running(self):
        with tempfile.TemporaryDirectory() as d:
            cand = Path(d) / "c.yaml"
            cand.write_text("set: {visual.max_tilt_deg: 30}\n")
            pool = Path(d) / "pool"
            pool.mkdir()
            rc = gate.main(["--candidate", str(cand), "--gate-dir", str(Path(d) / "g"), "--pool", str(pool)])
            self.assertEqual(rc, 1)
            self.assertIn("frozen", (Path(d) / "g" / "ledger.jsonl").read_text())


FAKE_ROUND = """
import json, sys, pathlib, numpy as np
a = sys.argv
out = pathlib.Path(a[a.index("--out") + 1]); out.mkdir(parents=True, exist_ok=True)
spec = pathlib.Path(a[a.index("--spec") + 1]).read_text()
only = a[a.index("--only") + 1].split(",")
i, n = map(int, a[a.index("--shard") + 1].split("/"))
for label in sorted(only)[i::n]:
    k = int(label.split("_s")[-1])
    ok = k < 25 if "tracking_gain: 1.0" in spec else k < 5
    if "sentinel" in str(out) and DRIFT:
        ok = not ok
    t = np.linspace(0, 1, 5)
    box = np.zeros((5, 7)); box[:, 3] = 1; box[:, 2] = 0.1 * t * ok
    pelvis = np.zeros((5, 7)); pelvis[:, 2] = 0.75
    stem = label.replace("sonic", "bfm", 1)
    np.savez(out / f"{stem}.npz", box=box, pelvis=pelvis)
    (out / f"{stem}.json").write_text(json.dumps(dict(ok=ok, why="" if ok else "reach_stalled", events=dict(end=20.0),
                                                      fields_read=["visual.tracking_gain"])))
"""


class GateRun(unittest.TestCase):
    def gate_on_fake_rounds(self, drift: bool):
        import functools

        import batch

        d = Path(tempfile.mkdtemp())
        pool = d / "pool"
        pool.mkdir()
        for k in range(gate.STAGE1 + 1):
            (pool / f"sonic_s{k:02d}.npz").touch()
        fake = d / "fake_round.py"
        fake.write_text(f"DRIFT = {drift}\n" + FAKE_ROUND)
        cand = d / "c.yaml"
        cand.write_text("set: {visual.tracking_gain: 1.0}\n")
        saved = batch.SIM_PY, gate.run_jobs
        batch.SIM_PY, gate.run_jobs = Path(sys.executable), functools.partial(batch.run_jobs, script=fake)
        try:
            rc = gate.main(["--candidate", str(cand), "--gate-dir", str(d / "g"), "--pool", str(pool), "--procs", "4"])
        finally:
            batch.SIM_PY, gate.run_jobs = saved
        import json
        import shutil

        verdict = json.loads((d / "g" / "ledger.jsonl").read_text().splitlines()[-1])
        shutil.rmtree(d)
        return rc, verdict

    def test_the_batched_sentinel_and_stage_1_accept_a_clear_winner(self):
        rc, verdict = self.gate_on_fake_rounds(drift=False)
        self.assertEqual(rc, 0, verdict["reason"])
        self.assertEqual((verdict["stage1"]["b"], verdict["stage1"]["c"]), (20, 0))
        self.assertEqual(verdict["result"]["n"], gate.STAGE1 + 1)

    def test_a_drifting_sentinel_voids_the_call_before_any_tally(self):
        rc, verdict = self.gate_on_fake_rounds(drift=True)
        self.assertEqual(rc, 1)
        self.assertIn("sentinel", verdict["reason"])
        self.assertNotIn("stage1", verdict)


class StopRule(unittest.TestCase):
    def test_phase3_when_the_incumbent_beats_sonic_by_five(self):
        import loop

        with tempfile.TemporaryDirectory() as d:
            old_state, old_sonic = loop.STATE, loop.sonic_accept_ok
            loop.STATE, loop.sonic_accept_ok = Path(d), lambda: 27
            try:
                meta = dict(started_s=__import__("time").time())
                loop.ledger(dict(step="gate", accepted=True, result=dict(candidate_ok=31)))
                self.assertIsNone(loop.stop_reason(meta))
                loop.ledger(dict(step="gate", accepted=True, result=dict(candidate_ok=32)))
                self.assertIn("phase 3", loop.stop_reason(meta))
            finally:
                loop.STATE, loop.sonic_accept_ok = old_state, old_sonic


class Fields(unittest.TestCase):
    def test_reads_are_logged_by_dotted_name(self):
        log = FieldLog()
        cfg = task_config(CONFIG, {"visual.tracking_gain": 0.25}, log)
        self.assertEqual(cfg.visual.tracking_gain, 0.25)
        _ = cfg.carry.waist_pitch_max
        self.assertIn("visual.tracking_gain", log.read)
        self.assertIn("carry.waist_pitch_max", log.read)
        self.assertNotIn("visual.walk_speed", log.read)
        self.assertEqual(cfg.visual, cfg.visual)  # equality and the dataclass interface still work


if __name__ == "__main__":
    unittest.main()
