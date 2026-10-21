"""TODO 32 (the scene loop made faster): batch.py's round queue hands every round to exactly one worker; the scene's
compiled-model cache is reused while the XML and the files it names are unchanged, and rebuilt when one changes;
replay.py finds a gate call's iteration from its candidate path and rebuilds the parks of a run without snapshots."""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# the warehouse loop has its own batch.py: load this loop's under its own name, leave sys.modules as it was
_saved = {k: sys.modules.pop(k, None) for k in ("batch", "sandbox")}
_path = list(sys.path)
sys.path.insert(0, str(HERE))
try:
    batch = _load("scene_rsi_batch", HERE / "batch.py")
    replay = _load("scene_rsi_replay", HERE / "replay.py")
finally:
    sys.path[:] = _path
    for _k, _v in _saved.items():
        sys.modules.pop(_k, None)
        if _v is not None:
            sys.modules[_k] = _v


class RoundQueue(unittest.TestCase):
    def test_every_round_goes_to_one_taker_in_job_then_label_order(self):
        with tempfile.TemporaryDirectory() as d:
            jobs = [batch.Job(Path(d) / "a", Path(d) / "out_a", {"X": 1}), batch.Job(Path(d) / "b", Path(d) / "out_b")]
            q = batch.make_queue(jobs, {0: ["s1", "s2"], 1: ["s1"]}, Path(d))
            self.assertEqual(batch.queued(q), 3)
            spec = json.loads((q / "jobs.json").read_text())
            self.assertEqual(spec[0]["overrides"], {"X": 1})
            self.assertEqual(spec[1]["overrides"], {})
            taken = [batch.claim_round(q) for _ in range(4)]
            self.assertEqual(taken[:3], [dict(job=0, label="s1"), dict(job=0, label="s2"), dict(job=1, label="s1")])
            self.assertIsNone(taken[3])
            self.assertEqual(batch.queued(q), 0)
            self.assertEqual(len(list((q / "taken").iterdir())), 3)

    def test_a_round_another_worker_took_first_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            q = batch.make_queue([batch.Job(Path(d), Path(d))], {0: ["s1", "s2"]}, Path(d))
            first = sorted((q / "todo").iterdir())[0]
            os.rename(first, q / "taken" / "other_worker.json")  # another process got there first
            self.assertEqual(batch.claim_round(q), dict(job=0, label="s2"))


class ModelCache(unittest.TestCase):
    XML = '<mujoco><asset><mesh name="m" file="m.stl"/></asset><worldbody><geom type="box" size="{s} .1 .1"/></worldbody></mujoco>'

    def setUp(self):
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco not installed")
        from locomani.backends.scalebfm import runtime

        self.runtime = runtime
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache"
        import locomani.paths as P

        self._old, P.MJB_CACHE = P.MJB_CACHE, str(self.cache)

    def tearDown(self):
        import locomani.paths as P

        P.MJB_CACHE = self._old
        self.tmp.cleanup()

    def _scene(self, d: Path, size: str) -> str:
        import struct

        tri = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]  # a tetrahedron, binary STL
        faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
        data = b"\0" * 80 + struct.pack("<I", len(faces))
        for f in faces:
            data += struct.pack("<3f", 0, 0, 0) + b"".join(struct.pack("<3f", *tri[i]) for i in f) + b"\0\0"
        (d / "m.stl").write_bytes(data)
        (d / "s.xml").write_text(self.XML.format(s=size))
        return str(d / "s.xml")

    def test_cache_is_used_until_the_xml_or_a_named_file_changes(self):
        d = Path(self.tmp.name) / "scene"
        d.mkdir()
        xml = self._scene(d, ".1")
        m1 = self.runtime.load_model(xml)
        self.assertEqual(len(list(self.cache.glob("*.mjb"))), 1)
        self.assertEqual(sorted(p.name for p in d.iterdir()), ["m.stl", "s.xml"])  # nothing beside the XML
        m2 = self.runtime.load_model(xml)  # from the cache
        self.assertEqual(m1.geom_size.tolist(), m2.geom_size.tolist())
        self.assertEqual(len(list(self.cache.glob("*.mjb"))), 1)
        Path(xml).write_text(self.XML.format(s=".2"))
        self.assertAlmostEqual(float(self.runtime.load_model(xml).geom_size[0][0]), 0.2)
        self.assertEqual(len(list(self.cache.glob("*.mjb"))), 2)
        os.utime(d / "m.stl", ns=(time.time_ns(), time.time_ns() + 10**9))  # a named file touched
        self.runtime.load_model(xml)
        self.assertEqual(len(list(self.cache.glob("*.mjb"))), 3)


class ReplayState(unittest.TestCase):
    def test_iteration_of_a_gate_call(self):
        self.assertEqual(replay.iter_of_candidate("/s/outputs/x/fam/loop/iter_03/candidate"), 3)
        self.assertIsNone(replay.iter_of_candidate("/s/outputs/x/fam/gate/incumbent_seed"))

    def test_parks_before_an_iteration_with_the_unparks_applied(self):
        claim = dict(task="pick_Mug", kind="current_skills", starts=["pick_Mug_s2", "pick_Mug_s1"],
                     bound=dict(statement="too wide"), actions_assumed="side grasp", refuted_by="a mug lifted")  # fmt: skip
        with tempfile.TemporaryDirectory() as d:
            loop = Path(d)
            for k in range(4):
                (loop / f"iter_{k:02d}").mkdir()
            (loop / "iter_01" / "hypotheses.json").write_text(json.dumps(dict(hypotheses=[], infeasible=[claim])))
            (loop / "iter_01" / "claims.json").write_text(json.dumps([dict(id="C1", parked=True, scope="task")]))
            (loop / "ledger.jsonl").write_text(json.dumps(dict(step="unpark", iter=3, claim="C1")) + "\n")
            self.assertEqual(replay.rebuild_parks(loop, 1), {})
            parks = replay.rebuild_parks(loop, 3)
            self.assertEqual(sorted(parks), ["C1"])
            self.assertEqual(parks["C1"]["starts"], ["pick_Mug_s1", "pick_Mug_s2"])
            self.assertEqual(parks["C1"]["iter"], 1)
            self.assertEqual(replay.rebuild_parks(loop, 4), {})  # unparked in iteration 3


if __name__ == "__main__":
    unittest.main()
