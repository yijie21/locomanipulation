"""TODO 40.20 (gso_pack_all.md D15, user 2026-11-07): the pack round's progress stage through the pipeline -- the gate
compares versions on it, the digest and the diagnostic summary count it, the sealed pool runs with traces and sums it,
the final video replays the sealed rounds with every object in place."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_names = ("gate", "batch", "family", "scene_round", "digest", "sealed", "loop")
_saved = {k: sys.modules.pop(k, None) for k in _names}
_path = list(sys.path)
sys.path.insert(0, str(HERE))
try:
    gate = _load("pack_stage_gate", HERE / "gate.py")
    digest = _load("pack_stage_digest", HERE / "digest.py")
    sealed = _load("pack_stage_sealed", HERE / "sealed.py")
    loop = _load("pack_stage_loop", HERE / "loop.py")
    B = gate.B
finally:
    sys.path[:] = _path
    for _k, _v in _saved.items():
        sys.modules.pop(_k, None)
        if _v is not None:
            sys.modules[_k] = _v


STAGES = ("none", "picked", "over_opening", "in_container", "goal")


def rec(ok=False, stage=None, task="pack_x", **kw):
    out = dict(ok=ok, fell=False, dropped=False, why="" if ok else "[goal] in(x, c): outside", plan_why="", task=task,
               t_end=20.0, failure_class="ok" if ok else "carry_dropped", **kw)  # fmt: skip
    if stage is not None:
        out["stage"], out["stage_name"] = stage, STAGES[stage]
    return out


class StageOf(unittest.TestCase):
    def test_a_round_without_a_stage_is_judged_on_ok(self):
        self.assertEqual((gate.stage_of(rec(True)), gate.stage_of(rec(False))), (4, 0))
        self.assertEqual(gate.stage_of(rec(False, stage=2)), 2)
        self.assertEqual(gate.stage_of(rec(True, stage=4)), 4)


class GateStage(unittest.TestCase):
    def test_a_higher_stage_wins_a_lower_one_loses_the_same_ties(self):
        inc = {"pack_x_s1": rec(stage=0), "pack_x_s2": rec(stage=1), "pack_x_s3": rec(stage=3), "pack_y_s4": rec(stage=1),
               "pack_y_s5": rec(True, stage=4), "pack_y_s6": rec(stage=0)}  # fmt: skip
        cand = {"pack_x_s1": rec(stage=1), "pack_x_s2": rec(stage=3), "pack_x_s3": rec(stage=3), "pack_y_s4": rec(stage=0),
                "pack_y_s5": rec(True, stage=4), "pack_y_s6": rec(stage=2), "extra_s9": rec(stage=4)}  # fmt: skip
        for d in (cand, inc):
            for k, r in d.items():
                r["task"] = k.rsplit("_s", 1)[0]
        t = gate.tally(cand, inc)
        self.assertEqual((t["n"], t["b"], t["c"]), (6, 3, 1))
        self.assertEqual((t["candidate_ok"], t["incumbent_ok"]), (1, 1))  # successes stay stage 4 alone
        st = t["stage"]
        self.assertEqual((st["wins"], st["losses"], st["ties"]), (3, 1, 2))
        self.assertEqual((st["incumbent_mean"], st["candidate_mean"]), (1.5, 2.17))
        self.assertEqual(st["per_task"], {"pack_x": [1.33, 2.33], "pack_y": [1.67, 2.0]})
        # the class tally counts the same wins and losses: a win under the incumbent's class, a loss under the candidate's
        pc = t["per_class"]
        self.assertEqual(sum(r["b"] for r in pc.values()), 3)
        self.assertEqual(sum(r["c"] for r in pc.values()), 1)

    def test_the_rule_is_the_usual_one_on_b_and_c(self):
        inc = {f"p_s{i}": rec(stage=0) for i in range(8)}
        cand = {f"p_s{i}": rec(stage=1 if i < 3 else 0) for i in range(8)}
        t = gate.tally(cand, inc)
        self.assertEqual((t["b"], t["c"], t["candidate_ok"]), (3, 0, 0))
        self.assertEqual(gate.verdict_fails(t), [])  # three starts carried further: accepted with no success anywhere
        cand["p_s7"] = rec(stage=0)
        inc["p_s7"] = rec(stage=3)
        self.assertEqual(gate.verdict_fails(gate.tally(cand, inc)), ["net wins 2 < 3"])

    def test_without_stages_the_tally_is_the_old_one(self):
        cand = {"a": rec(True), "b": rec(True), "c": rec(False), "d": rec(True)}
        inc = {"a": rec(False), "b": rec(True), "c": rec(True), "d": rec(False)}
        t = gate.tally(cand, inc)
        self.assertEqual((t["n"], t["b"], t["c"]), (4, 2, 1))
        self.assertIsNone(t["stage"])


class ProbeAndFlipsStage(unittest.TestCase):
    """The Prober and a candidate's diagnostic flips count fixed / broken as the gate does (stage_of), else a hypothesis
    that carries starts along the chain without finishing one would be refuted while the gate accepts it."""

    def _dirs(self):
        d = Path(tempfile.mkdtemp())
        inc, cand = d / "inc", d / "cand"
        inc.mkdir(), cand.mkdir()
        rows = {"p_s1": (0, 1), "p_s2": (1, 1), "p_s3": (3, 1), "p_s4": (0, 4), "q_s5": (None, None)}
        for k, (a, b) in rows.items():
            (inc / f"{k}.json").write_text(json.dumps(rec(a == 4, stage=a) if a is not None else rec(False)))
            (cand / f"{k}.json").write_text(json.dumps(rec(b == 4, stage=b) if b is not None else rec(True)))
        return d, inc, cand

    def test_flips_follow_the_stage(self):
        _, inc, cand = self._dirs()
        f = loop.flips(inc, cand)
        self.assertEqual((f["n"], f["ok"], f["incumbent_ok"]), (5, 2, 0))
        self.assertEqual((f["fixed"], f["broken"]), (["p_s1", "p_s4", "q_s5"], ["p_s3"]))

    def test_the_probe_verifies_a_hypothesis_that_carries_starts_further(self):
        d, inc, cand = self._dirs()
        real = loop.B.run_jobs
        loop.B.run_jobs = lambda *a, **k: {}
        it_dir = d / "iter"
        (it_dir / "probe").mkdir(parents=True)
        (it_dir / "probe" / "H1").symlink_to(cand)
        h = dict(id="H1", delta=dict(set={"x": 1}), predicted_fixed=["p_s1", "p_s2"], predicted_broken=[])
        try:
            res = loop.probe([h], dict(dir=str(d), set={}), inc, it_dir)
        finally:
            loop.B.run_jobs = real
        r = res[0]
        self.assertEqual(
            (r["ok_before"], r["ok_after"], r["fixed"], r["broken"]), (0, 2, ["p_s1", "p_s4", "q_s5"], ["p_s3"])
        )
        self.assertEqual((r["predicted_fixed_hit"], r["predicted_fixed_n"]), (1, 2))
        self.assertEqual(loop.probe_status(h, r), "mixed")
        r2 = dict(r, broken=[])
        self.assertEqual(loop.probe_status(h, r2), "verified")


class DigestStage(unittest.TestCase):
    def test_the_stage_line_counts_the_rounds_at_each_stage(self):
        rows = [rec(stage=0), rec(stage=1), rec(stage=1), rec(stage=3), rec(True, stage=4)]
        lines = digest.stage_line(rows)
        self.assertEqual(lines[0], "Stages reached: goal 1, in_container 1, picked 2, none 1 (mean 1.80 of 4)")
        self.assertEqual(digest.stage_line([rec(), rec(True)]), [])


class SummaryStage(unittest.TestCase):
    def test_the_diagnostic_summary_carries_the_stage_distribution(self):
        d = Path(tempfile.mkdtemp())
        rows = {"pack_x_s1": rec(stage=1), "pack_x_s2": rec(stage=3), "pack_y_s3": rec(True, stage=4),
                "pack_y_s4": rec(stage=1)}  # fmt: skip
        for k, r in rows.items():
            r["task"] = k.rsplit("_s", 1)[0]
            (d / f"{k}.json").write_text(json.dumps(r))
        s = loop.summarize(d)
        self.assertEqual((s["ok"], s["n"]), (1, 4))
        self.assertEqual(s["stages"], {"picked": 2, "in_container": 1, "goal": 1})
        self.assertEqual(s["per_task_stage"], {"pack_x": 2.0, "pack_y": 2.5})
        (d / "pick_z_s5.json").write_text(json.dumps(dict(rec(), task="pick_z")))
        self.assertEqual(loop.summarize(d)["stages"], {"picked": 2, "in_container": 1, "goal": 1})
        for p in d.glob("*.json"):
            p.unlink()
        (d / "pick_z_s5.json").write_text(json.dumps(dict(rec(), task="pick_z")))
        self.assertNotIn("stages", loop.summarize(d))


class SealedTraces(unittest.TestCase):
    def test_the_sealed_pool_runs_with_traces_and_sums_the_stages(self):
        tmp = Path(tempfile.mkdtemp())
        hidden = tmp / "outputs" / "scene_rsi_20261016_hidden" / "fam"
        pool = hidden / "pool_sealed"
        pool.mkdir(parents=True)
        for i in range(4):
            (pool / f"pack_x_s{i}.json").write_text("{}")
        seed = hidden / "gate" / "seed"
        (seed / "skills").mkdir(parents=True)
        (seed / "candidate.json").write_text('{"set": {}}')
        seen = []

        def fake(jobs, family, pool_, procs, log, gpus="1", stop=None):
            for job in jobs:
                seen.append(job.npz)
                job.out.mkdir(parents=True, exist_ok=True)
                for i in range(4):
                    (job.out / f"pack_x_s{i}.json").write_text(json.dumps(rec(i == 3, stage=min(i + 1, 4))))
            return {}

        real = (sealed.B.run_jobs, sealed.versions, sealed.REPO, sealed.body.score)
        sealed.B.run_jobs = fake
        sealed.versions = lambda gate_dir: [("seed", seed)]
        sealed.REPO = tmp
        sealed.body.score = lambda rounds: {}
        try:
            rc = sealed.main(["fam"])
        finally:
            sealed.B.run_jobs, sealed.versions, sealed.REPO, sealed.body.score = real
        self.assertEqual(rc, 0)
        self.assertEqual(seen, [True])
        summary = json.loads((hidden / "sealed" / "summary.json").read_text())
        self.assertEqual(summary["stages"], {"seed": {"1": 1, "2": 1, "3": 1, "4": 1}})
        self.assertEqual(summary["final_rate"], 0.25)  # stage 4 alone


class MediaRounds(unittest.TestCase):
    def test_rounds_replaces_the_final_versions_rounds_and_shows_it_alone(self):
        sys.path.insert(0, str(HERE / "report"))
        try:
            import evolution_media as EM
        finally:
            sys.path.pop(0)
        tmp = Path(tempfile.mkdtemp())
        (tmp / "scripts" / "rsi" / "families").mkdir(parents=True)
        (tmp / "scripts" / "rsi" / "families" / "fam.json").write_text(json.dumps(dict(scene="s")))
        sealed_rounds = tmp / "sealed" / "final"
        sealed_rounds.mkdir(parents=True)
        got = {}

        def fake_videos(sc, vs, out_dir, names, final_only=False):
            got.update(vs=vs, final_only=final_only)
            return []

        real = (EM.versions, EM.task_videos, EM.V.Scene)
        EM.versions = lambda loop_: [
            dict(name="种子", hash="a", diag=tmp / "d0"),
            dict(name="第 1 版", hash="b", diag=tmp / "d1"),
        ]
        EM.task_videos = fake_videos
        EM.V.Scene = lambda path: None
        try:
            EM.main(["--sandbox", str(tmp), "--family", "fam", "--out", str(tmp / "out"), "--only", "task",
                     "--rounds", str(sealed_rounds)])  # fmt: skip
            self.assertEqual([v["hash"] for v in got["vs"]], ["b"])
            self.assertEqual(got["vs"][0]["diag"], sealed_rounds.resolve())
            self.assertTrue(got["final_only"])
            EM.main(["--sandbox", str(tmp), "--family", "fam", "--out", str(tmp / "out"), "--only", "task"])
            self.assertEqual([v["hash"] for v in got["vs"]], ["a", "b"])
            self.assertFalse(got["final_only"])
        finally:
            EM.versions, EM.task_videos, EM.V.Scene = real


class SceneSet(unittest.TestCase):
    def test_every_traced_object_is_placed_and_unknown_bodies_are_skipped(self):
        import mujoco

        sys.path.insert(0, str(HERE / "report"))
        try:
            import visualize as V
        finally:
            sys.path.pop(0)
        xml = """<mujoco><worldbody>
          <body name="pelvis"><freejoint/><geom type="sphere" size="0.1"/></body>
          <body name="x" pos="1 0 0"><freejoint/><geom type="box" size="0.03 0.03 0.05"/></body>
          <body name="c" pos="2 0 0"><freejoint/><geom type="box" size="0.2 0.15 0.15"/></body>
        </worldbody></mujoco>"""
        sc = V.Scene.__new__(V.Scene)
        sc.m = mujoco.MjModel.from_xml_string(xml)
        sc.d = mujoco.MjData(sc.m)
        sc.free, sc.joints, sc.grip = 0, [], {"left": [], "right": []}
        q = lambda x, y, z: np.array([x, y, z, 1.0, 0, 0, 0])  # noqa: E731
        z = dict(t=np.array([0.0, 1.0]), robot=np.stack([q(0, 0, 1), q(0.5, 0, 1)]), joints=np.zeros((2, 0)),
                 jaw=np.zeros((2, 2)), obj_x=np.stack([q(1, 0, 0), q(1, 1, 0.5)]), obj_c=np.stack([q(2, 0, 0), q(2, 2, 0)]),
                 obj_gone=np.stack([q(9, 9, 9), q(9, 9, 9)]))  # fmt: skip
        r = SimpleNamespace(z=z, key="obj_x")
        sc.set(r, 1)
        self.assertTrue(np.allclose(sc.d.xpos[sc.m.body("x").id], [1, 1, 0.5]))
        self.assertTrue(np.allclose(sc.d.xpos[sc.m.body("c").id], [2, 2, 0]))  # the carton moves too, not only the key
        self.assertTrue(np.allclose(sc.d.xpos[sc.m.body("pelvis").id], [0.5, 0, 1]))


if __name__ == "__main__":
    unittest.main()
