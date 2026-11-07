"""TODO 33.8, 33.13, 33.14 (user 2026-10-22: "选 B，改完直接开跑吧"): the relaxed gate (b - c >= 3 and c <= b / 3, no
p; a candidate rejected under the older rule may be judged again), the candidate pool (keep, rank, clear, rebuild,
`base`, the Composer's `adopted_pool`), the Manager's repeats and the reopen limit, and the object-name lint."""

import json
import os
import sys
import tempfile
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


def _isolated(name: str, path: Path):
    """A scripts/rsi module under its own name (the warehouse loop has a loop.py, gate.py, batch.py too)."""
    names = ("gate", "batch", "family", "scene_round", "digest", "loop")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return _load(name, path)
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _tally(b, c, **kw):
    t = dict(n=40, candidate_ok=20 + b, incumbent_ok=20 + c, b=b, c=c, p_one_sided=0.5, candidate_fell=0,
             incumbent_fell=0, candidate_dropped=0, incumbent_dropped=0)  # fmt: skip
    t.update(kw)
    return t


def _cand(d: Path, marker: str = "") -> Path:
    (d / "skills").mkdir(parents=True, exist_ok=True)
    (d / "skills" / "__init__.py").write_text(f"A = 1\n{marker}\n")
    (d / "candidate.json").write_text('{"set": {}}')
    return d


class RelaxedGate(unittest.TestCase):
    def setUp(self):
        self.gate = _isolated("rsi_relaxed_gate_gate", HERE / "gate.py")

    def test_verdict_is_net_three_and_losses_at_most_a_third(self):
        ok = lambda b, c: not self.gate.verdict_fails(_tally(b, c))  # noqa: E731
        for b, c in ((5, 1), (4, 0), (3, 0), (6, 2)):
            self.assertTrue(ok(b, c), (b, c))
        for b, c in ((3, 1), (2, 0), (6, 3)):
            self.assertFalse(ok(b, c), (b, c))
        self.assertIn("losses 3 > wins 6 / 3", "; ".join(self.gate.verdict_fails(_tally(6, 3))))
        self.assertTrue(self.gate.verdict_fails(_tally(5, 0, candidate_fell=2)))  # falls and drops as before
        self.assertTrue(self.gate.verdict_fails(_tally(5, 0, candidate_dropped=2)))

    def _dry(self, history: list[dict]) -> str:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            pool = tmp / "pool"
            pool.mkdir()
            for i in range(31):
                (pool / f"t_s{i}.json").write_text("{}")
            cand = tmp / "cand"
            (cand / "skills").mkdir(parents=True)
            (cand / "skills" / "__init__.py").write_text("def apply_set(pkg, ov):\n    return []\n")
            (cand / "candidate.json").write_text('{"set": {}}')
            g = tmp / "gate"
            g.mkdir()
            h = self.gate.candidate_hash(cand)
            (g / "ledger.jsonl").write_text("".join(json.dumps(dict(x, hash=h)) + "\n" for x in history))
            self.gate.main(["f", "--gate", str(g), "--pool", str(pool), "--candidate", str(cand), "--dry-run"])
            last = json.loads((g / "ledger.jsonl").read_text().splitlines()[-1])
        self.assertEqual(last["rule"], self.gate.GATE_RULE)
        return last["reason"]

    def test_a_candidate_rejected_under_the_older_rule_may_be_judged_again(self):
        self.assertIn("dry run", self._dry([dict(accepted=False, reason="p 0.11 > 0.05", call=3)]))  # no rule: rule 1
        self.assertIn("dry run", self._dry([dict(accepted=False, void=True, reason="pairing incomplete", rule=2)]))
        self.assertIn("rejected before", self._dry([dict(accepted=False, reason="net wins 2 < 3", rule=2)]))


class _LoopCase(unittest.TestCase):
    def setUp(self):
        from unittest import mock

        self.lp = _isolated("rsi_relaxed_gate_loop", HERE / "loop.py")
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = Path(self.d.name)
        hid = self.root / "outputs" / "scene_rsi_20261016_hidden" / "fam"  # never HIDDEN at Path() (TODO 36)
        hid.mkdir(parents=True)
        (self.root / "state").mkdir()
        self.lp.FAMILY = "fam"
        for p in (mock.patch.multiple(self.lp, STATE=self.root / "state", LOOP=self.root, HIDDEN=hid,
                                      GATE_DIR=hid / "gate", accept_pool=lambda: self.root, redraw_accept=lambda: None,
                                      unparked_view=lambda r: None, put_skills=lambda src: None,
                                      revert_strays=lambda base: [], changed_files=lambda: [],
                                      transcript_leaks=lambda s: []),
                  mock.patch.dict(os.environ)):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)
        (hid / "gate").mkdir()

    def verdict(self, h, call, b, c, accepted=False, rule=2, **kw):
        return dict(hash=h, call=call, rule=rule, accepted=accepted, reason="x", result=_tally(b, c, **kw))


class CandidatePool(_LoopCase):
    def test_keep_rank_and_clear(self):
        lp = self.lp
        for h, call, b, c, kw in (("a1", 2, 3, 1, {}), ("a2", 3, 5, 1, {}), ("a3", 4, 3, 3, {}),
                                  ("a4", 5, 6, 0, dict(candidate_fell=2)), ("a5", 6, 5, 1, {})):  # fmt: skip
            lp.pool_update(call, _cand(self.root / h), self.verdict(h, call, b, c, **kw), False)
        pool = lp.pool_load()
        self.assertEqual([e["hash"] for e in pool], ["a5", "a2"])  # b - c 4 twice, the most recent first; a3, a4 out
        self.assertTrue(Path(pool[0]["dir"], "skills", "__init__.py").exists())
        lp.pool_update(7, _cand(self.root / "a6"), self.verdict("a6", 7, 9, 0), True)  # a new incumbent
        self.assertEqual(lp.pool_load(), [])
        rows = [e for e in lp.entries() if e["step"] == "pool"]
        self.assertEqual(rows[-1]["cleared"], ["a5", "a2"])

    def test_rebuild_from_the_gate_ledger_since_the_incumbent(self):
        lp = self.lp
        g = lp.GATE_DIR
        lines = [self.verdict("i1", 1, 21, 0, accepted=True, rule=1), self.verdict("r2", 2, 3, 1, rule=1),
                 self.verdict("r3", 3, 5, 1, rule=1), self.verdict("r4", 4, 5, 1, rule=1, candidate_fell=1),
                 self.verdict("r5", 5, 5, 1, rule=1)]  # fmt: skip
        for v in lines:
            v["candidate"] = str(self.root / "loop" / f"iter_{v['call']:02d}" / "candidate")
            _cand(g / f"{v['call']:02d}_{v['hash']}" / "cand")
        (g / "ledger.jsonl").write_text("".join(json.dumps(v) + "\n" for v in lines))
        (g / "incumbent.json").write_text(json.dumps(dict(dir="x", rounds="y", since=1)))
        kept = lp.rebuild_pool()
        self.assertEqual([(e["hash"], e["iter"]) for e in kept], [("r5", 5), ("r4", 4)])
        view = lp.pool_view(dict(dir=str(_cand(self.root / "inc"))), None)
        self.assertTrue(all(v["older_rule"] for v in view))
        self.assertTrue(Path(view[0]["patch"]).exists())


class PoolBase(_LoopCase):
    """`base`: the Engineers' working copies and the Composer's start from the pool candidate; `adopted_pool` sends a
    pool candidate as it stands; a draft that names an object is void."""

    def _iteration(self, comp: dict, engineer_names: dict | None = None):
        from unittest import mock

        lp = self.lp
        lp.pool_update(2, _cand(self.root / "pooled", "BASE = 1"), self.verdict("p1", 2, 5, 1, rule=1), False)
        inc = dict(dir=str(_cand(self.root / "incdir")), hash="inc0", set={})
        it = self.root / "state" / "iter_03"
        it.mkdir(parents=True)
        hyps = dict(hypotheses=[dict(id="H1", claim="c", delta={"set": {}})])
        sent = []

        def fake_claude(role, prompt, schema, it_dir, tools, stub=None, stop=None):  # stop: TODO 37.3
            if role.startswith("engineer_"):
                k = int(role.split("_")[1])
                w = it_dir / f"cand_{k}" / "skills"
                self.assertIn("BASE = 1", (w / "__init__.py").read_text())  # started from the pool candidate
                ctx = json.loads((it_dir / f"cand_{k}" / "engineer_context.json").read_text())
                self.assertEqual(ctx["base"]["hash"], "p1")
                (w / "__init__.py").write_text((w / "__init__.py").read_text() + f"K = {k}\n")
                if engineer_names and k in engineer_names:
                    (w / "named.py").write_text(f"X = {engineer_names[k]!r}\n")
                return dict(mechanism=f"m{k}", files_changed=[], set={}, tests=""), dict(cost_usd=0.0, session=None)
            self.assertIn("BASE = 1", (it_dir / "compose" / "skills" / "__init__.py").read_text())
            return comp, dict(cost_usd=0.0, session=None)

        with mock.patch.multiple(lp, claude=fake_claude, CANDIDATES=3,
                                 gate_candidate=lambda n, it_dir, cand, stub, reopen=None: sent.append(cand)),\
             mock.patch.object(lp.B, "run_jobs", lambda *a, **k: {}),\
             mock.patch.object(lp, "object_names", lambda: {"Mug_2", "Mug"}):  # fmt: skip
            lp.multi_engineer_step(3, it, inc, hyps, [], dict(decision="engineer", hypotheses=["H1"], base="p1"),
                                   {}, [], False)  # fmt: skip
        return sent

    def test_engineers_and_composer_start_from_the_base(self):
        sent = self._iteration(dict(decision="adopt", adopted=1, notes=""))
        self.assertEqual(len(sent), 1)
        self.assertIn("K = 1", (sent[0] / "skills" / "__init__.py").read_text())
        rows = [e for e in self.lp.entries() if e["step"] == "engineer"]
        self.assertTrue(all(e.get("base") == "p1" for e in rows))

    def test_composer_adopts_a_pool_candidate_as_it_stands(self):
        sent = self._iteration(dict(decision="adopt", adopted_pool="p1", notes=""))
        self.assertEqual(sent, [Path(self.lp.pool_entry("p1")["dir"])])

    def test_a_draft_that_names_an_object_is_void(self):
        self._iteration(dict(decision="adopt", adopted=1, notes=""), engineer_names={2: "Mug_2"})
        rows = {e["candidate"]: e for e in self.lp.entries() if e["step"] == "engineer"}
        self.assertTrue(rows[2]["void"])
        self.assertEqual(rows[2]["named_objects"], ["named.py:1 'Mug_2'"])
        self.assertFalse(rows[1]["void"])


class ManagerRules(_LoopCase):
    def test_repeats_link_ids_and_built_mechanisms_drop_out(self):
        lp = self.lp
        (lp.STATE / "iter_05").mkdir()
        lp.jdump(lp.STATE / "iter_05" / "hypotheses.json", dict(hypotheses=[dict(id="H1", claim="soap narrow side")]))
        es = [
            dict(step="manager", iter=1, decision="engineer", hypotheses=["H1", "H3"]),
            dict(step="manager", iter=2, decision="engineer", hypotheses=["H1"],
                 repeats=[dict(now="H3", same_as=["1:H2"]), dict(now="H4", same_as=["01:H3"])]),
            dict(step="manager", iter=5, decision="reopen", repeats=[dict(now="H1", same_as=["2:H3", "iter 1:H2"])]),
            dict(step="manager", iter=6, decision="engineer", hypothesis="H9", void=True,
                 repeats=[dict(now="H2", same_as=["05:H1"])]),
        ]  # fmt: skip
        rec = lp.recurring(es)
        self.assertEqual(len(rec), 1)  # 01:H3 was built, so its mechanism is not listed; the void row adds nothing
        self.assertEqual(
            (rec[0]["proposed"], rec[0]["ids"], rec[0]["latest"]), (3, ["01:H2", "02:H3", "05:H1"], "05:H1")
        )
        self.assertEqual(rec[0]["claim"], "soap narrow side")

    def test_no_third_reopen_in_a_row(self):
        lp = self.lp
        m = lambda n, d, **kw: dict(step="manager", iter=n, decision=d, **kw)  # noqa: E731
        self.assertTrue(lp.reopen_allowed([m(3, "reopen")]))
        self.assertFalse(lp.reopen_allowed([m(3, "reopen"), m(4, "reopen")]))
        self.assertTrue(lp.reopen_allowed([m(3, "reopen"), m(4, "reopen"), m(5, "engineer")]))
        self.assertFalse(lp.reopen_allowed([m(3, "reopen"), m(4, "engineer", void=True), m(4, "reopen")]))
        self.assertNotIn("reopen", lp.manager_schema(False)["properties"]["decision"]["enum"])
        self.assertIn("reopen", lp.manager_schema(True)["properties"]["decision"]["enum"])

    def test_a_blocked_reopen_ends_the_iteration_and_the_context_says_so(self):
        from unittest import mock

        lp = self.lp
        for n in (3, 4):
            lp.ledger(dict(step="manager", iter=n, decision="reopen"))
        it = self.root / "state" / "iter_05"
        it.mkdir()
        seen = {}

        def fake_claude(role, prompt, schema, it_dir, tools, stub=None, stop=None):  # stop: TODO 37.3
            seen["schema"] = schema
            seen["ctx"] = json.loads((it_dir / "manager_context.json").read_text())
            return dict(decision="reopen", reopen_claim="C1", reason="r", summary_zh="z"), dict(
                cost_usd=0.0, session=None
            )

        with mock.patch.multiple(lp, claude=fake_claude, probe=lambda *a: [], gap_candidates=lambda: {},
                                 engineer_step=lambda *a: self.fail("no Engineer after a blocked reopen")):  # fmt: skip
            why = lp.after_review(5, it, dict(dir=str(_cand(self.root / "inc")), hash="h", set={}), self.root, [],
                                  dict(hypotheses=[]), {}, {}, False)  # fmt: skip
        self.assertIsNone(why)
        self.assertEqual(list(seen["ctx"])[:2], ["recurring", "reopen_blocked"])
        self.assertNotIn("reopen", seen["schema"]["properties"]["decision"]["enum"])
        self.assertEqual(lp.entries()[-1]["step"], "manager_invalid")


class NameLint(_LoopCase):
    def test_literals_void_docstrings_do_not(self):
        sk = self.root / "sk"
        (sk / "recipes").mkdir(parents=True)
        (sk / "__init__.py").write_text(
            '"""Built for Mug_2 (a tub)."""\n\ndef f(key):\n    """Mug_2 again."""\n    return 1\n'
        )
        (sk / "recipes" / "cup.py").write_text('def match(o, key=""):\n    return key == "Mug_2" or f"{key}" == "x"\n')
        (sk / "recipes" / "stem.py").write_text('def g(key):\n    return f"pre{key}" if "Mug" in key else None\n')
        names = {"Mug_2", "Mug_2_object", "Mug"}
        self.assertEqual(self.lp.named_objects(sk, names), ["recipes/cup.py:2 'Mug_2'", "recipes/stem.py:2 'Mug'"])

    def test_reading_the_recorder_voids_the_draft(self):
        """TODO 39 (rules.md 38): importing the PAST builder or naming its files is as void as an object name."""
        sk = self.root / "sk2"
        sk.mkdir(parents=True)
        (sk / "__init__.py").write_text(
            "from locomani.tasks.scene import past\nimport locomani.tasks.scene.past as p2\n"
            'def f(label):\n    return open(f"{label}.past.jsonl")\n'
        )
        (sk / "clean.py").write_text('def g():\n    return "the past is another country"\n')
        hits = self.lp.named_objects(sk, set())
        self.assertEqual(len(hits), 3, hits)
        self.assertTrue(all(h.endswith(self.lp.RECORDER_TAG) for h in hits))
        self.assertTrue(any("import locomani.tasks.scene" in h for h in hits))
        self.assertTrue(any("past.jsonl" in h for h in hits))

    def test_object_names_keep_semantic_stems_allowed(self):
        from unittest import mock

        scene = self.root / "scene"
        scene.mkdir()
        objs = {
            "Mug_2": dict(body="Mug_2_object", semantic=dict(adt=dict(instance_name="Mug_2", prototype_name="WhiteMug",
                                                                       category="cup"))),
            "Orange_A": dict(body="Orange_A_object", semantic=dict(adt=dict(prototype_name="Orange", category="food"))),
        }  # fmt: skip
        (scene / "scene_meta.json").write_text(json.dumps(dict(objects=objs)))
        with mock.patch.object(self.lp, "family_spec", lambda: dict(scene=str(scene))):
            names = self.lp.object_names()
        self.assertEqual(names, {"Mug_2", "Mug_2_object", "Mug", "Orange_A", "Orange_A_object"})

    def test_object_names_keep_record_field_stems_allowed(self):
        from unittest import mock

        scene = self.root / "scene"
        scene.mkdir()
        objs = {"gso_0050": dict(body="gso_0050", gso=dict(id=50), semantic=dict(mass_kg=0.2))}
        (scene / "scene_meta.json").write_text(json.dumps(dict(objects=objs)))
        with mock.patch.object(self.lp, "family_spec", lambda: dict(scene=str(scene))):
            names = self.lp.object_names()
        self.assertEqual(names, {"gso_0050"})


if __name__ == "__main__":
    unittest.main()
