"""TODO 56.2 (user 2026-11-27, main's TODO 54 F8): under `compose: program` the loop folds the positive drafts into
the gate's candidate by a 3-way merge (scripts/rsi/merge.py) and calls the Composer only on a conflict or when no
draft is positive; the compose ledger row says who decided; `--compose` / the family file / a resume set the mode."""

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"
BASE = "".join(f"line {i}\n" for i in range(12))


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _scene_loop():
    """scripts/rsi/loop.py under its own name (the warehouse loop has a loop.py too), sys.modules left as it was."""
    names = ("gate", "batch", "family", "scene_round", "digest", "loop", "merge", "verdict")
    saved = {k: sys.modules.pop(k, None) for k in names}
    sys.path.insert(0, str(HERE))
    try:
        return _load("scene_rsi_loop_56", HERE / "loop.py")
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def edit(text: str, i: int, to: str) -> str:
    lines = text.splitlines(keepends=True)
    lines[i] = to + "\n"
    return "".join(lines)


class ProgramCompose(unittest.TestCase):
    def setUp(self):
        self.lp = _scene_loop()
        self.d = tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.root = Path(self.d.name)
        (self.root / "pool_diag").mkdir()
        hid = self.root / "outputs" / "scene_rsi_20261016_hidden" / "fam"
        hid.mkdir(parents=True)
        for p in (mock.patch.multiple(self.lp, STATE=self.root / "state", LOOP=self.root, HIDDEN=hid,
                                      GATE_DIR=hid / "gate", accept_pool=lambda: self.root),):  # fmt: skip
            p.start()
            self.addCleanup(p.stop)
        (self.root / "state").mkdir()

    def compose(self, drafts: dict, flips: dict, mode: str = "program", composer: dict | None = None) -> dict:
        """multi_engineer_step with three Engineers whose drafts are `drafts` {k: (text of __init__.py, set)} and
        whose diagnostic flips are `flips` {k: (fixed, broken)}; the Composer, when called, answers `composer`.
        Returns the ledger rows, the Composer's prompts and contexts, the working copy as the Composer saw it, the
        candidate the gate was given and the iteration directory."""
        lp = self.lp
        it_dir = lp.STATE / "iter_00"
        it_dir.mkdir(parents=True)
        inc_dir = self.root / "inc"
        (inc_dir / "skills").mkdir(parents=True)
        (inc_dir / "skills" / "__init__.py").write_text(BASE)
        (inc_dir / "candidate.json").write_text('{"set": {}}')
        inc = dict(dir=str(inc_dir), hash=lp.candidate_hash(inc_dir), set={})
        hyps = dict(hypotheses=[dict(id="H1", claim="c", instead="i", delta={"set": {}})])
        dec = dict(decision="engineer", hypotheses=["H1", "H1", "H1"], reason="r")
        seen = dict(rows=[], prompts=[], contexts=[], wc=None, gated=[], it_dir=it_dir)

        def fake_claude(role, prompt, schema, it_dir_, tools, stub=None, stop=None):
            if role.startswith("engineer_"):
                k = int(role.split("_")[1])
                text, st = drafts[k]
                (it_dir / f"cand_{k}" / "skills" / "__init__.py").write_text(text)
                return (dict(mechanism=f"m{k}", files_changed=["__init__.py"], set=st, tests="t"),
                        dict(cost_usd=1.0, session=f"s{k}", seconds=1))  # fmt: skip
            seen["prompts"].append(prompt)
            seen["contexts"].append(json.loads((it_dir / "composer_context.json").read_text()))
            seen["wc"] = (it_dir / "compose" / "skills" / "__init__.py").read_text()
            (it_dir / "composer_raw.json").write_text("{}")
            return (composer or dict(decision="adopt", adopted=1, notes="n", mechanism="m1"),
                    dict(cost_usd=2.0, session="sc", seconds=1))  # fmt: skip

        def fake_flips(inc_diag, cand_diag):
            h = Path(cand_diag).name.split("_", 1)[1]
            k = next(k for k in drafts if lp.candidate_hash(it_dir / f"candidate_{k}") == h)
            fixed, broken = flips[k]
            return dict(ok=fixed, incumbent_ok=0, n=10, fixed=[f"f{i}" for i in range(fixed)],
                        broken=[f"b{i}" for i in range(broken)])  # fmt: skip

        with (
            mock.patch.multiple(lp, FAMILY="fam", CANDIDATES=3, ENGINEER_LATE_MIN=0, SECOND_SHOT=False, COMPOSE=mode,
                                claude=fake_claude, ledger=seen["rows"].append, transcript_leaks=lambda s, samples=None: [],
                                revert_strays=lambda b: [], put_skills=lambda p: None,
                                named_objects=lambda s, names=None: [], context=lambda inc_, extra, compact=True: extra,
                                pool_view=lambda *a, **k: [], flips=fake_flips,
                                gate_candidate=lambda n, it, cand, stub, **k: seen["gated"].append(cand)),
            mock.patch.object(lp.B, "run_jobs", lambda *a, **k: {}),
        ):  # fmt: skip
            self.assertIsNone(lp.multi_engineer_step(0, it_dir, inc, hyps, [], dec, dict(
                RULES="r", PLAN="p", SIM_PY="py", POOL="pool", FAMILY="fam", FAMILY_FILE="ff", N_DIAG=30, LIBRARY="",
                DIAG_INC="d", TMP="t"), [], stub=False))  # fmt: skip
        return seen

    def test_two_clean_positive_drafts_are_merged_without_the_composer(self):
        s = self.compose({1: (edit(BASE, 1, "A"), {"p": 1}), 2: (edit(BASE, 9, "B"), {"q": 2}),
                          3: (edit(BASE, 5, "C"), {})}, {1: (3, 0), 2: (1, 0), 3: (0, 2)})  # fmt: skip
        self.assertEqual(s["prompts"], [])  # no model call
        row = next(r for r in s["rows"] if r["step"] == "compose")
        self.assertEqual(
            (row["decision"], row["by"], row["mode"], row["merged"]), ("merge", "program", "merge", [1, 2])
        )
        self.assertEqual((row["merged_cleanly"], row["conflicts"], row["merged_set"]), ([1, 2], {}, {"p": 1, "q": 2}))
        self.assertEqual((row["cost_usd"], row["session"], row["void"]), (0.0, None, False))
        self.assertIn("draft 1: 3 fixed, 0 broken", row["notes"])
        self.assertEqual(s["gated"], [s["it_dir"] / "candidate"])
        cand = s["it_dir"] / "candidate"
        self.assertEqual((cand / "skills" / "__init__.py").read_text(), edit(edit(BASE, 1, "A"), 9, "B"))
        self.assertEqual(json.loads((cand / "candidate.json").read_text()), {"set": {"p": 1, "q": 2}})
        self.assertTrue((s["it_dir"] / "code.patch").read_text())
        comp = json.loads((s["it_dir"] / "composer.json").read_text())
        self.assertEqual((comp["by"], comp["mode"], comp["merged"]), ("program", "merge", [1, 2]))
        self.assertFalse((s["it_dir"] / "composer_context.json").exists())

    def test_one_positive_draft_goes_as_it_stands(self):
        s = self.compose({1: (edit(BASE, 1, "A"), {"p": 1}), 2: (edit(BASE, 9, "B"), {}), 3: (edit(BASE, 5, "C"), {})},
                         {1: (2, 0), 2: (0, 0), 3: (0, 1)})  # fmt: skip
        self.assertEqual(s["prompts"], [])
        row = next(r for r in s["rows"] if r["step"] == "compose")
        self.assertEqual((row["decision"], row["adopted"], row["by"], row["mode"]), ("adopt", 1, "program", "adopt"))
        self.assertEqual(row["merged"], None)
        self.assertEqual(s["gated"], [s["it_dir"] / "candidate_1"])

    def test_overlapping_edits_call_the_composer_with_the_clean_merge_in_its_working_copy(self):
        s = self.compose({1: (edit(BASE, 1, "A"), {}), 2: (edit(BASE, 1, "B"), {}), 3: (edit(BASE, 9, "C"), {})},
                         {1: (3, 0), 2: (2, 0), 3: (1, 0)},
                         composer=dict(decision="merge", merged=[1, 3], set={}, notes="n", mechanism="m"))  # fmt: skip
        self.assertEqual(len(s["prompts"]), 1)
        ctx = s["contexts"][0]
        self.assertEqual(
            (ctx["compose_mode"], ctx["merged_cleanly"], sorted(ctx["conflicts"])), ("conflict", [1, 3], ["2"])
        )
        self.assertIn("the edits overlap", ctx["conflicts"]["2"][0])
        self.assertEqual(s["wc"], edit(edit(BASE, 1, "A"), 9, "C"))  # the clean part, as the Composer found it
        row = next(r for r in s["rows"] if r["step"] == "compose")
        self.assertEqual((row["decision"], row["by"], row["mode"], row["merged"]), ("merge", "llm", "conflict", [1, 3]))
        self.assertEqual((row["merged_cleanly"], row["cost_usd"]), ([1, 3], 2.0))
        self.assertEqual(s["gated"], [s["it_dir"] / "candidate"])

    def test_no_positive_draft_leaves_the_decision_to_the_composer(self):
        s = self.compose({1: (edit(BASE, 1, "A"), {}), 2: (edit(BASE, 9, "B"), {}), 3: (BASE + "# 3\n", {})},
                         {1: (0, 0), 2: (1, 1), 3: (0, 2)}, composer=dict(decision="keep", notes="n"))  # fmt: skip
        self.assertEqual(len(s["prompts"]), 1)
        self.assertEqual(s["contexts"][0]["compose_mode"], "none_positive")
        self.assertNotIn("merged_cleanly", s["contexts"][0])
        self.assertEqual(s["wc"], BASE)  # the incumbent's code
        row = next(r for r in s["rows"] if r["step"] == "compose")
        self.assertEqual((row["decision"], row["by"], row["mode"]), ("keep", "llm", "none_positive"))
        self.assertEqual(s["gated"], [])

    def test_llm_mode_is_the_composer_as_before(self):
        s = self.compose({1: (edit(BASE, 1, "A"), {}), 2: (edit(BASE, 9, "B"), {}), 3: (edit(BASE, 5, "C"), {})},
                         {1: (3, 0), 2: (1, 0), 3: (0, 2)}, mode="llm")  # fmt: skip
        self.assertEqual(len(s["prompts"]), 1)
        self.assertEqual(s["contexts"][0]["compose_mode"], "llm")
        row = next(r for r in s["rows"] if r["step"] == "compose")
        self.assertEqual((row["decision"], row["by"], row["mode"], row["adopted"]), ("adopt", "llm", "llm", 1))
        self.assertNotIn("merged_cleanly", row)

    def test_positive_drafts_are_ordered_like_the_second_shot(self):
        c = lambda k, f, b: dict(number=k, fixed=["x"] * f, broken=["y"] * b)  # noqa: E731
        got = self.lp.positive_drafts([c(1, 2, 1), c(2, 3, 0), c(3, 1, 1), c(4, 4, 1), dict(number=5)])
        self.assertEqual([x["number"] for x in got], [2, 4, 1])  # net 3, 3 (fewer broken first), 1; 3 and 5 are out

    def test_the_mode_comes_from_the_cli_the_family_file_or_the_resume(self):
        lp = self.lp
        none = Namespace(compose=None)
        with mock.patch.object(lp, "family_spec", lambda: {}):
            self.assertEqual(lp.resolve_settings(none, {}, False)["compose"], "llm")
            self.assertEqual(
                lp.resolve_settings(none, dict(started="x", compose="program"), True)["compose"], "program"
            )
            self.assertEqual(lp.resolve_settings(none, dict(started="x", compose="program"), False)["compose"], "llm")
            self.assertEqual(lp.resolve_settings(Namespace(compose="program"), {}, False)["compose"], "program")
        with (
            mock.patch.object(lp, "family_spec", lambda: dict(compose="program")),
            mock.patch.object(lp, "FAMILY", "f"),
        ):
            self.assertEqual(lp.resolve_settings(none, {}, False)["compose"], "program")
            self.assertEqual(lp.resolve_settings(Namespace(compose="llm"), {}, False)["compose"], "llm")
        seen = {}
        with (
            mock.patch.object(lp, "cmd_run", lambda a: seen.update(vars(a)) or 0),
            mock.patch.object(lp, "configure", lambda f: None),
        ):
            lp.main(["run", "gso_single_grasp", "--compose", "program"])
            self.assertEqual(seen["compose"], "program")
            lp.main(["run", "gso_single_grasp"])
            self.assertIsNone(seen["compose"])
        for fam in ("gso_single_grasp", "gso_bimanual_lift"):
            spec = json.loads((HERE / "families" / f"{fam}.json").read_text())
            self.assertEqual(spec["compose"], "program")
            self.assertIn("scripts/rsi/merge.py", spec["sandbox"]["keep"])


if __name__ == "__main__":
    unittest.main()
