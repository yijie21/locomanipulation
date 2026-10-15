"""TODO 25.9.1.6's isolation (docs/handoff/scalebfm-warehouse-rsi-20261014.md §17): the transcript audit that ends an arm
when a role names an off-limits path, and the sandbox scrub that keeps phase 1-3's answers out of a cold-start arm."""

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "experiments" / "scalebfm_warehouse_rsi_20261014"))
import loop  # noqa: E402
import phase4_sandbox as sandbox  # noqa: E402


def tool_use(cmd: str) -> str:
    return '{"type":"assistant","message":{"content":[{"type":"tool_use","input":{"command":"%s"}}]}}' % cmd


class Audit(unittest.TestCase):
    def setUp(self):
        self.saved = loop.ARM, loop.ORIGIN, loop.REPO
        loop.ARM, loop.ORIGIN, loop.REPO = "A", Path("/workspace/code/origin"), Path("/workspace/code/rsi-phase4/arm-a")

    def tearDown(self):
        loop.ARM, loop.ORIGIN, loop.REPO = self.saved

    def test_origin_reads_are_leaks_but_its_third_party_is_not(self):
        pats = loop.leak_patterns()
        self.assertEqual(loop.leaks_in([tool_use("cat /workspace/code/origin/STATE.md")], pats), ["origin"])
        venv = tool_use("/workspace/code/origin/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python x.py")
        self.assertEqual(loop.leaks_in([venv], pats), [])

    def test_climbs_and_home_and_gate_are_leaks(self):
        pats = loop.leak_patterns()
        lines = [tool_use("ls third_party/../outputs"), tool_use("cat ~/.claude/projects/x/memory/a.md"),
                 tool_use("ls outputs/scalebfm_warehouse_rsi_20261014_hidden/gate")]  # fmt: skip
        self.assertEqual(loop.leaks_in(lines, pats), ["claude_home", "climb", "gate"])

    def test_only_tool_inputs_count(self):
        text = '{"type":"user","message":{"content":"do not read /workspace/code/origin or ~/.claude"}}'
        self.assertEqual(loop.leaks_in([text], loop.leak_patterns()), [])

    def test_phase_two_checks_only_the_gate(self):
        loop.ARM = None
        self.assertEqual(sorted(loop.leak_patterns()), ["gate"])

    def test_arm_prompts_fall_back_to_phase_two(self):
        self.assertEqual(loop.prompt_file("reviewer").parent.name, "phase4")
        self.assertEqual(loop.prompt_file("manager").name, "manager.md")  # arm A keeps phase 2's Manager
        loop.ARM = "B"
        self.assertEqual(loop.prompt_file("manager").name, "manager_B.md")


class StructuredOutput(unittest.TestCase):
    def test_the_last_structured_output_in_the_transcript_is_used(self):
        import json
        import tempfile

        def call(name, inp):
            return json.dumps(
                {"type": "assistant", "message": {"content": [{"type": "tool_use", "name": name, "input": inp}]}}
            )

        with tempfile.TemporaryDirectory() as d:
            t = Path(d) / "s.jsonl"
            t.write_text("\n".join([call("StructuredOutput", {"v": 1}), call("Bash", {"command": "x"}),
                                     call("StructuredOutput", {"v": 2}), call("Bash", {"command": "y"})]))  # fmt: skip
            saved = loop.transcript
            loop.transcript = lambda session: t if session else None
            try:
                self.assertEqual(loop.structured_from_transcript("s"), {"v": 2})
                self.assertIsNone(loop.structured_from_transcript(None))
            finally:
                loop.transcript = saved


class Labels(unittest.TestCase):
    def test_every_way_a_reviewer_names_a_start_is_one_label(self):
        for x in ("sonic_s47000_01", "bfm_s47000_01", "s47000_01"):
            self.assertEqual(loop.start_label(x), "sonic_s47000_01")


class Restore(unittest.TestCase):
    def test_rejected_edits_go_back_and_tracked_files_survive(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
            git("init", "-q")
            (repo / "clean.py").write_text("committed\n")
            (repo / "dirty.py").write_text("committed\n")
            git("add", "clean.py", "dirty.py")
            git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x")
            (repo / "dirty.py").write_text("someone else's work\n")
            saved = {"dirty.py": b"someone else's work\n"}  # what the driver saves: files already modified
            (repo / "clean.py").write_text("engineer\n")
            (repo / "dirty.py").write_text("engineer\n")
            (repo / "new.py").write_text("engineer\n")
            old, loop.REPO = loop.REPO, repo
            try:
                loop.restore(["clean.py", "dirty.py", "new.py"], saved)
            finally:
                loop.REPO = old
            self.assertEqual((repo / "clean.py").read_text(), "committed\n")
            self.assertEqual((repo / "dirty.py").read_text(), "someone else's work\n")
            self.assertFalse((repo / "new.py").exists())


class Batch(unittest.TestCase):
    def test_all_shards_of_all_jobs_run_and_the_pool_never_exceeds_procs(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake = d / "fake_round.py"  # records its own start and end; argv: pool --spec S --out O --shard i/n ...
            fake.write_text(
                "import sys, time, pathlib\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)\n"
                "shard = sys.argv[sys.argv.index('--shard') + 1].replace('/', 'of')\n"
                "t0 = time.time(); time.sleep(0.3)\n"
                "import os; (out / shard).write_text(f'{t0} {time.time()} {os.environ[\"CUDA_VISIBLE_DEVICES\"]}')\n"
            )
            saved = loop.SIM_PY, loop.PROCS, loop.LOOP
            loop.SIM_PY, loop.PROCS, loop.LOOP = Path(sys.executable), 3, d
            try:
                jobs = [(d / "spec.yaml", d / f"out{j}", None, None) for j in range(4)]
                loop.run_batch(jobs, d / "log", script=fake)
            finally:
                loop.SIM_PY, loop.PROCS, loop.LOOP = saved
            spans = [tuple(map(float, f.read_text().split()[:2])) for j in range(4) for f in (d / f"out{j}").iterdir()]
            self.assertEqual(len(spans), 4 * 3)
            peak = max(sum(a <= t < b for a, b in spans) for t, _ in spans)
            self.assertLessEqual(peak, 3)

    def test_a_gpu_list_spreads_the_pool_over_the_devices(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake = d / "fake_round.py"
            fake.write_text(
                "import os, sys, time, pathlib\n"
                "out = pathlib.Path(sys.argv[sys.argv.index('--out') + 1]); out.mkdir(parents=True, exist_ok=True)\n"
                "shard = sys.argv[sys.argv.index('--shard') + 1].replace('/', 'of')\n"
                "t0 = time.time(); time.sleep(0.3)\n"
                "(out / shard).write_text(f'{t0} {time.time()} {os.environ[\"CUDA_VISIBLE_DEVICES\"]}')\n"
            )
            saved = loop.SIM_PY, loop.PROCS, loop.LOOP
            loop.SIM_PY, loop.PROCS, loop.LOOP = Path(sys.executable), 4, d
            try:
                loop.run_batch([(d / "spec.yaml", d / f"out{j}", None, None) for j in range(3)], d / "log", "0,1", fake)
            finally:
                loop.SIM_PY, loop.PROCS, loop.LOOP = saved
            runs = [f.read_text().split() for j in range(3) for f in (d / f"out{j}").iterdir()]
            self.assertEqual(len(runs), 3 * 4)
            for g in ("0", "1"):
                mine = [(float(a), float(b)) for a, b, x in runs if x == g]
                self.assertTrue(mine)
                self.assertLessEqual(max(sum(a <= t < b for a, b in mine) for t, _ in mine), 2)


class Scrub(unittest.TestCase):
    def test_cold_plan_stops_before_the_results(self):
        for arm in ("A", "B"):
            text = sandbox.cold_plan(arm)
            self.assertIn(f"## 0. 本次运行（第 4 阶段，臂 {arm}）", text)
            self.assertIn("## 11.", text)
            self.assertNotIn("## 12.", text)
            self.assertIsNone(sandbox.MARKERS.search(text))
            self.assertLess(text.index("## 0."), text.index("## 1."))

    def test_extensions_keep_the_registry_and_lose_the_hooks(self):
        src = (REPO / "scripts" / "experiments" / "scalebfm_warehouse_rsi_20261014" / "extensions.py").read_text()
        out = sandbox.strip_hooks(src)
        self.assertIn("def hook(", out)
        self.assertNotIn("\n@hook(", out)  # the docstring still shows how to register one
        ns = {}
        exec(compile(out, "extensions.py", "exec"), ns)
        self.assertEqual(ns["HOOKS"], {})


if __name__ == "__main__":
    unittest.main()
