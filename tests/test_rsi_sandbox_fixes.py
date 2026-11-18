"""TODO 36 (review 2026-10-22): the scene RSI sandbox's fixes (scripts/rsi/): the acceptance and sealed seeds leave the
sandbox's family file for the hidden tree, the experience library of a family with a `sandbox` block carries no earlier
grasp code, every sandbox is built without CLAUDE.md / AGENTS.md / report (36.8), and reflect.py scrubs the paths a
role may not name from a library entry."""

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[1]
HERE = REPO / "scripts" / "rsi"


def _load(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _fresh(name: str, file: str):
    """One of the loop's scripts under its own module name; the generic names it imports are left as they were."""
    saved = {k: sys.modules.pop(k, None) for k in ("gate", "batch", "family", "scene_round", "sandbox", "reflect")}
    sys.path.insert(0, str(HERE))
    try:
        return _load(name, HERE / file)
    finally:
        sys.path.remove(str(HERE))
        for k, v in saved.items():
            sys.modules.pop(k, None)
            if v is not None:
                sys.modules[k] = v


def _write(p: Path, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def _family_file(root: Path, name: str, **extra) -> Path:
    fam = dict(
        name=name, pools=dict(diag=dict(seed=1, n=3), accept=dict(seed=2, n=4), sealed=dict(seed=3, n=5)), **extra
    )
    return _write(root / "scripts" / "rsi" / "families" / f"{name}.json", json.dumps(fam, indent=1))


class HiddenSeeds(unittest.TestCase):
    """S1: a role that reads the sandbox's family file and runs family.py draw must not get the hidden starts."""

    def setUp(self):
        self.sb = _fresh("rsi_sandbox_s1", "sandbox.py")

    def test_the_seeds_move_to_the_hidden_file_and_the_diag_seed_stays(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f = _family_file(root, "fam")
            out = self.sb.hide_seeds(root, "fam")
            self.assertEqual(out, dict(accept=2, sealed=3))
            seeds = json.loads((root / self.sb.HIDDEN / "fam" / "seeds.json").read_text())
            self.assertEqual(seeds, dict(accept=2, sealed=3))
            pools = json.loads(f.read_text())["pools"]
            self.assertEqual(
                pools, dict(diag=dict(seed=1, n=3), accept=dict(seed=None, n=4), sealed=dict(seed=None, n=5))
            )
            self.assertEqual(self.sb.hide_seeds(root, "fam"), {})  # already blank: nothing is overwritten
            self.assertEqual(json.loads((root / self.sb.HIDDEN / "fam" / "seeds.json").read_text()), seeds)

    def test_family_load_reads_blank_seeds_from_the_hidden_file(self):
        F = _fresh("rsi_family_s1", "family.py")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _family_file(root, "fam", scene="scene", tasks_file="t.jsonl", tasks=["a"])
            _write(root / "scene" / "scene_meta.json", "{}")
            _write(root / "t.jsonl", json.dumps(dict(id="a")) + "\n")
            self.sb.hide_seeds(root, "fam")
            with mock.patch.object(F, "HERE", root / "scripts" / "rsi"), mock.patch.object(F, "REPO", root):
                fam = F.load("fam")
                self.assertEqual([fam["pools"][p]["seed"] for p in ("diag", "accept", "sealed")], [1, 2, 3])
                (root / self.sb.HIDDEN / "fam" / "seeds.json").unlink()  # not in this tree: blank stays blank
                fam = F.load("fam")
                self.assertIsNone(fam["pools"]["accept"]["seed"])
                with self.assertRaises(SystemExit):
                    F.draw(fam, "accept", root / "pool")


class SandboxLibrary(unittest.TestCase):
    """S2 / S3: no earlier grasp code in a sandbox with a `sandbox` block; no CLAUDE.md / AGENTS.md / tools anywhere."""

    def setUp(self):
        self.sb = _fresh("rsi_sandbox_s2", "sandbox.py")

    def _library(self, root: Path, names=("fam", "other", "third")) -> Path:
        lib = root / "scripts" / "rsi" / "experience"
        for n in names:
            _write(lib / n / "entry.json", json.dumps(dict(experience=dict(cue=f"cue of {n}", confidence="high"))))
            _write(lib / n / "final.patch", "--- a/pick.py\n")
            _write(
                lib / n / "EXPERIENCE.md",
                f"# {n}\n\n**Cue:** c\n\nThe code the run ended with, against its seed: `final.patch`.\n",
            )
        return lib

    def test_a_sandbox_block_drops_every_final_patch_and_the_line_pointing_at_it(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _family_file(root, "fam", sandbox=dict(seed="s", keep=[]))
            lib = self._library(root)
            self.sb.library_without(root, "fam")
            self.assertFalse((lib / "fam").exists())
            for n in ("other", "third"):
                self.assertFalse((lib / n / "final.patch").exists())
                md = (lib / n / "EXPERIENCE.md").read_text()
                self.assertNotIn("final.patch", md)
                self.assertIn("**Cue:** c", md)
            index = (lib / "INDEX.md").read_text()
            self.assertIn("cue of other", index)
            self.assertNotIn("final.patch", index)

    def test_library_exclude_removes_the_named_families(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _family_file(root, "fam", sandbox=dict(seed="s", keep=[]), library_exclude=["other"])
            lib = self._library(root)
            self.sb.library_without(root, "fam")
            self.assertEqual(sorted(p.name for p in lib.iterdir() if p.is_dir()), ["third"])
            self.assertNotIn("cue of other", (lib / "INDEX.md").read_text())

    def test_a_family_without_a_block_keeps_the_patches(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _family_file(root, "fam")
            lib = self._library(root)
            self.sb.library_without(root, "fam")
            self.assertTrue((lib / "other" / "final.patch").exists())

    def test_the_default_strip(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for rel in (
                "CLAUDE.md",
                "AGENTS.md",
                "scripts/rsi/report/steps.py",
                "scripts/rsi/tools/steps.py",
                "scripts/rsi/library/a.md",
            ):
                _write(root / rel)
            _write(root / "Makefile")
            _write(root / "scripts" / "rsi" / "loop.py")
            self.sb.default_strip(root)
            self.assertEqual(
                sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()),
                ["Makefile", "scripts/rsi/loop.py"],
            )
            self.sb.default_strip(root)  # again: nothing to remove

    def test_grasp_atomic_keeps_neither_guide(self):
        keep = json.loads((HERE / "families" / "grasp_atomic.json").read_text())["sandbox"]["keep"]
        self.assertNotIn("CLAUDE.md", keep)
        self.assertNotIn("AGENTS.md", keep)

    def test_nothing_the_loop_runs_needs_the_stripped_files(self):
        # no role prompt names the report/ tools (TODO 36.8 dropped the tools note); nothing but docs / lint scripts
        # name the guides, and nothing the loop runs imports from report/
        loop = (HERE / "loop.py").read_text()
        self.assertNotIn("TOOLS_NOTE", loop)
        for f in (HERE / "prompts").glob("*.md"):
            self.assertNotRegex(f.read_text(), r"\{TOOLS\}|rsi/report|rsi/tools", f.name)
        for name in ("loop", "gate", "batch", "scene_round", "sealed", "digest", "claim_check", "family"):
            self.assertNotRegex(
                (HERE / f"{name}.py").read_text(), r"CLAUDE\.md|AGENTS\.md|rsi/library|rsi/report", name
            )


class ReflectScrub(unittest.TestCase):
    """S6: the library names no path a role may not name."""

    def setUp(self):
        self.rf = _fresh("rsi_reflect_s6", "reflect.py")

    def test_path_tokens_are_replaced(self):
        text = (
            'see /workspace/code/rsi-scene/x/outputs/scene_rsi_20261016_hidden/fam/gate and "rsi_kit/foo.py", '
            "also .claude/projects/y. Plain text stays."
        )
        out = self.rf.scrub(text, "t")
        self.assertIsNone(re.search(r"_hidden|rsi_kit|\.claude/|/workspace/", out))
        self.assertIn(self.rf.PLACEHOLDER, out)
        self.assertIn("Plain text stays.", out)
        self.assertEqual(self.rf.scrub("nothing here", "t"), "nothing here")

    def test_json_stays_valid(self):
        raw = json.dumps(dict(experience=dict(lesson="read /workspace/tmp/a_hidden/x.json now")), indent=1)
        out = json.loads(self.rf.scrub(raw, "entry.json"))
        self.assertIn(self.rf.PLACEHOLDER, out["experience"]["lesson"])

    def test_a_patch_line_keeps_its_shape(self):
        out = self.rf.scrub('+    root = "/workspace/tmp/x"\n-    y = 1\n', "final.patch")
        self.assertEqual(out, f'+    root = "{self.rf.PLACEHOLDER}"\n-    y = 1\n')


if __name__ == "__main__":
    unittest.main()


class PhysicsLink(unittest.TestCase):
    def test_the_sandbox_links_an_existing_physics_directory_so_the_first_round_can_write_the_copy(self):
        """2026-11-17: a sandbox links `<scene>_physics` to the repository's; when that directory was missing (the
        scene and its copy deleted and rebuilt, no round run yet) the link dangled and world.physics_scene's
        mkdir(exist_ok=True) raised FileExistsError in the sandbox's first round. sandbox.physics_target creates it."""
        from locomani.tasks.scene.world import physics_scene

        sandbox = _load("sandbox_physics_link", HERE / "sandbox.py")
        with tempfile.TemporaryDirectory() as tmp:
            repo, box = Path(tmp) / "repo", Path(tmp) / "box"
            scene = "outputs/scene_discovery/s"
            (repo / scene).mkdir(parents=True)
            (repo / scene / "scene_locomani.xml").write_text(
                '<mujoco><asset/><worldbody><body name="pelvis"><freejoint/><geom type="box" size="0.1 0.1 0.1" mass="1"/>'
                "</body></worldbody></mujoco>"
            )
            (repo / scene / "scene_meta.json").write_text("{}")
            (box / scene).parent.mkdir(parents=True)
            (box / scene).symlink_to(repo / scene)
            dangling = box / f"{scene}_physics"
            dangling.symlink_to(repo / f"{scene}_physics")
            with self.assertRaises(FileExistsError):  # the bug: a link to nothing
                physics_scene(box / scene)
            dangling.unlink()
            dangling.symlink_to(sandbox.physics_target(repo, scene))
            out = physics_scene(box / scene)
            self.assertTrue((repo / f"{scene}_physics" / "robot.xml").is_file() and out == dangling)
