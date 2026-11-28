"""TODO 56.2 (user 2026-11-27): scripts/rsi/merge.py, the program's 3-way merge of the Engineers' drafts."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "scripts" / "rsi"
spec = importlib.util.spec_from_file_location("rsi_merge_under_test", HERE / "merge.py")
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)

BASE = "".join(f"line {i}\n" for i in range(12))


def pkg(root: Path, text: str, **extra) -> Path:
    root.mkdir(parents=True)
    (root / "skill.py").write_text(text)
    for k, v in extra.items():
        (root / k).write_text(v)
    return root


def edit(text: str, i: int, to: str) -> str:
    lines = text.splitlines(keepends=True)
    lines[i] = to + "\n"
    return "".join(lines)


class MergeDrafts(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.tmp = Path(d.name)
        self.env = {k: v for k, v in os.environ.items() if k.startswith("GIT_")}  # a test runner inside a git hook
        for k in self.env:
            del os.environ[k]
        self.addCleanup(os.environ.update, self.env)

    def test_edits_in_different_places_merge_and_overlapping_ones_conflict(self):
        base = pkg(self.tmp / "base", BASE)
        a = pkg(self.tmp / "a", edit(BASE, 1, "A"))
        b = pkg(self.tmp / "b", edit(BASE, 9, "B"))
        c = pkg(self.tmp / "c", edit(BASE, 1, "C"))  # overlaps a
        r = M.merge_drafts(base, {}, [(1, a, {}), (2, b, {}), (3, c, {})], self.tmp / "out")
        self.assertEqual((r["merged"], list(r["conflicts"])), ([1, 2], [3]))
        out = (self.tmp / "out" / "skill.py").read_text()
        self.assertTrue("A\n" in out and "B\n" in out and "C\n" not in out)
        self.assertEqual(out, edit(edit(BASE, 1, "A"), 9, "B"))
        self.assertIn("the edits overlap", r["conflicts"][3][0])

    def test_added_files_and_set_keys(self):
        base = pkg(self.tmp / "base", BASE)
        a = pkg(self.tmp / "a", BASE, **{"new_a.py": "x\n"})
        b = pkg(self.tmp / "b", BASE, **{"new_a.py": "different\n"})
        c = pkg(self.tmp / "c", BASE, **{"new_c.py": "y\n"})
        r = M.merge_drafts(base, {"k": 1}, [(1, a, {"k": 1, "p": 2}), (2, b, {"k": 1}), (3, c, {"k": 1, "p": 3}),
                                            (4, pkg(self.tmp / "d", BASE), {"k": 1, "q": 5})], self.tmp / "out")  # fmt: skip
        self.assertEqual((r["merged"], sorted(r["conflicts"])), ([1, 4], [2, 3]))
        self.assertEqual(r["set"], {"k": 1, "p": 2, "q": 5})
        self.assertEqual((self.tmp / "out" / "new_a.py").read_text(), "x\n")
        self.assertFalse((self.tmp / "out" / "new_c.py").exists())
        self.assertIn("set key p", r["conflicts"][3][0])

    def test_a_removed_set_key_and_a_deleted_file(self):
        base = pkg(self.tmp / "base", BASE, **{"old.py": "o\n"})
        a = pkg(self.tmp / "a", BASE)  # deletes old.py
        r = M.merge_drafts(base, {"k": 1, "z": 1}, [(1, a, {"z": 1})], self.tmp / "out")
        self.assertEqual(r["merged"], [1])
        self.assertFalse((self.tmp / "out" / "old.py").exists())
        self.assertEqual(r["set"], {"z": 1})

    def test_the_same_edit_in_two_drafts_is_no_conflict_and_out_is_replaced(self):
        base = pkg(self.tmp / "base", BASE)
        a = pkg(self.tmp / "a", edit(BASE, 1, "A"))
        b = pkg(self.tmp / "b", edit(BASE, 1, "A"))
        out = self.tmp / "out"
        out.mkdir()
        (out / "stale.py").write_text("stale\n")
        r = M.merge_drafts(base, {}, [(1, a, {"p": 1}), (2, b, {"p": 1})], out)
        self.assertEqual((r["merged"], r["conflicts"], r["set"]), ([1, 2], {}, {"p": 1}))
        self.assertFalse((out / "stale.py").exists())


if __name__ == "__main__":
    unittest.main()
