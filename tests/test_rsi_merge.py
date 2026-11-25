"""TODO 54: scripts/rsi/merge.py, the program's 3-way merge of the Engineers' drafts."""

import importlib.util
import sys
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


def test_edits_in_different_places_merge_and_overlapping_ones_conflict(tmp_path, monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("GIT_"):
            monkeypatch.delenv(k)
    base = pkg(tmp_path / "base", BASE)
    a = pkg(tmp_path / "a", edit(BASE, 1, "A"))
    b = pkg(tmp_path / "b", edit(BASE, 9, "B"))
    c = pkg(tmp_path / "c", edit(BASE, 1, "C"))  # overlaps a
    r = M.merge_drafts(base, {}, [(1, a, {}), (2, b, {}), (3, c, {})], tmp_path / "out")
    assert r["merged"] == [1, 2] and list(r["conflicts"]) == [3]
    out = (tmp_path / "out" / "skill.py").read_text()
    assert "A\n" in out and "B\n" in out and "C\n" not in out
    assert "the edits overlap" in r["conflicts"][3][0]


def test_added_files_and_set_keys(tmp_path, monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("GIT_"):
            monkeypatch.delenv(k)
    base = pkg(tmp_path / "base", BASE)
    a = pkg(tmp_path / "a", BASE, **{"new_a.py": "x\n"})
    b = pkg(tmp_path / "b", BASE, **{"new_a.py": "different\n"})
    c = pkg(tmp_path / "c", BASE, **{"new_c.py": "y\n"})
    r = M.merge_drafts(base, {"k": 1}, [(1, a, {"k": 1, "p": 2}), (2, b, {"k": 1}), (3, c, {"k": 1, "p": 3}),
                                        (4, pkg(tmp_path / "d", BASE), {"k": 1, "q": 5})], tmp_path / "out")  # fmt: skip
    assert r["merged"] == [1, 4] and sorted(r["conflicts"]) == [2, 3]
    assert r["set"] == {"k": 1, "p": 2, "q": 5}
    assert (tmp_path / "out" / "new_a.py").read_text() == "x\n" and not (tmp_path / "out" / "new_c.py").exists()


def test_a_removed_set_key_and_a_deleted_file(tmp_path):
    base = pkg(tmp_path / "base", BASE, **{"old.py": "o\n"})
    a = pkg(tmp_path / "a", BASE)  # deletes old.py
    r = M.merge_drafts(base, {"k": 1, "z": 1}, [(1, a, {"z": 1})], tmp_path / "out")
    assert r["merged"] == [1] and not (tmp_path / "out" / "old.py").exists() and r["set"] == {"z": 1}
