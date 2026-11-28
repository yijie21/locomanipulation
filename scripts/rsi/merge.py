"""TODO 56.2 (user 2026-11-27, from main's TODO 54 F8): the program merges the Engineers' drafts; the Composer is
called only when a draft does not apply (loop.compose_by_program).

Each draft is a skills package changed from the same base (the incumbent's, or the pool candidate the Manager named).
`merge_drafts` folds the drafts into one package in the order given (best diagnostic flips first) by a 3-way merge
of every changed file against the base (`git merge-file`), and merges their parameter sets the same way (a key two
drafts set to different values clashes). A draft whose files or keys do not apply to what is merged so far is left
out and named in `conflicts`; the rest are in `merged`. Pure functions of directories and dicts: no loop state, no
simulator, no model call."""

from __future__ import annotations

import filecmp
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def _files(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            and p.suffix != ".pyc"}  # fmt: skip


def _same(a: Path, b: Path) -> bool:
    return a.exists() == b.exists() and (not a.exists() or filecmp.cmp(a, b, shallow=False))


def _git_env() -> dict:
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def merge_file(cur: Path, base: Path, other: Path) -> bool:
    """3-way merge of `other`'s changes (against `base`) into `cur`, in place; False (cur untouched) on a conflict."""
    r = subprocess.run(["git", "merge-file", "-p", str(cur), str(base), str(other)], capture_output=True,
                       env=_git_env(), check=False)  # fmt: skip
    if r.returncode != 0:
        return False
    cur.write_bytes(r.stdout)
    return True


def set_delta(base: dict, new: dict) -> dict:
    """What a draft's set changes against the base: {key: value} in `changed`, the keys it removed in `gone`."""
    return dict(changed={k: v for k, v in new.items() if k not in base or base[k] != v},
                gone=sorted(k for k in base if k not in new))  # fmt: skip


def merge_sets(base: dict, current: dict, new: dict, touched: dict) -> list[str]:
    """Apply a draft's set delta to `current` in place; returns the keys that clash (changed by an earlier draft to a
    different value, or removed by one and changed by this one). `touched` {key: value | None (removed)} is updated."""
    d = set_delta(base, new)
    bad = [k for k, v in d["changed"].items() if k in touched and touched[k] != v]
    bad += [k for k in d["gone"] if k in touched and touched[k] is not None]
    if bad:
        return sorted(set(bad))
    for k, v in d["changed"].items():
        current[k] = v
        touched[k] = v
    for k in d["gone"]:
        current.pop(k, None)
        touched[k] = None
    return []


def merge_drafts(base: Path, base_set: dict, drafts: list[tuple[int, Path, dict]], out: Path) -> dict:
    """`drafts`: [(number, skills dir, set)] best first. Writes the merged package to `out` (a skills directory, made
    from `base`; replaced when it exists) and returns {merged: [numbers], conflicts: {number: [why]}, set: merged set}."""
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(base, out, ignore=IGNORE)
    cur_set: dict = dict(base_set)
    touched: dict = {}
    merged: list[int] = []
    conflicts: dict[int, list[str]] = {}
    for k, d, dset in drafts:
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp) / "t"
            shutil.copytree(out, trial, ignore=IGNORE)
            bad: list[str] = []
            for rel in sorted(f for f in _files(base) | _files(d) if not _same(base / f, d / f)):
                b, o, c = base / rel, d / rel, trial / rel
                if not o.exists():  # the draft deleted the file
                    if c.exists() and _same(c, b):
                        c.unlink()
                    elif c.exists():
                        bad.append(f"{rel}: deleted by this draft, edited by an earlier one")
                elif not b.exists():  # the draft added it
                    if not c.exists():
                        c.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(o, c)
                    elif not _same(c, o):
                        bad.append(f"{rel}: added by both drafts with different content")
                elif not c.exists():
                    bad.append(f"{rel}: edited by this draft, deleted by an earlier one")
                elif _same(c, b):
                    shutil.copy2(o, c)
                elif not _same(c, o) and not merge_file(c, b, o):
                    bad.append(f"{rel}: the edits overlap")
            trial_set, trial_touched = dict(cur_set), dict(touched)
            bad += [f"set key {x}: set to another value by an earlier draft"
                    for x in merge_sets(base_set, trial_set, dset, trial_touched)]  # fmt: skip
            if bad:
                conflicts[k] = bad
                continue
            shutil.rmtree(out)
            shutil.copytree(trial, out, ignore=IGNORE)
            cur_set, touched = trial_set, trial_touched
            merged.append(k)
    return dict(merged=merged, conflicts=conflicts, set=cur_set)
