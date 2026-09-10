#!/usr/bin/env python3
"""Size caps from docs/reference/standards.md: a file <= 600 lines, a function <= 80.

    python scripts/lint_sizes.py            # report (exit 0)
    python scripts/lint_sizes.py --strict   # exit 1 on any violation (pre-commit after the grace week)

Baseline on 2026-09-09 (refactor step 1): 12 files and 70 functions over the caps; the
skeleton extraction (step 4) is what brings them under. Excluded: third_party, src/gr00t,
outputs, tests.
"""

import ast
import glob
import sys

FILE_MAX, FUNC_MAX = 600, 80
EXCLUDE = ("third_party/", "src/gr00t/", "outputs/", ".claude/", "legacy/")


def files():
    for f in sorted(set(glob.glob("src/**/*.py", recursive=True) + glob.glob("scripts/*.py"))):
        if not any(x in f for x in EXCLUDE):
            yield f


def main():
    strict = "--strict" in sys.argv
    bad_files, bad_funcs = [], []
    for f in files():
        src = open(f).read()
        n = src.count("\n") + 1
        if n > FILE_MAX:
            bad_files.append((n, f))
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            print(f"SYNTAX {f}: {e}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                length = node.end_lineno - node.lineno + 1
                if length > FUNC_MAX:
                    bad_funcs.append((length, f, node.name))
    for n, f in sorted(bad_files, reverse=True):
        print(f"FILE {n:5d} > {FILE_MAX}  {f}")
    for n, f, name in sorted(bad_funcs, reverse=True):
        print(f"FUNC {n:5d} > {FUNC_MAX}   {f}:{name}")
    fail = strict and (bad_files or bad_funcs)
    print(
        f"[lint_sizes] {len(bad_files)} files over {FILE_MAX} lines, {len(bad_funcs)} functions over {FUNC_MAX}"
        + (" -- FAIL" if fail else "")
    )
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
