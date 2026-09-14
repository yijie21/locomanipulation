#!/usr/bin/env python3
"""Final validation shared by make check and pre-commit; no result caching.

Pre-commit hides unstaged tracked edits before invoking this entry. A direct
invocation checks both staged and unstaged edits, plus untracked inputs.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def changed_paths(root: Path) -> set[str]:
    paths = set()
    # Disable rename detection so both the old and new path affect the scope.
    for args in (
        ["diff", "--name-only", "--no-renames", "-z"],
        ["diff", "--cached", "--name-only", "--no-renames", "-z"],
        ["ls-files", "--others", "--exclude-standard", "-z"],
    ):
        output = subprocess.check_output(["git", *args], cwd=root)
        paths.update(p.decode("utf-8", errors="surrogateescape") for p in output.split(b"\0") if p)
    return paths


def documentation_only(paths: set[str]) -> bool:
    # Unknown paths, Markdown fixtures and executable docs take the full route.
    return bool(paths) and all(
        Path(p).suffix == ".md" and (len(Path(p).parts) == 1 or Path(p).parts[0] == "docs") for p in paths
    )


def run(root: Path, *, full: bool = False) -> int:
    paths = changed_paths(root)
    docs = not full and documentation_only(paths)
    print(f"[check] {'documentation' if docs else 'full'} validation ({len(paths)} changed paths)", flush=True)
    commands = [["make", "check-docs"]] if docs else [["make", "lint"], ["make", "test", "TESTS=tests"]]
    for command in commands:
        result = subprocess.run(command, cwd=root)
        if result.returncode:
            return result.returncode
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="run all checks regardless of changed paths")
    args = parser.parse_args()
    return run(ROOT, full=args.all)


if __name__ == "__main__":
    raise SystemExit(main())
