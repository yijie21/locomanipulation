#!/usr/bin/env python3
"""Check the small navigation surface and prevent new forbidden import edges.

Existing dependency debt is explicit; stale exceptions fail so the list shrinks.
This is a static import check, not proof about arbitrary runtime/dynamic imports.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DEBT = "tests/fixtures/architecture_debt.json"
GUIDES = {
    "AGENTS.md": 80,
    "CLAUDE.md": 12,
    "README.md": 100,
    "docs/STATE.md": 60,
    "docs/ROADMAP.md": 60,
    "docs/README.md": 80,
    "docs/reference/change-map.md": 75,
    "docs/reference/standards.md": 100,
    "docs/reference/runbook.md": 100,
    "docs/agents/session-workflow.md": 50,
    "docs/handoff/agent-maintenance-plan.md": 120,
}


def imports(source: str, filename: str) -> set[str]:
    """Include function-local imports and resolve package-relative imports."""
    package = Path(filename).with_suffix("").parts[1:-1]
    targets = set()
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = ".".join((*package[: len(package) - node.level + 1], *base.split("."))).rstrip(".")
            targets.update(f"{base}.{alias.name}" for alias in node.names)
    return targets


def forbidden(source: str, target: str) -> bool:
    if source.startswith("src/locomani/backends/") and target.startswith("locomani.tasks"):
        return True
    if source.startswith("src/locomani/primitives/") and target.startswith(("locomani.tasks", "locomani.pipeline")):
        return True
    if target.startswith("scripts.experiments"):
        return True
    if source.endswith("/cli_args.py") and target.startswith("locomani.tasks.bottle.grasp_gym"):
        return True
    if source.endswith(("config.py", "config_base.py", "/cli_args.py")) and target.split(".")[0] in {
        "mujoco",
        "torch",
        "isaacsim",
    }:
        return True
    return False


def dependency_edges(root: Path) -> set[tuple[str, str]]:
    edges = set()
    for path in sorted((root / "src/locomani").rglob("*.py")):
        name = path.relative_to(root).as_posix()
        targets = imports(path.read_text(), name)
        edges.update((name, target) for target in targets if forbidden(name, target))
    return edges


def dependency_errors(root: Path) -> list[str]:
    rows = json.loads((root / DEBT).read_text())
    allowed = {(row["source"], row["target"]) for row in rows}
    errors = []
    if len(allowed) != len(rows) or any(not row.get("reason", "").strip() for row in rows):
        errors.append(f"{DEBT}: exceptions must be unique and have a reason")
    actual = dependency_edges(root)
    for source, target in sorted(actual - allowed):
        errors.append(f"{source}: forbidden import {target}; pass data through the owning interface (see standards)")
    for source, target in sorted(allowed - actual):
        errors.append(f"{DEBT}: remove resolved exception {source} -> {target}")
    return errors


def navigation_errors(
    root: Path, guides: dict[str, int] | None = None, documents: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    errors = []
    checked: dict[str, int | None] = dict(GUIDES if guides is None else guides)
    for name in documents:
        if name.endswith(".md") and (root / name).is_file():
            checked.setdefault(name, None)
    for name, limit in checked.items():
        path = root / name
        if not path.is_file():
            errors.append(f"{name}: missing navigation document")
            continue
        body = path.read_text()
        if limit is not None and len(body.splitlines()) > limit:
            errors.append(f"{name}: over {limit} lines; move task-specific detail behind a relevant link")
        for link in re.findall(r"\[[^\]]*\]\(([^)]+)\)", body):
            if link.startswith(("https://", "http://", "mailto:", "#")):
                continue
            target = unquote(link.split("#", 1)[0])
            if not (path.parent / target).exists():
                errors.append(f"{name}: broken link {link}")
        if name == "docs/reference/change-map.md":
            for test in re.findall(r"`(test_[\w]+\.py)`", body):
                if not (root / "tests" / test).is_file():
                    errors.append(f"{name}: missing tests/{test}; name an actual verification entry")
    return errors


def artifact_errors(tracked: list[str]) -> list[str]:
    return [
        f"{path}: runtime artifacts and their symlinks must remain untracked"
        for path in tracked
        if path.split("/", 1)[0] in {"outputs", "third_party", "datasets"}
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navigation-only", action="store_true", help="check documentation without scanning code")
    parser.add_argument("--changed-docs", action="store_true", help="also check links in modified Markdown files")
    args = parser.parse_args()
    documents = set()
    if args.changed_docs:
        from check_changes import changed_paths

        documents = changed_paths(ROOT)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT, text=True).split("\0")
    errors = navigation_errors(ROOT, documents=documents) + artifact_errors(tracked)
    if not args.navigation_only:
        errors += dependency_errors(ROOT)
    for error in errors:
        print(f"[repository] {error}")
    print(f"[repository] {'FAIL' if errors else 'PASS'}: {len(errors)} issues")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
