"""Real Git diffs and a disposable make target check final-validation selection."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # A Git hook exports GIT_DIR/GIT_INDEX_FILE etc. cwd alone does not isolate
    # git commands: inherited variables can redirect init/add/commit to the host.
    for key in tuple(os.environ):
        if key.startswith("GIT_"):
            monkeypatch.delenv(key)

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True)

    git("init", "-q")
    (tmp_path / "scripts").mkdir()
    shutil.copyfile(SOURCE / "scripts/check_changes.py", tmp_path / "scripts/check_changes.py")
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("old\n")
    (tmp_path / "code.py").write_text("value = 1\n")
    (tmp_path / ".gitignore").write_text("*.log\n__pycache__/\n")
    (tmp_path / "Makefile").write_text(
        ".PHONY: check-docs lint test\n"
        "check-docs:\n\t@echo docs >> checks.log\n"
        "lint:\n\t@echo lint >> checks.log\n"
        "test:\n\t@echo tests=$(TESTS) >> checks.log\n"
    )
    git("add", ".")
    # Only the disposable fixture repo: never invoke the real repository's hooks.
    git(
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return tmp_path, git


def invoke(repo, *args):
    root, _ = repo
    result = subprocess.run(
        [sys.executable, "scripts/check_changes.py", *args],
        cwd=root,
        env=dict(os.environ, TESTS="one_test.py"),
        capture_output=True,
        text=True,
    )
    calls = (root / "checks.log").read_text().splitlines()
    return result, calls


def test_markdown_changes_only_check_navigation(repo):
    root, git = repo
    (root / "README.md").write_text("new\n")
    git("add", "README.md")
    (root / "docs/new guide.md").write_text("new\n")
    result, calls = invoke(repo)
    assert result.returncode == 0, result.stderr
    assert calls == ["docs"]


@pytest.mark.parametrize("kind", ["unstaged", "staged", "untracked", "rename", "fixture"])
def test_code_inputs_require_one_full_suite_even_with_docs_changes(repo, kind):
    root, git = repo
    (root / "README.md").write_text("new\n")
    git("add", "README.md")
    if kind in {"unstaged", "staged"}:
        (root / "code.py").write_text("value = 2\n")
        if kind == "staged":
            git("add", "code.py")
    elif kind == "untracked":
        (root / "new.py").touch()
    elif kind == "rename":
        git("mv", "code.py", "docs/old-code.md")
    else:
        (root / "tests").mkdir()
        (root / "tests/fixture.md").touch()
    result, calls = invoke(repo)
    assert result.returncode == 0, result.stderr
    assert calls == ["lint", "tests=tests"]


@pytest.mark.parametrize("force", [False, True])
def test_clean_checkout_or_explicit_all_runs_full_validation(repo, force):
    root, _ = repo
    if force:
        (root / "README.md").write_text("new\n")
    result, calls = invoke(repo, *(["--all"] if force else []))
    assert result.returncode == 0
    assert calls == ["lint", "tests=tests"]


def test_failed_lint_stops_before_cpu_suite(repo):
    root, _ = repo
    p = root / "Makefile"
    p.write_text(p.read_text().replace("@echo lint >> checks.log", "@echo lint >> checks.log\n\t@exit 9"))
    result, calls = invoke(repo)
    assert result.returncode != 0
    assert calls == ["lint"]
