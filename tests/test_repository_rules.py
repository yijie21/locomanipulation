"""Mutation checks: the guards must detect actual violations and resolved debt."""

import importlib.util
import json
import tempfile
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "lint_repository", Path(__file__).resolve().parents[1] / "scripts/lint_repository.py"
)
rules = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rules)


def test_import_styles_and_local_imports_cannot_bypass_direction_check():
    filename = "src/locomani/backends/sonic/example.py"
    for code in (
        "import locomani.tasks.bottle as bottle",
        "from locomani import tasks",
        "from ...tasks.bottle import config",
        "def f():\n    from locomani.tasks.bottle.config import BottleTaskConfig",
    ):
        assert any(rules.forbidden(filename, target) for target in rules.imports(code, filename)), code
    assert not any(
        rules.forbidden(filename, target)
        for target in rules.imports("from locomani.primitives import arm_ik", filename)
    )


def test_debt_rejects_new_edges_and_requires_removing_fixed_exceptions():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "src/locomani/backends/sonic/example.py"
        path.parent.mkdir(parents=True)
        debt = root / rules.DEBT
        debt.parent.mkdir(parents=True)
        debt.write_text("[]")
        path.write_text("from locomani.tasks.bottle import config")
        assert "forbidden import" in rules.dependency_errors(root)[0]
        debt.write_text(
            json.dumps(
                [
                    dict(
                        source=path.relative_to(root).as_posix(),
                        target="locomani.tasks.bottle.config",
                        reason="pending migration",
                    )
                ]
            )
        )
        assert rules.dependency_errors(root) == []
        path.write_text("from locomani.primitives import arm_ik")
        assert "remove resolved exception" in rules.dependency_errors(root)[0]


def test_navigation_detects_broken_links_missing_tests_and_budget_overflow():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = root / "docs/reference/change-map.md"
        path.parent.mkdir(parents=True)
        path.write_text("[broken](missing.md)\n`test_missing.py`\nextra\n")
        errors = rules.navigation_errors(root, {"docs/reference/change-map.md": 2})
        assert len(errors) == 3
        (path.parent / "missing.md").touch()
        (root / "tests").mkdir()
        (root / "tests/test_missing.py").touch()
        assert rules.navigation_errors(root, {"docs/reference/change-map.md": 3}) == []


def test_runtime_paths_are_rejected_including_directory_replacing_symlinks():
    assert len(rules.artifact_errors(["outputs", "third_party", "datasets", "outputs/run/data.json"])) == 4
    assert rules.artifact_errors(["src/locomani/paths.py", "assets/robot.xml"]) == []


def test_changed_markdown_outside_entry_guides_is_checked(tmp_path):
    guide = tmp_path / "docs/task.md"
    guide.parent.mkdir()
    guide.write_text("[missing](missing.md)\n")
    assert "broken link" in rules.navigation_errors(tmp_path, {}, {"docs/task.md"})[0]
    (guide.parent / "missing.md").touch()
    assert rules.navigation_errors(tmp_path, {}, {"docs/task.md", "docs/deleted.md"}) == []
