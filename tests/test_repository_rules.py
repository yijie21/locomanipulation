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
        path.parent.mkdir(parents=True, exist_ok=True)
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
        path.parent.mkdir(parents=True, exist_ok=True)
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


def test_domain_maps_keep_link_and_test_checks(tmp_path):
    name = "docs/reference/change-map-crate.md"
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[owner](missing.py) `test_missing.py`\n")
    errors = rules.navigation_errors(tmp_path, {name: 35})
    assert any("broken link" in error for error in errors)
    assert any("missing tests/test_missing.py" in error for error in errors)
    (path.parent / "missing.py").touch()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_missing.py").touch()
    assert rules.navigation_errors(tmp_path, {name: 35}) == []


def test_router_cannot_hide_excessive_reading_in_one_long_line(tmp_path):
    name = "docs/reference/change-map.md"
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("腰" * (rules.BYTE_BUDGETS[name] // 3 + 1))
    assert len(path.read_text().splitlines()) == 1
    errors = rules.navigation_errors(tmp_path, {name: 40})
    assert len(errors) == 1 and "bytes" in errors[0]


def test_changed_markdown_outside_entry_guides_is_checked(tmp_path):
    guide = tmp_path / "docs/task.md"
    guide.parent.mkdir()
    guide.write_text("[missing](missing.md)\n")
    assert "broken link" in rules.navigation_errors(tmp_path, {}, {"docs/task.md"})[0]
    (guide.parent / "missing.md").touch()
    assert rules.navigation_errors(tmp_path, {}, {"docs/task.md", "docs/deleted.md"}) == []


def test_runbooks_may_only_name_paths_that_exist(tmp_path):
    name = rules.COMMAND_DOCUMENTS[0]
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)  # README.md's parent is the tree root itself
    path.write_text(
        "见 `scripts/run_wall_pilot.sh`：\n\n"
        "    bash scripts/single_arm/run_agent_server.sh configs/experiments/x.yaml --gpu 1\n\n"
        "产物在 outputs/runs/<name>/collect/，模板见 configs/experiments/<name>.yaml，"
        "代码在 src/locomani/ 下。\n"
    )
    errors = rules.navigation_errors(tmp_path, {name: 180})
    named = {error.split("names ")[1].split(",")[0] for error in errors if "names " in error}
    # `outputs/` is runtime, so it is never checked. A path interrupted by a `<name>`
    # placeholder truncates to its parent directory, which is checked as a directory --
    # not a false positive, and it still catches a renamed `configs/` or `src/` tree.
    assert named == {
        "scripts/run_wall_pilot.sh",
        "scripts/single_arm/run_agent_server.sh",
        "configs/experiments/x.yaml",
        "configs/experiments",
        "src/locomani",
    }
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_wall_pilot.sh").touch()
    (tmp_path / "scripts/single_arm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts/single_arm/run_agent_server.sh").touch()
    (tmp_path / "configs/experiments").mkdir(parents=True)
    (tmp_path / "configs/experiments/x.yaml").touch()
    (tmp_path / "src/locomani").mkdir(parents=True)
    assert rules.navigation_errors(tmp_path, {name: 180}) == []


def test_a_runbook_may_name_the_experiment_file_it_tells_you_to_create(tmp_path):
    name = rules.COMMAND_DOCUMENTS[0]
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "    cp configs/experiments/real.yaml configs/experiments/mine.yaml\n"
        "    bash scripts/run_overnight.sh configs/experiments/mine.yaml --gpu 1\n"
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run_overnight.sh").touch()
    (tmp_path / "configs/experiments").mkdir(parents=True)
    (tmp_path / "configs/experiments/real.yaml").touch()
    assert rules.navigation_errors(tmp_path, {name: 180}) == []
    # ... but only the copy's destination; the template it copies still has to be there.
    (tmp_path / "configs/experiments/real.yaml").unlink()
    errors = rules.navigation_errors(tmp_path, {name: 180})
    assert len(errors) == 1 and "configs/experiments/real.yaml" in errors[0]


def test_a_runbook_may_only_name_output_that_something_prints(tmp_path):
    name = rules.COMMAND_DOCUMENTS[0]
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts/recipe.py").write_text(
        '"""A docstring mentioning Saving is not output."""\n'
        "print(f\"  finger pads {'HARDENED (priority 1)' if hard else 'SOFT'}\")\n"
        "# a comment mentioning Vanished is not output either\n"
    )
    path.write_text(
        "装完回显 `finger pads HARDENED`；训练日志里找 `Saving checkpoint` 和 `Vanished marker`。\n"
        "命令 `python -m locomani.stack.cli status --instance N` 和配置 `max_batches: 1` 不算输出。\n"
    )
    errors = rules.navigation_errors(tmp_path, {name: 180})
    assert len(errors) == 2, errors
    # Assembled at runtime, so the phrase is never contiguous -- but every word is there.
    assert not any("finger pads" in e for e in errors)
    assert any("`Saving checkpoint` names" in e and "'Saving'" in e for e in errors)
    assert any("`Vanished marker` names" in e and "'Vanished'" in e for e in errors)
