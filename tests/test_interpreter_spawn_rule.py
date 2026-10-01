"""A subprocess is allowed only across a venv boundary, or for a reason written down.

The 2026-09-22 refactor deleted four drivers whose whole body was assembling a command line for a
second interpreter OF THE SAME VENV, which then parsed it straight back
(docs/handoff/flat-functional-scripts-20260922.md). Nothing stopped a fifth from being written,
so `lint_repository.spawn_errors` checks every command line built for a python interpreter against
`tests/fixtures/interpreter_spawns.json`, where each entry carries its reason.

A check that cannot fail is worse than no check, so the first test here rebuilds the deleted shape
and asserts it is rejected.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lint_repository as rules  # noqa: E402

ALLOWED = json.loads((ROOT / "tests/fixtures/interpreter_spawns.json").read_text())


@pytest.fixture
def tree(tmp_path):
    """A source tree carrying only this repository's allow-list."""
    (tmp_path / "src/locomani/pipeline").mkdir(parents=True)
    (tmp_path / "tests/fixtures").mkdir(parents=True)
    shutil.copy(ROOT / "tests/fixtures/interpreter_spawns.json", tmp_path / "tests/fixtures/interpreter_spawns.json")
    return tmp_path


def test_a_reintroduced_same_venv_driver_is_rejected(tree):
    """Exactly the shape that was deleted: an argv for the venv this process is already in."""
    (tree / "src/locomani/pipeline/run_collect_single_arm.py").write_text(
        "from locomani import paths\n"
        "SIM_PY = paths.VENV_SIM_PY\n"
        "def gym_argv(cfg):\n"
        "    return [SIM_PY, '/src/locomani/tasks/bottle/grasp_gym.py', '--rounds', '10']\n"
    )
    errors = [e for e in rules.spawn_errors(tree) if "run_collect_single_arm" in e]
    assert errors, "a same-venv driver was not rejected"
    assert "venv boundary" in errors[0]


def test_an_interpreter_hidden_behind_a_local_name_is_still_found(tree):
    """Naming the interpreter something else must not get a spawn past the rule."""
    (tree / "src/locomani/pipeline/sneaky.py").write_text(
        'ROOT = "/repo"\n'
        'WORKER = ROOT + "/third_party/GR00T-WholeBodyControl/.venv_sim/bin/python"\n'
        'def command():\n    return [WORKER, "-m", "locomani.tasks.bottle.collect"]\n'
    )
    assert any("sneaky" in e for e in rules.spawn_errors(tree))


def test_spawning_an_external_tool_is_never_flagged(tree):
    """The rule is about starting OUR code in another python, not about subprocesses."""
    (tree / "src/locomani/pipeline/tools.py").write_text(
        "import subprocess\n"
        "def free_memory():\n"
        '    return subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv"])\n'
        'def deploy_running():\n    return subprocess.run(["pgrep", "-af", "g1_deploy"])\n'
    )
    assert [e for e in rules.spawn_errors(tree) if "tools.py" in e] == []


def test_a_resolved_entry_is_reported_so_the_list_shrinks(tree):
    """Same discipline as the dependency debt list: stale exceptions fail."""
    stale = [e for e in rules.spawn_errors(tree) if "remove the resolved entry" in e]
    assert len(stale) == len(ALLOWED)


def test_every_allow_listed_spawn_still_exists_and_carries_a_reason():
    """The live repository: no unrecorded spawn, no stale entry, every reason written."""
    assert rules.spawn_errors(ROOT) == []
    assert all(row["reason"].strip() for row in ALLOWED)
    assert len({(row["source"], row["interpreter"]) for row in ALLOWED}) == len(ALLOWED)


def test_the_boundary_crossings_are_the_ones_we_expect():
    """The gr00t venv is the real boundary; no same-interpreter debt is left."""
    by_interpreter = {}
    for row in ALLOWED:
        by_interpreter.setdefault(row["interpreter"], set()).add(Path(row["source"]).name)
    assert by_interpreter["isaac_gr00t"] == {
        "run_train.py",
        "run_vla_server.py",
        "eval_crate_closed_loop.py",
    }
    # The last three (scene builders, boundary search) became in-process calls on 2026-10-01.
    assert "same-interpreter" not in by_interpreter
