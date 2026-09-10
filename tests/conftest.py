"""Test tiers (docs/reference/standards.md): unit tests run everywhere in seconds;
`sim` tests need a live stack, `torch` tests need the GR00T venv. Both are skipped
unless asked for, so pre-commit and `make check` stay fast."""

import sys
from pathlib import Path

import pytest

# Every test imports `locomani.*` by its qualified package path (refactor step 4a);
# this is the one place "src" goes on sys.path instead of each test file doing it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

SIM_TEST_FILES = {"test_ready_pilot.py"}
TORCH_TEST_FILES = {"test_action_prefix_loss.py"}
# The frozen ScaleBFM line (decision 6, Codex-owned, src/teleop_free/scalebfm/) ships its own
# tests/test_bfm_*.py; they need that line's extra packages and are not part of this tier.
SCALEBFM_PREFIX = "test_bfm_"


def pytest_addoption(parser):
    parser.addoption("--run-sim", action="store_true", default=False, help="run tests that need a live stack")
    parser.addoption("--run-torch", action="store_true", default=False, help="run tests that need torch")
    parser.addoption(
        "--run-scalebfm", action="store_true", default=False, help="collect the frozen ScaleBFM line's tests"
    )


def pytest_ignore_collect(collection_path, config):
    return collection_path.name.startswith(SCALEBFM_PREFIX) and not config.getoption("--run-scalebfm")


def pytest_collection_modifyitems(config, items):
    for item in items:
        name = item.fspath.basename
        if name in SIM_TEST_FILES:
            item.add_marker(pytest.mark.sim)
            if not config.getoption("--run-sim"):
                item.add_marker(pytest.mark.skip(reason="needs a live stack (--run-sim)"))
        if name in TORCH_TEST_FILES:
            item.add_marker(pytest.mark.torch)
            if not config.getoption("--run-torch"):
                item.add_marker(pytest.mark.skip(reason="needs torch (--run-torch)"))
