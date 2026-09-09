"""Test tiers (docs/reference/standards.md): unit tests run everywhere in seconds;
`sim` tests need a live stack, `torch` tests need the GR00T venv. Both are skipped
unless asked for, so pre-commit and `make check` stay fast."""

import pytest

SIM_TEST_FILES = {"test_ready_pilot.py"}
TORCH_TEST_FILES = {"test_action_prefix_loss.py"}


def pytest_addoption(parser):
    parser.addoption("--run-sim", action="store_true", default=False, help="run tests that need a live stack")
    parser.addoption("--run-torch", action="store_true", default=False, help="run tests that need torch")


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
