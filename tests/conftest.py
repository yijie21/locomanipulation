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
# test_visual_training / test_visual_validation import gr00t + omegaconf + pandas (the GR00T venv), the
# rest of the ScaleBFM line's tests run in .venv_sim like every other unit test (2026-09-10 reorg).
TORCH_TEST_FILES = {
    "test_action_prefix_loss.py",
    "test_backbone_path.py",
    "test_visual_training.py",
    "test_visual_validation.py",
}


def pytest_addoption(parser):
    parser.addoption("--run-sim", action="store_true", default=False, help="run tests that need a live stack")
    parser.addoption("--run-torch", action="store_true", default=False, help="run tests that need torch")


@pytest.fixture
def bottle_geometry(monkeypatch):
    """Opt-in fixtures for geometry callers that construct their own IK helper."""
    import numpy as np
    from geometry_fixture import DIRECTORY, scene_ik

    from locomani.tasks.bottle import adaptive_grasp, ready_grasp_policy

    monkeypatch.setattr(ready_grasp_policy, "ArmIK", scene_ik)
    monkeypatch.setattr(adaptive_grasp, "_POINTS", np.load(DIRECTORY / "hand_points_dex1.npy"))
    ready_grasp_policy.templates.cache_clear()
    ready_grasp_policy.lower_grip_samples.cache_clear()
    yield
    ready_grasp_policy.templates.cache_clear()
    ready_grasp_policy.lower_grip_samples.cache_clear()


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
