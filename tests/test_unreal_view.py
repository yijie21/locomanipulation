"""unreal.view: which sim joint a URLab joint is. URLab prefixes every name with its actor, so the match is by suffix;
it must take the most specific sim name, never whichever the scan meets first (review of 30cb422, 2026-09-29)."""

import pytest

pytest.importorskip("mujoco")

from locomani.sim.unreal.view import sim_joint  # noqa: E402


def test_the_exact_name_wins_then_the_longest_suffix():
    sim = ["hip_joint", "left_hip_joint", "knee_joint", "nor_cal_box_object"]
    assert sim_joint("left_hip_joint", sim) == "left_hip_joint"
    assert sim_joint("g1_left_hip_joint", sim) == "left_hip_joint"
    assert sim_joint("g1_left_hip_joint", list(reversed(sim))) == "left_hip_joint"  # order does not matter
    assert sim_joint("dynamic_C_0/knee_joint", sim) == "knee_joint"


def test_a_joint_the_sim_does_not_have_is_none():
    assert sim_joint("g1_elbow_joint", ["hip_joint"]) is None
    assert sim_joint("xhip_joint", ["hip_joint"]) is None  # a suffix must follow a separator
