"""Grasp profiles resolve to the same numbers the underlying task configs already carry."""

import pytest

from locomani.skills.profiles import PROFILE_SOURCES, GraspProfile, load_profile, load_profiles


def test_profile_sources_lists_exactly_three_profiles():
    assert set(PROFILE_SOURCES) == {"crate_low_table", "warehouse_box", "bottle_simple"}


def test_load_profiles_returns_one_graspprofile_per_source():
    profiles = load_profiles()
    assert set(profiles) == set(PROFILE_SOURCES)
    assert all(isinstance(p, GraspProfile) for p in profiles.values())


def test_crate_low_table_matches_configs_task_crate_visual_low_reach():
    profile = load_profile("crate_low_table")
    assert profile.approach["stand_off"] == pytest.approx(0.39)
    assert profile.approach["walk_brake_margin"] == pytest.approx(0.1)
    assert profile.grasp["close_seconds"] == pytest.approx(0.8)
    # not overridden by the profile yaml, so it is CrateVisualConfig's code default
    assert profile.ready["ready_pose"] == "forward"


def test_warehouse_box_matches_box_centre_base_20260927():
    profile = load_profile("warehouse_box")
    assert profile.approach["approach_stop"] == "guard"
    assert profile.approach["stand_off"] == pytest.approx(0.5)
    assert profile.ready["ready_pose"] == "lifted"
    assert profile.ready["elbow_min_deg"] == pytest.approx(-60.0)
    assert profile.grasp["close_trigger"] == "contact"
    assert profile.grasp["close_target"] == pytest.approx(-0.06)


def test_bottle_simple_matches_bottle_simple_200():
    profile = load_profile("bottle_simple")
    assert profile.approach["approach_stop"] == "guard"
    assert profile.approach["walk_speed"] == pytest.approx(0.25)
    assert profile.ready["ready_pitch_deg"] == pytest.approx(30.0)
    assert profile.grasp["lift_height_m"] == pytest.approx(0.08)


def test_profile_dicts_are_read_only():
    profile = load_profile("bottle_simple")
    with pytest.raises(TypeError):
        profile.approach["walk_speed"] = 1.0


def test_unknown_profile_name_raises_key_error():
    with pytest.raises(KeyError):
        load_profile("no_such_profile")


def test_profiles_carry_hands_and_the_resolved_task_config():
    profiles = load_profiles()
    assert profiles["bottle_simple"].hands == "single" and profiles["warehouse_box"].hands == "bimanual"
    assert profiles["warehouse_box"].config.visual.close_trigger == "contact"
    assert profiles["warehouse_box"].grasp["carry"]["waist_dofs"] == profiles["warehouse_box"].config.carry.waist_dofs
