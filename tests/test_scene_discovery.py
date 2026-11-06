"""Scene file v2 validation and goal predicates over an inline two-room scene (a yawed table, a bowl, a mug)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from locomani.scene_discovery import predicates, scene_file


def _quat_about_x(deg: float) -> list[float]:
    h = np.radians(deg) / 2
    return [float(np.cos(h)), float(np.sin(h)), 0.0, 0.0]


SEMANTIC = {"asset": "objects/Mug_1/3d-asset.glb", "phrase": "white mug", "room": "kitchen", "movable": True}
PROV = {"asset": "adt", "phrase": "claude", "room": "rule", "movable": "rule"}

META = {
    "schema_version": 2,
    "source": {"kind": "adt", "sequence": "Apartment_release_clean_seq131_M1292"},
    "robot_start": {"xy": [0.0, -1.0], "yaw_deg": 90.0},
    "surfaces": {
        # a 1.2 x 0.4 m table turned 45 deg: its far corner is outside any axis-aligned reading of half_extent
        "table": {
            "top": 0.75,
            "xy": [0.0, 0.0],
            "half_extent": [0.6, 0.2],
            "yaw": float(np.pi / 4),
            "room": "kitchen",
            "approach_edges": ["-y"],
            "provenance": {"yaw": "adt", "room": "rule", "approach_edges": "rule"},
        },
    },
    "objects": {
        "Mug_1": {
            "box": {"half_extent": [0.04, 0.04, 0.05], "centre": [0.0, 0.0, 0.0]},
            "surface": "table",
            "container": False,
            "semantic": dict(SEMANTIC, graspable="single", mass_kg=0.3, friction=1.0),
            "provenance": dict(PROV, graspable="rule", mass_kg="claude", friction="claude"),
        },
        "WhiteBowl": {
            "box": {"half_extent": [0.08, 0.08, 0.04], "centre": [0.0, 0.0, 0.0], "wall": 0.01},
            "surface": "table",
            "container": True,
            "semantic": dict(SEMANTIC, phrase="white bowl", graspable="single"),
            "provenance": dict(PROV, graspable="rule"),
        },
    },
    "room": {
        "bounds": {"x": [-3.0, 3.0], "y": [-3.0, 3.0]},
        "walls": [[[0.0, -3.0], [0.0, -2.0]]],
        "rooms": {
            "kitchen": {"polygon": [[-3, -3], [3, -3], [3, 1.5], [-3, 1.5]], "name_source": "claude"},
            "hall": {"polygon": [[-3, 1.5], [3, 1.5], [3, 3], [-3, 3]], "name_source": "claude"},
        },
        "spawn": {"kitchen": [[[-1, -2], [1, -2], [1, -1], [-1, -1]]]},
    },
    "randomization": {"movable_pool": ["Mug_1"], "placements": {"Mug_1": ["table"]}, "spawn_rooms": ["kitchen"]},
}


def state(mug=(0.0, 0.0, 0.80), mug_quat=(1, 0, 0, 0), touching=(), bowl=(0.3, 0.3, 0.79), pelvis=(0.0, -1.0, 0.79),
          gripped=(), bowl_quat=(1, 0, 0, 0)):  # fmt: skip
    return {
        "fallen": False,
        "pelvis": {"pos": list(pelvis)},
        "objects": {
            "Mug_1": {"pos": list(mug), "quat": list(mug_quat), "touching": list(touching), "gripped": list(gripped)},
            "WhiteBowl": {"pos": list(bowl), "quat": list(bowl_quat), "touching": []},
        },
    }


# -- scene file -----------------------------------------------------------------------------------


def test_a_complete_v2_file_validates_and_loads(tmp_path):
    assert scene_file.validate(META) == []
    path = tmp_path / "scene_meta.json"
    path.write_text(json.dumps(META))
    assert scene_file.load(path)["schema_version"] == 2


def test_validate_lists_every_problem_at_once():
    bad = copy.deepcopy(META)
    bad["schema_version"] = 1
    del bad["objects"]["Mug_1"]["provenance"]["mass_kg"]
    bad["objects"]["WhiteBowl"]["provenance"]["phrase"] = "guess"
    bad["objects"]["WhiteBowl"]["semantic"]["room"] = "garage"
    bad["surfaces"]["table"].pop("yaw")
    bad["randomization"]["placements"]["Mug_1"] = ["shelf"]
    problems = scene_file.validate(bad)
    assert len(problems) == 6, problems
    assert any("no provenance for 'mass_kg'" in p for p in problems)
    assert any("'guess'" in p for p in problems) and any("garage" in p for p in problems)
    assert any("yaw missing" in p for p in problems) and any("'shelf' is not a surface" in p for p in problems)


def test_load_refuses_a_file_with_problems(tmp_path):
    bad = copy.deepcopy(META)
    bad["room"]["rooms"]["hall"]["polygon"] = [[0, 0], [1, 1]]
    path = tmp_path / "scene_meta.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="polygon needs >= 3"):
        scene_file.load(path)


# -- predicates -----------------------------------------------------------------------------------


def test_parse_and_compile_reject_bad_goals_before_any_episode():
    assert predicates.parse("near(robot, table, 0.5)").args == ("robot", "table", 0.5)
    for goal, msg in [
        (["inside(Mug_1, WhiteBowl)"], "unknown predicate"),
        (["on(Mug_1)"], "takes 2 arguments"),
        (["on(Mug_2, table)"], "must be a object"),
        (["in(WhiteBowl, Mug_1)"], "must be a container"),
        (["in_room(robot, garage)"], "must be a room"),
        ([], "at least one"),
    ]:
        with pytest.raises(ValueError, match=msg):
            predicates.compile_goal(goal, META)


def test_on_uses_the_surface_yaw():
    check = predicates.compile_goal(["on(Mug_1, table)"], META)
    # 0.5 m along the table's own long axis: inside the yawed table, outside x in [-0.6, 0.6] x y in [-0.2, 0.2]
    along = 0.5 * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])
    assert check(state(mug=(along[0], along[1], 0.80))) == (True, "")
    ok, why = check(state(mug=(0.5, -0.3, 0.80)))
    assert not ok and "outside table's footprint" in why
    ok, why = check(state(mug=(0.0, 0.0, 1.20)))
    assert not ok and why.startswith("[goal] on(Mug_1, table): underside z")


def test_in_held_upright_near_in_room():
    assert predicates.compile_goal(["in(Mug_1, WhiteBowl)"], META)(state(mug=(0.3, 0.3, 0.80)))[0]
    assert not predicates.compile_goal(["in(Mug_1, WhiteBowl)"], META)(state())[0]
    held = predicates.compile_goal(["held(Mug_1)"], META)
    assert held(state(mug=(0.0, 0.0, 0.95), touching=["right"], gripped=["right"]))[0]
    assert held(state(mug=(0.0, 0.0, 0.95), touching=["left", "right"]))[0]  # a two-hand carry
    assert "between two fingers" in held(state(mug=(0.0, 0.0, 0.95), touching=["right"]))[1]  # hanging on one finger
    assert "resting on table" in held(state(mug=(0.0, 0.0, 0.80), touching=["right"], gripped=["right"]))[1]
    assert "no hand" in held(state(mug=(0.0, 0.0, 0.95)))[1]
    upright = predicates.compile_goal(["upright(Mug_1)"], META)
    assert upright(state(mug_quat=_quat_about_x(10)))[0]
    assert "tilted 30 deg" in upright(state(mug_quat=_quat_about_x(30)))[1]
    assert predicates.compile_goal(["near(robot, table, 1.1)"], META)(state())[0]
    assert not predicates.compile_goal(["near(robot, table, 0.9)"], META)(state())[0]
    in_room = predicates.compile_goal(["in_room(robot, kitchen)", "in_room(Mug_1, kitchen)"], META)
    assert in_room(state())[0]
    ok, why = in_room(state(pelvis=(0.0, 2.0, 0.79)))
    assert not ok and why.startswith("[goal] in_room(robot, kitchen)")


def test_in_reads_the_box_centres_of_object_and_container_turned_with_their_bodies():
    """A scanned object's body origin is its mesh origin, not its centre (2026-11-05): `in` judges where the box is."""
    import copy

    meta = copy.deepcopy(META)
    meta["objects"]["Mug_1"]["box"]["centre"] = [0.3, 0.0, 0.0]  # the box 30 cm along the body's x from its origin
    meta["objects"]["WhiteBowl"]["box"]["centre"] = [0.1, 0.0, 0.0]
    goal = predicates.compile_goal(["in(Mug_1, WhiteBowl)"], meta)
    # the bowl at (0.3, 0.3) unturned: its inner box is centred at (0.4, 0.3)
    assert goal(state(mug=(0.1, 0.3, 0.80)))[0]  # the mug's box centre at (0.4, 0.3)
    assert not goal(state(mug=(0.4, 0.3, 0.80)))[0]  # its origin there, its box 30 cm beyond the bowl
    assert goal(state(mug=(0.7, 0.3, 0.80), mug_quat=(0, 0, 0, 1)))[0]  # turned 180 deg: the box back at (0.4, 0.3)
    # the bowl turned 90 deg: its centre offset turns with it, the inner box is centred at (0.3, 0.4)
    quarter = (float(np.cos(np.pi / 4)), 0.0, 0.0, float(np.sin(np.pi / 4)))
    assert goal(state(mug=(0.0, 0.4, 0.80), bowl_quat=quarter))[0]
    assert not goal(state(mug=(0.1, 0.3, 0.80), bowl_quat=quarter))[0]


def test_a_fall_fails_every_goal():
    s = state(mug=(0.3, 0.3, 0.80))
    s["fallen"] = True
    assert predicates.compile_goal(["in(Mug_1, WhiteBowl)"], META)(s) == (False, "[goal] robot fell")
