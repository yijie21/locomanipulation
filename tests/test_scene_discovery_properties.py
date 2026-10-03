"""The rules around Claude's estimates: clamps, refusals, unique names and room renaming (replies are faked)."""

from __future__ import annotations

import copy

from locomani.scene_discovery import properties, scene_file


def obj(category, half, movable=True, room="room_1"):
    return {
        "box": {"half_extent": list(half), "centre": [0.0, 0.0, 0.0]},
        "surface": None,
        "container": False,
        "semantic": {
            "asset": "x.glb",
            "adt": {"category": category, "prototype_name": category.title()},
            "room": room,
            "movable": movable,
            "graspable": "single" if movable else "none",
        },
        "provenance": {"asset": "adt", "adt": "adt", "room": "rule", "movable": "rule", "graspable": "rule"},
    }


META = {
    "schema_version": 2,
    "source": {"kind": "adt"},
    "robot_start": {"xy": [0.0, 0.0]},
    "surfaces": {
        "KitchIsland__0": {
            "top": 0.9,
            "xy": [0, 0],
            "half_extent": [0.6, 0.4],
            "yaw": 0.0,
            "of_object": "KitchIsland",
            "room": "room_1",
            "name": "table KitchIsland",
            "provenance": {"yaw": "rule", "of_object": "rule", "room": "rule"},
        },  # fmt: skip
    },
    "objects": {
        "KitchIsland": obj("table", (0.63, 0.4, 0.45), movable=False),
        "WoodenBowl": obj("bowl", (0.14, 0.14, 0.04)),
        "SpiceJar": obj("jar", (0.02, 0.02, 0.05)),
        "Mug_1": obj("cup", (0.05, 0.04, 0.05)),
        "Mug_2": obj("cup", (0.05, 0.04, 0.05)),
        "Mug_3": obj("cup", (0.05, 0.04, 0.05), room="room_2"),
    },
    "room": {
        "bounds": {"x": [-5, 5], "y": [-5, 5]},
        "rooms": {
            "room_1": {"polygon": [[-5, -5], [0, -5], [0, 5], [-5, 5]], "name_source": "rule"},
            "room_2": {"polygon": [[0, -5], [5, -5], [5, 5], [0, 5]], "name_source": "rule"},
        },
        "spawn": {"room_1": [], "room_2": []},
    },
    "randomization": {"spawn_rooms": ["room_1", "room_2"]},
}

REPLY = [
    {"key": "KitchIsland", "phrase": "the kitchen island", "mass_kg": 120.0, "friction": 0.6, "container": False},
    {"key": "WoodenBowl", "phrase": "the wooden bowl", "mass_kg": 0.4, "friction": 0.6, "container": True},
    {"key": "SpiceJar", "phrase": "the spice jar", "mass_kg": 0.08, "friction": 3.5, "container": True},
    {"key": "Mug_1", "phrase": "the white mug", "mass_kg": 0.3, "friction": 0.4, "container": False},
    {"key": "Mug_2", "phrase": "The white mug", "mass_kg": 0.3, "friction": 0.4, "container": False},
    {"key": "Mug_3", "phrase": "the white mug", "mass_kg": 40.0, "friction": 0.4, "container": False},
]


def test_estimates_land_with_provenance_and_the_rules_correct_what_cannot_hold():
    meta = copy.deepcopy(META)
    notes = properties.apply_objects(meta, REPLY)
    bowl, jar = meta["objects"]["WoodenBowl"], meta["objects"]["SpiceJar"]
    assert bowl["container"] is True and bowl["box"]["wall"] == properties.CONTAINER_WALL
    assert bowl["semantic"]["container"]["opening_half_extent"] == [0.13, 0.13]
    assert bowl["provenance"]["container"] == "claude"
    assert jar["container"] is False and jar["provenance"]["container"] == "rule"  # 4 cm across: holds nothing
    assert jar["semantic"]["friction"] == 2.0
    assert any("SpiceJar: friction 3.5 clamped" in n for n in notes)
    assert any(n.startswith("Mug_3: 40.000 kg") for n in notes)  # implausible density reported, value kept
    assert meta["objects"]["Mug_3"]["semantic"]["mass_kg"] == 40.0
    # Mug_2 repeats Mug_1's phrase in room_1 (case-insensitively) and gets a suffix; Mug_3 is in another room
    assert meta["objects"]["Mug_2"]["semantic"]["phrase"] == "The white mug 2"
    assert meta["objects"]["Mug_2"]["provenance"]["phrase"] == "rule"
    assert meta["objects"]["Mug_3"]["semantic"]["phrase"] == "the white mug"
    assert scene_file.validate(meta) == []


def test_rows_for_claude_carry_what_it_needs_and_skip_the_shell():
    meta = copy.deepcopy(META)
    meta["objects"]["ApartmentEnv"] = obj("shelter", (5, 5, 3), movable=False, room=None)
    rows = properties.object_rows(meta)
    assert "ApartmentEnv" not in {r["key"] for r in rows}
    bowl = next(r for r in rows if r["key"] == "WoodenBowl")
    assert bowl == {"key": "WoodenBowl", "category": "bowl", "prototype": "Bowl", "size_m": [0.28, 0.28, 0.08],
                    "room": "room_1", "fixed": False}  # fmt: skip
    surface = properties.surface_rows(meta, {"KitchIsland": "the kitchen island"})[0]
    assert surface["of"] == "the kitchen island" and surface["top_m"] == 0.9
    assert properties.room_rows(meta)[0] == {"key": "room_1", "area_m2": 50.0, "furniture": ["table"]}


def test_rooms_rename_everywhere_or_not_at_all():
    meta = copy.deepcopy(META)
    assert properties.rename_rooms(meta, [{"key": "room_1", "name": "kitchen"}])  # room_2 missing: refused
    assert set(meta["room"]["rooms"]) == {"room_1", "room_2"}
    assert properties.rename_rooms(meta, [{"key": "room_1", "name": "kitchen"}, {"key": "room_2", "name": "kitchen"}])
    assert (
        properties.rename_rooms(meta, [{"key": "room_1", "name": "kitchen"}, {"key": "room_2", "name": "bedroom"}])
        == []
    )
    assert set(meta["room"]["rooms"]) == {"kitchen", "bedroom"}
    assert meta["room"]["rooms"]["kitchen"]["name_source"] == "claude"
    assert meta["objects"]["Mug_3"]["semantic"]["room"] == "bedroom"
    assert meta["surfaces"]["KitchIsland__0"]["room"] == "kitchen"
    assert meta["randomization"]["spawn_rooms"] == ["kitchen", "bedroom"]
    assert scene_file.validate(meta) == []


def test_surface_names_apply_and_missing_ones_are_reported():
    meta = copy.deepcopy(META)
    notes = properties.apply_surfaces(meta, [{"key": "Nope__0", "phrase": "the floor"}])
    assert any("unknown surface" in n for n in notes) and any("no name for 1 surfaces" in n for n in notes)
    assert properties.apply_surfaces(meta, [{"key": "KitchIsland__0", "phrase": "the kitchen island"}]) == []
    assert meta["surfaces"]["KitchIsland__0"]["provenance"]["phrase"] == "claude"
