"""Existing readers on a v2 scene file: the navigator's map, the stand plan's edge, SceneModel's put-on check.

A hand-built scene (no `room`, no `yaw`) must read exactly as before; a scene_discovery one brings its own
bounds, walls, furniture footprints and turned tables (docs/handoff/scene-file-format-20261003.md §8).
"""

from __future__ import annotations

import numpy as np
import pytest

from locomani.primitives.navigator import ROOM_XLIM, ROOM_YLIM, OccupancyMap
from locomani.skills.scene_model import SceneModel, TaskInstance
from locomani.tasks.bottle.plan import nearest_edge

TABLE = {"top": 0.75, "xy": [0.0, 0.0], "half_extent": [0.6, 0.2]}


def test_a_hand_built_scene_keeps_the_fixed_room_box_and_axis_aligned_tables():
    occ = OccupancyMap({"surfaces": {"A": TABLE}})
    assert occ.origin.tolist() == [ROOM_XLIM[0], ROOM_YLIM[0]]
    assert not occ.is_free([0.0, 0.0]) and occ.occupied[tuple(occ.cell_of([0.55, 0.0]))]
    assert not occ.occupied[tuple(occ.cell_of([0.0, 0.5]))]


def test_a_scene_file_brings_bounds_walls_obstacles_and_turned_tables():
    meta = {
        "surfaces": {"T": dict(TABLE, yaw=float(np.pi / 2))},  # the 1.2 m side now runs along y
        "room": {
            "bounds": {"x": [-2.0, 6.0], "y": [-3.0, 3.0]},
            "walls": [[[3.0, -3.0], [3.0, 3.0]]],
            "obstacles": [[[4.5, -0.5], [5.5, -0.5], [5.5, 0.5], [4.5, 0.5]]],
        },
    }
    occ = OccupancyMap(meta)
    assert occ.origin.tolist() == [-2.0, -3.0]
    at = lambda x, y: bool(occ.occupied[tuple(occ.cell_of([x, y]))])  # noqa: E731
    assert at(0.0, 0.55) and not at(0.55, 0.0)  # turned table
    assert at(3.0, 1.0) and at(3.0, -2.5) and not at(3.3, 1.0)  # 10 cm wall
    assert at(5.0, 0.0) and not at(5.0, 0.8)  # sofa footprint


def test_the_stand_edge_is_found_in_the_table_frame_and_only_among_open_edges():
    assert nearest_edge([0.5, 0.05], TABLE) == (0, 1.0)  # near +x of an axis-aligned table
    assert nearest_edge([0.1, -0.15], TABLE) == (1, -1.0)
    turned = dict(TABLE, yaw=float(np.pi / 2))
    assert nearest_edge([0.05, 0.5], turned) == (0, 1.0)  # world +y is the turned table's +x
    against_wall = dict(TABLE, approach_edges=["-y"])
    assert nearest_edge([0.5, 0.05], against_wall) == (1, -1.0)  # +x is nearer but blocked
    with pytest.raises(ValueError, match="no edge"):
        nearest_edge([0.0, 0.0], dict(TABLE, approach_edges=[]))


def test_put_on_judges_a_turned_table_in_its_own_frame():
    meta = {
        "surfaces": {"T": dict(TABLE, yaw=float(np.pi / 2))},
        "objects": {"mug": {"box": {"half_extent": [0.04, 0.04, 0.05], "centre": [0, 0, 0]}, "surface": "T"}},
    }
    scene = SceneModel(meta)
    check = scene.criterion(TaskInstance("put <obj> on <surface>", "mug", "T", "on", "bottle_simple", ""))
    on = {"fallen": False, "objects": {"mug": {"pos": [0.0, 0.5, 0.80]}}}
    off = {"fallen": False, "objects": {"mug": {"pos": [0.5, 0.0, 0.80]}}}
    assert check(on) == (True, "")
    assert not check(off)[0]
    assert scene.front_line("T")["yaw"] == pytest.approx(np.pi / 2)
