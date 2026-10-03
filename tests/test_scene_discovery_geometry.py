"""Floor plan and support-surface geometry on synthetic inputs: a two-room plan with a 0.9 m doorway, tilted tops."""

from __future__ import annotations

import numpy as np
import pytest

from locomani.scene_discovery import floorplan, surfaces

BOUNDS = {"x": [-1.0, 9.0], "y": [-1.0, 5.0]}
# 8 x 4 m outer wall, an inner wall at x = 4 with a 0.9 m doorway (y 1.5-2.4), a window gap in the outer wall
# would leak the flood fill, so the outer loop is closed
WALLS = [
    [[0, 0], [8, 0]],
    [[8, 0], [8, 4]],
    [[8, 4], [0, 4]],
    [[0, 4], [0, 0]],
    [[4, 0], [4, 1.5]],
    [[4, 2.4], [4, 4]],
]


def plan():
    grid = floorplan.Grid(BOUNDS)
    walls = grid.segments(WALLS, thickness=0.1)
    return grid, walls, floorplan.segment_rooms(walls)


def test_a_doorway_splits_two_rooms_and_each_gets_a_polygon():
    grid, walls, labels = plan()
    rooms = sorted(int(k) for k in np.unique(labels[labels > 0]))
    assert rooms == [1, 2]
    for k in rooms:
        poly = np.asarray(grid.contour(labels == k))
        assert len(poly) >= 4
        assert poly[:, 0].max() - poly[:, 0].min() == pytest.approx(3.9, abs=0.2)  # one 4 m room, wall excluded
    # the doorway cells belong to one room or the other, nothing is left unassigned inside
    inside = floorplan.interior(walls)
    assert not (inside & (labels == 0)).any()


def test_an_entrance_gap_in_the_outer_wall_needs_the_floor_mask():
    grid = floorplan.Grid(BOUNDS)
    walls = grid.segments([[[0, 0], [7, 0]]] + WALLS[1:], thickness=0.1)  # 1 m entrance at x 7-8 on the bottom wall
    assert not (floorplan.segment_rooms(walls) > 0).any()  # the flood fill leaks through the entrance
    floor = grid.polygons([[[0, 0], [8, 0], [8, 4], [0, 4]]])
    assert sorted(np.unique(floorplan.segment_rooms(walls, inside=floor)).tolist()) == [0, 1, 2]


def test_a_wide_opening_does_not_split():
    grid = floorplan.Grid(BOUNDS)
    walls = grid.segments(WALLS[:4] + [[[4, 0], [4, 1.0]], [[4, 3.0], [4, 4]]], thickness=0.1)
    labels = floorplan.segment_rooms(walls)  # 2 m opening > door_close 1.0
    assert sorted(np.unique(labels[labels > 0]).tolist()) == [1]


def test_spawn_regions_keep_clear_of_walls_and_furniture():
    grid, walls, labels = plan()
    table = grid.polygons([[[1, 1], [3, 1], [3, 3], [1, 3]]])
    clear = floorplan.clearance(walls | table)
    spawn = floorplan.spawn_regions(labels, clear, need=0.4, grid=grid)
    assert set(spawn) == {1, 2}
    left = 1 if np.asarray(grid.contour(labels == 1))[:, 0].mean() < 4 else 2
    pts = np.vstack([np.asarray(p) for p in spawn[left]])
    assert pts[:, 0].min() >= 0.35 and pts[:, 1].min() >= 0.35  # >= 0.4 m from the outer wall (one-cell slack)
    inside_table = (pts[:, 0] > 1.2) & (pts[:, 0] < 2.8) & (pts[:, 1] > 1.2) & (pts[:, 1] < 2.8)
    assert not inside_table.any()


def test_approach_edges_drop_the_side_against_a_wall():
    grid, walls, _ = plan()
    rect = {"xy": [6.0, 3.6], "half_extent": [0.6, 0.3], "yaw": 0.0}  # a sideboard against the top wall
    table = grid.polygons([[[5.4, 3.3], [6.6, 3.3], [6.6, 3.9], [5.4, 3.9]]])
    edges = floorplan.approach_edges(rect, floorplan.clearance(walls | table), grid, body_radius=0.26)
    assert "-y" in edges and "+y" not in edges


def _box_top(cx, cy, lx, ly, yaw, z):
    """Two triangles of a rectangle top, rotated by yaw, at height z."""
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[-lx, -ly], [lx, -ly], [lx, ly], [-lx, ly]]) @ np.array([[c, s], [-s, c]]) + [cx, cy]
    p = np.column_stack([corners, np.full(4, z)])
    return np.array([[p[0], p[1], p[2]], [p[0], p[2], p[3]]])


def test_a_tilted_table_top_comes_back_as_one_oriented_rectangle():
    tris = np.concatenate([_box_top(1.0, 2.0, 0.6, 0.3, np.radians(30), 0.75), _box_top(1.0, 2.0, 0.6, 0.3, 0, 0.0)])
    regions = surfaces.horizontal_regions(tris)  # the z = 0 copy is below z_range and dropped
    assert len(regions) == 1
    r = regions[0]
    assert r["top"] == pytest.approx(0.75)
    assert r["xy"] == pytest.approx([1.0, 2.0], abs=0.02)
    assert r["half_extent"] == pytest.approx([0.6, 0.3], abs=0.02)
    assert r["yaw"] == pytest.approx(np.radians(30), abs=np.radians(2))
    assert r["fill"] > 0.9


def test_shelf_boards_split_by_height_and_small_or_downward_faces_drop():
    boards = [_box_top(0, 0, 0.4, 0.15, 0, z) for z in (0.30, 0.302, 0.70)]  # 0.30 and 0.302 are one board
    tiny = _box_top(2, 2, 0.05, 0.05, 0, 0.5)
    down = _box_top(4, 4, 0.5, 0.5, 0, 0.9)[:, ::-1]  # reversed winding: normal points down
    regions = surfaces.horizontal_regions(np.concatenate(boards + [tiny, down]))
    assert sorted(r["top"] for r in regions) == pytest.approx([0.301, 0.70], abs=0.002)


def test_a_seat_inside_its_sofa_still_gets_its_front_edge():
    grid, walls, _ = plan()
    sofa = grid.polygons([[[5.0, 3.0], [7.0, 3.0], [7.0, 3.9], [5.0, 3.9]]])  # against the top wall
    seat = {"xy": [6.0, 3.4], "half_extent": [0.8, 0.25], "yaw": 0.0}  # 0.15 m inside the sofa's front
    clear = floorplan.clearance(walls | sofa)
    assert floorplan.approach_edges(seat, clear, grid, body_radius=0.26) == []
    edges = floorplan.approach_edges(seat, clear, grid, body_radius=0.26, own=sofa)
    assert "-y" in edges and "+y" not in edges  # the front, never the wall side; the arm ends have floor beside them


def test_stairs_drop_and_a_two_level_counter_stays():
    steps = [{"top": 0.18 * (k + 1), "xy": [0.0, 0.3 * k], "half_extent": [0.5, 0.15], "yaw": 0.0} for k in range(4)]
    counter = [  # counter | sink basin a riser lower | lower counter section: a riser pair, not a flight
        {"top": 0.92, "xy": [3.0, 0.0], "half_extent": [0.8, 0.3], "yaw": 0.0},
        {"top": 0.73, "xy": [4.05, 0.0], "half_extent": [0.25, 0.2], "yaw": 0.0},
        {"top": 0.88, "xy": [4.6, 0.0], "half_extent": [0.3, 0.3], "yaw": 0.0},
    ]
    kept = surfaces.drop_stairs(steps + counter)
    assert [r["top"] for r in kept] == [0.92, 0.73, 0.88]


def test_an_edge_needs_a_runup_not_just_a_stand():
    grid, walls, _ = plan()
    rect = {"xy": [6.0, 2.0], "half_extent": [0.4, 0.3], "yaw": 0.0}  # a cart in the right room
    chair = grid.polygons([[[4.35, 1.7], [4.85, 1.7], [4.85, 2.3], [4.35, 2.3]]])  # 0.75 m off its -x edge
    clear = floorplan.clearance(walls | chair)
    assert "-x" in floorplan.approach_edges(rect, clear, grid, body_radius=0.26)
    assert "-x" not in floorplan.approach_edges(rect, clear, grid, body_radius=0.26, runup=0.6)
