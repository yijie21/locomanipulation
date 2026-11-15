"""scene_discovery.scan_room (TODO 38.1): a scanned room's `room` block from a top-down occupancy grid."""

import numpy as np
import pytest

from locomani.scene_discovery import scan_room


def grid():
    """A 6 x 6 m room on a 5 cm grid: its four walls one cell thick, a 1 x 1 m rack in the north-east corner."""
    n = 120
    occ = np.zeros((n, n), dtype=bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    occ[90:110, 90:110] = True
    return occ, np.array([-3.0, -3.0]), 0.05


def test_rectangles_cover_exactly_the_true_cells_and_merge_runs():
    occ, lo, res = grid()
    rects = scan_room.rectangles(occ, lo, res)
    # the two long walls along y are one rectangle each, the rack one, the two walls along x split by the corners
    assert len(rects) <= 8
    back = np.zeros_like(occ)
    for (xa, ya), _, (xb, yb), _ in rects:
        i0, j0 = int(round((xa - lo[0]) / res)), int(round((ya - lo[1]) / res))
        i1, j1 = int(round((xb - lo[0]) / res)), int(round((yb - lo[1]) / res))
        back[i0:i1, j0:j1] = True
    np.testing.assert_array_equal(back, occ)
    assert any(xb - xa == pytest.approx(1.0) and yb - ya == pytest.approx(1.0) for (xa, ya), _, (xb, yb), _ in rects)


def test_room_block_spawns_only_clear_of_walls_and_rack_in_the_start_region():
    occ, lo, res = grid()
    room = scan_room.room_block(occ, lo, res, (0.0, -1.0), body_radius=0.35, margin=0.10, name="r")
    assert room["bounds"] == {"x": [-3.0, 3.0], "y": [-3.0, 3.0]}
    assert room["walls"] == [] and set(room["spawn"]) == {"r"} and set(room["rooms"]) == {"r"}
    from locomani.tasks.scene.start import _inside

    polys = room["spawn"]["r"]
    inside = lambda xy: any(_inside(p, xy) for p in polys)  # noqa: E731
    assert inside((0.0, -1.0)) and inside((-2.4, -2.4))
    assert not inside((-2.7, 0.0)), "0.3 m from a wall is inside the body radius"
    assert not inside((1.3, 1.3)), "0.2 m from the rack"
    assert not inside((2.0, 2.0)), "the rack itself"
    assert all(xa >= -2.55 and xb <= 2.55 for (xa, _), _, (xb, _), _ in polys)
    poly = room["rooms"]["r"]["polygon"]
    assert min(x for x, _ in poly) <= -2.5 and max(x for x, _ in poly) >= 2.5


def test_room_block_falls_back_to_the_nearest_clear_region_when_the_start_is_not_clear():
    occ, lo, res = grid()
    occ[:, 60] = True  # a wall across the room: two regions
    room = scan_room.room_block(occ, lo, res, (0.0, 1.0), body_radius=0.35, name="r")
    from locomani.tasks.scene.start import _inside

    polys = room["spawn"]["r"]
    assert any(_inside(p, (0.0, 1.5)) for p in polys)
    assert not any(_inside(p, (0.0, -1.5)) for p in polys), "the start's side of the wall only"
    blocked = scan_room.room_block(occ, lo, res, (0.0, 0.02), body_radius=0.35, name="r")  # on the wall itself
    assert blocked["spawn"]["r"]


def test_sample_edges_fills_a_panel_the_vertices_alone_would_not():
    a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    b = np.array([[2.0, 0.0, 2.0], [0.0, 0.0, 2.0]])  # the two diagonals of a 2 x 2 m vertical panel
    p = scan_room._sample_edges(a, b, (0.1, 1.5), 0.025)
    assert p[:, 2].min() > 0.1 and p[:, 2].max() < 1.5
    xs = np.sort(p[:, 0])
    assert np.diff(xs).max() < 0.05, "no 5 cm cell along the panel is left empty"
    assert len(scan_room._sample_edges(a, b, (3.0, 4.0), 0.025)) == 0
