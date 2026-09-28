"""The four-edge warehouse line: each round turns the walk-in to one side of the platform (visual.start_edges).

The warehouse layout below is the real one (platform 1.2 x 0.9 m centred at (0, -0.105), carton 0.45 x 0.32 m flush
with the -y edge, robot start facing +y) with the free floor warehouse_box_scene measured on 2026-09-28.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from locomani.sim.scene_edges import round_edge
from locomani.tasks.crate.expert import facing_edge, start_placement
from locomani.tasks.crate.visual_approach import approach_heading, edge_distance, near_edge
from locomani.tasks.crate.visual_config import CrateVisualConfig

EDGES = {
    "ym": {"heading_deg": 90.0, "clear_m": 1.285, "video_camera": "edge_ym"},
    "yp": {"heading_deg": -90.0, "clear_m": 1.629, "video_camera": "edge_yp"},
    "xm": {"heading_deg": 0.0, "clear_m": 1.112, "video_camera": "edge_xm"},
    "xp": {"heading_deg": 180.0, "clear_m": 1.048, "video_camera": "edge_xp"},
}
META = {
    "robot_start": {"xy": [0.0, -1.205], "yaw_deg": 90.00000000000001},
    "target_object": "nor_cal_box",
    "objects": {
        "nor_cal_box": {
            "surface": "A",
            "init_pos": [0.0, -0.395, 0.371],
            "init_quat": [1, 0, 0, 0],
            "box": {"half_extent": [0.225, 0.16, 0.165], "centre": [0.0, 0.0, 0.165]},
        }
    },
    "surfaces": {"A": {"xy": [0.0, -0.105], "flat_half_extent": [0.6, 0.45], "edges": EDGES}},
}
# the box_platform default's layout ranges with the envelope's "我打算用" column (user, 2026-09-28)
FOUR = CrateVisualConfig(
    position_jitter=0.0,
    yaw_jitter_deg=0.0,
    start_edge_min=0.6,
    start_edge_max=2.0,
    start_yaw_range_deg=(-20.0, 20.0),
    start_lateral_range=(-0.08, 0.0),
    object_setback_range=(0.0, 0.025),
    object_yaw_range_deg=(0.0, 3.5),
    start_edges=("xm", "xp", "ym", "yp"),
    object_along_range=(-1.0, 1.0),
)


class RecordingReset:
    def __init__(self):
        self.calls = []

    def reset(self, **kwargs):
        self.calls.append(kwargs)


def _round(cfg, seed, meta=META):
    rng = np.random.default_rng(seed)
    view = facing_edge(meta, cfg, rng)
    reset = RecordingReset()
    placement, record = start_placement(reset, view, cfg, rng)
    placement("initial_placement")
    return view, reset.calls[0], record, rng


def _box_pose(call):
    """World xy and yaw of the carton after the reset: the scene pose plus the reset's nudge."""
    obj = call["objects"]["nor_cal_box"]
    return np.asarray(META["objects"]["nor_cal_box"]["init_pos"][:2]) + obj["dxy"], obj["yaw"]


def test_no_start_edges_is_the_scene_itself_and_draws_nothing():
    rng = np.random.default_rng(5)
    assert facing_edge(META, CrateVisualConfig(), rng) is META
    assert rng.random() == np.random.default_rng(5).random()


def test_one_named_edge_draws_nothing_and_reads_that_side():
    rng = np.random.default_rng(5)
    view = facing_edge(META, replace(FOUR, start_edges=("xp",)), rng)
    assert rng.random() == np.random.default_rng(5).random()
    assert view["approach"]["side"] == "xp"
    assert approach_heading(view) == pytest.approx(math.pi)
    edge, axis = near_edge(view)
    np.testing.assert_allclose(edge, [0.6, -0.105])
    np.testing.assert_allclose(axis, [-1.0, 0.0], atol=1e-12)
    assert "approach" not in META  # the scene meta is never written


def test_the_scenes_own_edge_places_the_box_where_the_scene_has_it():
    cfg = replace(FOUR, start_edges=("ym",), object_along_range=(0.0, 0.0), object_setback_range=(0.0, 0.0))
    cfg = replace(cfg, object_yaw_range_deg=(0.0, 0.0))
    _view, call, record, _rng = _round(cfg, 1)
    np.testing.assert_array_equal(call["objects"]["nor_cal_box"]["dxy"], [0.0, 0.0])
    assert call["objects"]["nor_cal_box"]["yaw"] == 0.0
    assert record["edge"] == "ym"


@pytest.mark.parametrize("side", ["xm", "xp", "ym", "yp"])
def test_every_edge_carries_the_box_flush_to_it_and_starts_the_robot_in_front(side):
    cfg = replace(FOUR, start_edges=(side,))
    for seed in range(40):
        view, call, record, _rng = _round(cfg, seed)
        heading = approach_heading(view)
        edge, axis = near_edge(view)
        left = np.array([-axis[1], axis[0]])
        box_xy, box_yaw = _box_pose(call)
        # the box: its front face setback behind the edge (flush in the scene), square to the walk-in up to the
        # drawn yaw, and wholly on the edge
        assert axis @ (box_xy - edge) - 0.16 == pytest.approx(record["object_setback_m"], abs=1e-9)
        turn = (box_yaw - (heading - math.pi / 2) + math.pi) % (2 * math.pi) - math.pi
        assert math.degrees(turn) == pytest.approx(record["object_yaw_offset_deg"], abs=1e-9)
        half = 0.6 if side in ("ym", "yp") else 0.45
        assert abs(left @ (box_xy - edge)) + 0.225 <= half + 1e-9
        # the robot: before the edge by a distance its free floor allows, lateral -8..0 cm from the box centre line
        start = np.asarray(record["start_xy"])
        distance = edge_distance(view, start)
        assert 0.6 <= distance <= min(2.0, EDGES[side]["clear_m"] - 0.25) + 1e-9
        assert -0.08 - 1e-9 <= left @ (start - box_xy) <= 1e-9
        assert call["robot"]["yaw"] == pytest.approx(heading + math.radians(record["start_yaw_offset_deg"]))
        np.testing.assert_allclose(np.asarray(call["robot"]["dxy"]) + META["robot_start"]["xy"], start)


def test_the_draw_covers_all_four_edges_and_the_whole_edge_length():
    seen, along = set(), []
    for seed in range(400):
        _view, _call, record, _rng = _round(FOUR, seed)
        seen.add(record["edge"])
        if record["edge"] == "ym":
            along.append(record["object_along_m"])
    assert seen == {"xm", "xp", "ym", "yp"}
    assert min(along) < -0.3 and max(along) > 0.3  # +-0.375 m of room on the 1.2 m edge for a 0.45 m box


def test_an_edge_without_room_for_the_nearest_start_is_refused():
    with pytest.raises(ValueError, match="free floor"):
        facing_edge(META, replace(FOUR, start_edges=("xp",), start_edge_min=0.9), np.random.default_rng(0))


def test_a_scene_without_edges_is_refused():
    meta = {**META, "surfaces": {"A": {"xy": [0.0, -0.105], "flat_half_extent": [0.6, 0.45]}}}
    with pytest.raises(ValueError, match="lists its edges"):
        facing_edge(meta, FOUR, np.random.default_rng(0))


def test_start_edges_is_validated():
    with pytest.raises(ValueError, match="distinct sides"):
        replace(FOUR, start_edges=("ym", "north"))
    with pytest.raises(ValueError, match="distinct sides"):
        replace(FOUR, start_edges=("ym", "ym"))
    with pytest.raises(ValueError, match="start_pose sampled"):
        replace(FOUR, start_pose="scene")
    with pytest.raises(ValueError, match="-45, 45"):
        replace(FOUR, start_yaw_range_deg=(-50.0, 0.0))
    with pytest.raises(ValueError, match="within \\[-1, 1\\]"):
        replace(FOUR, object_along_range=(-1.5, 0.0))


@pytest.mark.parametrize("side", ["xm", "xp", "ym", "yp"])
def test_the_renderers_recover_the_edge_from_the_first_recorded_yaw(side):
    for seed in range(20):
        _view, call, record, _rng = _round(replace(FOUR, start_edges=(side,)), seed)
        yaw = call["robot"]["yaw"]
        qpos = [0.0, 0.0, 0.8, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        assert round_edge(META, qpos)["side"] == side
        assert round_edge(META, qpos)["video_camera"] == f"edge_{side}"
    assert round_edge({**META, "surfaces": {"A": {}}}, [0, 0, 0, 1, 0, 0, 0]) is None
